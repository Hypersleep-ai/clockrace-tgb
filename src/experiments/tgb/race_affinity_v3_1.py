#!/usr/bin/env python3
"""race_affinity v3.1 (3 Nov 2026) = race_affinity_v3.py plus --late-clamp {on,off}.

Why: tgbn-genre fires one TRAIN label time late (label time index 320: 19 labelled users already hold edges up to 12.5 h
after it). The genre model race_affinity_v2.py computes those users' features with the negative gap; the large-dataset
driver race_affinity_v2_large_v2.py (change 5), and therefore v3, evaluates them at their last edge time. So v3 with the
clocks off equals the large driver exactly but moves the genre v2 validation history by 3.3e-6, and the pre-registered
genre parity (1e-12) failed. --late-clamp off uses race_affinity_v2.State,
so v3.1 with the clocks off is race_affinity_v2.py on genre operation for operation; --late-clamp on is v3.
Everything else (clock bank, race readout, inputs, exact NDCG reporting, stream) is imported unchanged from v3.
"""
import argparse, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/pylib_tgb')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_tgb
import tgb_sparse_labels
import race_affinity_v3 as v3
from race_affinity_v2 import Readout, TAU_DAYS, TAU_LAB, _PF, State as V2State
from race_affinity_v2_large_v2 import State as LargeState


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', required=True)
    ap.add_argument('--root', default='../../../data/tgb/root'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--max-label-times', type=int, default=0)
    ap.add_argument('--edge-weight', choices=('count', 'feat'), default='feat')
    ap.add_argument('--chunk-users', type=int, default=1024)
    ap.add_argument('--extra-clocks', choices=sorted(v3.EXTRA), default='none')
    ap.add_argument('--late-clamp', choices=('on', 'off'), required=True)
    ap.add_argument('--baseline-checkpoint', default='')
    ap.add_argument('--score-test', action='store_true')
    a = ap.parse_args(); t0 = time.time(); torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(1)
    v3.State = LargeState if a.late_clamp == 'on' else V2State
    safe_tgb.patch_tgb(); tgb_sparse_labels.patch()
    from tgb.nodeproppred.dataset import NodePropPredDataset
    from tgb.nodeproppred.evaluate import Evaluator
    ds = NodePropPredDataset(name=a.dataset, root=a.root, preprocess=True); ev = Evaluator(name=a.dataset)
    d = ds.full_data; C = ds.num_classes; load_s = time.time() - t0
    assert int(d['destinations'].max()) < C and int(d['sources'].min()) >= C, 'expected destination id = class index, users >= C'
    U = int(d['sources'].max()) + 1 - C
    n_t = len(d['timestamps'])
    w_all = np.ones(n_t) if a.edge_weight == 'count' else np.asarray(d['edge_feat'], float).reshape(n_t, -1)[:, 0]
    nf = 2 * (3 + len(TAU_LAB) + len(TAU_DAYS) + 2) + 1 + 4
    extra = v3.EXTRA[a.extra_clocks]
    model = v3.RaceReadout(nf + 2 * len(extra), a.hidden, len(TAU_DAYS) + len(extra)) if extra else Readout(nf, a.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    baseline, base_info = None, None
    if a.baseline_checkpoint:
        baseline = Readout(nf, a.hidden); bp = ROOT / a.baseline_checkpoint
        baseline.load_state_dict(torch.load(bp)); baseline.eval()
        base_info = dict(checkpoint=a.baseline_checkpoint, sha256=hashlib.sha256(bp.read_bytes()).hexdigest())
    od = ROOT / 'experiments/results/tgb'; od.mkdir(parents=True, exist_ok=True); ckpt = od / f'{a.tag}.pt'
    kw = dict(max_label_times=a.max_label_times, chunk=a.chunk_users, dataset=a.dataset)
    p0 = time.time()
    pf = v3.run_stream(ds, ev, _PF(), None, [('train', ds.train_mask), ('val', ds.val_mask)], C, U, False, w_all, **kw)
    print('persistent forecast (our replay)', json.dumps(pf['val'][:2]), f'{time.time() - p0:.0f}s', flush=True)
    hist, best, best_rep = [], -1.0, None
    for ep in range(a.epochs):
        e0 = time.time()
        r = v3.run_stream(ds, ev, model, opt, [('train', ds.train_mask), ('val', ds.val_mask)], C, U, True, w_all,
                          extra=extra, baseline=baseline, **kw)
        hist.append(dict(epoch=ep, val_ndcg=r['val'][0], val_label_times=r['val'][1], epoch_s=time.time() - e0,
                         vmhwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, val_report=r['val'][2]))
        print(json.dumps({k: v for k, v in hist[-1].items() if k != 'val_report'}), flush=True)
        if r['val'][0] is not None and r['val'][0] > best:
            best = r['val'][0]; best_rep = r['val'][2]; torch.save(model.state_dict(), ckpt)
    res = dict(status='completed', benchmark='TGB', tag=a.tag, dataset=a.dataset,
               model=f'race_affinity v3.1 (multi-clock bank, soft per-user clock race; late clamp {a.late_clamp})',
               args=vars(a), classes=C, users=U, features=nf + 2 * len(extra) + (2 if extra else 0), extra_clocks=[list(k) for k in extra],
               parameters=sum(p.numel() for p in model.parameters()),
               persistent_forecast_val_ndcg=pf['val'][0], persistent_forecast_val_label_times=pf['val'][1],
               best_val_ndcg=best, best_val_report=best_rep, history=hist, checkpoint=str(ckpt.relative_to(ROOT)),
               baseline=base_info, load_s=load_s, rate_edges_log10=v3.RATE_EDGES.get(a.dataset),
               extra_state_bytes=int(len(extra) * U * C * 4))
    if a.score_test:
        model.load_state_dict(torch.load(ckpt))
        r = v3.run_stream(ds, ev, model, None, [('train', ds.train_mask), ('val', ds.val_mask), ('test', ds.test_mask)], C, U, False,
                          w_all, extra=extra, **kw)
        res['replay_val_ndcg'] = r['val'][0]; res['test_ndcg'] = r['test'][0]; res['test_label_times'] = r['test'][1]
        res['test_report'] = r['test'][2]
    res['wall_s'] = time.time() - t0; res['vmhwm_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    here = Path(__file__).resolve().parent
    res['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__).resolve(), here / 'race_affinity_v3.py', here / 'race_affinity_v2.py',
                                      here / 'race_affinity_v2_large_v2.py', here / 'safe_tgb.py', here / 'tgb_sparse_labels.py')}
    (od / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('RESULT', json.dumps({k: res.get(k) for k in ('persistent_forecast_val_ndcg', 'best_val_ndcg', 'test_ndcg', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
