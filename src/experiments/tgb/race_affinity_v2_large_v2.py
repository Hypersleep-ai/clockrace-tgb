#!/usr/bin/env python3
"""race_affinity v2 on the large TGB node-affinity datasets, large-dataset driver v2 (2 Nov 2026): the first large-dataset
driver plus change 5 below (late label times); everything else is that driver verbatim.

The model is the tgbn-genre model `race_affinity_v2.py`, imported unchanged (State, Readout, the persistent-forecast
reference, feature count): same per-user lazily decayed state, features, MLP race, CE loss, AdamW, configuration
(--epochs 4 --hidden 32 --lr 0.003, one thread) and selection (best VALIDATION epoch; TEST only with --score-test).
Same TGB loop (batches of 200, one label pointer, labels revealed after their label time is scored).

Changes, each required by a measured dataset property:
  1. --edge-weight: v2 accumulates the official edge feature as affinity mass (`feat`, the default, unchanged). On genre
     it is a listening weight; on token the official loader returns log(transfer value), measured >= 0 and finite on
     every edge, and it tracks the labels slightly better than counts (L1 to the next-7-day distribution 0.393 vs
     0.409), so token keeps `feat`. Reddit's edge feature is the comment score, negative on 5.3% and zero on 6.0% of
     edges: as affinity mass it breaks the row normalisation and the log features (NaN). Reddit uses `count` (every
     edge weight 1, decayed interaction counts), the quantity its labels track (L1 1.077 next 7 days vs 1.431 previous 7).
  2. Label-time work in user chunks (--chunk-users): token fires up to 24,375 labelled users x 1,001 classes at one
     label time; v2's one-shot feature tensor and autograd graph would need tens of GB. Each row is scored
     independently (LayerNorm/MLP per (user, class), softmax per user), so chunked predictions equal the one-shot ones;
     training accumulates sum(chunk CE)/n over chunks before the single optimiser step (the same mean-CE gradient).
  3. Labels through tgb_sparse_labels (exact sparse stand-in for py-tgb's load_label_dict; equality of every label time
     verified with `python tgb_sparse_labels.py`): the official dense dict alone is
     25-26 GB.
  4. Replay passes without a weight update (persistent-forecast reference, the final TEST replay) skip predictions on
     TRAIN label times (v2 computed and discarded them); state and revealed labels are updated exactly as before.
  5. Late label times: the official loop fires at most one label time per batch, so after a gap in the edge stream a
     label time can fire when the state already holds edges after it (token: 21 TRAIN label times, up to 53 h late;
     none in reddit, genre, or any validation/test split). v2 then computes a
     negative time since the user's last edge, log1p of it is NaN below -1 day, and one NaN row makes the loss and
     every weight NaN (observed in the first token development run, stopped). The State subclass evaluates such users at
     their last edge time (time since last edge clamped at 0, no decay applied); with no late label time it is v2.
"""
import argparse, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/pylib_tgb')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_tgb
import tgb_sparse_labels
from race_affinity_v2 import State as V2State, Readout, TAU_DAYS, TAU_LAB, _PF


class State(V2State):
    """v2 state; features of users whose last edge is after the label time t are taken at that last edge time."""

    def features(self, u, t):
        el = self.elast[u]
        late = el > t
        if not late.any():
            return super().features(u, t)
        self.elast[u] = np.where(late, t, el)
        try:
            return super().features(u, t)
        finally:
            self.elast[u] = el


def run_stream(ds, ev, model, opt, splits, C, U, train_ok, w_all, bs=200, max_label_times=0, chunk=1024):
    """One streaming pass over the given splits; returns per-split mean NDCG over scored label times."""
    d = ds.full_data; src_all = d['sources'].astype(np.int64); dst_all = d['destinations'].astype(np.int64)
    t_all = d['timestamps'].astype(float)
    st = State(U, C); ds.reset_label_time(); out = {}; fired = 0
    for name, mask in splits:
        idx = np.where(mask)[0]; res = []; label_t = ds.return_label_ts()
        for b in range(0, len(idx), bs):
            ix = idx[b:b + bs]; tq = t_all[ix][-1]
            if tq > label_t:
                r = ds.find_next_labels_batch(tq)
                if r is None:
                    break
                lt, nodes, y = r[0][0], np.asarray(r[1], np.int64), np.asarray(r[2], np.float32)
                u = nodes - C; assert u.min() >= 0
                n = len(u)
                if name == 'train' and train_ok and opt is not None:
                    model.train(); opt.zero_grad()
                    for s in range(0, n, chunk):
                        x = torch.from_numpy(st.features(u[s:s + chunk], float(lt))); yt = torch.from_numpy(y[s:s + chunk])
                        loss = -(yt * torch.log_softmax(model(x), -1)).sum(-1).sum() / n
                        loss.backward()
                    opt.step()
                elif name != 'train':
                    model.eval()
                    with torch.no_grad():
                        pred = np.concatenate([model(torch.from_numpy(st.features(u[s:s + chunk], float(lt)))).numpy()
                                               for s in range(0, n, chunk)])
                    res.append(float(ev.eval({'y_true': y, 'y_pred': pred, 'eval_metric': ['ndcg']})['ndcg']))
                st.reveal(u, y); label_t = ds.return_label_ts(); fired += 1
                if max_label_times and fired >= max_label_times:
                    break
            s = src_all[ix] - C; keep = s >= 0
            st.add_edges(s[keep], dst_all[ix][keep], w_all[ix][keep], tq)
        out[name] = (float(np.mean(res)) if res else None, len(res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', required=True)
    ap.add_argument('--root', default='../../../data/tgb/root'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--max-label-times', type=int, default=0)
    ap.add_argument('--edge-weight', choices=('count', 'feat'), default='feat')
    ap.add_argument('--chunk-users', type=int, default=1024)
    ap.add_argument('--score-test', action='store_true')
    a = ap.parse_args(); t0 = time.time(); torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(1)
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
    model = Readout(nf, a.hidden); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    od = ROOT / 'experiments/results/tgb'; od.mkdir(parents=True, exist_ok=True); ckpt = od / f'{a.tag}.pt'
    kw = dict(max_label_times=a.max_label_times, chunk=a.chunk_users)
    p0 = time.time()
    pf = run_stream(ds, ev, _PF(), None, [('train', ds.train_mask), ('val', ds.val_mask)], C, U, False, w_all, **kw)
    print('persistent forecast (our replay)', json.dumps(pf), f'{time.time() - p0:.0f}s', flush=True)
    hist, best = [], -1.0
    for ep in range(a.epochs):
        e0 = time.time()
        r = run_stream(ds, ev, model, opt, [('train', ds.train_mask), ('val', ds.val_mask)], C, U, True, w_all, **kw)
        hist.append(dict(epoch=ep, val_ndcg=r['val'][0], val_label_times=r['val'][1], epoch_s=time.time() - e0,
                         vmhwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)); print(json.dumps(hist[-1]), flush=True)
        if r['val'][0] is not None and r['val'][0] > best:
            best = r['val'][0]; torch.save(model.state_dict(), ckpt)
    res = dict(status='completed', benchmark='TGB', tag=a.tag, dataset=a.dataset, model='race_affinity v2 (streaming), large-dataset driver v2',
               args=vars(a), classes=C, users=U, features=nf, parameters=sum(p.numel() for p in model.parameters()),
               persistent_forecast_val_ndcg=pf['val'][0], persistent_forecast_val_label_times=pf['val'][1],
               best_val_ndcg=best, history=hist, checkpoint=str(ckpt.relative_to(ROOT)), load_s=load_s)
    if a.score_test:
        model.load_state_dict(torch.load(ckpt))
        r = run_stream(ds, ev, model, None, [('train', ds.train_mask), ('val', ds.val_mask), ('test', ds.test_mask)], C, U, False, w_all, **kw)
        res['replay_val_ndcg'] = r['val'][0]; res['test_ndcg'] = r['test'][0]; res['test_label_times'] = r['test'][1]
    res['wall_s'] = time.time() - t0; res['vmhwm_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    here = Path(__file__).resolve().parent
    res['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__).resolve(), here / 'race_affinity_v2.py', here / 'safe_tgb.py', here / 'tgb_sparse_labels.py')}
    (od / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('RESULT', json.dumps({k: res.get(k) for k in ('persistent_forecast_val_ndcg', 'best_val_ndcg', 'test_ndcg', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
