#!/usr/bin/env python3
"""race_link for TGB 2.0 THG, v2 (3 Nov 2026): race_link_thg.py (unchanged) + two
addressed reads required by the thgl-forum / thgl-software diagnosis.

  --reverse      reverse pair read: the candidate's own pair slot toward the source, (c, s) any relation, read with the
                 same exact causal clocks (1 h, 1 d, 7 d, 30 d), count, seen flag and time since its last event. No new
                 memory: the (s, c) slot family read at the swapped address. Reciprocity of replies (forum user -> user)
                 and of repo forks.
  --src-clocks   decayed clocks on the source slot family: the candidate's (and the source's) recent activity as a source
                 at the four timescales (v1 reads only its count and time since its last event). A replied-to author has
                 just posted.
  --neg K        size of the evaluation-shaped training race (v1 default 20; the official negatives are 100 per query on
                 forum and 1,000 on software).
Both flags off = race_link_thg.py exactly (asserted by an equivalence smoke against v1 on the same seed and prefix).
Retained from v1 unchanged: strict causal per-query state, relation-addressed slots and relation embedding (or --untyped),
evaluation-shaped race, exact vectorised slot reads, compact official negatives (equal to query_batch on 2,000 queries),
Evaluator agreement, best VALIDATION epoch, TEST only with --score-test through the sealed path.
"""
import argparse, gc, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import race_link_thg as v1


class Memory(v1.Memory):
    def __init__(self, src, dst, ts, et, n, typed, reverse=False, src_clocks=False):
        super().__init__(src, dst, ts, et, n, typed)
        self.reverse, self.src_clocks = reverse, src_clocks
        if src_clocks:
            f = self.fam['src']
            f.A = v1.decay_scan(f.start, np.diff(np.append(f.start, len(f.comp))), f.t)

    def nf(self):
        return super().nf() + (7 if self.reverse else 0) + (2 * len(v1.TAUS) if self.src_clocks else 0)

    def features(self, s, r, cand, t):
        base = super().features(s, r, cand, t)
        if not (self.reverse or self.src_clocks):
            return base
        Q, C = cand.shape; n = self.n
        qr = np.searchsorted(self.ut, t).astype(np.int64)
        QR, T, S = (np.broadcast_to(x[:, None], (Q, C)) for x in (qr, t, s))
        cols = [base]
        if self.reverse:
            cnt, dt, clk = self.fam['pair'].read(cand * n + S, QR, T)
            cols += [np.log1p(clk), np.log1p(cnt)[..., None], (cnt > 0)[..., None], v1.slog(dt)[..., None]]
        if self.src_clocks:
            cols.append(np.log1p(self.fam['src'].read(cand, QR, T)[2]))
            own = np.log1p(self.fam['src'].read(s, qr, t)[2])
            cols.append(np.broadcast_to(own[:, None], (Q, C, own.shape[-1])))
        return np.concatenate([np.asarray(x, np.float32) for x in cols], -1)


def build(a, ds, upto, n):
    return Memory(*v1.stream(ds, upto), n, typed=not a.untyped, reverse=a.reverse, src_clocks=a.src_clocks)


def score_sealed(a, ckpt, split):
    """v1.score_sealed with this Memory (identical code path otherwise)."""
    t0 = time.time(); ds, ev, negs = v1.load(a, split); d = ds.full_data
    n = int(max(d['sources'].max(), d['destinations'].max())) + 1; R = int(d['edge_type'].max()) + 1
    mask = ds.test_mask if split == 'test' else ds.val_mask
    upto = int(np.flatnonzero(mask)[-1]) + 1
    assert split != 'test' or upto == len(d['sources'])
    mem = build(a, ds, upto, n)
    model = v1.Model(mem.nf(), R, a.hidden, 0 if a.untyped else a.rdim); model.load_state_dict(torch.load(ckpt))
    mrr, nq, wall, prof, by_rel = v1.evaluate(ds, ev, model, mem, mask, split, negs)
    p = 'test' if split == 'test' else 'sealed_path_smoke_val'
    return {f'{p}_mrr': mrr, f'{p}_queries': nq, f'{p}_query_profile': prof, f'{p}_by_relation': by_rel,
            f'{p}_wall_s': time.time() - t0, f'{p}_eval_wall_s': wall, f'{p}_vm': v1.vm(),
            f'{p}_negatives_equal_to_query_batch': negs.checked,
            f'{p}_protocol': 'best VALIDATION epoch checkpoint; fresh load; slots rebuilt over every event up to the end '
                             f'of {split.upper()} and read strictly causally (time < t_query); official {split.upper()} '
                             'negatives and Evaluator (agreement on the first 50 queries); scored once'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', default='thgl-forum')
    ap.add_argument('--root', default='../../../data/tgb/root')
    for k, v in (('seed', 0), ('epochs', 3), ('neg', 20), ('hidden', 64), ('rdim', 8), ('max_train', 0), ('max_val', 0)):
        ap.add_argument('--' + k.replace('_', '-'), type=int, default=v)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--untyped', action='store_true'); ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--src-clocks', action='store_true')
    ap.add_argument('--score-test', action='store_true'); ap.add_argument('--sealed-path-smoke', action='store_true')
    a = ap.parse_args(); start = time.time(); torch.set_num_threads(1)
    assert not (a.score_test and (a.sealed_path_smoke or a.max_train or a.max_val)), 'sealed runs use the full data'
    out = ROOT / 'experiments/results/tgb' / f'{a.tag}.json'; assert not out.exists(), 'fresh tag'
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    ds, ev, negs = v1.load(a, 'val'); d = ds.full_data
    n = int(max(d['sources'].max(), d['destinations'].max())) + 1; R = int(d['edge_type'].max()) + 1
    tr, va = np.flatnonzero(ds.train_mask), np.flatnonzero(ds.val_mask)
    assert tr[0] == 0 and va[0] == tr[-1] + 1 and np.all(np.diff(va) == 1)
    t_load = time.time() - start
    mem = build(a, ds, va[-1] + 1, n)
    t_mem = time.time() - start - t_load; mem_vm = v1.vm()
    ntype = np.asarray(ds.node_type).astype(np.int64); sampler = v1.TypeSampler(ntype)
    model = v1.Model(mem.nf(), R, a.hidden, 0 if a.untyped else a.rdim)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    ckpt = out.with_suffix('.pt'); best, best_ep, history = -1.0, -1, []
    lim = tr[:a.max_train] if a.max_train else tr
    s_all, c_all, t_all, r_all = v1.stream(ds, len(lim))
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); total = 0.0; nb = 0
        for b in range(0, len(lim), 200):
            s, c, t, r = s_all[b:b + 200], c_all[b:b + 200], t_all[b:b + 200], r_all[b:b + 200]
            cand = np.concatenate([c[:, None], sampler(c, a.neg, rng)], 1)
            sc = model(torch.from_numpy(r), torch.from_numpy(mem.features(s, r, cand, t)))
            loss = F.cross_entropy(sc, torch.zeros(len(s), dtype=torch.long)); assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach()); nb += 1
        tw = time.time() - t0
        mrr, nq, vw, prof, by_rel = v1.evaluate(ds, ev, model, mem, ds.val_mask, 'val', negs, a.max_val)
        row = dict(epoch=ep, train_loss=total / max(nb, 1), val_mrr=mrr, val_queries=nq, train_wall_s=tw,
                   val_wall_s=vw, query_profile=prof, by_relation=by_rel)
        history.append(row); print(json.dumps(row), flush=True)
        if mrr > best:
            best, best_ep = mrr, ep; torch.save(model.state_dict(), ckpt)
    res = dict(status='completed', args=vars(a), dataset=a.dataset, nodes=n, relations=R, nf=mem.nf(),
               parameters=sum(p.numel() for p in model.parameters()), val_mrr=best, best_epoch=best_ep,
               history=history, checkpoint=str(ckpt.relative_to(ROOT)), trained_events=len(lim) * a.epochs,
               load_wall_s=t_load, memory_build_wall_s=t_mem, memory_vm=mem_vm,
               negatives_equal_to_query_batch=negs.checked,
               protocol='Per-query features from events strictly earlier than the query time (ties excluded); '
                        'evaluation-shaped race (uniform over the destination node type); official node-type-filtered '
                        'validation negatives and Evaluator; no TEST')
    if a.score_test or a.sealed_path_smoke:
        del mem, ds, d, negs, sampler, s_all, c_all, t_all, r_all; gc.collect(); v1.trim()
        res.update(score_sealed(a, ckpt, 'test' if a.score_test else 'val'))
        if a.score_test:
            res['protocol'] = res['protocol'].replace('; no TEST', '; TEST scored once with the best VALIDATION checkpoint')
    res.update(tag=a.tag, benchmark='TGB 2.0 THG', driver='race_link_thg_v2.py', score_test=a.score_test,
               wall_s=time.time() - start, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__).resolve(), HERE / 'race_link_thg.py', HERE / 'safe_tgb.py')})
    with out.open('x') as f:
        json.dump(res, f, indent=1); f.write('\n')
    print(json.dumps({k: v for k, v in res.items() if k not in ('history', 'args')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
