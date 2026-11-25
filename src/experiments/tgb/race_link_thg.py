#!/usr/bin/env python3
"""race_link for TGB 2.0 temporal heterogeneous graphs (thgl-github, thgl-myket): a race among candidate destination
clocks read from sparse addressed temporal memory, with relation-addressed slots and an evaluation-shaped race.

Lineage: race_link v4 (tgbl-wiki-v2) and race_link_review v2-v7 (tgbl-review-v2). Retained: strict causal per-query
state (every event with time < t_query, ties excluded), sparse addressed pair / destination / source clocks at several
timescales, the race readout with losing-candidate credit, no identity tables (inductive; removed on review for overfit).
New for THG, each required by a measured property of the data:
  1. relation-addressed slots: pair (s, r, c) and destination (r, c) clocks beside the untyped pair (s, c) and
     destination c clocks, plus source-under-relation (s, r) activity; a relation embedding conditions the readout.
     --untyped drops every relation-addressed slot and the embedding (the v4/v7 slot set).
  2. evaluation-shaped race: the official negatives are drawn uniformly from ALL nodes of the true destination's type
     (thg_negative_generator.py, node-type-filtered) and almost never come from the source's own history, so each
     training race is the positive + K competitors drawn the same way (uniform over the destination's node type; the
     positive and duplicates redrawn).
  3. exact vectorised reads of the addressed memory: each slot family is its event list sorted by (slot, time) with the
     clock values after each event (decayed counts at 1 h, 1 d, 7 d, 30 d by A_j = A_{j-1} exp(-(t_j - t_{j-1})/tau) + 1);
     a query at time t reads its slot's last event with time < t and decays it to t: the state an event-by-event memory
     holds at t, for a whole batch at once (asserted against direct sums by --contract).
Evaluation: official py-tgb loader, node-type-filtered negatives (the official evaluation set copied into compact arrays,
equality with query_batch asserted on the first 2,000 queries, then the dict released), Evaluator agreement on the
first 50 queries. Selection on VALIDATION MRR (best epoch). TEST only with --score-test: after training, the slots are
rebuilt over TRAIN + VALIDATION + TEST edges (read strictly causally), the best checkpoint scores the official TEST
negatives once.
"""
import argparse, gc, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / '.cache/pylib_tgb')); sys.path.insert(0, str(HERE))
import safe_tgb

HOUR, DAY = 3600.0, 86400.0
TAUS = np.array([HOUR, DAY, 7 * DAY, 30 * DAY])
NO_EVENT_DT = 1e9
CLASSES = ('repeat_pair', 'new_pair_known_source', 'new_source')


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def vm():
    out = {}
    for line in open('/proc/self/status'):
        if line.startswith(('VmRSS', 'VmHWM', 'VmPeak')):
            k, v = line.split(':'); out[k + '_kb'] = int(v.split()[0])
    return out


def decay_scan(start, length, t):
    """Clock values after each event: A_j = A_{j-1} exp(-(t_j - t_{j-1}) / tau) + 1 within each slot (vectorised over
    slots, one step per position within a slot)."""
    A = np.ones((len(t), len(TAUS)), np.float32)
    if not len(length):
        return A
    o = np.argsort(-length, kind='stable'); st = start[o]; neg = -length[o]
    for p in range(1, int(-neg[0])):
        m = int(np.searchsorted(neg, -p, 'left'))
        ix = st[:m] + p
        dt = (t[ix] - t[ix - 1]).astype(np.float64)
        A[ix] = (A[ix - 1] * np.exp(-dt[:, None] / TAUS) + 1.0).astype(np.float32)
    return A


class Slots:
    """One addressed slot family over a time-ordered event stream."""
    def __init__(self, keys, rank, t, nbits, clocks=True):
        self.u, inv = np.unique(keys, return_inverse=True); self.nbits = nbits
        comp = (inv.astype(np.int64) << nbits) | rank
        o = np.argsort(comp, kind='stable'); del inv
        self.comp = comp[o]; self.t = t[o]; del comp, o
        self.start = np.searchsorted(self.comp, np.arange(len(self.u), dtype=np.int64) << nbits)
        self.A = decay_scan(self.start, np.diff(np.append(self.start, len(self.comp))), self.t) if clocks else None

    def read(self, q, qr, t):
        """Slot reads at query ranks qr (events with rank < qr): count, time since last event, decayed clocks."""
        i = np.minimum(np.searchsorted(self.u, q), len(self.u) - 1); found = self.u[i] == q
        hi = np.searchsorted(self.comp, (i.astype(np.int64) << self.nbits) | qr)
        cnt = np.where(found, hi - self.start[i], 0); has = cnt > 0; j = np.maximum(hi - 1, 0)
        dt = np.where(has, t - self.t[j], NO_EVENT_DT).astype(np.float64)
        clk = None
        if self.A is not None:
            clk = np.where(has[..., None], self.A[j] * np.exp(-np.where(has, dt, 0.0)[..., None] / TAUS), 0.0)
        return cnt, dt, clk


class Memory:
    """Addressed slot families over the events at positions [0, E) of the stream (time ordered)."""
    def __init__(self, src, dst, ts, et, n, typed):
        self.n, self.typed = n, typed
        self.ut = np.unique(ts); rank = np.searchsorted(self.ut, ts).astype(np.int64)
        nb = int(np.ceil(np.log2(len(self.ut) + 2)))
        self.fam = dict(pair=Slots(src * n + dst, rank, ts, nb), dst=Slots(dst, rank, ts, nb),
                        src=Slots(src, rank, ts, nb, clocks=False))
        if typed:
            self.fam.update(pair_r=Slots((et * n + src) * n + dst, rank, ts, nb), dst_r=Slots(et * n + dst, rank, ts, nb),
                            src_r=Slots(et * n + src, rank, ts, nb, clocks=False))

    def nf(self):
        return 2 * 7 + 4 + (2 * 7 + 2 if self.typed else 0)

    def features(self, s, r, cand, t):
        """s, r, t [Q]; cand [Q, C] -> float32 [Q, C, NF]; reads only events with time < t."""
        n = self.n; Q, C = cand.shape
        qr = np.searchsorted(self.ut, t).astype(np.int64)
        QR, T, S, R = (np.broadcast_to(x[:, None], (Q, C)) for x in (qr, t, s, r))
        cols = []

        def clocked(name, key):
            cnt, dt, clk = self.fam[name].read(key, QR, T)
            cols.extend([np.log1p(clk), np.log1p(cnt)[..., None], (cnt > 0)[..., None], slog(dt)[..., None]])

        def activity(name, key, qrk, tk):
            cnt, dt, _ = self.fam[name].read(key, qrk, tk)
            return [np.log1p(cnt)[..., None], slog(dt)[..., None]]

        clocked('pair', S * n + cand); clocked('dst', cand)
        a = activity('src', s, qr, t)
        cols.extend(np.broadcast_to(x[:, None], (Q, C, 1)) for x in a)
        cols.extend(activity('src', cand, QR, T))
        if self.typed:
            clocked('pair_r', (R * n + S) * n + cand); clocked('dst_r', R * n + cand)
            cols.extend(np.broadcast_to(x[:, None], (Q, C, 1)) for x in activity('src_r', r * n + s, qr, t))
        return np.concatenate([np.asarray(x, np.float32) for x in cols], -1)

    def query_class(self, s, c, t):
        qr = np.searchsorted(self.ut, t).astype(np.int64)
        seen = self.fam['pair'].read(s * self.n + c, qr, t)[0] > 0
        known = self.fam['src'].read(s, qr, t)[0] > 0
        return np.where(seen, 0, np.where(known, 1, 2))


class Model(nn.Module):
    def __init__(self, nf, R, hidden, rdim):
        super().__init__()
        self.norm = nn.LayerNorm(nf); self.rel = nn.Embedding(R, rdim) if rdim else None
        self.mlp = nn.Sequential(nn.Linear(nf + rdim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, r, feat):
        x = self.norm(feat)
        if self.rel is not None:
            x = torch.cat([x, self.rel(r)[:, None].expand(-1, x.shape[1], -1)], -1)
        return self.mlp(x).squeeze(-1)


def rr(sc, valid):
    pos = sc[:, :1]; neg = sc[:, 1:]; v = valid[:, 1:]
    return 1.0 / (0.5 * (((neg > pos) & v).sum(1) + ((neg >= pos) & v).sum(1)) + 1.0)


class TypeSampler:
    """Competitors drawn as the official generator draws negatives: uniform over all nodes of the destination's type."""
    def __init__(self, ntype):
        self.ntype = ntype; self.nodes = {int(k): np.flatnonzero(ntype == k) for k in np.unique(ntype)}

    def __call__(self, c, K, rng):
        out = np.empty((len(c), K), np.int64); ty = self.ntype[c]
        for k in np.unique(ty):
            rows = np.flatnonzero(ty == k); pool = self.nodes[int(k)]
            out[rows] = pool[rng.integers(0, len(pool), (len(rows), K))]
        for q in np.flatnonzero((out == c[:, None]).any(1) | (np.diff(np.sort(out, 1), axis=1) == 0).any(1)):
            pool = self.nodes[int(ty[q])]; row = set()
            while len(row) < K:
                x = int(pool[rng.integers(len(pool))])
                if x != c[q]:
                    row.add(x)
            out[q] = sorted(row)
        return out


class CompactNegatives:
    """The official evaluation set {(t, s, r): negatives} (node-type-filtered, py-tgb THGNegativeEdgeSampler) as sorted
    int64 keys and an int32 matrix, copied row by row; equality with the official query_batch is asserted on the first
    2,000 queries of the split before the dict is released (about 11.5 GB on thgl-myket's 8M validation keys)."""
    def __init__(self, ds, split, n, R):
        es = ds.negative_sampler.eval_set[split]; self.n, self.R = n, R
        keys = np.array(list(es.keys()), dtype=np.int64)
        lens = np.fromiter((len(v) for v in es.values()), np.int64, len(es))
        vals = np.full((len(es), int(lens.max())), -1, np.int32)
        for i, v in enumerate(es.values()):
            vals[i, :len(v)] = v
        self.t0 = int(keys[:, 0].min())
        code = self.code(keys[:, 0], keys[:, 1], keys[:, 2]); o = np.argsort(code)
        self.codes, self.vals, self.lens = code[o], vals[o], lens[o]; del keys, vals, code, o
        assert np.all(np.diff(self.codes) > 0)
        d = ds.full_data; ix = np.flatnonzero(ds.val_mask if split == 'val' else ds.test_mask)[:2000]
        for b in range(0, len(ix), 200):
            j = ix[b:b + 200]
            off = ds.negative_sampler.query_batch(d['sources'][j], d['destinations'][j], d['timestamps'][j],
                                                  d['edge_type'][j], split_mode=split)
            mine = self(*(np.asarray(d[k][j], np.int64) for k in ('timestamps', 'sources', 'edge_type')))
            assert all(np.array_equal(np.asarray(x), m[m >= 0]) for x, m in zip(off, mine)), 'compact != official'
        self.checked = len(ix)
        ds.negative_sampler.eval_set[split] = None; del es; gc.collect(); trim()

    def code(self, t, s, r):
        return ((t - self.t0) * self.n + s) * self.R + r

    def __call__(self, t, s, r):
        c = self.code(t, s, r); i = np.minimum(np.searchsorted(self.codes, c), len(self.codes) - 1)
        assert np.all(self.codes[i] == c), 'query not in the official evaluation set'
        return self.vals[i]


def trim():
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass


def evaluate(ds, ev, model, mem, mask, split, negatives, max_q=0):
    d = ds.full_data; ix = np.flatnonzero(mask)
    if max_q:
        ix = ix[:max_q]
    model.eval(); out, cls, rel = [], [], []; checked = 0; t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(ix), 200):
            j = ix[b:b + 200]
            s, c, t, r = (np.asarray(d[k][j], np.int64) for k in ('sources', 'destinations', 'timestamps', 'edge_type'))
            neg = negatives(t, s, r).astype(np.int64)
            cand = np.concatenate([c[:, None], neg], 1)
            valid = cand >= 0; cc = np.where(valid, cand, 0)
            sc = model(torch.from_numpy(r), torch.from_numpy(mem.features(s, r, cc, t))).numpy().astype(np.float64)
            sc = np.where(valid, sc, -np.inf); v = rr(sc, valid)
            while checked < 50 and checked < len(j):
                o = ev.eval({'y_pred_pos': sc[checked, :1], 'y_pred_neg': sc[checked, 1:][valid[checked, 1:]][None],
                             'eval_metric': ['mrr']})['mrr']
                assert abs(float(o) - v[checked]) < 1e-6, (o, v[checked]); checked += 1
            out.append(v); cls.append(mem.query_class(s, c, t)); rel.append(r)
    v, cls, rel = np.concatenate(out), np.concatenate(cls), np.concatenate(rel)
    prof = {cl: dict(queries=int((cls == i).sum()), mrr=float(v[cls == i].mean()) if (cls == i).any() else None)
            for i, cl in enumerate(CLASSES)}
    by_rel = {int(x): dict(queries=int((rel == x).sum()), mrr=float(v[rel == x].mean())) for x in np.unique(rel)}
    return float(v.mean()), len(v), time.time() - t0, prof, by_rel


def contracts():
    """Vectorised slot reads equal direct sums over strictly earlier events (ties excluded)."""
    rng = np.random.default_rng(0); E, n = 4000, 30
    ts = np.sort(rng.integers(0, 40 * DAY, E)); ts[100:110] = ts[100]
    src, dst, et = rng.integers(0, n, E), rng.integers(0, n, E), rng.integers(0, 3, E)
    mem = Memory(src, dst, ts, et, n, typed=True)
    s, r, c = rng.integers(0, n, 300), rng.integers(0, 3, 300), rng.integers(0, n, (300, 5))
    t = np.sort(rng.choice(ts, 300)).astype(np.int64); t[:5] = ts[100]
    feat = mem.features(s, r, c, t); worst = 0.0
    for q in range(len(s)):
        for k in range(c.shape[1]):
            m = (src == s[q]) & (dst == c[q, k]) & (ts < t[q])
            ref = np.log1p(np.exp(-(t[q] - ts[m])[:, None] / TAUS).sum(0)) if m.any() else np.zeros(len(TAUS))
            worst = max(worst, float(np.abs(feat[q, k, :4] - ref).max()))
            assert feat[q, k, 4] == np.float32(np.log1p(m.sum()))
            m = (src == s[q]) & (dst == c[q, k]) & (et == r[q]) & (ts < t[q])
            ref = np.log1p(np.exp(-(t[q] - ts[m])[:, None] / TAUS).sum(0)) if m.any() else np.zeros(len(TAUS))
            worst = max(worst, float(np.abs(feat[q, k, 18:22] - ref).max()))
    assert worst < 1e-4, worst
    return dict(max_abs_clock_error=worst, queries=int(len(s) * c.shape[1]), ties_excluded=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(a, split):
    safe_tgb.patch_tgb()
    import tgb.linkproppred.thg_negative_sampler as thg_ns
    thg_ns.load_pkl = safe_tgb.load_pkl
    import tgb.linkproppred.dataset as dsm
    from tgb.linkproppred.evaluate import Evaluator
    assert dsm.load_pkl is safe_tgb.load_pkl
    ds = dsm.LinkPropPredDataset(name=a.dataset, root=a.root, preprocess=True, download=False)
    d = ds.full_data
    for k in ('edge_feat', 'w', 'edge_label', 'edge_idxs'):
        d.pop(k, None)
    ds._edge_feat = None; gc.collect(); trim()
    (ds.load_val_ns if split == 'val' else ds.load_test_ns)()
    n = int(max(d['sources'].max(), d['destinations'].max())) + 1; R = int(d['edge_type'].max()) + 1
    return ds, Evaluator(name=a.dataset), CompactNegatives(ds, split, n, R)


def stream(ds, upto):
    d = ds.full_data
    return tuple(np.asarray(d[k][:upto], np.int64) for k in ('sources', 'destinations', 'timestamps', 'edge_type'))


def score_sealed(a, ckpt, split):
    """The sealed path: fresh load, slots rebuilt over every event up to the end of `split`, best checkpoint, official
    negatives of `split`. split='test' only under --score-test; split='val' (--sealed-path-smoke, development only) runs
    the identical code on VALIDATION and must reproduce the best epoch's validation MRR."""
    t0 = time.time(); ds, ev, negs = load(a, split); d = ds.full_data
    n = int(max(d['sources'].max(), d['destinations'].max())) + 1; R = int(d['edge_type'].max()) + 1
    mask = ds.test_mask if split == 'test' else ds.val_mask
    upto = int(np.flatnonzero(mask)[-1]) + 1
    assert split != 'test' or upto == len(d['sources'])
    mem = Memory(*stream(ds, upto), n, typed=not a.untyped)
    model = Model(mem.nf(), R, a.hidden, 0 if a.untyped else a.rdim); model.load_state_dict(torch.load(ckpt))
    mrr, nq, wall, prof, by_rel = evaluate(ds, ev, model, mem, mask, split, negs)
    p = 'test' if split == 'test' else 'sealed_path_smoke_val'
    return {f'{p}_mrr': mrr, f'{p}_queries': nq, f'{p}_query_profile': prof, f'{p}_by_relation': by_rel,
            f'{p}_wall_s': time.time() - t0, f'{p}_eval_wall_s': wall, f'{p}_vm': vm(),
            f'{p}_negatives_equal_to_query_batch': negs.checked,
            f'{p}_protocol': 'best VALIDATION epoch checkpoint; fresh load; slots rebuilt over every event up to the end '
                             f'of {split.upper()} and read strictly causally (time < t_query); official {split.upper()} '
                             'negatives and Evaluator (agreement on the first 50 queries); scored once'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', default='thgl-github')
    ap.add_argument('--root', default='../../../data/tgb/root'); ap.add_argument('--contract', action='store_true')
    for k, v in (('seed', 0), ('epochs', 3), ('neg', 20), ('hidden', 64), ('rdim', 8), ('max_train', 0), ('max_val', 0)):
        ap.add_argument('--' + k.replace('_', '-'), type=int, default=v)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--untyped', action='store_true'); ap.add_argument('--score-test', action='store_true')
    ap.add_argument('--sealed-path-smoke', action='store_true')
    a = ap.parse_args(); start = time.time(); torch.set_num_threads(1)
    assert not (a.score_test and (a.sealed_path_smoke or a.max_train or a.max_val)), 'sealed runs use the full data'
    out = ROOT / 'experiments/results/tgb' / f'{a.tag}.json'; assert not out.exists(), 'fresh tag'
    if a.contract:
        res = dict(status='completed', checks=contracts(), scope='vectorised slot reads vs direct sums; no data, no fit')
    else:
        torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
        ds, ev, negs = load(a, 'val'); d = ds.full_data
        n = int(max(d['sources'].max(), d['destinations'].max())) + 1; R = int(d['edge_type'].max()) + 1
        tr, va = np.flatnonzero(ds.train_mask), np.flatnonzero(ds.val_mask)
        assert tr[0] == 0 and va[0] == tr[-1] + 1 and np.all(np.diff(va) == 1)
        t_load = time.time() - start
        mem = Memory(*stream(ds, va[-1] + 1), n, typed=not a.untyped)
        t_mem = time.time() - start - t_load; mem_vm = vm()
        ntype = np.asarray(ds.node_type).astype(np.int64); sampler = TypeSampler(ntype)
        model = Model(mem.nf(), R, a.hidden, 0 if a.untyped else a.rdim)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        ckpt = out.with_suffix('.pt'); best, best_ep, history = -1.0, -1, []
        lim = tr[:a.max_train] if a.max_train else tr
        s_all, c_all, t_all, r_all = stream(ds, len(lim))
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
            mrr, nq, vw, prof, by_rel = evaluate(ds, ev, model, mem, ds.val_mask, 'val', negs, a.max_val)
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
            del mem, ds, d, negs, sampler, s_all, c_all, t_all, r_all; gc.collect(); trim()
            res.update(score_sealed(a, ckpt, 'test' if a.score_test else 'val'))
            if a.score_test:
                res['protocol'] = res['protocol'].replace('; no TEST', '; TEST scored once with the best VALIDATION checkpoint')
    res.update(tag=a.tag, benchmark='TGB 2.0 THG', driver='race_link_thg.py', score_test=a.score_test,
               wall_s=time.time() - start, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__).resolve(), HERE / 'safe_tgb.py')})
    with out.open('x') as f:
        json.dump(res, f, indent=1); f.write('\n')
    print(json.dumps({k: v for k, v in res.items() if k not in ('history', 'args')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
