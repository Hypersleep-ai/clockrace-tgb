#!/usr/bin/env python3
"""race_link for tgbl-flight-v2 (5 Nov 2026): a race among candidate destination airports read from sparse addressed
temporal memory, with schedule lags, return-flight reads and a source-history training race.

Lineage: race_link v4 (tgbl-wiki-v2), race_link_review v7 (tgbl-review-v2: the source's own past destinations as
training competitors) and race_link_thg (thgl-github/myket: exact vectorised reads of sorted slot families).
Retained: strict causal per-query state (every event with time < t_query, ties excluded), sparse addressed pair /
destination / source clocks, the race readout with losing-candidate credit, no identity tables, official negatives and
Evaluator, best VALIDATION epoch, TEST only with --score-test through the sealed path.
New for tgbl-flight, each required by a property of the data:
  1. day-scale clocks (1, 7, 30, 365 days): timestamps are calendar days (all flights of a day share one stamp).
  2. schedule lags: the pair's (and the destination's) event counts on exactly day t-1, t-7 and t-14, read as the
     difference of two addressed slot reads (delayed reads of the same memory); flights run on weekly schedules.
  3. return-flight read: the candidate's own pair slot toward the source (c, s), same clocks and lags.
  4. hist_rnd-shaped training race: the official negatives (hist_rnd, ratio 0.5) are up to 10 of the source's own TRAIN
     destinations plus random airports outside that history; each training race is the positive plus up to floor(K/2)
     competitors drawn from the source's destinations first seen before the query, the rest uniform over airports the
     source has not flown to before the query.
--diagnose runs no fit: the scorer proof (EdgeBank unlimited and time-window with TGB's batch-of-200 memory against the
leaderboard's validation 0.166 / 0.363), strict-causal heuristics and the query classes, on VALIDATION only.
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

DAY = 86400.0
TAUS = np.array([DAY, 7 * DAY, 30 * DAY, 365 * DAY])
LAGS = (1, 7, 14)
NO_EVENT_DT = 1e9
CLASSES = ('repeat_pair', 'new_pair_known_source', 'new_source')
DATASET = 'tgbl-flight'


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def vm():
    out = {}
    for line in open('/proc/self/status'):
        if line.startswith(('VmRSS', 'VmHWM', 'VmPeak')):
            k, v = line.split(':'); out[k + '_kb'] = int(v.split()[0])
    return out


def trim():
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass


def decay_scan(start, length, t):
    """Clock values after each event within each slot: A_j = A_{j-1} exp(-(t_j - t_{j-1}) / tau) + 1."""
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
    """One addressed slot family over a time-ordered event stream; reads return the state of events with rank < qr."""
    def __init__(self, keys, rank, t, nbits, clocks=True):
        self.u, inv = np.unique(keys, return_inverse=True); self.nbits = nbits
        comp = (inv.astype(np.int64) << nbits) | rank
        o = np.argsort(comp, kind='stable'); del inv
        self.comp = comp[o]; self.t = t[o]; del comp, o
        self.start = np.searchsorted(self.comp, np.arange(len(self.u), dtype=np.int64) << nbits)
        self.A = decay_scan(self.start, np.diff(np.append(self.start, len(self.comp))), self.t) if clocks else None

    def slot(self, q):
        i = np.minimum(np.searchsorted(self.u, q), len(self.u) - 1)
        return i, self.u[i] == q

    def count(self, q, qr):
        i, found = self.slot(q)
        hi = np.searchsorted(self.comp, (i.astype(np.int64) << self.nbits) | qr)
        return np.where(found, hi - self.start[i], 0)

    def read(self, q, qr, t):
        i, found = self.slot(q)
        hi = np.searchsorted(self.comp, (i.astype(np.int64) << self.nbits) | qr)
        cnt = np.where(found, hi - self.start[i], 0); has = cnt > 0; j = np.maximum(hi - 1, 0)
        dt = np.where(has, t - self.t[j], NO_EVENT_DT).astype(np.float64)
        clk = None
        if self.A is not None:
            clk = np.where(has[..., None], self.A[j] * np.exp(-np.where(has, dt, 0.0)[..., None] / TAUS), 0.0)
        return cnt, dt, clk


class Memory:
    """Addressed slot families over the events at positions [0, E) of the stream (time ordered)."""
    def __init__(self, src, dst, ts, n):
        self.n = n
        self.ut = np.unique(ts); rank = np.searchsorted(self.ut, ts).astype(np.int64)
        nb = int(np.ceil(np.log2(len(self.ut) + 2)))
        self.fam = dict(pair=Slots(src * n + dst, rank, ts, nb), dst=Slots(dst, rank, ts, nb), src=Slots(src, rank, ts, nb))
        p = self.fam['pair']
        self.first_t = p.t[p.start]; srcof = p.u // n
        o = np.lexsort((self.first_t, srcof))
        self.h_dst = (p.u % n)[o]; self.h_first = self.first_t[o]; self.h_src = srcof[o]
        self.h_lo = np.searchsorted(self.h_src, np.arange(n + 1))
        self.h_key = self.h_src * (len(self.ut) + 1) + self.rank(self.h_first)

    def rank(self, x):
        return np.searchsorted(self.ut, x).astype(np.int64)

    def nf(self):
        return 2 * (7 + len(LAGS)) + (6 + 2) + 6 + 6

    def lags(self, fam, key, t):
        return [np.log1p(self.fam[fam].count(key, self.rank(t - (L - 1) * DAY)) - self.fam[fam].count(key, self.rank(t - L * DAY)))[..., None]
                for L in LAGS[:2 if fam == 'dst' else 3]]

    def features(self, s, cand, t):
        """s, t [Q]; cand [Q, C] -> float32 [Q, C, NF]; reads only events with time < t."""
        n = self.n; Q, C = cand.shape
        qr = self.rank(t)
        QR, T, S = (np.broadcast_to(x[:, None], (Q, C)) for x in (qr, t, s))
        cols = []
        for key in (S * n + cand, cand * n + S):
            cnt, dt, clk = self.fam['pair'].read(key, QR, T)
            cols += [np.log1p(clk), np.log1p(cnt)[..., None], (cnt > 0)[..., None], slog(dt)[..., None]]
            cols += self.lags('pair', key, T)
        cnt, dt, clk = self.fam['dst'].read(cand, QR, T)
        cols += [np.log1p(clk), np.log1p(cnt)[..., None], slog(dt)[..., None]] + self.lags('dst', cand, T)
        cnt, dt, clk = self.fam['src'].read(s, qr, t)
        own = np.concatenate([np.log1p(clk), np.log1p(cnt)[:, None], slog(dt)[:, None]], 1)
        cols.append(np.broadcast_to(own[:, None], (Q, C, own.shape[1])))
        cnt, dt, clk = self.fam['src'].read(cand, QR, T)
        cols += [np.log1p(clk), np.log1p(cnt)[..., None], slog(dt)[..., None]]
        return np.concatenate([np.asarray(x, np.float32) for x in cols], -1)

    def query_class(self, s, c, t):
        qr = self.rank(t)
        seen = self.fam['pair'].count(s * self.n + c, qr) > 0
        known = self.fam['src'].count(s, qr) > 0
        return np.where(seen, 0, np.where(known, 1, 2))

    def competitors(self, s, c, t, K, rng):
        """hist_rnd-shaped race: up to floor(K/2) of the source's destinations first seen before t (positive excluded),
        the rest uniform over airports without a pair event before t (positive excluded). Draws with replacement."""
        Q = len(s); kh = K // 2; out = np.empty((Q, K), np.int64)
        lo = self.h_lo[s]
        nv = np.searchsorted(self.h_key, s * (len(self.ut) + 1) + self.rank(t)) - lo
        seen_pos = self.fam['pair'].count(s * self.n + c, self.rank(t)) > 0
        nh = np.minimum(kh, np.maximum(nv - seen_pos, 0))
        draw = lo[:, None] + (rng.random((Q, kh)) * np.maximum(nv, 1)[:, None]).astype(np.int64)
        hist = self.h_dst[np.minimum(draw, len(self.h_dst) - 1)]
        for _ in range(4):
            bad = (hist == c[:, None]) & (nv[:, None] > 1)
            if not bad.any():
                break
            redraw = lo[:, None] + (rng.random((Q, kh)) * np.maximum(nv, 1)[:, None]).astype(np.int64)
            hist = np.where(bad, self.h_dst[np.minimum(redraw, len(self.h_dst) - 1)], hist)
        rnd = rng.integers(0, self.n, (Q, K)); qr = self.rank(t)
        for _ in range(6):
            bad = (rnd == c[:, None]) | (self.fam['pair'].count(s[:, None] * self.n + rnd, qr[:, None]) > 0)
            if not bad.any():
                break
            rnd = np.where(bad, rng.integers(0, self.n, (Q, K)), rnd)
        rnd = np.where(rnd == c[:, None], (c[:, None] + 1 + rng.integers(0, self.n - 1, (Q, K))) % self.n, rnd)
        hist = np.where(hist == c[:, None], rnd[:, :kh], hist)
        use_h = np.arange(K)[None, :] < nh[:, None]
        out[:] = np.where(use_h, np.pad(hist, ((0, 0), (0, K - kh)))[:, :K], rnd)
        return out


class Model(nn.Module):
    def __init__(self, nf, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(nf)
        self.mlp = nn.Sequential(nn.Linear(nf, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, feat):
        return self.mlp(self.norm(feat)).squeeze(-1)


def rr(sc, valid):
    pos = sc[:, :1]; neg = sc[:, 1:]; v = valid[:, 1:]
    return 1.0 / (0.5 * (((neg > pos) & v).sum(1) + ((neg >= pos) & v).sum(1)) + 1.0)


class CompactNegatives:
    """The official evaluation set {(s, d, t): negatives} (hist_rnd, py-tgb NegativeEdgeSampler) as sorted int64 codes and
    an int32 matrix; equality with query_batch asserted on the first 2,000 queries of the split, then the dict released."""
    def __init__(self, ds, split, ut, n):
        es = ds.negative_sampler.eval_set[split]; self.ut, self.n = ut, n
        N = len(es); K = max(len(v) for v in es.values())
        keys = np.empty((N, 3), np.int64); vals = np.full((N, K), -1, np.int32)
        for i, (k, v) in enumerate(es.items()):
            keys[i] = k; vals[i, :len(v)] = v
        code = self.code(keys[:, 2], keys[:, 0], keys[:, 1]); o = np.argsort(code)
        self.codes, self.vals = code[o], vals[o]; del keys, vals, code, o
        assert np.all(np.diff(self.codes) > 0)
        d = ds.full_data; ix = np.flatnonzero(ds.val_mask if split == 'val' else ds.test_mask)[:2000]
        for b in range(0, len(ix), 200):
            j = ix[b:b + 200]
            off = ds.negative_sampler.query_batch(d['sources'][j], d['destinations'][j], d['timestamps'][j], split_mode=split)
            mine = self(*(np.asarray(d[k][j], np.int64) for k in ('timestamps', 'sources', 'destinations')))
            assert all(np.array_equal(np.asarray(x), m[m >= 0]) for x, m in zip(off, mine)), 'compact != official'
        self.checked = len(ix); self.K = K
        ds.negative_sampler.eval_set[split] = None; del es; gc.collect(); trim()

    def code(self, t, s, d):
        r = np.searchsorted(self.ut, t)
        assert np.all(self.ut[np.minimum(r, len(self.ut) - 1)] == t)
        return (r.astype(np.int64) * self.n + s) * self.n + d

    def __call__(self, t, s, d):
        c = self.code(t, s, d); i = np.minimum(np.searchsorted(self.codes, c), len(self.codes) - 1)
        assert np.all(self.codes[i] == c), 'query not in the official evaluation set'
        return self.vals[i]


def load(root, split):
    safe_tgb.patch_tgb()
    import tgb.linkproppred.dataset as dsm
    from tgb.linkproppred.evaluate import Evaluator
    assert dsm.load_pkl is safe_tgb.load_pkl
    ds = dsm.LinkPropPredDataset(name=DATASET, root=root, preprocess=True, download=False)
    d = ds.full_data
    for k in ('edge_feat', 'w', 'edge_label', 'edge_idxs'):
        d.pop(k, None)
    ds._edge_feat = None; gc.collect(); trim()
    (ds.load_val_ns if split == 'val' else ds.load_test_ns)()
    src, dst, ts = (np.asarray(d[k], np.int64) for k in ('sources', 'destinations', 'timestamps'))
    n = int(max(src.max(), dst.max())) + 1
    negs = CompactNegatives(ds, split, np.unique(ts), n)
    return ds, Evaluator(name=DATASET), negs, src, dst, ts, n


def evaluate(ev, model, mem, idx, split, negatives, src, dst, ts, scorer=None):
    out, cls = [], []; checked = 0; t0 = time.time()
    if model is not None:
        model.eval()
    with torch.no_grad():
        for b in range(0, len(idx), 200):
            j = idx[b:b + 200]; s, c, t = src[j], dst[j], ts[j]
            neg = negatives(t, s, c).astype(np.int64)
            cand = np.concatenate([c[:, None], neg], 1); valid = cand >= 0; cc = np.where(valid, cand, 0)
            if scorer is None:
                sc = model(torch.from_numpy(mem.features(s, cc, t))).numpy().astype(np.float64)
            else:
                sc = scorer(s, cc, t, j).astype(np.float64)
            sc = np.where(valid, sc, -np.inf); v = rr(sc, valid)
            while checked < 50 and checked < len(j):
                o = ev.eval({'y_pred_pos': sc[checked, :1], 'y_pred_neg': sc[checked, 1:][valid[checked, 1:]][None],
                             'eval_metric': ['mrr']})['mrr']
                assert abs(float(o) - v[checked]) < 1e-6, (o, v[checked]); checked += 1
            out.append(v); cls.append(mem.query_class(s, c, t))
    v, cls = np.concatenate(out), np.concatenate(cls)
    prof = {cl: dict(queries=int((cls == i).sum()), mrr=float(v[cls == i].mean()) if (cls == i).any() else None)
            for i, cl in enumerate(CLASSES)}
    return float(v.mean()), len(v), time.time() - t0, prof


def contracts():
    """Vectorised slot and lag reads equal direct sums over strictly earlier events; competitors obey the race rules."""
    rng = np.random.default_rng(0); E, n = 6000, 25
    ts = np.sort(rng.integers(0, 60, E)).astype(np.int64) * int(DAY)
    src, dst = rng.integers(0, n, E), rng.integers(0, n, E)
    mem = Memory(src, dst, ts, n)
    s, c = rng.integers(0, n, 300), rng.integers(0, n, (300, 6))
    t = np.sort(rng.choice(ts, 300)); t[:5] = ts[100]
    feat = mem.features(s, c, t); worst = 0.0
    for q in range(len(s)):
        for k in range(c.shape[1]):
            m = (src == s[q]) & (dst == c[q, k]) & (ts < t[q])
            ref = np.log1p(np.exp(-(t[q] - ts[m])[:, None] / TAUS).sum(0)) if m.any() else np.zeros(len(TAUS))
            worst = max(worst, float(np.abs(feat[q, k, :4] - ref).max()))
            assert feat[q, k, 4] == np.float32(np.log1p(m.sum()))
            for li, L in enumerate(LAGS):
                lag = ((src == s[q]) & (dst == c[q, k]) & (ts >= t[q] - L * DAY) & (ts < t[q] - (L - 1) * DAY)).sum()
                assert feat[q, k, 7 + li] == np.float32(np.log1p(lag)), (q, k, L)
            m = (src == c[q, k]) & (dst == s[q]) & (ts < t[q])
            assert feat[q, k, 10 + 4] == np.float32(np.log1p(m.sum()))
    assert worst < 1e-4, worst
    big = Memory(src, dst, ts, 40 * n)
    cs = mem.competitors(s, c[:, 0], t, 20, rng); cb = big.competitors(s, c[:, 0], t, 20, rng)
    assert not (cs == c[:, :1]).any() and not (cb == c[:, :1]).any(), 'the positive is never a competitor'
    hist_ok, hist_n, rnd_seen = 0, 0, 0
    for q in range(len(s)):
        hist = set(dst[(src == s[q]) & (ts < t[q])].tolist()) - {int(c[q, 0])}
        k = min(10, len(hist)); hist_n += k
        hist_ok += sum(int(x) in hist for x in cb[q, :k])
        rnd_seen += sum(int(x) in hist for x in cb[q, k:])
    assert hist_ok == hist_n and rnd_seen == 0, (hist_ok, hist_n, rnd_seen)
    return dict(max_abs_clock_error=worst, queries=int(len(s) * c.shape[1]), ties_excluded=True, lag_reads_exact=True,
                positive_never_competitor=True, history_competitors_from_prior_history=hist_n,
                random_competitors_outside_history=True)


def diagnose(a, start):
    """Scorer proof and query anatomy on VALIDATION (no fit)."""
    ds, ev, negs, src, dst, ts, n = load(a.root, 'val')
    tr, va = np.flatnonzero(ds.train_mask), np.flatnonzero(ds.val_mask)
    assert tr[0] == 0 and va[0] == tr[-1] + 1 and np.all(np.diff(va) == 1)
    t_load = time.time() - start
    mem = Memory(src[:va[-1] + 1], dst[:va[-1] + 1], ts[:va[-1] + 1], n); t_mem = time.time() - start - t_load
    E = va[-1] + 1; pos = np.arange(E, dtype=np.int64)
    bank = Slots(src[:E] * n + dst[:E], pos, ts[:E], int(np.ceil(np.log2(E + 2))), clocks=False)
    tmin, tmax = float(ts[tr].min()), float(ts[tr].max()); dur = (tmax - tmin) * 0.15
    if a.max_val:
        va = va[:a.max_val]

    def edgebank(window):
        def score(s, cc, t, j):
            b0 = (j[0] - va[0]) // 200 * 200 + va[0]
            qr = np.full(cc.shape, b0, np.int64); key = s[:, None] * n + cc
            cnt, _, _ = bank.read(key, qr, np.broadcast_to(t[:, None], cc.shape).astype(np.float64))
            if not window:
                return (cnt > 0).astype(np.float64)
            i, found = bank.slot(key); hi = np.searchsorted(bank.comp, (i.astype(np.int64) << bank.nbits) | qr)
            last = bank.t[np.maximum(hi - 1, 0)]; cur = max(tmax, float(ts[b0 - 1]))
            return ((cnt > 0) & (last >= cur - dur)).astype(np.float64)
        return score

    def recency(s, cc, t, j):
        cnt, dt, _ = mem.fam['pair'].read(s[:, None] * n + cc, np.broadcast_to(mem.rank(t)[:, None], cc.shape),
                                         np.broadcast_to(t[:, None], cc.shape).astype(np.float64))
        pop = mem.fam['dst'].read(cc, np.broadcast_to(mem.rank(t)[:, None], cc.shape),
                                  np.broadcast_to(t[:, None], cc.shape).astype(np.float64))[2][..., 1]
        return np.where(cnt > 0, 1e6 - dt / DAY, 0.0) + 1e-3 * np.log1p(pop)

    def weekly(s, cc, t, j):
        f = mem.features(s, cc, t)
        return 1e3 * f[..., 8] + recency(s, cc, t, j)

    res = dict(load_wall_s=t_load, memory_build_wall_s=t_mem, memory_vm=vm(), nodes=n, events=int(len(src)),
               train=int(len(tr)), val=int(np.sum(ds.val_mask)), scored_val=int(len(va)), negatives_per_query=int(negs.K),
               negatives_equal_to_query_batch=negs.checked, unique_days=int(len(np.unique(ts))))
    for name, sc in (('edgebank_unlimited_bs200', edgebank(False)), ('edgebank_tw015_bs200', edgebank(True)),
                     ('recency_then_popularity_strict', recency), ('lag7_then_recency_strict', weekly)):
        mrr, nq, w, prof = evaluate(ev, None, mem, va, 'val', negs, src, dst, ts, scorer=sc)
        res[name] = dict(val_mrr=mrr, queries=nq, wall_s=w, query_profile=prof); print(json.dumps({name: res[name]}), flush=True)
    j = va[:min(len(va), 200000)]; neg = negs(ts[j], src[j], dst[j]).astype(np.int64)
    hist = mem.fam['pair'].count(src[j][:, None] * n + np.maximum(neg, 0), np.broadcast_to(mem.rank(ts[j])[:, None], neg.shape)) > 0
    res['negatives_seen_pair_share_first_200k'] = float(hist[neg >= 0].mean())
    res['leaderboard_validation'] = dict(TPNet=0.892, LocalRecencyLocalPopularity=0.884, HyperEvent=0.885,
                                         EdgeBank_tw=0.363, EdgeBank_unlimited=0.166)
    return res


def score_sealed(a, ckpt, split):
    t0 = time.time(); ds, ev, negs, src, dst, ts, n = load(a.root, split)
    mask = ds.test_mask if split == 'test' else ds.val_mask
    upto = int(np.flatnonzero(mask)[-1]) + 1
    assert split != 'test' or upto == len(src)
    mem = Memory(src[:upto], dst[:upto], ts[:upto], n)
    model = Model(mem.nf(), a.hidden); model.load_state_dict(torch.load(ckpt))
    mrr, nq, wall, prof = evaluate(ev, model, mem, np.flatnonzero(mask), split, negs, src, dst, ts)
    p = 'test' if split == 'test' else 'sealed_path_smoke_val'
    return {f'{p}_mrr': mrr, f'{p}_queries': nq, f'{p}_query_profile': prof, f'{p}_wall_s': time.time() - t0,
            f'{p}_eval_wall_s': wall, f'{p}_vm': vm(), f'{p}_negatives_equal_to_query_batch': negs.checked,
            f'{p}_protocol': 'best VALIDATION epoch checkpoint; fresh load; slots rebuilt over every event up to the end '
                             f'of {split.upper()} and read strictly causally (time < t_query); official {split.upper()} '
                             'negatives and Evaluator (agreement on the first 50 queries); scored once'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--root', default='../../../data/tgb/root')
    ap.add_argument('--contract', action='store_true'); ap.add_argument('--diagnose', action='store_true')
    for k, v in (('seed', 0), ('epochs', 3), ('neg', 20), ('hidden', 64), ('train_last', 0), ('max_val', 0)):
        ap.add_argument('--' + k.replace('_', '-'), type=int, default=v)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--score-test', action='store_true'); ap.add_argument('--sealed-path-smoke', action='store_true')
    a = ap.parse_args(); start = time.time(); torch.set_num_threads(1)
    assert not (a.score_test and (a.sealed_path_smoke or a.max_val)), 'sealed runs score the full TEST'
    out = ROOT / 'experiments/results/tgb' / f'{a.tag}.json'; assert not out.exists(), 'fresh tag'
    if a.contract:
        res = dict(status='completed', checks=contracts(), scope='vectorised slot and lag reads vs direct sums; competitor rules; no data, no fit')
    elif a.diagnose:
        res = dict(status='completed', diagnosis=diagnose(a, start))
    else:
        torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
        ds, ev, negs, src, dst, ts, n = load(a.root, 'val')
        tr, va = np.flatnonzero(ds.train_mask), np.flatnonzero(ds.val_mask)
        assert tr[0] == 0 and va[0] == tr[-1] + 1 and np.all(np.diff(va) == 1)
        t_load = time.time() - start
        mem = Memory(src[:va[-1] + 1], dst[:va[-1] + 1], ts[:va[-1] + 1], n)
        t_mem = time.time() - start - t_load; mem_vm = vm()
        model = Model(mem.nf(), a.hidden)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        ckpt = out.with_suffix('.pt'); best, best_ep, history = -1.0, -1, []
        lim = tr[-a.train_last:] if a.train_last else tr
        vq = va[:a.max_val] if a.max_val else va
        for ep in range(a.epochs):
            model.train(); t0 = time.time(); total = 0.0; nb = 0
            for b in range(0, len(lim), 200):
                j = lim[b:b + 200]; s, c, t = src[j], dst[j], ts[j]
                cand = np.concatenate([c[:, None], mem.competitors(s, c, t, a.neg, rng)], 1)
                sc = model(torch.from_numpy(mem.features(s, cand, t)))
                loss = F.cross_entropy(sc, torch.zeros(len(s), dtype=torch.long)); assert torch.isfinite(loss)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                total += float(loss.detach()); nb += 1
            tw = time.time() - t0
            mrr, nq, vw, prof = evaluate(ev, model, mem, vq, 'val', negs, src, dst, ts)
            row = dict(epoch=ep, train_loss=total / max(nb, 1), val_mrr=mrr, val_queries=nq, train_wall_s=tw,
                       val_wall_s=vw, query_profile=prof, vm=vm())
            history.append(row); print(json.dumps(row), flush=True)
            if mrr > best:
                best, best_ep = mrr, ep; torch.save(model.state_dict(), ckpt)
        res = dict(status='completed', args=vars(a), dataset=DATASET, nodes=n, nf=mem.nf(),
                   parameters=sum(p.numel() for p in model.parameters()), val_mrr=best, best_epoch=best_ep,
                   history=history, checkpoint=str(ckpt.relative_to(ROOT)), trained_queries=len(lim) * a.epochs,
                   load_wall_s=t_load, memory_build_wall_s=t_mem, memory_vm=mem_vm,
                   negatives_equal_to_query_batch=negs.checked,
                   protocol='Per-query features from events strictly earlier than the query day (ties excluded); '
                            'hist_rnd-shaped training race; official hist_rnd validation negatives and Evaluator; no TEST')
        if a.score_test or a.sealed_path_smoke:
            del mem, ds, negs, src, dst, ts; gc.collect(); trim()
            res.update(score_sealed(a, ckpt, 'test' if a.score_test else 'val'))
            if a.score_test:
                res['protocol'] = res['protocol'].replace('; no TEST', '; TEST scored once with the best VALIDATION checkpoint')
    res.update(tag=a.tag, benchmark='TGB tgbl-flight-v2', driver='race_link_flight.py', score_test=a.score_test,
               wall_s=time.time() - start, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__).resolve(), HERE / 'safe_tgb.py')})
    with out.open('x') as f:
        json.dump(res, f, indent=1); f.write('\n')
    print(json.dumps({k: v for k, v in res.items() if k not in ('history', 'args')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
