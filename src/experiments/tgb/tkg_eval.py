#!/usr/bin/env python3
"""TGB 2.0 temporal knowledge graphs (tkgl-smallpedia, tkgl-polecat, tkgl-icews): the official evaluation protocol
in compact, exact, vectorised form, shared by the drivers (race_tkg.py and its successors).

Official protocol (py-tgb 2.3.0, .cache/pylib_tgb; TGB examples/linkproppred/tkgl-*):
  * LinkPropPredDataset adds an inverse quadruple (o, r + R, s, t) for every (s, r, o, t) (utils.add_inverse_quadruples),
    so every split edge is a query in both directions: queries = all val (test) rows of full_data, 2x the quadruples.
  * splits by timestamp quantiles 0.70 / 0.85 (train ts <= q70 < val ts <= q85 < test ts): whole snapshots per split.
  * negatives, strategy 'time-filtered' (TKGNegativeEdgeSampler.query_batch): eval_set[(t, s, r)] is the conflict set,
    the true objects of (s, r) at t; the negatives are np.delete(arange(min_dst, max_dst + 1), conflict): one-vs-all over
    every entity except the true objects at t (time-aware filtered setting).
  * Evaluator MRR: rank = 0.5 * (#neg > pos + #neg >= pos) + 1, RR = float32(1 / float32(rank)), mean over queries.

Compact form used here (equal to the official one, asserted by checks()):
  * Conflicts: eval_set copied into sorted int64 codes (t-index, s, r) + CSR objects; equality with the true objects of
    the split asserted for every key, and with query_batch's negative arrays on the first 2,000 queries.
  * filtered_rr: a scorer gives each query a background score b(e) shared by a group of queries (one dense vector per
    group, stored as its sorted distinct values and cumulative counts) and a full score f(e) for a sparse set of special
    entities (every entity whose score differs from b, plus the conflict set). Then, exactly,
      #{neg e : score(e) > x} = #{all e : b(e) > x} - #{special e : b(e) > x} + #{special negative e : f(e) > x}
    (same for >=), so a one-vs-all rank over N entities costs O(log N + #specials). dense_rr is the direct definition,
    used by the contracts.
"""
import gc, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / '.cache/pylib_tgb')); sys.path.insert(0, str(HERE))
import safe_tgb


def trim():
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass


def vm():
    out = {}
    for line in open('/proc/self/status'):
        if line.startswith(('VmRSS', 'VmHWM', 'VmPeak')):
            k, v = line.split(':'); out[k + '_kb'] = int(v.split()[0])
    return out


def reformat_ts(timestamps, dataset_name='tkgl'):
    """Verbatim logic of TGB modules/tkg_utils.reformat_ts (RecB's time steps): start at 0, step = dataset granularity."""
    all_ts = np.unique(timestamps); ts_min = all_ts.min()
    diffs = np.diff(all_ts); dmean, dmin = diffs.mean(), diffs.min()
    if dmean != dmin:
        if 'icews' in dataset_name or 'polecat' in dataset_name:
            ts_dist = 86400
        elif 'wiki' in dataset_name or 'yago' in dataset_name:
            ts_dist = 31536000
        else:
            ts_dist = dmean
        if ts_dist - dmean > 0.1 * dmean:
            ts_dist = dmean
    else:
        ts_dist = dmean
    return np.ceil((timestamps - ts_min) / ts_dist).astype(np.int64)


class TKG:
    """Official loader + arrays. s, o, t, r over all rows of full_data (originals then inverses, as py-tgb orders them);
    tix = index of t among the distinct timestamps; rts = RecB's reformatted time step."""
    def __init__(self, name, root='../../../data/tgb/root'):
        safe_tgb.patch_tgb()
        import tgb.linkproppred.tkg_negative_sampler as tkg_ns
        tkg_ns.load_pkl = safe_tgb.load_pkl
        import tgb.linkproppred.dataset as dsm
        from tgb.linkproppred.evaluate import Evaluator
        dsm.load_pkl = safe_tgb.load_pkl
        self.name = name
        self.ds = ds = dsm.LinkPropPredDataset(name=name, root=root, preprocess=True, download=False)
        self.ev = Evaluator(name=name)
        d = ds.full_data
        for k in ('edge_feat', 'w', 'edge_label', 'edge_idxs'):
            d.pop(k, None)
        ds._edge_feat = None; gc.collect(); trim()
        self.s, self.o, self.t, self.r = (np.asarray(d[k], np.int64) for k in ('sources', 'destinations', 'timestamps', 'edge_type'))
        self.n = int(max(self.s.max(), self.o.max())) + 1; self.R2 = int(self.r.max()) + 1; self.R = self.R2 // 2
        self.ut = np.unique(self.t); self.tix = np.searchsorted(self.ut, self.t).astype(np.int64)
        self.rts = reformat_ts(self.t, name)
        self.masks = dict(train=np.asarray(ds.train_mask), val=np.asarray(ds.val_mask), test=np.asarray(ds.test_mask))
        self.first_dst, self.last_dst = int(ds.min_dst_idx), int(ds.max_dst_idx)

    def checks(self):
        s, o, t, r, h = self.s, self.o, self.t, self.r, len(self.s) // 2
        out = dict(rows=len(s), quadruples=h, entities=self.n, relations=self.R, relations_with_inverse=self.R2,
                   timestamps=len(self.ut), first_dst=self.first_dst, last_dst=self.last_dst)
        assert len(np.unique(np.concatenate([s, o]))) == self.n, 'entity ids not contiguous'
        assert self.first_dst == 0 and self.last_dst == self.n - 1, 'one-vs-all range != all entities'
        assert np.array_equal(s[:h], o[h:]) and np.array_equal(o[:h], s[h:]) and np.array_equal(t[:h], t[h:])
        assert np.array_equal(r[:h] + self.R, r[h:]) and r[:h].max() < self.R, 'inverse quadruples'
        assert np.all(np.diff(t[:h]) >= 0), 'original rows time sorted'
        for k, m in self.masks.items():
            out[k] = dict(rows=int(m.sum()), t_first=int(t[m].min()), t_last=int(t[m].max()),
                          snapshots=int(len(np.unique(t[m]))))
        assert t[self.masks['train']].max() < t[self.masks['val']].min() and t[self.masks['val']].max() < t[self.masks['test']].min()
        assert np.all(self.masks['train'] | self.masks['val'] | self.masks['test'])
        return out

    def queries(self, split):
        """Query rows of a split in py-tgb order (originals then inverses, each time sorted)."""
        return np.flatnonzero(self.masks[split])


class Conflicts:
    """eval_set[(t, s, r)] -> true objects, as sorted codes + CSR; the dict is released after the checks."""
    def __init__(self, g, split, check=True, keep=False):
        ds = g.ds; (ds.load_val_ns if split == 'val' else ds.load_test_ns)()
        es = ds.negative_sampler.eval_set[split]; self.g, self.split = g, split
        keys = np.array(list(es.keys()), dtype=np.int64)
        lens = np.fromiter((len(v) for v in es.values()), np.int64, len(es))
        vals = np.concatenate([np.asarray(v, np.int64).ravel() for v in es.values()])
        code = self.code(keys[:, 0], keys[:, 1], keys[:, 2]); o = np.argsort(code)
        ptr = np.concatenate([[0], np.cumsum(lens)])
        self.codes = code[o]; self.lens = lens[o]
        self.ptr = np.concatenate([[0], np.cumsum(self.lens)])
        idx = np.repeat(ptr[:-1][o], self.lens) + (np.arange(self.lens.sum()) - np.repeat(self.ptr[:-1], self.lens))
        self.vals = vals[idx]
        assert np.all(np.diff(self.codes) > 0), 'duplicate keys'
        self.report = dict(keys=int(len(self.codes)), objects=int(self.lens.sum()), max_conflict=int(self.lens.max()))
        if check:
            self._check(es)
        del es
        if not keep:
            self.release()

    def release(self):
        self.g.ds.negative_sampler.eval_set[self.split] = None; gc.collect(); trim()

    def code(self, t, s, r):
        g = self.g
        tix = np.searchsorted(g.ut, t)
        assert np.all(g.ut[np.minimum(tix, len(g.ut) - 1)] == t)
        return (tix.astype(np.int64) * g.n + s) * g.R2 + r

    def lookup(self, rows):
        """Key index of each query row (asserted present)."""
        g = self.g; c = (g.tix[rows] * g.n + g.s[rows]) * g.R2 + g.r[rows]
        i = np.minimum(np.searchsorted(self.codes, c), len(self.codes) - 1)
        assert np.all(self.codes[i] == c), 'query not in the official evaluation set'
        return i

    def flat(self, rows):
        """(query position, object) pairs of every query's conflict set."""
        k = self.lookup(rows); L = self.lens[k]
        q = np.repeat(np.arange(len(rows)), L)
        idx = np.repeat(self.ptr[k], L) + (np.arange(L.sum()) - np.repeat(np.cumsum(L) - L, L))
        return q, self.vals[idx]

    def _check(self, es):
        g = self.g; rows = g.queries(self.split)
        c = (g.tix[rows] * g.n + g.s[rows]) * g.R2 + g.r[rows]
        uc = np.unique(c)
        assert np.array_equal(uc, self.codes), 'eval_set keys != (t, s, r) of the split rows'
        oo = np.lexsort((g.o[rows], c)); cs, os_ = c[oo], g.o[rows][oo]
        keep = np.concatenate([[True], (np.diff(cs) != 0) | (np.diff(os_) != 0)])
        cs, os_ = cs[keep], os_[keep]
        mine = np.lexsort((self.vals, np.repeat(self.codes, self.lens)))
        assert np.array_equal(np.repeat(self.codes, self.lens)[mine], cs) and np.array_equal(self.vals[mine], os_), \
            'conflict sets != true objects at t'
        ns = g.ds.negative_sampler; first = rows[:2000]; allv = np.arange(g.first_dst, g.last_dst + 1)
        for b in range(0, len(first), 200):
            j = first[b:b + 200]
            off = ns.query_batch(g.s[j], g.o[j], g.t[j], g.r[j], split_mode=self.split)
            k = self.lookup(j)
            for x, kk in zip(off, k):
                conf = self.vals[self.ptr[kk]:self.ptr[kk + 1]]
                assert np.array_equal(np.asarray(x), np.delete(allv, conf)), 'query_batch != all minus conflict'
        self.report.update(keys_equal_split=True, conflict_sets_equal_true_objects=True,
                           query_batch_equal_first=int(len(first)))


class Background:
    """A group's background score vector over all N entities as sorted distinct values + cumulative counts."""
    def __init__(self, values, n, zero_count=0):
        """values: background scores of the entities listed explicitly; zero_count further entities score 0.0."""
        if zero_count:
            values = np.concatenate([values, [0.0]]); w = np.ones(len(values), np.int64); w[-1] = zero_count
        else:
            w = np.ones(len(values), np.int64)
        u, inv = np.unique(values, return_inverse=True)
        self.u = u; self.cum = np.cumsum(np.bincount(inv, weights=w, minlength=len(u)).astype(np.int64))
        self.n = n; assert self.cum[-1] == n

    def counts(self, x, scale=None):
        """(#{e : b(e) > x}, #{e : b(e) >= x}); scale multiplies the stored values first (monotone, ties exact)."""
        u = self.u if scale is None else scale * self.u
        cr = np.searchsorted(u, x, 'right'); cl = np.searchsorted(u, x, 'left')
        le = np.where(cr > 0, self.cum[np.maximum(cr - 1, 0)], 0); lt = np.where(cl > 0, self.cum[np.maximum(cl - 1, 0)], 0)
        return self.n - le, self.n - lt


def filtered_rr(x, gt_all, ge_all, sq, s_bg, s_full, s_neg):
    """Exact official RR for each query. x [Q] positive score; gt_all / ge_all [Q] background counts over all N entities;
    specials (flat, unique per query, must include the whole conflict set): query position sq, background score s_bg,
    full score s_full, s_neg = the special is a negative (not in the conflict set)."""
    Q = len(x); xq = x[sq]
    gt = gt_all - np.bincount(sq, s_bg > xq, Q) + np.bincount(sq, s_neg & (s_full > xq), Q)
    ge = ge_all - np.bincount(sq, s_bg >= xq, Q) + np.bincount(sq, s_neg & (s_full >= xq), Q)
    rank = 0.5 * (np.rint(gt) + np.rint(ge)) + 1.0
    return (1.0 / rank.astype(np.float32)).astype(np.float32)


def dense_rr(scores, pos, conflict):
    """Direct definition for one query: scores [N] over all entities, pos id, conflict ids."""
    neg = np.ones(len(scores), bool); neg[conflict] = False
    x = scores[pos]; gt = int((scores[neg] > x).sum()); ge = int((scores[neg] >= x).sum())
    return np.float32(1.0 / np.float32(0.5 * (gt + ge) + 1.0))


def official_rr(g, split, row, scores):
    """The official path for one query: query_batch negatives and the Evaluator."""
    neg = g.ds.negative_sampler.query_batch(g.s[[row]], g.o[[row]], g.t[[row]], g.r[[row]], split_mode=split)[0]
    return float(g.ev.eval({'y_pred_pos': np.array([scores[g.o[row]]]), 'y_pred_neg': scores[neg],
                            'eval_metric': ['mrr']})['mrr'])


class SlotIndex:
    """Addressed slot family: events with key k at time index tix, sorted by (key, tix, original order); a read at
    (key, qtix) returns the range [lo, hi) of the key's events with tix < qtix (strictly earlier snapshots)."""
    def __init__(self, keys, tix, payload=None):
        o = np.lexsort((tix, keys)); self.k = keys[o]; self.tix = tix[o]; self.order = o
        self.payload = None if payload is None else payload[o]
        self.u, self.start = np.unique(self.k, return_index=True)
        self.end = np.append(self.start[1:], len(self.k))
        self.comp = self.k * (int(self.tix.max()) + 2) + self.tix if len(self.k) else self.k
        self.mult = int(self.tix.max()) + 2 if len(self.k) else 1

    def range(self, keys, qtix):
        qtix = np.minimum(qtix, self.mult - 1)
        hi = np.searchsorted(self.comp, keys * self.mult + qtix, 'left')
        lo = np.searchsorted(self.comp, keys * self.mult, 'left')
        return lo, hi

    def count_last(self, keys, qtix):
        lo, hi = self.range(keys, qtix); has = hi > lo
        last = np.where(has, self.tix[np.maximum(hi - 1, 0)], -1)
        return hi - lo, last


def expand(lo, hi):
    """Flatten ranges: (owner position, index) for every index in [lo_i, hi_i)."""
    L = hi - lo; q = np.repeat(np.arange(len(lo)), L)
    return q, np.repeat(lo, L) + (np.arange(L.sum()) - np.repeat(np.cumsum(L) - L, L))


class Slots:
    """Addressed slot family keyed by (owner, object): events sorted by (slot, tix). Gives, for a query (owner, tix),
    the distinct objects with an event strictly before tix (enumerated from first occurrences, not from events), and per
    (slot, tix) the last event index before tix; scan() computes per-slot running sums exactly (one vectorised step per
    position within a slot, so slots with identical event times get bit-identical values)."""
    def __init__(self, owner, obj, tix, n):
        key = owner * n + obj; ev = np.lexsort((tix, key)); k = key[ev]
        start = np.flatnonzero(np.concatenate([[True], k[1:] != k[:-1]]))
        self.ev, self.n = ev, n; self.key = k[start]; self.start = start
        self.len = np.diff(np.append(start, len(k))); self.slot = np.repeat(np.arange(len(start)), self.len)
        self.tix = tix[ev]; self.M = int(tix.max()) + 2
        self.comp = self.slot * self.M + self.tix
        self.first = SlotIndex(self.key // n, self.tix[start], payload=np.arange(len(start)))

    def candidates(self, owner_q, tix_q):
        """(query position, slot id) for every object of each query's owner seen before its snapshot."""
        lo, hi = self.first.range(owner_q, tix_q); q, i = expand(lo, hi)
        return q, self.first.payload[i]

    def find(self, owner_q, obj_q):
        """Slot id of (owner, object) or -1."""
        c = owner_q * self.n + obj_q; i = np.minimum(np.searchsorted(self.key, c), len(self.key) - 1)
        return np.where(self.key[i] == c, i, -1)

    def last(self, slot, tix_q):
        """Index (into the sorted events) of the slot's last event before tix_q, or -1 (also for slot -1)."""
        i = np.searchsorted(self.comp, np.maximum(slot, 0) * self.M + np.minimum(tix_q, self.M - 1), 'left') - 1
        ok = (slot >= 0) & (i >= 0)
        ok &= self.slot[np.maximum(i, 0)] == np.maximum(slot, 0)
        return np.where(ok, i, -1)

    def scan(self, w, decay=None):
        """Per-slot running sums A_j = A_{j-1} * decay_j + w_j over the sorted events (decay None: plain sums)."""
        A = np.asarray(w, np.float64).copy()
        if not len(A):
            return A
        o = np.argsort(-self.len, kind='stable'); st = self.start[o]; neg = -self.len[o]
        for p in range(1, int(-neg[0])):
            m = int(np.searchsorted(neg, -p, 'left')); ix = st[:m] + p
            A[ix] = (A[ix - 1] * decay[ix] if decay is not None else A[ix - 1]) + A[ix]
        return A

