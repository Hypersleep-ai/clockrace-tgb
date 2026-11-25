#!/usr/bin/env python3
"""race_tkg: a race of clocks among candidate causes, all-destination (one-vs-all) for TGB 2.0 temporal
knowledge graphs (tkgl-smallpedia, tkgl-polecat, tkgl-icews).

Query (s, r, ?, t): every entity e is a candidate; its hazard is the sum of the hazards of the causes that propose it
(a race of exponential clocks: P(e wins) = lambda(e) / sum_e' lambda(e')):
  REC  recurrency of the addressed (s, r, e) slot: an MLP over the slot's clocks (decayed counts at 1/4/16/64/256 time
       steps, count, age, presence in the last snapshot of (s, r), share of (s, r)) and the relation embedding:
       relation-specific learned clocks; sparse (only objects (s, r) has had).
  TR   relation-transferred recurrency: every (s, r', e) slot with r' != r is a cause keyed by the pair (r, r') (query
       relation embedding and the slot's relation embedding as separate key inputs; the slot clocks as the value);
       hazards of all r' add up; sparse (only objects s has had under another relation, inverse relations included).
  TRANS object-to-object transitions under r: when an (s', r) slot whose last snapshot held objects A (at most 4)
       gains a new object B, the addressed transition slot (r, A -> B) records it; a query (s, r) reads the transition
       slots of the objects in its own last snapshot (a sequential key read); an MLP over the transition clocks gives
       each (A, B) a hazard; sparse (only B some A has led to).
  POP  popularity of e under r: log-linear in the (r, e) slot clocks with relation-specific weights; sparse per
       (relation, snapshot) group over the objects r has had; shared by every query of the group.
  BG   background over ALL entities: log-linear in each entity's activity clocks and its relation bag (an inductive
       type: which relations the entity took part in, projected; no identity tables), relation-specific weights; dense,
       computed once per (relation, snapshot) group.
Exactness: log Z = logsumexp(sum_e BG + sum_supp POP + sum_candidates (REC + TR)) over all N entities, because REC and TR
add hazard only on their sparse candidates. Training minimises the time-aware filtered cross-entropy over all entities
(the other true objects of (s, r, t) removed from Z exactly as the official negatives remove them); its gradient gives
every losing candidate's causes and the background their counterfactual credit (minus their share of Z), and the
causes of the true object their responsibility share. Evaluation ranks the true object against all entities with the
official tie rule via tkg_eval.filtered_rr (dense background per group + sparse specials).

Memory: persistent addressed slot families over every event strictly before the query snapshot (ties excluded), never
reset; reads are exact (decayed clocks A_j = A_{j-1} exp(-(t_j - t_{j-1}) / T) + 1 per slot, read at t and decayed;
asserted against direct sums by --contract). Weights train over TRAIN in time order and stay fixed on VALIDATION and
TEST, while the memory keeps every earlier snapshot (the TKG single-step protocol of the leaderboard).
Selection on VALIDATION MRR (best epoch). TEST only with --score-test: fresh load, slots rebuilt over every event,
best checkpoint, official TEST conflicts, scored once.
"""
import argparse, gc, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import tkg_eval as te

TAUS = np.array([1.0, 4.0, 16.0, 64.0, 256.0]); J = len(TAUS)
NINF = float('-inf')
F_CAND = J + 2
F_REC, F_TR, F_POP, F_ENT, F_TRANS = J + 8 + F_CAND, J + 6 + F_CAND, J + 4, J + 3, J + 5 + F_CAND
K_TRANS = 4


def l1p(x):
    return np.log1p(np.maximum(x, 0)).astype(np.float32)


class Family:
    """Addressed slot family (owner -> object) with exact causal reads of its clocks."""
    def __init__(self, owner, obj, tix, rts, nobj, clocks=True):
        self.sl = sl = te.Slots(owner, obj, tix, nobj); self.nobj = nobj
        self.rts = rts[sl.ev].astype(np.float64)
        self.pos = np.arange(len(sl.ev)) - np.repeat(sl.start, sl.len)
        self.A = None
        if clocks:
            gap = np.diff(self.rts, prepend=self.rts[:1]); gap[sl.start] = 0.0
            self.A = sl.scan(np.ones((len(gap), J)), decay=np.exp(-gap[:, None] / TAUS)).astype(np.float32)

    def read(self, slot, tix_q, rts_q):
        """has, count, age (time steps since the last event), clocks decayed to the query time, last event time."""
        i = self.sl.last(slot, tix_q); has = i >= 0; j = np.maximum(i, 0)
        dt = np.where(has, rts_q - self.rts[j], 0.0)
        clk = None if self.A is None else np.where(has[:, None], self.A[j] * np.exp(-dt[:, None] / TAUS), 0.0)
        return has, np.where(has, self.pos[j] + 1, 0), dt, clk, np.where(has, self.rts[j], -1.0)

    def count_at(self, slot, tix_q):
        i = self.sl.last(slot, tix_q)
        return np.where(i >= 0, self.pos[np.maximum(i, 0)] + 1, 0)

    def distinct(self, owner_q, tix_q):
        lo, hi = self.sl.first.range(owner_q, tix_q)
        return hi - lo


class Memory:
    """Persistent addressed memory over the events [ev] (time indices tix, time steps rts)."""
    def __init__(self, s, r, o, tix, rts, n, R2, transfer=True):
        self.n, self.R2, self.transfer = n, R2, transfer; z = np.zeros_like(s); rts = rts.astype(np.float64)
        self.sre = Family(r * n + s, o, tix, rts, n)
        self.re = Family(r, o, tix, rts, n)
        self.sr = Family(r * n + s, z, tix, rts, 1)
        self.ent = Family(s, z, tix, rts, 1)
        self.rel = Family(r, z, tix, rts, 1)
        self.bag = Family(s, r, tix, rts, R2, clocks=False)
        self.ent_slot = self.ent.sl.find(np.arange(n), np.zeros(n, np.int64))
        self.n_trans = 0
        if not transfer:
            return
        self.s_re = Family(s, r * n + o, tix, rts, R2 * n)
        self.sr_ev = te.SlotIndex(r * n + s, tix, payload=o)
        c_b, _ = te.SlotIndex((r * n + s) * n + o, tix).count_last((r * n + s) * n + o, tix)
        A, row, _, _ = self.last_objects(s, r, tix, keep=c_b == 0)
        u = np.unique(row * n + A); row, A = u // n, u % n; self.n_trans = len(A)
        self.trans = Family(r[row] * n + A, o[row], tix[row], rts[row], n)
        self.trans_out = Family(r[row] * n + A, np.zeros_like(A), tix[row], rts[row], 1)

    def cand_feats(self, e, tix, rts):
        """The candidate entity's own activity (both directions): clocks, count, age."""
        has, cnt, dt, clk, _ = self.ent.read(self.ent_slot[e], tix, rts)
        return np.concatenate([l1p(clk), l1p(cnt)[:, None], l1p(dt)[:, None]], 1).astype(np.float32)

    def rec_items(self, s, r, tix, rts):
        """REC: every object e the (s, r) slot has had before the query snapshot, with its slot clocks."""
        n = self.n; key = r * n + s
        q, slot = self.sre.sl.candidates(key, tix)
        has, cnt, dt, clk, last = self.sre.read(slot, tix[q], rts[q])
        sslot = self.sr.sl.find(key, np.zeros_like(key))
        _, scnt, sdt, _, slast = self.sr.read(sslot, tix, rts)
        i = self.sr.sl.last(sslot, tix); ltix = np.where(i >= 0, self.sr.sl.tix[np.maximum(i, 0)], 0)
        n_last = scnt - np.where(i >= 0, self.sr.count_at(sslot, ltix), 0)
        dist = self.sre.distinct(key, tix)
        f = np.concatenate([l1p(clk), l1p(cnt)[:, None], l1p(dt)[:, None], (last == slast[q])[:, None],
                            (cnt / np.maximum(scnt[q], 1))[:, None], l1p(scnt[q])[:, None], l1p(sdt[q])[:, None],
                            l1p(n_last[q])[:, None], l1p(dist[q])[:, None]], 1).astype(np.float32)
        e = self.sre.sl.key[slot] % n
        return q, e, np.concatenate([f, self.cand_feats(e, tix[q], rts[q])], 1)

    def tr_items(self, s, r, tix, rts):
        """TR: every (s, r', e) slot with r' != r before the query snapshot."""
        n = self.n; q, slot = self.s_re.sl.candidates(s, tix)
        obj = self.s_re.sl.key[slot] % (self.R2 * n); rp = obj // n; e = obj % n
        keep = rp != r[q]; q, slot, rp, e = q[keep], slot[keep], rp[keep], e[keep]
        has, cnt, dt, clk, last = self.s_re.read(slot, tix[q], rts[q])
        _, ecnt, edt, _, elast = self.ent.read(self.ent_slot[s], tix, rts)
        dist = self.s_re.distinct(s, tix)
        f = np.concatenate([l1p(clk), l1p(cnt)[:, None], l1p(dt)[:, None], (last == elast[q])[:, None],
                            l1p(ecnt[q])[:, None], l1p(edt[q])[:, None], l1p(dist[q])[:, None]], 1).astype(np.float32)
        return q, e, rp, np.concatenate([f, self.cand_feats(e, tix[q], rts[q])], 1)

    def last_objects(self, s, r, tix, keep=None):
        """Objects A in the last snapshot (before tix) of each (s, r) with at most K_TRANS of them: returns
        (A, B = row's own object, relation, tix) per (row, A) for transitions, or the query positions for reads."""
        key = r * self.n + s; cnt, last = self.sr_ev.count_last(key, tix)
        _, a = self.sr_ev.range(key, last); _, b = self.sr_ev.range(key, last + 1)
        ok = (cnt > 0) & (b - a <= K_TRANS) & (b > a)
        if keep is not None:
            ok &= keep
        q, i = te.expand(np.where(ok, a, 0), np.where(ok, b, 0))
        return self.sr_ev.payload[i], q, r[q], tix[q]

    def trans_items(self, s, r, tix, rts):
        """TRANS: every B reached by a transition slot (r, A -> B) from an object A of the query's last (s, r) snapshot."""
        n = self.n; A, qa, ra, ta = self.last_objects(s, r, tix)
        u = np.unique(qa * n + A); qa, A = u // n, u % n
        k, slot = self.trans.sl.candidates(r[qa] * n + A, tix[qa])
        q = qa[k]; B = self.trans.sl.key[slot] % n
        has, cnt, dt, clk, last = self.trans.read(slot, tix[q], rts[q])
        oslot = self.trans_out.sl.find(r[q] * n + A[k], np.zeros(len(k), np.int64))
        ocnt = self.trans_out.count_at(oslot, tix[q])
        nA = np.bincount(qa, minlength=len(s))
        f = np.concatenate([l1p(clk), l1p(cnt)[:, None], l1p(dt)[:, None], l1p(ocnt)[:, None],
                            (cnt / np.maximum(ocnt, 1))[:, None], l1p(nA[q])[:, None]], 1).astype(np.float32)
        return q, B, np.concatenate([f, self.cand_feats(B, tix[q], rts[q])], 1)

    def pop_items(self, ru, tix, rts):
        """POP: every object of each relation in ru before the snapshot (one group per relation)."""
        X = np.full(len(ru), tix); k, slot = self.re.sl.candidates(ru, X)
        has, cnt, dt, clk, last = self.re.read(slot, np.full(len(k), tix), np.full(len(k), rts, np.float64))
        _, rcnt, _, _, rlast = self.rel.read(self.rel.sl.find(ru, np.zeros_like(ru)), X, np.full(len(ru), rts, np.float64))
        f = np.concatenate([l1p(clk), l1p(cnt)[:, None], l1p(dt)[:, None], (last == rlast[k])[:, None],
                            l1p(rcnt[k])[:, None]], 1).astype(np.float32)
        return k, self.re.sl.key[slot] % self.n, f

    def ent_feats(self, tix, rts):
        """BG: dense entity activity features at one snapshot, and the relation bag (entity, relation, log1p count)."""
        n = self.n; X = np.full(n, tix)
        has, cnt, dt, clk, _ = self.ent.read(self.ent_slot, X, np.full(n, rts, np.float64))
        f = np.concatenate([l1p(clk), l1p(cnt)[:, None], l1p(dt)[:, None], (~has)[:, None]], 1).astype(np.float32)
        e, slot = self.bag.sl.candidates(np.arange(n), X)
        return f, e, self.bag.sl.key[slot] % self.R2, l1p(self.bag.count_at(slot, X[e]))

    def query_class(self, s, r, o, tix):
        n = self.n
        rep = self.sre.sl.last(self.sre.sl.find(r * n + s, o), tix) >= 0
        srk = self.sr.sl.last(self.sr.sl.find(r * n + s, np.zeros_like(s)), tix) >= 0
        ek = self.ent.sl.last(self.ent_slot[s], tix) >= 0
        return np.where(rep, 0, np.where(srk, 1, np.where(ek, 2, 3)))


class MLP(nn.Module):
    def __init__(self, nin, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Race(nn.Module):
    def __init__(self, R2, d=16, hidden=32, k=16, n=1):
        super().__init__()
        self.rel = nn.Embedding(R2, d); nn.init.normal_(self.rel.weight, std=0.1)
        self.n_rec, self.n_tr = nn.LayerNorm(F_REC), nn.LayerNorm(F_TR)
        self.rec = MLP(F_REC + d, hidden); self.tr = MLP(F_TR + 2 * d, hidden)
        self.n_trans = nn.LayerNorm(F_TRANS); self.trans = MLP(F_TRANS + d, hidden)
        self.pop_u = nn.Linear(d, F_POP); self.pop_c = nn.Embedding(R2, 1)
        self.ent = nn.Linear(F_ENT, k); self.typ = nn.Parameter(torch.randn(R2, k) * 0.05)
        self.bg_a = nn.Linear(d, k); self.bg_b = nn.Embedding(R2, 1)
        nn.init.constant_(self.bg_b.weight, -float(np.log(max(n, 2)))); nn.init.constant_(self.pop_c.weight, -0.5 * float(np.log(max(n, 2))))

    def forward(self, b):
        """b: batch dict of numpy arrays (see build_batch). Returns float64 tensors."""
        R = self.rel.weight; t = torch.from_numpy
        rq = t(b['r'])
        lrec = self.rec(torch.cat([self.n_rec(t(b['rec_f'])), R[rq[t(b['rec_q'])]]], 1)).double()
        ltr = self.tr(torch.cat([self.n_tr(t(b['tr_f'])), R[rq[t(b['tr_q'])]], R[t(b['tr_rp'])]], 1)).double()
        ltrans = self.trans(torch.cat([self.n_trans(t(b['trans_f'])), R[rq[t(b['trans_q'])]]], 1)).double()
        ru = t(b['ru'])
        u = self.pop_u(R[ru])
        lpop = (self.pop_c.weight[ru, 0][t(b['pop_k'])] + (t(b['pop_f']) * u[t(b['pop_k'])]).sum(1)).double()
        bag = torch.sparse_coo_tensor(torch.stack([t(b['bag_e']), t(b['bag_r'])]), t(b['bag_c']),
                                      (b['n'], self.typ.shape[0])).coalesce()
        H = torch.tanh(self.ent(t(b['ent_f'])) + torch.sparse.mm(bag, self.typ))
        lg = (self.bg_b.weight[ru] + self.bg_a(R[ru]) @ H.T).double()
        return lrec, ltr, ltrans, lpop, lg


def seg_lse(x, seg, nseg):
    """Segment logsumexp (empty segments -> -inf)."""
    m = torch.full((nseg,), NINF, dtype=x.dtype).scatter_reduce(0, seg, x.detach(), 'amax', include_self=True)
    mm = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s = torch.zeros(nseg, dtype=x.dtype).index_add(0, seg, torch.exp(x - mm[seg]))
    return torch.where(s > 0, torch.log(s) + mm, torch.full_like(s, NINF))


def build_batch(mem, s, r, o, tix, rts, conf_q, conf_e):
    """Items of a batch of queries at ONE snapshot (time index tix, time step rts). conf_q / conf_e: (query position,
    object) of each query's conflict set (the true objects of (s, r) at t, positive included)."""
    n = mem.n; Q = len(s); X = np.full(Q, tix); RT = np.full(Q, rts, np.float64)
    rq, rqe, rf = mem.rec_items(s, r, X, RT)
    if mem.transfer:
        tq, tqe, trp, tf = mem.tr_items(s, r, X, RT)
        xq, xqe, xf = mem.trans_items(s, r, X, RT)
    else:
        tq = tqe = trp = xq = xqe = np.zeros(0, np.int64); tf = np.zeros((0, F_TR), np.float32)
        xf = np.zeros((0, F_TRANS), np.float32)
    ru, rinv = np.unique(r, return_inverse=True)
    pk, pe, pf = mem.pop_items(ru, tix, rts)
    ef, be, br, bc = mem.ent_feats(tix, rts)
    u = np.unique(np.concatenate([rq * n + rqe, tq * n + tqe, xq * n + xqe, conf_q * n + conf_e, np.arange(Q) * n + o]))
    sq, se = u // n, u % n
    pos = np.searchsorted(u, np.arange(Q) * n + o)
    rec_s = np.searchsorted(u, rq * n + rqe); tr_s = np.searchsorted(u, tq * n + tqe); x_s = np.searchsorted(u, xq * n + xqe)
    conf = np.isin(u, conf_q * n + conf_e); assert conf[pos].all()
    s_pop = np.full(len(u), -1)
    if len(pk):
        pcode = pk * n + pe; po = np.argsort(pcode); pcs = pcode[po]
        scode = rinv[sq] * n + se; k = np.minimum(np.searchsorted(pcs, scode), len(pcs) - 1)
        s_pop = np.where(pcs[k] == scode, po[k], -1)
    is_cand = np.zeros(len(u), bool); is_cand[rec_s] = True; is_cand[tr_s] = True; is_cand[x_s] = True
    return dict(n=n, Q=Q, r=r, rinv=rinv, ru=ru, rec_q=rq, rec_f=rf, rec_s=rec_s, tr_q=tq, tr_rp=trp, tr_f=tf,
                tr_s=tr_s, trans_q=xq, trans_f=xf, trans_s=x_s, pop_k=pk, pop_e=pe, pop_f=pf, ent_f=ef, bag_e=be, bag_r=br, bag_c=bc, sq=sq, se=se,
                pos=pos, conf=conf, s_pop=s_pop, is_cand=is_cand)


def combine(b, lrec, ltr, ltrans, lpop, lg):
    """Per special: log lambda (all causes) and its parts; per query: log Z over all N entities (exact)."""
    t = torch.from_numpy; S = len(b['sq']); Q = b['Q']; f64 = torch.float64
    l_rec = torch.full((S,), NINF, dtype=f64).index_put((t(b['rec_s']),), lrec)
    l_tr = seg_lse(ltr, t(b['tr_s']), S)
    l_x = seg_lse(ltrans, t(b['trans_s']), S)
    l_extra = torch.logaddexp(torch.logaddexp(l_rec, l_tr), l_x)
    l_g = lg[t(b['rinv'][b['sq']]), t(b['se'])]
    sp = t(b['s_pop'])
    l_p = lpop[sp.clamp(min=0)] if len(lpop) else torch.zeros(S, dtype=f64)
    l_p = torch.where(sp >= 0, l_p, torch.full((S,), NINF, dtype=f64))
    l_bg = torch.logaddexp(l_g, l_p)
    l_all = torch.logaddexp(l_bg, l_extra)
    LG = torch.logsumexp(lg, 1)
    LP = seg_lse(lpop, t(b['pop_k']), len(b['ru'])) if len(lpop) else torch.full((len(b['ru']),), NINF, dtype=f64)
    rq = t(b['rinv']); cand = t(b['is_cand'])
    lc = seg_lse(torch.where(cand, l_extra, torch.full_like(l_extra, NINF)), t(b['sq']), Q)
    logZ = torch.logsumexp(torch.stack([LG[rq], LP[rq], lc]), 0)
    return l_all, l_bg, l_extra, logZ


def filtered_loss(b, l_all, logZ):
    """-log P(o) over all entities minus the other true objects of (s, r, t) (time-aware filtered softmax)."""
    t = torch.from_numpy; sq = t(b['sq']); Q = b['Q']
    lo = l_all[t(b['pos'])]
    other = t(b['conf']).clone(); other[t(b['pos'])] = False
    frac = torch.zeros(Q, dtype=torch.float64).index_add(0, sq[other], torch.exp(l_all[other] - logZ[sq[other]]))
    one_minus = torch.maximum(1.0 - frac, torch.exp(lo - logZ))
    return -lo + logZ + torch.log(one_minus)


def dense_bg(b, lg, lpop):
    """Dense log background per relation group: logaddexp(BG, POP) over all N entities (float64 numpy)."""
    bg = lg.detach().numpy().copy()
    if len(lpop):
        lp = lpop.detach().numpy(); bg[b['pop_k'], b['pop_e']] = np.logaddexp(bg[b['pop_k'], b['pop_e']], lp)
    return bg


def ranks(b, l_extra, lg, lpop):
    """Exact official filtered RR per query: dense background per relation group + sparse specials."""
    bg = dense_bg(b, lg, lpop); n = b['n']
    s_bg = bg[b['rinv'][b['sq']], b['se']]
    s_full = np.logaddexp(s_bg, l_extra.detach().numpy())
    x = s_full[b['pos']]
    gt = np.empty(b['Q']); ge = np.empty(b['Q'])
    o = np.argsort(b['rinv'], kind='stable'); cuts = np.searchsorted(b['rinv'][o], np.arange(len(b['ru']) + 1))
    for i in range(len(b['ru'])):
        m = o[cuts[i]:cuts[i + 1]]
        if len(m):
            gt[m], ge[m] = te.Background(bg[i], n).counts(x[m])
    return te.filtered_rr(x, gt, ge, b['sq'], s_bg, s_full, ~b['conf'])


class DataConflicts:
    """True objects of (s, r) at t among the given rows (the official conflict-set definition, used for TRAIN)."""
    def __init__(self, g, rows):
        c = (g.tix[rows] * g.n + g.s[rows]) * g.R2 + g.r[rows]; o = np.lexsort((g.o[rows], c))
        self.c, self.v = c[o], g.o[rows][o]; self.g = g

    def flat(self, rows):
        g = self.g; c = (g.tix[rows] * g.n + g.s[rows]) * g.R2 + g.r[rows]
        lo = np.searchsorted(self.c, c, 'left'); hi = np.searchsorted(self.c, c, 'right'); assert (hi > lo).all()
        q, i = te.expand(lo, hi)
        return q, self.v[i]


def snapshot_batches(g, rows, bs, rng=None):
    """Batches of query rows within one snapshot each, grouped by relation; snapshots in time order."""
    o = np.lexsort((g.r[rows], g.tix[rows])); rows = rows[o]; tx = g.tix[rows]
    cuts = np.flatnonzero(np.diff(tx)) + 1; out = []
    for chunk in np.split(rows, cuts):
        parts = [chunk[i:i + bs] for i in range(0, len(chunk), bs)]
        if rng is not None:
            parts = [parts[i] for i in rng.permutation(len(parts))]
        out.extend(parts)
    return out


def run_batch(g, mem, model, rows, conflicts):
    s, r, o = g.s[rows], g.r[rows], g.o[rows]; tix = int(g.tix[rows[0]]); rts = float(g.rts[rows[0]])
    assert (g.tix[rows] == tix).all()
    cq, ce = conflicts.flat(rows)
    b = build_batch(mem, s, r, o, tix, rts, cq, ce)
    parts = model(b)
    return b, parts, combine(b, *parts)


def evaluate(g, mem, model, rows, conflicts, bs, max_q=0):
    if max_q:
        rows = rows[:max_q]
    model.eval(); out, cls, nitems = [], [], dict(rec=0, tr=0, trans=0, pop=0, specials=0); t0 = time.time(); order = []
    with torch.no_grad():
        for rb in snapshot_batches(g, rows, bs):
            b, parts, (l_all, l_bg, l_extra, logZ) = run_batch(g, mem, model, rb, conflicts)
            out.append(ranks(b, l_extra, parts[4], parts[3])); order.append(rb)
            cls.append(mem.query_class(g.s[rb], g.r[rb], g.o[rb], g.tix[rb]))
            for k, key in (('rec', 'rec_q'), ('tr', 'tr_q'), ('trans', 'trans_q'), ('pop', 'pop_k'), ('specials', 'sq')):
                nitems[k] += len(b[key])
    v = np.concatenate(out).astype(np.float64); cls = np.concatenate(cls); order = np.concatenate(order)
    names = ('repeat_sro', 'new_object_known_sr', 'new_sr_known_subject', 'new_subject')
    prof = {nm: dict(queries=int((cls == i).sum()), mrr=float(v[cls == i].mean()) if (cls == i).any() else None)
            for i, nm in enumerate(names)}
    inv = g.r[order] >= g.R
    prof['direction'] = dict(original=float(v[~inv].mean()), inverse=float(v[inv].mean()) if inv.any() else None)
    return float(v.mean()), len(v), time.time() - t0, prof, nitems


def build_memory(g, upto_split, transfer=True):
    m = g.masks['train'] | (g.masks['val'] if upto_split in ('val', 'test') else False) | (g.masks['test'] if upto_split == 'test' else False)
    ev = np.flatnonzero(m)
    return Memory(g.s[ev], g.r[ev], g.o[ev], g.tix[ev], g.rts[ev], g.n, g.R2, transfer)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_model(a, R2, n):
    return Race(R2, d=a.d, hidden=a.hidden, k=a.k, n=n)


class _G:
    """Tiny synthetic TKG with the attributes the driver reads."""
    def __init__(self, rng, n=25, R2=6, E=700, T=30):
        self.n, self.R2, self.R = n, R2, R2 // 2
        self.tix = np.sort(rng.integers(0, T, E)); self.rts = (self.tix * 2).astype(np.int64)
        self.s, self.r, self.o = rng.integers(0, n, E), rng.integers(0, R2, E), rng.integers(0, n, E)


def contracts():
    """(1) slot clocks and counts equal direct sums over strictly earlier snapshots (ties excluded); (2) the item sets
    equal brute-force enumerations; (3) the sparse + background decomposition of log Z and of the filtered loss equals a
    dense computation of every entity's hazard from brute-force features; (4) ranks equal the dense definition."""
    torch.manual_seed(0); rng = np.random.default_rng(0); g = _G(rng)
    mem = Memory(g.s, g.r, g.o, g.tix, g.rts, g.n, g.R2)
    model = Race(g.R2, d=4, hidden=8, k=4).double()
    for p in model.parameters():
        p.data.normal_(0, 0.5)
    model = model.float()
    n, R2 = g.n, g.R2; worst = dict(feat=0.0, logZ=0.0, loss=0.0, rr=0.0); nq = 0; trans_n = [0]

    def clocks(m, tq):
        dt = (tq - g.rts[m]).astype(np.float64)
        return np.exp(-dt[:, None] / TAUS).sum(0) if m.any() else np.zeros(J)

    def cand(e, tix, tq):
        m = (g.s == e) & (g.tix < tix)
        return np.concatenate([l1p(clocks(m, tq)), [l1p(m.sum()), l1p(tq - g.rts[m].max()) if m.any() else 0.0]])

    for tix in (5, 12, 21, 29):
        rows = np.flatnonzero(g.tix == tix)
        if not len(rows):
            continue
        rts = float(g.rts[rows[0]]); conf = DataConflicts(g, rows)
        cq, ce = conf.flat(rows)
        b = build_batch(mem, g.s[rows], g.r[rows], g.o[rows], tix, rts, cq, ce)
        with torch.no_grad():
            lrec, ltr, ltrans, lpop, lg = model(b)
            l_all, l_bg, l_extra, logZ = combine(b, lrec, ltr, ltrans, lpop, lg)
            loss = filtered_loss(b, l_all, logZ).numpy(); rr = ranks(b, l_extra, lg, lpop)
            R = model.rel.weight
            past = g.tix < tix
            H_f = np.zeros((n, F_ENT), np.float32); bagd = np.zeros((n, R2), np.float32)
            for e in range(n):
                m = past & (g.s == e)
                H_f[e, :J] = l1p(clocks(m, rts)); H_f[e, J] = l1p(m.sum())
                H_f[e, J + 1] = l1p(rts - g.rts[m].max()) if m.any() else 0.0; H_f[e, J + 2] = float(not m.any())
                for rr_ in range(R2):
                    bagd[e, rr_] = l1p((m & (g.r == rr_)).sum())
            assert np.abs(H_f - b['ent_f']).max() < 1e-5
            H = torch.tanh(model.ent(torch.from_numpy(H_f)) + torch.from_numpy(bagd) @ model.typ)
            for qi, row in enumerate(rows):
                s, r, o = g.s[row], g.r[row], g.o[row]; nq += 1
                lam = np.zeros(n)
                lam_g = (model.bg_b.weight[r] + model.bg_a(R[r]) @ H.T).double().numpy()
                lam += np.exp(lam_g)
                mr = past & (g.r == r); rlast = g.rts[mr].max() if mr.any() else -1
                rec_set, tr_set, pop_set = set(), set(), set()
                for e in range(n):
                    m = mr & (g.o == e)
                    if m.any():
                        pop_set.add(e)
                        f = np.concatenate([l1p(clocks(m, rts)), [l1p(m.sum()), l1p(rts - g.rts[m].max()),
                                            float(g.rts[m].max() == rlast), l1p(mr.sum())]]).astype(np.float32)
                        u = model.pop_u(R[r]); lam[e] += np.exp(float(model.pop_c.weight[r, 0] + (torch.from_numpy(f) * u).sum()))
                msr = past & (g.s == s) & (g.r == r)
                if msr.any():
                    slast = g.rts[msr].max(); ltx = g.tix[msr].max(); nl = (msr & (g.tix == ltx)).sum()
                    dist = len(set(g.o[msr].tolist()))
                    for e in set(g.o[msr].tolist()):
                        m = msr & (g.o == e); rec_set.add(e)
                        f = np.concatenate([l1p(clocks(m, rts)), [l1p(m.sum()), l1p(rts - g.rts[m].max()),
                                            float(g.rts[m].max() == slast), m.sum() / msr.sum(), l1p(msr.sum()),
                                            l1p(rts - slast), l1p(nl), l1p(dist)], cand(e, tix, rts)]).astype(np.float32)
                        x = torch.cat([model.n_rec(torch.from_numpy(f)[None]), R[r][None]], 1)
                        lam[e] += np.exp(float(model.rec(x)))
                ms = past & (g.s == s)
                if ms.any():
                    elast = g.rts[ms].max(); dist = len(set(zip(g.r[ms].tolist(), g.o[ms].tolist())))
                    for rp, e in set(zip(g.r[ms].tolist(), g.o[ms].tolist())):
                        if rp == r:
                            continue
                        m = ms & (g.r == rp) & (g.o == e); tr_set.add((rp, e))
                        f = np.concatenate([l1p(clocks(m, rts)), [l1p(m.sum()), l1p(rts - g.rts[m].max()),
                                            float(g.rts[m].max() == elast), l1p(ms.sum()), l1p(rts - elast), l1p(dist)],
                                            cand(e, tix, rts)]).astype(np.float32)
                        x = torch.cat([model.n_tr(torch.from_numpy(f)[None]), R[r][None], R[rp][None]], 1)
                        lam[e] += np.exp(float(model.tr(x)))
                if msr.any():
                    ltx = g.tix[msr].max(); Aset = sorted(set(g.o[msr & (g.tix == ltx)].tolist()))
                    if len(Aset) <= K_TRANS:
                        tx_items = {}
                        for j in np.flatnonzero(past & (g.r == r)):
                            mj = (g.s == g.s[j]) & (g.r == g.r[j]) & (g.tix < g.tix[j])
                            if not mj.any() or (mj & (g.o == g.o[j])).any():
                                continue
                            lj = g.tix[mj].max(); Aj = set(g.o[mj & (g.tix == lj)].tolist())
                            if len(Aj) > K_TRANS:
                                continue
                            for a_ in Aj:
                                tx_items.setdefault((a_, int(g.o[j])), set()).add(int(j))
                        out_cnt = {}
                        for (a_, b_), js in tx_items.items():
                            out_cnt.setdefault(a_, []).extend(js)
                        for (a_, b_), js in tx_items.items():
                            if a_ not in Aset:
                                continue
                            m = np.zeros(len(g.s), bool); m[list(js)] = True
                            oc = len(out_cnt[a_])
                            f = np.concatenate([l1p(clocks(m, rts)), [l1p(m.sum()), l1p(rts - g.rts[m].max()), l1p(oc),
                                                m.sum() / oc, l1p(len(Aset))], cand(b_, tix, rts)]).astype(np.float32)
                            x = torch.cat([model.n_trans(torch.from_numpy(f)[None]), R[r][None]], 1)
                            lam[b_] += np.exp(float(model.trans(x))); trans_n[0] += 1
                assert rec_set == set(b['se'][b['rec_s'][b['rec_q'] == qi]].tolist())
                assert tr_set == set(zip(b['tr_rp'][b['tr_q'] == qi].tolist(), b['se'][b['tr_s'][b['tr_q'] == qi]].tolist()))
                C = ce[cq == qi]
                keep = np.ones(n, bool); keep[C] = False; keep[o] = True
                lz = np.log(lam.sum()); worst['logZ'] = max(worst['logZ'], abs(lz - float(logZ[qi])))
                ref = -np.log(lam[o]) + np.log(lam[keep].sum()); worst['loss'] = max(worst['loss'], abs(ref - loss[qi]))
                worst['rr'] = max(worst['rr'], abs(float(te.dense_rr(np.log(lam), o, C)) - float(rr[qi])))
    assert worst['logZ'] < 1e-4 and worst['loss'] < 1e-4 and worst['rr'] < 1e-6, worst
    worst_clk = 0.0
    for _ in range(300):
        s, r, e = rng.integers(0, n), rng.integers(0, R2), rng.integers(0, n); tq = int(rng.integers(0, 31))
        tq_rts = 2.0 * tq
        slot = mem.sre.sl.find(np.array([r * n + s]), np.array([e]))
        has, cnt, dt, clk, _ = mem.sre.read(slot, np.array([tq]), np.array([tq_rts]))
        m = (g.s == s) & (g.r == r) & (g.o == e) & (g.tix < tq)
        worst_clk = max(worst_clk, float(np.abs(clk[0] - clocks(m, tq_rts)).max())); assert cnt[0] == m.sum()
    assert worst_clk < 1e-4
    worst['clock'] = worst_clk
    assert trans_n[0] > 0, 'contract exercised no transition'
    return dict(max_abs=worst, queries=nq, trans_items=trans_n[0], scope='synthetic TKG (25 entities, 6 relations, 700 events, ties); every '
                'entity hazard recomputed densely from brute-force features')


def score_sealed(a, ckpt, split):
    """Sealed path: fresh load, slots over every event up to the end of `split`, best checkpoint, official conflicts."""
    t0 = time.time(); c0 = time.process_time(); g = te.TKG(a.dataset, a.root); conf = te.Conflicts(g, split, check=True)
    mem = build_memory(g, split, not a.no_transfer); model = make_model(a, g.R2, g.n); model.load_state_dict(torch.load(ckpt))
    mrr, nq, wall, prof, nitems = evaluate(g, mem, model, g.queries(split), conf, a.eval_batch)
    p = 'test' if split == 'test' else 'sealed_path_smoke_val'
    return {f'{p}_mrr': mrr, f'{p}_queries': nq, f'{p}_profile': prof, f'{p}_items': nitems, f'{p}_wall_s': time.time() - t0,
            f'{p}_cpu_s': time.process_time() - c0,
            f'{p}_eval_wall_s': wall, f'{p}_vm': te.vm(), f'{p}_conflicts': conf.report,
            f'{p}_protocol': 'best VALIDATION epoch checkpoint; fresh load; slots rebuilt over every event up to the end of '
                             f'{split.upper()} and read strictly before each query snapshot; official {split.upper()} '
                             'time-filtered conflict sets (checked equal to the true objects and to query_batch), '
                             'one-vs-all over all entities, official tie rule; scored once'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', default='tkgl-smallpedia')
    ap.add_argument('--root', default='../../../data/tgb/root'); ap.add_argument('--contract', action='store_true')
    for k, v in (('seed', 0), ('epochs', 4), ('hidden', 32), ('d', 16), ('k', 16), ('batch', 1024), ('eval_batch', 4096),
                 ('max_train', 0), ('max_val', 0), ('train_from', 0)):
        ap.add_argument('--' + k.replace('_', '-'), type=int, default=v)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--score-test', action='store_true'); ap.add_argument('--sealed-path-smoke', action='store_true')
    ap.add_argument('--no-transfer', action='store_true', help='arm A0: drop the TR and TRANS causes')
    a = ap.parse_args(); start = time.time(); torch.set_num_threads(1)
    assert not (a.score_test and (a.sealed_path_smoke or a.max_train or a.max_val)), 'sealed runs use the full data'
    out = ROOT / 'experiments/results/tgb' / f'{a.tag}.json'; assert not out.exists(), 'fresh tag'
    if a.contract:
        t0 = time.time(); res = dict(status='completed', checks=contracts(), wall_s_contract=time.time() - t0,
                                     scope='race_tkg contracts; no data, no fit')
    else:
        torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
        g = te.TKG(a.dataset, a.root); conf_val = te.Conflicts(g, 'val', check=True)
        t_load = time.time() - start
        cm = time.process_time(); mem = build_memory(g, 'val', not a.no_transfer); t_mem_cpu = time.process_time() - cm
        t_mem = time.time() - start - t_load; mem_vm = te.vm()
        tr = g.queries('train'); tr = tr[g.tix[tr] >= a.train_from]
        if a.max_train:
            tr = tr[np.argsort(g.tix[tr], kind='stable')][-a.max_train:]
        conf_tr = DataConflicts(g, g.queries('train'))
        model = make_model(a, g.R2, g.n); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        ckpt = out.with_suffix('.pt'); best, best_ep, history = -1.0, -1, []
        for ep in range(a.epochs):
            model.train(); t0 = time.time(); c0 = time.process_time(); total = 0.0; nb = 0; nq = 0
            for rb in snapshot_batches(g, tr, a.batch, rng):
                b, parts, (l_all, l_bg, l_extra, logZ) = run_batch(g, mem, model, rb, conf_tr)
                loss = filtered_loss(b, l_all, logZ).mean(); assert torch.isfinite(loss)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                total += float(loss.detach()) * len(rb); nq += len(rb); nb += 1
            tw = time.time() - t0; tc = time.process_time() - c0; c1 = time.process_time()
            mrr, nv, vw, prof, nitems = evaluate(g, mem, model, g.queries('val'), conf_val, a.eval_batch, a.max_val)
            row = dict(epoch=ep, train_loss=total / max(nq, 1), batches=nb, train_queries=nq, val_mrr=mrr, val_queries=nv,
                       train_wall_s=tw, train_cpu_s=tc, val_wall_s=vw, val_cpu_s=time.process_time() - c1, profile=prof,
                       val_items=nitems, vm=te.vm())
            history.append(row); print(json.dumps(row), flush=True)
            if mrr > best:
                best, best_ep = mrr, ep; torch.save(model.state_dict(), ckpt)
        res = dict(status='completed', args=vars(a), dataset=a.dataset, entities=g.n, relations_with_inverse=g.R2,
                   transitions=int(mem.n_trans), memory_cpu_s=t_mem_cpu,
                   parameters=sum(p.numel() for p in model.parameters()), val_mrr=best, best_epoch=best_ep,
                   history=history, checkpoint=str(ckpt.relative_to(ROOT)), trained_queries=int(len(tr)) * a.epochs,
                   load_wall_s=t_load, memory_build_wall_s=t_mem, memory_vm=mem_vm, val_conflicts=conf_val.report,
                   protocol='Strictly causal addressed memory (every snapshot before the query, ties excluded); exact '
                            'one-vs-all filtered softmax over all entities for training; official time-filtered VALIDATION '
                            'conflict sets, one-vs-all over all entities, official tie rule; no TEST')
        if a.score_test or a.sealed_path_smoke:
            del mem, g, conf_val, conf_tr; gc.collect(); te.trim()
            res.update(score_sealed(a, ckpt, 'test' if a.score_test else 'val'))
            if a.score_test:
                res['protocol'] = res['protocol'].replace('; no TEST', '; TEST scored once with the best VALIDATION checkpoint')
    res.update(tag=a.tag, benchmark='TGB 2.0 TKG', driver='race_tkg.py', score_test=a.score_test,
               wall_s=time.time() - start, cpu_s=time.process_time(), max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__).resolve(), HERE / 'tkg_eval.py',
                                                                         HERE / 'safe_tgb.py')})
    with out.open('x') as f:
        json.dump(res, f, indent=1); f.write('\n')
    print(json.dumps({k: v for k, v in res.items() if k not in ('history', 'args')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
