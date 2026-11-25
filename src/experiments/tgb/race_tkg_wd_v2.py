#!/usr/bin/env python3
"""race_tkg_wd_v2: race_tkg_wd (v1, unchanged) + bit-reproducible evaluation (deterministic CUDA
algorithms and cuBLAS workspace while scoring, so equal candidates tie exactly and the sealed path reproduces the
in-run VALIDATION MRR), source hashes taken at start, optional cosine learning-rate decay (--lr-decay cos),
training from a year (--train-from, already in v1 of this driver) and a development arm --pop-mlp (POP hazard from an MLP
over the (r, e) popularity clocks and the candidate's activity instead of v1's log-linear POP). Without --pop-mlp the
model, memory, causes and objective are unchanged.

race_tkg_wd description follows.

race_tkg_wd: the tkgl-smallpedia model (race_tkg.py, arm A) ported to tkgl-wikidata on the GPU.

The model is race_tkg.Race itself (same modules, parameters, feature definitions and state dict) over race_tkg.Memory
itself (the persistent addressed slot families, built by the frozen v1 code over every event strictly before each query
snapshot, never reset). What changes is the execution, for a dataset 18x larger with 26x more entities:

  * GPU mirror of the memory (GMemory): every slot family's sorted event arrays, decayed clocks and first-occurrence
    indices are copied to CUDA tensors; finds, last-event reads, distinct counts and range expansions are the same
    searchsorted formulas as tkg_eval.Slots / SlotIndex, vectorised over a batch.
  * Candidate-addressed causes. The protocol ranks the true object against 1,000 listed negatives, so a query needs the
    hazard lambda(e) of its listed candidates only. Every cause is read per candidate by address: REC by the (s, r, e)
    slot, POP by the (r, e) slot, TRANS by the (r, A -> e) slot of each object A in the last (s, r) snapshot (<= 4),
    TR by enumerating s's (r', e) slots (r' != r) and keeping those whose e is listed, BG by the entity's activity clocks
    and relation bag. Each candidate's hazard equals v1's exactly: log lambda(e) = logaddexp(logaddexp(BG, POP),
    logaddexp(REC, TR, TRANS)); --contract checks it against the v1 code path (dense over all entities) on a synthetic
    TKG and on real tkgl-wikidata queries.
  * Training objective: the race over the protocol's own candidate distribution. For each training query the candidate
    set is its true object plus K destinations sampled like the official generator (uniformly from the objects relation
    r has in TRAIN; whole pool plus uniform fill when the pool is smaller than K), with the other true objects of
    (s, r, t) and duplicates removed (time-aware filtering). The loss is -log P(o) of the race among these candidates;
    its gradient credits every losing candidate and each of its causes. (v1's all-entity softmax would enumerate every
    past object of inverse hub slots: 2.1e11 REC items per TRAIN epoch here.)
  * Evaluation: candidates = the true object + the official 1,000 negatives from the memory-mapped store
    (tkg_ns_stream.NegStore), scored as listed, official tie rule; checked against the official Evaluator on 50 queries.
Weights train over TRAIN in time order (batches within one snapshot) and stay fixed on VALIDATION; selection on
VALIDATION MRR (best epoch). TEST only with --score-test: fresh load, memory rebuilt over every event, best checkpoint,
official TEST negatives converted by the sealed path, EdgeBank(tw) TEST computed on the same files, scored once.
"""
import argparse, gc, hashlib, json, os, resource, sys, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import tkg_eval as te
import tkg_ns_stream as ns
import race_tkg as v1
from race_tkg import TAUS, J, K_TRANS, NINF

NAME = 'tkgl-wikidata'
CLASS_NAMES = ('repeat_sro', 'new_object_known_sr', 'new_sr_known_subject', 'new_subject')


def tgb_root():
    from tgb.utils.info import PROJ_DIR
    return os.path.relpath(ROOT / 'data/tgb/root', PROJ_DIR)


def l1p(x):
    return torch.log1p(x.double().clamp(min=0)).float()


def expand(lo, hi):
    """(owner position, index) for every index in [lo_i, hi_i) (torch)."""
    L = (hi - lo).clamp(min=0); dev = lo.device
    q = torch.repeat_interleave(torch.arange(len(lo), device=dev), L)
    st = torch.cumsum(L, 0) - L
    return q, torch.repeat_interleave(lo, L) + (torch.arange(int(L.sum()), device=dev) - torch.repeat_interleave(st, L))


def seg_lse(x, seg, nseg):
    """Segment logsumexp (empty segments -> -inf), device-aware copy of race_tkg.seg_lse."""
    dev = x.device
    m = torch.full((nseg,), NINF, dtype=x.dtype, device=dev).scatter_reduce(0, seg, x.detach(), 'amax', include_self=True)
    mm = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s = torch.zeros(nseg, dtype=x.dtype, device=dev).index_add(0, seg, torch.exp(x - mm[seg]))
    return torch.where(s > 0, torch.log(s) + mm, torch.full_like(s, NINF))


class GF:
    """GPU mirror of a race_tkg.Family (tkg_eval.Slots): same arrays, same read formulas."""
    def __init__(self, fam, dev, clocks=True, first=False, tix=False):
        sl = fam.sl
        T = lambda x, dt: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=dev)
        self.M, self.nobj = int(sl.M), int(sl.n)
        self.key = T(sl.key, torch.int64); self.comp = T(sl.comp, torch.int64)
        self.rts = T(fam.rts, torch.float64); self.pos = T(fam.pos, torch.int32)
        self.A = T(fam.A, torch.float32) if (clocks and fam.A is not None) else None
        if first:
            self.f_comp = T(sl.first.comp, torch.int64); self.f_mult = int(sl.first.mult)
            self.f_pay = T(sl.first.payload, torch.int64)
        if tix:
            self.tix = T(sl.tix, torch.int64)

    def find(self, owner, obj):
        c = owner * self.nobj + obj
        i = torch.searchsorted(self.key, c).clamp(max=len(self.key) - 1)
        return torch.where(self.key[i] == c, i, torch.full_like(i, -1))

    def last(self, slot, tixq):
        sc = slot.clamp(min=0)
        i = torch.searchsorted(self.comp, sc * self.M + tixq.clamp(max=self.M - 1)) - 1
        ic = i.clamp(min=0)
        ok = (slot >= 0) & (i >= 0) & (torch.div(self.comp[ic], self.M, rounding_mode='floor') == sc)
        return torch.where(ok, i, torch.full_like(i, -1))

    def read(self, slot, tixq, rtsq):
        i = self.last(slot, tixq); has = i >= 0; j = i.clamp(min=0)
        dt = torch.where(has, rtsq - self.rts[j], torch.zeros_like(rtsq))
        clk = None
        if self.A is not None:
            clk = torch.where(has[:, None], self.A[j].double() * torch.exp(-dt[:, None] / TAUS_T['t']), 0.0)
        cnt = torch.where(has, self.pos[j].long() + 1, 0)
        return has, cnt, dt, clk, torch.where(has, self.rts[j], -1.0)

    def count_at(self, slot, tixq):
        i = self.last(slot, tixq)
        return torch.where(i >= 0, self.pos[i.clamp(min=0)].long() + 1, 0)

    def first_range(self, owner, tixq):
        tq = tixq.clamp(max=self.f_mult - 1)
        return (torch.searchsorted(self.f_comp, owner * self.f_mult),
                torch.searchsorted(self.f_comp, owner * self.f_mult + tq))


TAUS_T = {}


class GMemory:
    """GPU mirror of race_tkg.Memory (arm A: REC, TR, TRANS, POP, BG)."""
    def __init__(self, mem, dev):
        assert mem.transfer, 'arm A'
        TAUS_T['t'] = torch.as_tensor(TAUS, dtype=torch.float64, device=dev)
        self.n, self.R2, self.dev = mem.n, mem.R2, dev
        self.sre = GF(mem.sre, dev, first=True)
        self.re = GF(mem.re, dev)
        self.sr = GF(mem.sr, dev, clocks=False, tix=True)
        self.ent = GF(mem.ent, dev)
        self.rel = GF(mem.rel, dev, clocks=False)
        self.bag = GF(mem.bag, dev, clocks=False)
        self.s_re = GF(mem.s_re, dev, first=True)
        self.trans = GF(mem.trans, dev); self.trans_out = GF(mem.trans_out, dev, clocks=False)
        si = mem.sr_ev
        self.ev_comp = torch.as_tensor(si.comp, device=dev); self.ev_mult = int(si.mult)
        self.ev_tix = torch.as_tensor(si.tix, dtype=torch.int64, device=dev)
        self.ev_pay = torch.as_tensor(si.payload, dtype=torch.int64, device=dev)
        self.ent_slot = torch.as_tensor(mem.ent_slot, dtype=torch.int64, device=dev)
        self.bag_slots = torch.arange(len(self.bag.key), device=dev)
        self._snap = (None, None)

    def gpu_bytes(self):
        tot = 0
        for v in self.__dict__.values():
            if isinstance(v, GF):
                tot += sum(t.numel() * t.element_size() for t in v.__dict__.values() if isinstance(t, torch.Tensor))
            elif isinstance(v, torch.Tensor):
                tot += v.numel() * v.element_size()
        return tot

    def snapshot(self, tix, rts):
        """Dense entity activity features [N, 8] and the relation bag (entity, relation, log1p count) at a snapshot
        (race_tkg.Memory.ent_feats), cached per snapshot."""
        if self._snap[0] == tix:
            return self._snap[1]
        n, dev = self.n, self.dev; X = torch.full((n,), tix, device=dev, dtype=torch.int64)
        has, cnt, dt, clk, _ = self.ent.read(self.ent_slot, X, torch.full((n,), rts, dtype=torch.float64, device=dev))
        f = torch.cat([l1p(clk), l1p(cnt)[:, None], l1p(dt)[:, None], (~has).float()[:, None]], 1)
        c = self.bag.count_at(self.bag_slots, torch.full_like(self.bag_slots, tix)); m = c > 0
        key = self.bag.key[m]
        B = torch.sparse_coo_tensor(torch.stack([torch.div(key, self.R2, rounding_mode='floor'), key % self.R2]),
                                    l1p(c[m]).double(), (n, self.R2)).coalesce()
        out = (f, B)
        self._snap = (tix, out)
        return out

    def last_objects(self, key, tixq):
        """Objects A of each query's last (s, r) snapshot when it has at most K_TRANS of them (race_tkg.last_objects)."""
        mult = self.ev_mult; tq = tixq.clamp(max=mult - 1)
        lo = torch.searchsorted(self.ev_comp, key * mult); hi = torch.searchsorted(self.ev_comp, key * mult + tq)
        cnt = hi - lo; last = torch.where(cnt > 0, self.ev_tix[(hi - 1).clamp(min=0)], torch.full_like(hi, -1))
        a = torch.searchsorted(self.ev_comp, key * mult + last.clamp(max=mult - 1))
        b = torch.searchsorted(self.ev_comp, key * mult + (last + 1).clamp(max=mult - 1))
        ok = (cnt > 0) & (b - a <= K_TRANS) & (b > a)
        q, i = expand(torch.where(ok, a, 0), torch.where(ok, b, 0))
        return self.ev_pay[i], q


def score(model, gm, s, r, tix, rts, C, stats=None):
    """log lambda(e) [Q, L] (float64) of every listed candidate C [Q, L] for queries (s, r) at one snapshot: the hazard
    of race_tkg.Race for that entity, all causes read by address."""
    dev = gm.dev; n = gm.n; Q, L = C.shape; R = model.rel.weight
    X = torch.full((Q,), tix, dtype=torch.int64, device=dev); RT = torch.full((Q,), rts, dtype=torch.float64, device=dev)
    F, B = gm.snapshot(tix, rts)
    H = (model.ent(F) + torch.sparse.mm(B, model.typ.double()).float()).tanh()
    cf = C.reshape(-1); qf = torch.arange(Q, device=dev).repeat_interleave(L)
    a = model.bg_a(emb(r, R))
    Hc = H.index_select(0, cf).view(Q, L, -1)
    lg = (emb(r, model.bg_b.weight) + torch.bmm(Hc, a[:, :, None])[:, :, 0]).double()
    lp = torch.full((Q * L,), NINF, dtype=torch.float64, device=dev)
    pslot = gm.re.find(r[qf], cf); pi = torch.nonzero(gm.re.last(pslot, X[qf]) >= 0)[:, 0]
    if len(pi):
        q = qf[pi]; has, cnt, dt, clk, last = gm.re.read(pslot[pi], X[q], RT[q])
        _, rcnt, _, _, rlast = gm.rel.read(gm.rel.find(r, torch.zeros_like(r)), X, RT)
        f = torch.cat([l1p(clk), l1p(cnt)[:, None], l1p(dt)[:, None], (last == rlast[q]).float()[:, None],
                       l1p(rcnt[q])[:, None]], 1)
        fd = torch.zeros(Q * L, f.shape[1], device=dev).index_put((pi,), f).view(Q, L, -1)
        u = model.pop_u(emb(r, R))
        lpd = (emb(r, model.pop_c.weight) + torch.bmm(fd, u[:, :, None])[:, :, 0]).reshape(-1)
        if getattr(model, 'popm', None) is not None:
            e = cf[pi]
            x = torch.cat([model.n_pop(torch.cat([f, F[e, :7]], 1)), emb(r[q], R)], 1)
            lpd = lpd.index_put((pi,), model.popm(x))
        lp = torch.where(lp.new_zeros(Q * L, dtype=torch.bool).index_fill(0, pi, True), lpd.double(), lp)
    key = r * n + s
    lrec = torch.full((Q * L,), NINF, dtype=torch.float64, device=dev)
    slot = gm.sre.find(key[qf], cf); ri = torch.nonzero(gm.sre.last(slot, X[qf]) >= 0)[:, 0]
    if len(ri):
        q = qf[ri]; e = cf[ri]
        has, cnt, dt, clk, last = gm.sre.read(slot[ri], X[q], RT[q])
        sslot = gm.sr.find(key, torch.zeros_like(key))
        _, scnt, sdt, _, slast = gm.sr.read(sslot, X, RT)
        i = gm.sr.last(sslot, X); ltix = torch.where(i >= 0, gm.sr.tix[i.clamp(min=0)], 0)
        n_last = scnt - torch.where(i >= 0, gm.sr.count_at(sslot, ltix), 0)
        lo, hi = gm.sre.first_range(key, X); dist = hi - lo
        f = torch.cat([l1p(clk), l1p(cnt)[:, None], l1p(dt)[:, None], (last == slast[q]).float()[:, None],
                       (cnt.double() / scnt[q].clamp(min=1).double()).float()[:, None], l1p(scnt[q])[:, None],
                       l1p(sdt[q])[:, None], l1p(n_last[q])[:, None], l1p(dist[q])[:, None], F[e, :7]], 1)
        lrec = lrec.index_put((ri,), model.rec(torch.cat([model.n_rec(f), emb(r[q], R)], 1)).double())
    ltr = torch.full((Q * L,), NINF, dtype=torch.float64, device=dev)
    lo, hi = gm.s_re.first_range(s, X); tq, idx = expand(lo, hi); n_tr = 0
    if len(tq):
        tslot = gm.s_re.f_pay[idx]; obj = gm.s_re.key[tslot] % (gm.R2 * n)
        rp = torch.div(obj, n, rounding_mode='floor'); e = obj % n
        cc, co = torch.sort(qf * n + cf); code = tq * n + e
        j = torch.searchsorted(cc, code).clamp(max=len(cc) - 1)
        keep = (rp != r[tq]) & (cc[j] == code)
        tq, tslot, rp, e, code = tq[keep], tslot[keep], rp[keep], e[keep], code[keep]; n_tr = len(tq)
        if n_tr:
            has, cnt, dt, clk, last = gm.s_re.read(tslot, X[tq], RT[tq])
            _, ecnt, edt, _, elast = gm.ent.read(gm.ent_slot[s], X, RT)
            lo2, hi2 = gm.s_re.first_range(s, X); dist = hi2 - lo2
            f = torch.cat([l1p(clk), l1p(cnt)[:, None], l1p(dt)[:, None], (last == elast[tq]).float()[:, None],
                           l1p(ecnt[tq])[:, None], l1p(edt[tq])[:, None], l1p(dist[tq])[:, None], F[e, :7]], 1)
            lt = model.tr(torch.cat([model.n_tr(f), emb(r[tq], R), emb(rp, R)], 1)).double()
            uc, inv = torch.unique(code, return_inverse=True)
            lu = seg_lse(lt, inv, len(uc))
            jj = torch.searchsorted(uc, qf * n + cf).clamp(max=len(uc) - 1)
            hit = torch.nonzero(uc[jj] == qf * n + cf)[:, 0]
            ltr = ltr.index_put((hit,), lu[jj[hit]])
    lx = torch.full((Q * L,), NINF, dtype=torch.float64, device=dev); n_x = 0
    A, qa = gm.last_objects(key, X)
    if len(qa):
        nA = torch.bincount(qa, minlength=Q)
        P = len(qa); pf = torch.arange(P, device=dev).repeat_interleave(L); ent_ = C[qa].reshape(-1)
        xslot = gm.trans.find((r[qa] * n + A)[pf], ent_); xi = torch.nonzero(gm.trans.last(xslot, X[qa][pf]) >= 0)[:, 0]
        n_x = len(xi)
        if n_x:
            p = pf[xi]; q = qa[p]; B = ent_[xi]
            has, cnt, dt, clk, last = gm.trans.read(xslot[xi], X[q], RT[q])
            oslot = gm.trans_out.find(r[q] * n + A[p], torch.zeros_like(q)); ocnt = gm.trans_out.count_at(oslot, X[q])
            f = torch.cat([l1p(clk), l1p(cnt)[:, None], l1p(dt)[:, None], l1p(ocnt)[:, None],
                           (cnt.double() / ocnt.clamp(min=1).double()).float()[:, None], l1p(nA[q])[:, None], F[B, :7]], 1)
            lt = model.trans(torch.cat([model.n_trans(f), emb(r[q], R)], 1)).double()
            entry = q * L + (xi % L)
            lx = seg_lse(lt, entry, Q * L)
    l_extra = torch.logaddexp(torch.logaddexp(lrec, ltr), lx)
    l_bg = torch.logaddexp(lg.reshape(-1), lp)
    if stats is not None:
        for k_, v_ in (('pop', len(pi)), ('rec', len(ri)), ('tr', n_tr), ('trans', n_x), ('candidates', Q * L)):
            stats[k_] = stats.get(k_, 0) + int(v_)
    return torch.logaddexp(l_bg, l_extra).reshape(Q, L)


def make_model(R2, d=16, hidden=32, k=16, n=1, pop_mlp=False):
    """race_tkg.Race (v1 parameters, same names); --pop-mlp adds an MLP POP hazard over the (r, e) slot clocks and the
    candidate's activity (replacing the log-linear POP term of v1 for that run)."""
    m = v1.Race(R2, d=d, hidden=hidden, k=k, n=n)
    if pop_mlp:
        m.n_pop = torch.nn.LayerNorm(v1.F_POP + 7); m.popm = v1.MLP(v1.F_POP + 7 + d, hidden)
        torch.nn.init.zeros_(m.popm.net[-1].weight)
        torch.nn.init.constant_(m.popm.net[-1].bias, -0.5 * float(np.log(max(n, 2))))
    return m


def emb(idx, W):
    """W[idx] through F.embedding: its backward sums repeated rows with the embedding kernel (all items of a batch share
    a handful of relations; plain indexing's backward serialises on such repeats)."""
    return torch.nn.functional.embedding(idx, W)


def rr_official(S):
    """Official tie rule on scores S [Q, 1 + negatives] (column 0 = the true object)."""
    pos = S[:, :1]; neg = S[:, 1:]
    rank = 0.5 * ((neg > pos).sum(1) + (neg >= pos).sum(1)).double() + 1.0
    return (1.0 / rank.float()).float()


class Pool:
    """Training negatives like the official dst-time-filtered generator: uniform over the objects relation r has in
    TRAIN; when the pool is smaller than K, the whole pool plus uniform fill from all entities."""
    def __init__(self, g, rows, dev):
        codes = np.unique(g.r[rows] * g.n + g.o[rows])
        self.ent = torch.as_tensor(codes % g.n, device=dev)
        self.ptr = torch.as_tensor(np.searchsorted(codes, np.arange(g.R2 + 1) * g.n), device=dev)
        self.n, self.dev = g.n, dev

    def sample(self, r, K, gen):
        Q = len(r); dev = self.dev; lo = self.ptr[r]; P = self.ptr[r + 1] - lo
        u = torch.rand(Q, K, generator=gen, device=dev)
        big = (lo[:, None] + (u * P[:, None]).long().clamp(max=(P - 1).clamp(min=0)[:, None]))
        col = torch.arange(K, device=dev)[None, :]
        small = (P[:, None] <= K) & (col < P[:, None])
        fill = torch.randint(0, self.n, (Q, K), generator=gen, device=dev)
        neg = torch.where(P[:, None] > K, self.ent[big.clamp(max=len(self.ent) - 1)],
                          torch.where(small, self.ent[(lo[:, None] + col).clamp(max=len(self.ent) - 1)], fill))
        return neg


def train_batch(model, gm, g, rows, conf, pool, K, gen, dev):
    s, r, o = (torch.as_tensor(x, device=dev) for x in (g.s[rows], g.r[rows], g.o[rows]))
    tix = int(g.tix[rows[0]]); rts = float(g.rts[rows[0]]); assert (g.tix[rows] == tix).all()
    neg = pool.sample(r, K, gen)
    srt, _ = torch.sort(neg, 1); dup = torch.zeros_like(srt, dtype=torch.bool); dup[:, 1:] = srt[:, 1:] == srt[:, :-1]
    cq, ce = conf.flat(rows); cc = torch.sort(torch.as_tensor(cq * g.n + ce, device=dev)).values
    Q = len(rows); code = torch.arange(Q, device=dev)[:, None] * g.n + srt
    j = torch.searchsorted(cc, code).clamp(max=len(cc) - 1)
    valid = ~(dup | (cc[j] == code))
    C = torch.cat([o[:, None], srt], 1)
    S = score(model, gm, s, r, tix, rts, C)
    mask = torch.cat([torch.ones(Q, 1, dtype=torch.bool, device=dev), valid], 1)
    Sm = torch.where(mask, S, torch.full_like(S, NINF))
    return (torch.logsumexp(Sm, 1) - S[:, 0]), int(valid.sum())


def evaluate(model, gm, g, store, rows, bs, dev, max_q=0, return_rr=False):
    if max_q:
        rows = rows[:max_q]
    model.eval(); out, order = [], []; stats = {}; t0 = time.time()
    torch.use_deterministic_algorithms(True, warn_only=True)
    with torch.no_grad():
        for rb in v1.snapshot_batches(g, rows, bs):
            k = store.lookup(g.tix[rb], g.s[rb], g.r[rb]); negs, _ = store.negs(k)
            C = torch.as_tensor(np.concatenate([g.o[rb][:, None], negs.astype(np.int64)], 1), device=dev)
            s, r = torch.as_tensor(g.s[rb], device=dev), torch.as_tensor(g.r[rb], device=dev)
            S = score(model, gm, s, r, int(g.tix[rb[0]]), float(g.rts[rb[0]]), C, stats)
            out.append(rr_official(S).cpu().numpy()); order.append(rb)
    torch.use_deterministic_algorithms(False)
    v = np.concatenate(out).astype(np.float64); order = np.concatenate(order)
    return v, order, stats, time.time() - t0


def profile(v, order, cls_all, g):
    cls = cls_all[order]
    p = {nm: dict(queries=int((cls == i).sum()), mrr=float(v[cls == i].mean()) if (cls == i).any() else None)
         for i, nm in enumerate(CLASS_NAMES)}
    inv = g.r[order] >= g.R
    p['direction'] = dict(original=float(v[~inv].mean()), inverse=float(v[inv].mean()) if inv.any() else None)
    return p


def classes(mem, g, rows):
    out = np.empty(len(rows), np.int64)
    for c0 in range(0, len(rows), 500000):
        j = rows[c0:c0 + 500000]
        out[c0:c0 + len(j)] = mem.query_class(g.s[j], g.r[j], g.o[j], g.tix[j])
    full = np.full(len(g.s), -1); full[rows] = out
    return full


def v1_dense_scores(mem, model_cpu, s, r, o, tix, rts, cq, ce, n):
    """v1's own code path: log lambda of every entity for each query (specials: s_full; others: dense background)."""
    b = v1.build_batch(mem, s, r, o, tix, rts, cq, ce)
    with torch.no_grad():
        lrec, ltr, ltrans, lpop, lg = model_cpu(b)
        l_all, l_bg, l_extra, logZ = v1.combine(b, lrec, ltr, ltrans, lpop, lg)
        bg = v1.dense_bg(b, lg, lpop)
    D = bg[b['rinv']]
    s_full = np.logaddexp(bg[b['rinv'][b['sq']], b['se']], l_extra.numpy())
    D[b['sq'], b['se']] = s_full
    return D, len(b['rec_q']), len(b['tr_q']), len(b['trans_q'])


def contract_synthetic(dev):
    torch.manual_seed(0); rng = np.random.default_rng(0); g = v1._G(rng)
    mem = v1.Memory(g.s, g.r, g.o, g.tix, g.rts, g.n, g.R2); gm = GMemory(mem, dev)
    model = v1.Race(g.R2, d=4, hidden=8, k=4)
    for p in model.parameters():
        p.data.normal_(0, 0.5)
    mdev = v1.Race(g.R2, d=4, hidden=8, k=4).to(dev); mdev.load_state_dict(model.state_dict())
    worst, nq, items = 0.0, 0, [0, 0, 0]
    for tix in (5, 12, 21, 29):
        rows = np.flatnonzero(g.tix == tix)
        if not len(rows):
            continue
        rts = float(g.rts[rows[0]]); conf = v1.DataConflicts(g, rows); cq, ce = conf.flat(rows)
        D, a, b_, c = v1_dense_scores(mem, model, g.s[rows], g.r[rows], g.o[rows], tix, rts, cq, ce, g.n)
        items[0] += a; items[1] += b_; items[2] += c
        C = torch.arange(g.n, device=dev)[None, :].repeat(len(rows), 1)
        with torch.no_grad():
            S = score(mdev, gm, torch.as_tensor(g.s[rows], device=dev), torch.as_tensor(g.r[rows], device=dev), tix, rts, C)
        worst = max(worst, float(np.abs(S.cpu().numpy() - D).max())); nq += len(rows)
    assert worst < 1e-5, worst
    assert min(items) > 0, items
    return dict(max_abs_log_hazard=worst, queries=nq, v1_items_rec_tr_trans=items,
                scope='synthetic TKG (25 entities, 6 relations, 700 events, ties): every entity of every query scored by '
                      'the GPU candidate path vs race_tkg v1 build_batch/combine/dense_bg')


def contract_real(g, mem, gm, model, store, rows, dev, nq=256, max_rec=1500):
    """Real tkgl-wikidata queries of one VALIDATION snapshot (non-hub slots, so v1 enumerates them on the CPU): the GPU
    candidate path vs v1's dense scores at the official negatives; and the official Evaluator on 50 queries."""
    from tgb.linkproppred.evaluate import Evaluator
    tix0 = g.tix[rows].min(); rr_ = rows[g.tix[rows] == tix0]
    dist = mem.sre.distinct(g.r[rr_] * g.n + g.s[rr_], g.tix[rr_]); ntr = mem.s_re.distinct(g.s[rr_], g.tix[rr_])
    rng = np.random.default_rng(1); cand = rr_[(dist <= max_rec) & (ntr <= max_rec)]
    rel, cnt = np.unique(g.r[cand], return_counts=True); top = rel[np.argsort(-cnt)[:4]]
    pick = np.concatenate([np.sort(rng.choice(cand[g.r[cand] == x], min(nq // 4, int((g.r[cand] == x).sum())), replace=False))
                           for x in top])
    if getattr(model, 'popm', None) is not None:
        return dict(skipped='the --pop-mlp arm has no v1 counterpart; the v1 contract runs on the v1 model')
    model_cpu = v1.Race(g.R2, d=model.rel.weight.shape[1], hidden=model.rec.net[0].out_features, k=model.typ.shape[1])
    model_cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    worst = 0.0; rr_ok = True; t0 = time.time(); ev = Evaluator(name=NAME); ev_diff = 0.0
    conf = v1.DataConflicts(g, rows)
    for r0 in np.unique(g.r[pick]):
        j = pick[g.r[pick] == r0]
        cq, ce = conf.flat(j)
        D, _, _, _ = v1_dense_scores(mem, model_cpu, g.s[j], g.r[j], g.o[j], int(tix0), float(g.rts[j[0]]), cq, ce, g.n)
        k = store.lookup(g.tix[j], g.s[j], g.r[j]); negs, _ = store.negs(k)
        Cn = np.concatenate([g.o[j][:, None], negs.astype(np.int64)], 1)
        with torch.no_grad():
            S = score(model, gm, torch.as_tensor(g.s[j], device=dev), torch.as_tensor(g.r[j], device=dev), int(tix0),
                      float(g.rts[j[0]]), torch.as_tensor(Cn, device=dev)).cpu().numpy()
        ref = np.take_along_axis(D, Cn, 1)
        worst = max(worst, float(np.abs(S - ref).max()))
        rr_ok &= bool(np.array_equal(rr_official(torch.as_tensor(S)).numpy(), rr_official(torch.as_tensor(ref)).numpy()))
    j = pick[:50]; k = store.lookup(g.tix[j], g.s[j], g.r[j]); negs, _ = store.negs(k)
    Cn = np.concatenate([g.o[j][:, None], negs.astype(np.int64)], 1)
    with torch.no_grad():
        for rr0 in np.unique(g.r[j]):
            m = g.r[j] == rr0
            S = score(model, gm, torch.as_tensor(g.s[j][m], device=dev), torch.as_tensor(g.r[j][m], device=dev), int(tix0),
                      float(g.rts[j[0]]), torch.as_tensor(Cn[m], device=dev)).cpu().numpy()
            mine = rr_official(torch.as_tensor(S)).numpy()
            for x in range(len(S)):
                off = float(ev.eval({'y_pred_pos': S[x, :1], 'y_pred_neg': S[x, 1:], 'eval_metric': ['mrr']})['mrr'])
                ev_diff = max(ev_diff, abs(off - float(mine[x])))
    assert worst < 1e-4 and rr_ok and ev_diff < 1e-7, (worst, rr_ok, ev_diff)
    return dict(queries=int(len(pick)), snapshot_tix=int(tix0), max_abs_log_hazard=worst, rr_equal=rr_ok,
                evaluator_queries=50, evaluator_max_abs=ev_diff, wall_s=time.time() - t0,
                scope='real VALIDATION queries (first snapshot, slots with <= 1,500 objects): GPU candidate path vs race_tkg '
                      'v1 dense scores at the official negatives, with the model being trained')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(g, upto, dev):
    t0 = time.time(); mem = v1.build_memory(g, upto, True); t_np = time.time() - t0
    gm = GMemory(mem, dev)
    return mem, gm, dict(numpy_build_s=t_np, gpu_upload_s=time.time() - t0 - t_np, gpu_bytes=gm.gpu_bytes(),
                         transitions=int(mem.n_trans))


def edgebank_split(g, store, rows, hist_rows, dev, chunk=20000, bs=200):
    """EdgeBank, TGB convention (tkgl_wikidata_protocol.edgebank), on `rows` after the memory absorbed TRAIN
    at init and `hist_rows` in batches of 200 (the official script scores VALIDATION before TEST, updating memory and the
    time window as it goes). With hist_rows empty this is the VALIDATION reproduction."""
    import tkgl_wikidata_protocol as P
    tr = g.queries('train'); n = g.n
    src = np.concatenate([g.s[tr], g.s[hist_rows], g.s[rows]]); dst = np.concatenate([g.o[tr], g.o[hist_rows], g.o[rows]])
    ts = np.concatenate([g.t[tr], g.t[hist_rows], g.t[rows]])
    wr = P.Writes(src, dst, ts, n, dev)
    ttr = g.t[tr]; cur0 = ttr.max(); dur_all = ttr.max() - ttr.min()
    prev0 = ttr.min() + dur_all * (1 - 0.15); duration = cur0 - prev0
    cur_start = float(max(cur0, g.t[hist_rows].max())) if len(hist_rows) else float(cur0)
    Q = len(rows); b = np.arange(Q) // bs; nb = int(b.max()) + 1
    bmax = np.full(nb, -np.inf); np.maximum.at(bmax, b, g.t[rows].astype(np.float64))
    cur = np.maximum.accumulate(np.concatenate([[cur_start], bmax]))[:-1]
    prev = np.where(cur > cur0, cur - duration, prev0)
    cutoff = len(tr) + len(hist_rows) + b * bs
    key = store.lookup(g.tix[rows], g.s[rows], g.r[rows])
    out = {m: np.empty(Q, np.float32) for m in ('unlimited', 'tw')}
    for c0 in range(0, Q, chunk):
        sl = slice(c0, min(c0 + chunk, Q))
        negs, _ = store.negs(key[sl])
        cand = torch.as_tensor(np.concatenate([g.o[rows][sl][:, None], negs.astype(np.int64)], 1), device=dev)
        lt = wr.last_ts(torch.as_tensor(g.s[rows][sl], device=dev), cand, torch.as_tensor(cutoff[sl], device=dev))
        for m in ('unlimited', 'tw'):
            sc = (lt > float('-inf')) if m == 'unlimited' else (lt >= torch.as_tensor(prev[b[sl]], device=dev)[:, None])
            sc = sc.to(torch.float32)
            out[m][sl] = P.rr_from_scores(sc[:, 0], sc[:, 1:]).cpu().numpy()
    del wr; torch.cuda.empty_cache()
    return {m: float(out[m].astype(np.float64).mean()) for m in out}, dict(prev_t_start=float(prev[0]), duration=float(duration))


def score_sealed(a, ckpt, dev):
    """Sealed path: fresh load, memory over every event, best checkpoint, official TEST negatives (converted here),
    EdgeBank(tw) and EdgeBank(unlimited) on the same TEST files, scored once."""
    t0 = time.time(); g = te.TKG(NAME, tgb_root())
    split = 'test' if a.score_test else 'val'
    sdir = ns.store_dir(ROOT / 'data/tgb/root', NAME, split)
    if split == 'test' and not (sdir / 'receipt.json').exists():
        ns.convert(ROOT / f'data/tgb/root/tkgl_wikidata/{NAME}_test_ns.pkl', sdir)
    store = ns.NegStore(sdir, g.n, g.R2, g.ut); rows = g.queries(split)
    c = np.unique((g.tix[rows] * g.n + g.s[rows]) * g.R2 + g.r[rows]); assert np.array_equal(c, store.codes)
    mem, gm, info = build(g, split, dev)
    model = make_model(g.R2, a.d, a.hidden, a.k, g.n, a.pop_mlp).to(dev); model.load_state_dict(torch.load(ckpt))
    v, order, stats, wall = evaluate(model, gm, g, store, rows, a.eval_batch, dev)
    cls = classes(mem, g, rows)
    p = 'test' if split == 'test' else 'sealed_path_smoke_val'
    res = {f'{p}_mrr': float(v.mean()), f'{p}_queries': int(len(v)), f'{p}_profile': profile(v, order, cls, g),
           f'{p}_items': stats, f'{p}_eval_wall_s': wall, f'{p}_memory': info, f'{p}_store': store.receipt}
    del gm, mem; gc.collect(); torch.cuda.empty_cache()
    if split == 'test':
        res['test_edgebank'], res['test_edgebank_info'] = edgebank_split(g, store, rows, g.queries('val'), dev)
    res[f'{p}_wall_s'] = time.time() - t0
    res[f'{p}_protocol'] = (f'best VALIDATION epoch checkpoint; fresh load; memory rebuilt over every event up to the end of '
                            f'{split.upper()} and read strictly before each query snapshot; official {split.upper()} '
                            'dst-time-filtered negatives (1,000 per (t, s, r), memory-mapped from the official pickle); '
                            'official tie rule; scored once')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--contract', action='store_true')
    for k, v in (('seed', 0), ('epochs', 4), ('hidden', 32), ('d', 16), ('k', 16), ('batch', 1024), ('eval_batch', 2048),
                 ('max_train', 0), ('max_val', 0), ('train_from', 0), ('negatives', 1000)):
        ap.add_argument('--' + k.replace('_', '-'), type=int, default=v)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--gpu-mem-frac', type=float, default=0.25)
    ap.add_argument('--score-test', action='store_true'); ap.add_argument('--sealed-path-smoke', action='store_true')
    ap.add_argument('--lr-decay', choices=('none', 'cos'), default='none')
    ap.add_argument('--pop-mlp', action='store_true', help='POP hazard from an MLP (development arm)')
    a = ap.parse_args(); start = time.time(); torch.set_num_threads(2)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__).resolve(), HERE / 'race_tkg.py', HERE / 'tkg_eval.py',
               HERE / 'tkg_ns_stream.py', HERE / 'safe_tgb.py', HERE / 'tkgl_wikidata_protocol.py')}
    assert not (a.score_test and (a.sealed_path_smoke or a.max_train or a.max_val)), 'sealed runs use the full data'
    out = ROOT / 'experiments/results/tgb' / f'{a.tag}.json'; assert not out.exists(), 'fresh tag'
    dev = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(a.gpu_mem_frac)
    res = dict(tag=a.tag, benchmark='TGB 2.0 TKG', dataset=NAME, driver='race_tkg_wd_v2.py', arm='A (REC, TR, TRANS, POP, BG)',
               args=vars(a), host='local', device=torch.cuda.get_device_name(0))
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    if a.contract:
        res['contract_synthetic'] = contract_synthetic(dev); print(json.dumps(res['contract_synthetic']), flush=True)
    g = te.TKG(NAME, tgb_root()); t_load = time.time() - start
    store = ns.NegStore(ns.store_dir(ROOT / 'data/tgb/root', NAME, 'val'), g.n, g.R2, g.ut)
    vrows = g.queries('val')
    c = np.unique((g.tix[vrows] * g.n + g.s[vrows]) * g.R2 + g.r[vrows]); assert np.array_equal(c, store.codes)
    mem, gm, minfo = build(g, 'val', dev); res['memory'] = minfo; res['memory_vm'] = te.vm()
    print(json.dumps(minfo), flush=True)
    model = make_model(g.R2, a.d, a.hidden, a.k, g.n, a.pop_mlp).to(dev)
    if a.contract:
        res['contract_real'] = contract_real(g, mem, gm, model, store, vrows, dev); print(json.dumps(res['contract_real']), flush=True)
    if a.contract:
        eb, _ = edgebank_split(g, store, vrows, np.zeros(0, np.int64), dev)
        ref = json.loads((ROOT / 'experiments/results/tgb/tkgl_wikidata_protocol.json').read_text())['edgebank']
        assert eb == ref, (eb, ref)
        res['contract_edgebank_split'] = dict(val=eb, equals_protocol_result=True); print(json.dumps(eb), flush=True)
    cls = classes(mem, g, vrows)
    tr = g.queries('train'); tr = tr[g.t[tr] >= a.train_from]
    if a.max_train:
        tr = tr[np.argsort(g.tix[tr], kind='stable')][-a.max_train:]
    conf_tr = v1.DataConflicts(g, g.queries('train')); pool = Pool(g, g.queries('train'), dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    n_steps = a.epochs * len(v1.snapshot_batches(g, tr, a.batch, None)); step = 0
    ckpt = out.with_suffix('.pt'); best, best_ep, history = -1.0, -1, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); total = 0.0; nq = 0; nb = 0; nneg = 0
        for rb in v1.snapshot_batches(g, tr, a.batch, rng):
            loss, nv = train_batch(model, gm, g, rb, conf_tr, pool, a.negatives, gen, dev)
            loss = loss.mean(); assert torch.isfinite(loss)
            if a.lr_decay == 'cos':
                for pg in opt.param_groups:
                    pg['lr'] = a.lr * 0.5 * (1.0 + np.cos(np.pi * step / n_steps))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            total += float(loss.detach()) * len(rb); nq += len(rb); nb += 1; nneg += nv
            if nb % 2000 == 0:
                print(json.dumps(dict(epoch=ep, batches=nb, queries=nq, loss=total / nq, wall_s=time.time() - t0)), flush=True)
        tw = time.time() - t0
        v, order, stats, vw = evaluate(model, gm, g, store, vrows, a.eval_batch, dev, a.max_val)
        mrr = float(v.mean())
        row = dict(epoch=ep, train_loss=total / max(nq, 1), batches=nb, train_queries=nq, mean_valid_negatives=nneg / max(nq, 1),
                   val_mrr=mrr, val_queries=int(len(v)), train_wall_s=tw, val_wall_s=vw, profile=profile(v, order, cls, g),
                   val_items=stats, vm=te.vm(), gpu_max_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
        history.append(row); print(json.dumps(row), flush=True)
        if mrr > best:
            best, best_ep = mrr, ep; torch.save(model.state_dict(), ckpt)
    res.update(status='completed', entities=g.n, relations_with_inverse=g.R2,
               parameters=sum(p.numel() for p in model.parameters()), val_mrr=best, best_epoch=best_ep, history=history,
               checkpoint=str(ckpt.relative_to(ROOT)), trained_queries=int(len(tr)) * a.epochs, load_wall_s=t_load,
               protocol='Strictly causal addressed memory (every snapshot before the query, ties excluded); training race '
                        'over the true object and K negatives sampled like the official generator (filtered); official '
                        'VALIDATION dst-time-filtered negatives, official tie rule; no TEST')
    if a.score_test or a.sealed_path_smoke:
        del mem, gm, store; gc.collect(); torch.cuda.empty_cache()
        res.update(score_sealed(a, ckpt, dev))
        if a.score_test:
            res['protocol'] = res['protocol'].replace('; no TEST', '; TEST scored once with the best VALIDATION checkpoint')
    res.update(score_test=a.score_test, wall_s=time.time() - start, cpu_s=time.process_time(),
               max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               gpu_max_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
               source_sha256=sources)
    with out.open('x') as f:
        json.dump(res, f, indent=1); f.write('\n')
    print(json.dumps({k: v for k, v in res.items() if k not in ('history', 'args')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
