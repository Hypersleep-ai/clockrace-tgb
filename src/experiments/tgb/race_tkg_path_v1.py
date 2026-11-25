#!/usr/bin/env python3
"""race_tkg_path_v1 (19 Nov 2026): race_tkg_v2 (unchanged, imported) + a two-hop PATH cause.

Why: the earlier tkgl-polecat entry scores 0.077 TEST MRR on queries whose (subject, relation) is known but whose object is new
(42% of TEST queries). On VALIDATION, a two-hop history path s -> m -> o reaches the true object of 82% of those queries,
and a crude degree-normalised path count alone ranks them at 0.052 MRR. race_tkg has no path cause.

  PATH  composition of two addressed reads of one relation-agnostic pair slot family (s -> o, any relation, both directions
        through the inverse rows): the query subject's partners m active in the last W steps (at most --path-km, by hop clock
        strength), then each m's partners e active in the last W steps (at most --path-ke per m), e != s. Each path carries
        both hops' exact clocks (decayed counts at 1/4/16/64/256 steps, count, age), the recent degrees of s and of m and the
        candidate's activity; an MLP keyed by the query relation gives it a hazard; the paths reaching e add up (a race of
        races). Sparse: only the reached entities get hazard, so log Z stays exact (sum over all entities of background +
        popularity, plus the sparse causes over their candidates).
  The PATH head's output layer starts at weight 0 and bias -8: path_v1 starts as v2 (path hazard e^-8 per path) and the path
  cause earns its mass by gradient (a zero-initialised gate).

Without --path-window this driver computes v2 exactly (asserted by --contract on a synthetic TKG). Everything else (memory,
causes, windows, exact filtered softmax, official filtered evaluation, best VALIDATION epoch, sealed TEST path) is v2's.
"""
import argparse, gc, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import race_tkg_v2 as v2
import tkg_eval as te

J, NINF, F_CAND, l1p = v2.J, v2.NINF, v2.F_CAND, v2.l1p
F_PATH = 2 * (J + 2) + 2 + F_CAND


def topk_per_group(group, score, k):
    """Positions of the k highest scores within each group (ties: earlier position first)."""
    if not len(group):
        return np.zeros(0, np.int64)
    o = np.lexsort((np.arange(len(group)), -score, group)); g = group[o]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    rank = np.arange(len(g)) - np.repeat(starts, np.diff(np.r_[starts, len(g)]))
    return np.sort(o[rank < k])


class Memory(v2.Memory):
    def __init__(self, s, r, o, tix, rts, n, R2, transfer=True, windows=None, ut_rts=None, path=None):
        super().__init__(s, r, o, tix, rts, n, R2, transfer, windows, ut_rts)
        self.path = path
        if path:
            self.pair = v2.Family(s, o, tix, rts.astype(np.float64), n)

    def path_items(self, s, r, tix, rts, with_mid=False):
        n = self.n; W, KM, KE = self.path['window'], self.path['km'], self.path['ke']
        lo = np.searchsorted(self.ut_rts, rts - W, 'left')
        q1, sl1 = v2.recent(self.pair.sl, s, tix, lo)
        m = self.pair.sl.key[sl1] % n; keep = m != s[q1]; q1, sl1, m = q1[keep], sl1[keep], m[keep]
        deg_s = np.bincount(q1, minlength=len(s))
        _, c1, d1, k1, _ = self.pair.read(sl1, tix[q1], rts[q1])
        sel = topk_per_group(q1, k1.sum(1), KM); q1, m, c1, d1, k1 = q1[sel], m[sel], c1[sel], d1[sel], k1[sel]
        p, sl2 = v2.recent(self.pair.sl, m, tix[q1], lo[q1])
        e = self.pair.sl.key[sl2] % n; keep = e != s[q1[p]]; p, sl2, e = p[keep], sl2[keep], e[keep]
        deg_m = np.bincount(p, minlength=len(q1))
        _, c2, d2, k2, _ = self.pair.read(sl2, tix[q1[p]], rts[q1[p]])
        sel = topk_per_group(p, k2.sum(1), KE); p, e, c2, d2, k2 = p[sel], e[sel], c2[sel], d2[sel], k2[sel]
        q = q1[p]
        f = np.concatenate([l1p(k1[p]), l1p(c1[p])[:, None], l1p(d1[p])[:, None], l1p(k2), l1p(c2)[:, None], l1p(d2)[:, None],
                            l1p(deg_m[p])[:, None], l1p(deg_s[q])[:, None]], 1).astype(np.float32)
        f = np.concatenate([f, self.cand_feats(e, tix[q], rts[q])], 1)
        return (q, e, f, m[p]) if with_mid else (q, e, f)


class Race(v2.Race):
    def __init__(self, R2, d=16, hidden=32, k=16, n=1, path=False):
        super().__init__(R2, d, hidden, k, n)
        self.has_path = bool(path)
        if self.has_path:
            self.n_path = nn.LayerNorm(F_PATH); self.path = v2.MLP(F_PATH + d, hidden)
            last = self.path.net[-1]; nn.init.zeros_(last.weight); nn.init.constant_(last.bias, -8.0)

    def forward(self, b):
        lrec, ltr, ltrans, lpop, lg = super().forward(b)
        if not self.has_path:
            return lrec, ltr, ltrans, lpop, lg, None
        t = torch.from_numpy; rq = t(b['r'])
        lpath = self.path(torch.cat([self.n_path(t(b['path_f'])), self.rel.weight[rq[t(b['path_q'])]]], 1)).double()
        return lrec, ltr, ltrans, lpop, lg, lpath


def build_batch(mem, s, r, o, tix, rts, conf_q, conf_e):
    n = mem.n; Q = len(s); X = np.full(Q, tix); RT = np.full(Q, rts, np.float64)
    rq, rqe, rf = mem.rec_items(s, r, X, RT)
    if mem.transfer:
        tq, tqe, trp, tf = mem.tr_items(s, r, X, RT)
        xq, xqe, xf = mem.trans_items(s, r, X, RT)
    else:
        tq = tqe = trp = xq = xqe = np.zeros(0, np.int64); tf = np.zeros((0, v2.F_TR), np.float32)
        xf = np.zeros((0, v2.F_TRANS), np.float32)
    if getattr(mem, 'path', None):
        pq, pqe, pf = mem.path_items(s, r, X, RT)
    else:
        pq = pqe = np.zeros(0, np.int64); pf = np.zeros((0, F_PATH), np.float32)
    ru, rinv = np.unique(r, return_inverse=True)
    pk, pe, pof = mem.pop_items(ru, tix, rts)
    ef, be, br, bc = mem.ent_feats(tix, rts)
    u = np.unique(np.concatenate([rq * n + rqe, tq * n + tqe, xq * n + xqe, pq * n + pqe, conf_q * n + conf_e, np.arange(Q) * n + o]))
    sq, se = u // n, u % n
    pos = np.searchsorted(u, np.arange(Q) * n + o)
    rec_s = np.searchsorted(u, rq * n + rqe); tr_s = np.searchsorted(u, tq * n + tqe); x_s = np.searchsorted(u, xq * n + xqe)
    path_s = np.searchsorted(u, pq * n + pqe)
    conf = np.isin(u, conf_q * n + conf_e); assert conf[pos].all()
    s_pop = np.full(len(u), -1)
    if len(pk):
        pcode = pk * n + pe; po = np.argsort(pcode); pcs = pcode[po]
        scode = rinv[sq] * n + se; k = np.minimum(np.searchsorted(pcs, scode), len(pcs) - 1)
        s_pop = np.where(pcs[k] == scode, po[k], -1)
    is_cand = np.zeros(len(u), bool); is_cand[rec_s] = True; is_cand[tr_s] = True; is_cand[x_s] = True; is_cand[path_s] = True
    return dict(n=n, Q=Q, r=r, rinv=rinv, ru=ru, rec_q=rq, rec_f=rf, rec_s=rec_s, tr_q=tq, tr_rp=trp, tr_f=tf,
                tr_s=tr_s, trans_q=xq, trans_f=xf, trans_s=x_s, path_q=pq, path_f=pf, path_s=path_s,
                pop_k=pk, pop_e=pe, pop_f=pof, ent_f=ef, bag_e=be, bag_r=br, bag_c=bc, sq=sq, se=se,
                pos=pos, conf=conf, s_pop=s_pop, is_cand=is_cand)


def combine(b, lrec, ltr, ltrans, lpop, lg, lpath):
    t = torch.from_numpy; S = len(b['sq']); Q = b['Q']; f64 = torch.float64
    l_rec = torch.full((S,), NINF, dtype=f64).index_put((t(b['rec_s']),), lrec)
    l_tr = v2.seg_lse(ltr, t(b['tr_s']), S)
    l_x = v2.seg_lse(ltrans, t(b['trans_s']), S)
    l_extra = torch.logaddexp(torch.logaddexp(l_rec, l_tr), l_x)
    if lpath is not None and len(lpath):
        l_extra = torch.logaddexp(l_extra, v2.seg_lse(lpath, t(b['path_s']), S))
    l_g = lg[t(b['rinv'][b['sq']]), t(b['se'])]
    sp = t(b['s_pop'])
    l_p = lpop[sp.clamp(min=0)] if len(lpop) else torch.zeros(S, dtype=f64)
    l_p = torch.where(sp >= 0, l_p, torch.full((S,), NINF, dtype=f64))
    l_bg = torch.logaddexp(l_g, l_p)
    l_all = torch.logaddexp(l_bg, l_extra)
    LG = torch.logsumexp(lg, 1)
    LP = v2.seg_lse(lpop, t(b['pop_k']), len(b['ru'])) if len(lpop) else torch.full((len(b['ru']),), NINF, dtype=f64)
    rq = t(b['rinv']); cand = t(b['is_cand'])
    lc = v2.seg_lse(torch.where(cand, l_extra, torch.full_like(l_extra, NINF)), t(b['sq']), Q)
    logZ = torch.logsumexp(torch.stack([LG[rq], LP[rq], lc]), 0)
    return l_all, l_bg, l_extra, logZ


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
    model.eval(); out, cls, nitems = [], [], dict(rec=0, tr=0, trans=0, path=0, pop=0, specials=0); t0 = time.time(); order = []
    with torch.no_grad():
        for rb in v2.snapshot_batches(g, rows, bs):
            b, parts, (l_all, l_bg, l_extra, logZ) = run_batch(g, mem, model, rb, conflicts)
            out.append(v2.ranks(b, l_extra, parts[4], parts[3])); order.append(rb)
            cls.append(mem.query_class(g.s[rb], g.r[rb], g.o[rb], g.tix[rb]))
            for k, key in (('rec', 'rec_q'), ('tr', 'tr_q'), ('trans', 'trans_q'), ('path', 'path_q'), ('pop', 'pop_k'), ('specials', 'sq')):
                nitems[k] += len(b[key])
    v = np.concatenate(out).astype(np.float64); cls = np.concatenate(cls); order = np.concatenate(order)
    names = ('repeat_sro', 'new_object_known_sr', 'new_sr_known_subject', 'new_subject')
    prof = {nm: dict(queries=int((cls == i).sum()), mrr=float(v[cls == i].mean()) if (cls == i).any() else None)
            for i, nm in enumerate(names)}
    inv = g.r[order] >= g.R
    prof['direction'] = dict(original=float(v[~inv].mean()), inverse=float(v[inv].mean()) if inv.any() else None)
    return float(v.mean()), len(v), time.time() - t0, prof, nitems


def path_of(a):
    return None if a.path_window is None else dict(window=float(a.path_window), km=int(a.path_km), ke=int(a.path_ke))


def build_memory(g, upto_split, transfer=True, windows=None, path=None):
    m = g.masks['train'] | (g.masks['val'] if upto_split in ('val', 'test') else False) | (g.masks['test'] if upto_split == 'test' else False)
    ev = np.flatnonzero(m)
    ut_rts = np.zeros(len(g.ut)); ut_rts[g.tix] = g.rts
    return Memory(g.s[ev], g.r[ev], g.o[ev], g.tix[ev], g.rts[ev], g.n, g.R2, transfer, windows, ut_rts.astype(np.float64), path)


def make_model(a, R2, n):
    return Race(R2, d=a.d, hidden=a.hidden, k=a.k, n=n, path=a.path_window is not None)


def contracts():
    """(1) path items equal a brute-force two-hop enumeration (no caps), with exact hop clocks, counts, ages and degrees;
    (2) with the path cause on, log Z equals a dense log-sum over every entity of its summed hazards;
    (3) with the path cause off, this driver's losses equal v2's exactly."""
    rng = np.random.default_rng(0); g = v2._G(rng); W = 10.0
    ut_rts = 2.0 * np.arange(int(g.tix.max()) + 1)
    mem = Memory(g.s, g.r, g.o, g.tix, g.rts, g.n, g.R2, ut_rts=ut_rts, path=dict(window=W, km=10 ** 6, ke=10 ** 6))
    worst = 0.0; nq = 0; npaths = 0
    for tq in range(5, int(g.tix.max()) + 1, 3):
        s = np.arange(g.n); r = np.zeros(g.n, np.int64); X = np.full(g.n, tq); RT = np.full(g.n, 2.0 * tq)
        q, e, f, m = mem.path_items(s, r, X, RT, with_mid=True)
        got = {(int(a), int(b_), int(c)): f[i] for i, (a, b_, c) in enumerate(zip(q, m, e))}
        lo_t = int(np.ceil((2.0 * tq - W) / 2.0)); before = g.tix < tq; inwin = before & (g.tix >= lo_t)

        def hop(a, b_):
            ev = before & (g.s == a) & (g.o == b_)
            dt = 2.0 * tq - g.rts[ev].astype(np.float64)
            return np.exp(-dt[:, None] / v2.TAUS).sum(0), ev.sum(), (dt.min() if ev.any() else 0.0)

        want = {}
        for si in range(g.n):
            mids = np.unique(g.o[inwin & (g.s == si)]); mids = mids[mids != si]
            for mi in mids:
                es = np.unique(g.o[inwin & (g.s == mi)]); es = es[es != si]
                for ei in es:
                    want[(si, int(mi), int(ei))] = (len(mids), len(es))
        assert set(got) == set(want), ('path set', tq, len(got), len(want))
        for (si, mi, ei), (dsi, dmi) in want.items():
            k1, c1, d1 = hop(si, mi); k2, c2, d2 = hop(mi, ei)
            ref = np.concatenate([l1p(k1), [l1p(c1), l1p(d1)], l1p(k2), [l1p(c2), l1p(d2)], [l1p(dmi), l1p(dsi)]])
            worst = max(worst, float(np.abs(got[(si, mi, ei)][:len(ref)] - ref).max()))
        nq += g.n; npaths += len(got)
    assert worst < 1e-5, worst
    torch.manual_seed(0); model = Race(g.R2, d=8, hidden=16, k=8, n=g.n, path=True)
    with torch.no_grad():
        model.path.net[-1].bias.fill_(0.5); model.path.net[-1].weight.normal_(0, 0.3)
    tq = int(g.tix.max()); rows = np.flatnonzero(g.tix == tq); s, r, o = g.s[rows], g.r[rows], g.o[rows]
    b = build_batch(mem, s, r, o, tq, 2.0 * tq, np.arange(len(rows)), o)
    parts = model(b); l_all, l_bg, l_extra, logZ = combine(b, *parts)
    lrec, ltr, ltrans, lpop, lg, lpath = (p.detach().numpy() if p is not None else None for p in parts)
    dense = np.exp(lg[b['rinv']]).copy()
    for qi in range(len(rows)):
        rr = b['rinv'][qi]; pm = b['pop_k'] == rr; dense[qi, b['pop_e'][pm]] += np.exp(lpop[pm])
    for qq, ee, ll in ((b['rec_q'], b['se'][b['rec_s']], lrec), (b['tr_q'], b['se'][b['tr_s']], ltr),
                       (b['trans_q'], b['se'][b['trans_s']], ltrans), (b['path_q'], b['se'][b['path_s']], lpath)):
        np.add.at(dense, (qq, ee), np.exp(ll))
    dz = np.log(dense.sum(1))
    gap = float(np.abs(dz - logZ.detach().numpy()).max()); assert gap < 1e-8, gap
    torch.manual_seed(1); m2 = v2.Race(g.R2, d=8, hidden=16, k=8, n=g.n)
    torch.manual_seed(1); m3 = Race(g.R2, d=8, hidden=16, k=8, n=g.n, path=False)
    mem2 = v2.Memory(g.s, g.r, g.o, g.tix, g.rts, g.n, g.R2, ut_rts=ut_rts)
    mem3 = Memory(g.s, g.r, g.o, g.tix, g.rts, g.n, g.R2, ut_rts=ut_rts, path=None)
    b2 = v2.build_batch(mem2, s, r, o, tq, 2.0 * tq, np.arange(len(rows)), o)
    b3 = build_batch(mem3, s, r, o, tq, 2.0 * tq, np.arange(len(rows)), o)
    c2 = v2.combine(b2, *m2(b2)); l2 = v2.filtered_loss(b2, c2[0], c2[3])
    c3 = combine(b3, *m3(b3)); l3 = v2.filtered_loss(b3, c3[0], c3[3])
    eq = float((l2 - l3).abs().max()); assert eq == 0.0, eq
    return dict(path_queries=nq, paths=npaths, max_abs_feature_error=worst, dense_logZ_gap=gap, v2_equivalence_max_abs=eq,
                scope='synthetic TKG (25 entities, 6 relations, 700 events); path window 10 steps, no caps')


def score_sealed(a, ckpt, split):
    t0 = time.time(); c0 = time.process_time(); g = te.TKG(a.dataset, a.root); conf = te.Conflicts(g, split, check=True)
    mem = build_memory(g, split, not a.no_transfer, v2.windows_of(a), path_of(a)); model = make_model(a, g.R2, g.n)
    model.load_state_dict(torch.load(ckpt))
    mrr, nq, wall, prof, nitems = evaluate(g, mem, model, g.queries(split), conf, a.eval_batch)
    p = 'test' if split == 'test' else 'sealed_path_smoke_val'
    return {f'{p}_mrr': mrr, f'{p}_queries': nq, f'{p}_profile': prof, f'{p}_items': nitems, f'{p}_wall_s': time.time() - t0,
            f'{p}_cpu_s': time.process_time() - c0, f'{p}_eval_wall_s': wall, f'{p}_vm': te.vm(), f'{p}_conflicts': conf.report,
            f'{p}_protocol': 'best VALIDATION epoch checkpoint; fresh load; slots rebuilt over every event up to the end of '
                             f'{split.upper()} and read strictly before each query snapshot; official {split.upper()} '
                             'time-filtered conflict sets, one-vs-all over all entities, official tie rule; scored once'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', default='tkgl-polecat')
    ap.add_argument('--root', default='../../../data/tgb/root'); ap.add_argument('--contract', action='store_true')
    for k, v in (('seed', 0), ('epochs', 2), ('hidden', 32), ('d', 16), ('k', 16), ('batch', 1024), ('eval_batch', 4096),
                 ('max_train', 0), ('max_val', 0), ('train_from', 0), ('path_km', 16), ('path_ke', 16)):
        ap.add_argument('--' + k.replace('_', '-'), type=int, default=v)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--score-test', action='store_true'); ap.add_argument('--sealed-path-smoke', action='store_true')
    ap.add_argument('--no-transfer', action='store_true')
    for k in ('rec', 'tr', 'trans', 'pop'):
        ap.add_argument(f'--window-{k}', type=float, default=None)
    ap.add_argument('--path-window', type=float, default=None, help='PATH over pair slots active in the last W steps (off if unset)')
    a = ap.parse_args(); start = time.time(); torch.set_num_threads(1)
    assert not (a.score_test and (a.sealed_path_smoke or a.max_train or a.max_val)), 'sealed runs use the full data'
    out = ROOT / 'experiments/results/tgb' / f'{a.tag}.json'; assert not out.exists(), 'fresh tag'
    if a.contract:
        t0 = time.time(); res = dict(status='completed', checks=contracts(), wall_s_contract=time.time() - t0, scope='race_tkg_path_v1 contracts; no data, no fit')
    else:
        torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
        g = te.TKG(a.dataset, a.root); conf_val = te.Conflicts(g, 'val', check=True); t_load = time.time() - start
        cm = time.process_time(); mem = build_memory(g, 'val', not a.no_transfer, v2.windows_of(a), path_of(a)); t_mem_cpu = time.process_time() - cm
        t_mem = time.time() - start - t_load; mem_vm = te.vm()
        tr = g.queries('train'); tr = tr[g.tix[tr] >= a.train_from]
        if a.max_train:
            tr = tr[np.argsort(g.tix[tr], kind='stable')][-a.max_train:]
        conf_tr = v2.DataConflicts(g, g.queries('train'))
        model = make_model(a, g.R2, g.n); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        ckpt = out.with_suffix('.pt'); best, best_ep, history = -1.0, -1, []
        for ep in range(a.epochs):
            model.train(); t0 = time.time(); c0 = time.process_time(); total = 0.0; nb = 0; nq = 0
            for rb in v2.snapshot_batches(g, tr, a.batch, rng):
                b, parts, (l_all, l_bg, l_extra, logZ) = run_batch(g, mem, model, rb, conf_tr)
                loss = v2.filtered_loss(b, l_all, logZ).mean(); assert torch.isfinite(loss)
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
                   transitions=int(mem.n_trans), memory_cpu_s=t_mem_cpu, parameters=sum(p.numel() for p in model.parameters()),
                   val_mrr=best, best_epoch=best_ep, history=history, checkpoint=str(ckpt.relative_to(ROOT)),
                   trained_queries=int(len(tr)) * a.epochs, load_wall_s=t_load, memory_build_wall_s=t_mem, memory_vm=mem_vm,
                   val_conflicts=conf_val.report,
                   protocol='Strictly causal addressed memory (every snapshot before the query, ties excluded); exact one-vs-all '
                            'filtered softmax over all entities for training; official time-filtered VALIDATION conflict sets, '
                            'one-vs-all over all entities, official tie rule; no TEST')
        if a.score_test or a.sealed_path_smoke:
            del mem, g, conf_val, conf_tr; gc.collect(); te.trim()
            res.update(score_sealed(a, ckpt, 'test' if a.score_test else 'val'))
            if a.score_test:
                res['protocol'] = res['protocol'].replace('; no TEST', '; TEST scored once with the best VALIDATION checkpoint')
    res.update(tag=a.tag, benchmark='TGB 2.0 TKG', driver='race_tkg_path_v1.py', score_test=a.score_test,
               wall_s=time.time() - start, cpu_s=time.process_time(), max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               source_sha256={str(p.relative_to(ROOT)): v2.sha(p) for p in (Path(__file__).resolve(), HERE / 'race_tkg_v2.py',
                                                                            HERE / 'tkg_eval.py', HERE / 'safe_tgb.py')})
    with out.open('x') as f:
        json.dump(res, f, indent=1); f.write('\n')
    print(json.dumps({k: v for k, v in res.items() if k not in ('history', 'args')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
