#!/usr/bin/env python3
"""tkgl-wikidata (TGB 2.0) on the GPU: official protocol reproduction on VALIDATION with EdgeBank. No training, no TEST.

Official protocol (py-tgb 2.3.0; TGB examples/linkproppred/tkgl-wikidata):
  * LinkPropPredDataset('tkgl-wikidata', preprocess=True) adds an inverse quadruple (o, r + R, s, t) for every
    (s, r, o, t); splits by timestamp quantiles (TRAIN years 0-1998, VALIDATION 1999-2010, TEST 2011-2024). Every
    VALIDATION row is a query, in py-tgb row order (originals, then inverses).
  * Negatives: strategy 'dst-time-filtered' (DATA_NS_STRATEGY_DICT; TKGNegativeEdgeSampler.query_batch returns
    eval_set[(t, s, r)]): 1,000 pre-sampled destinations per (t, s, r) key, drawn by TKGNegativeEdgeGenerator from the
    destinations of relation r minus the true objects of (s, r) at t.
  * Evaluator MRR: rank = 0.5 * (#neg > pos + #neg >= pos) + 1, RR = 1 / float32(rank), mean over queries.
  * EdgeBank (TGB modules/edgebank_predictor.py, examples .../edgebank.py): memory keyed by (src, dst) from all TRAIN
    rows in row order; a query scores 1 for a destination whose pair is in memory ('unlimited'), or whose last written
    timestamp is >= prev_t ('fixed_time_window', ratio 0.15: prev_t = t_min + (t_max - t_min) * 0.85 over TRAIN,
    duration = t_max - prev_t; when a batch raises t_max, prev_t = t_max - duration). Queries run in batches of 200;
    the memory is updated with each batch's positive (src, dst, t) after scoring it (last write wins).

Negatives are read from the memory-mapped store written by tkg_ns_stream.py (streamed from the official pickle, int32,
exact). This script checks that store against the official pickle with an independent parser (pickletools.genops over
the whole stream: every key and every negative), checks the keys equal the (t, s, r) of the VALIDATION rows, counts the
generator invariants, computes EdgeBank vectorised on the GPU (exact pair lookups, write-order semantics), checks it
against a literal dict transcription of EdgeBankPredictor on the first queries and the official Evaluator on 50 queries,
and profiles EdgeBank by query class under strict causal state. Records wall, CPU, peak RSS and GPU memory.
"""
import argparse, hashlib, json, os, pickletools, resource, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import tkg_eval as te
import tkg_ns_stream as ns

LEADERBOARD = {'EdgeBank(tw)': (0.535, 0.633), 'EdgeBank(unlimited)': (0.535, 0.632)}
NAME = 'tkgl-wikidata'


def tgb_root():
    from tgb.utils.info import PROJ_DIR
    return os.path.relpath(ROOT / 'data/tgb/root', PROJ_DIR)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_store_genops(pkl, store, report_every=0):
    """Independent check: walk the official pickle with pickletools.genops; the 8-byte SHORT_BINBYTES payloads are the
    int64 key scalars (t, s, r) in order, every BYTEARRAY8 (in-band numpy buffer) payload is one negative array;
    compare all (counts asserted, so any other encoding fails loudly)."""
    K = len(store.t); keys = np.empty(3 * K, np.int64); nk = 0; na = 0; t0 = time.time()
    flat = store.flat
    with open(pkl, 'rb', buffering=1 << 24) as f:
        for op, arg, _ in pickletools.genops(f):
            nm = op.name
            if nm == 'SHORT_BINBYTES' and len(arg) == 8:
                keys[nk] = int.from_bytes(arg, 'little', signed=True); nk += 1
            elif nm in ('BYTEARRAY8', 'BINBYTES8', 'BINBYTES'):
                a = np.frombuffer(arg, '<i8')
                o, l = int(store.off[na]), int(store.len[na])
                assert len(a) == l and np.array_equal(a, flat[o:o + l]), f'array {na} differs'
                na += 1
                if report_every and na % report_every == 0:
                    print(f'genops {na}/{K} {time.time() - t0:.0f}s', flush=True)
    assert nk == 3 * K and na == K, (nk, na, K)
    k = keys.reshape(K, 3)
    assert np.array_equal(k[:, 0], store.t) and np.array_equal(k[:, 1], store.s) and np.array_equal(k[:, 2], store.r)
    return dict(keys=int(K), arrays=int(na), all_keys_equal=True, all_negatives_equal=True, wall_s=time.time() - t0,
                parser='pickletools.genops (stdlib opcode walker, independent of the unpickler used for the store)')


class Writes:
    """EdgeBank memory as sorted write records: pair code s * n + d, write order w, timestamp."""
    def __init__(self, src, dst, ts, n, dev):
        self.n, self.dev = n, dev
        code = torch.as_tensor(src, device=dev) * n + torch.as_tensor(dst, device=dev)
        self.U, pid = torch.unique(code, sorted=True, return_inverse=True)
        self.W = len(code); w = torch.arange(len(code), device=dev)
        comp = pid * self.W + w; o = torch.argsort(comp)
        self.comp = comp[o]; self.ts = torch.as_tensor(ts, device=dev, dtype=torch.float64)[o]

    def last_ts(self, s, cand, cutoff):
        """s [Q], cand [Q, C], cutoff [Q]: timestamp of the last write of pair (s, cand) with w < cutoff, or -inf."""
        code = s[:, None] * self.n + cand
        i = torch.searchsorted(self.U, code).clamp(max=len(self.U) - 1)
        found = self.U[i] == code
        q = i * self.W + cutoff[:, None]
        j = torch.searchsorted(self.comp, q) - 1
        jj = j.clamp(min=0)
        ok = found & (j >= 0) & (torch.div(self.comp[jj], self.W, rounding_mode='floor') == i)
        return torch.where(ok, self.ts[jj], torch.full_like(self.ts[jj], float('-inf')))


def rr_from_scores(pos, neg):
    """Official tie rule; pos [Q], neg [Q, C] (torch)."""
    opt = (neg > pos[:, None]).sum(1); pes = (neg >= pos[:, None]).sum(1)
    rank = 0.5 * (opt + pes).to(torch.float64) + 1.0
    return (1.0 / rank.to(torch.float32)).to(torch.float32)


def edgebank(g, store, rows, dev, chunk=20000, bs=200):
    """Both EdgeBank modes on the given query rows (py-tgb order) with the official negatives. Returns rr per mode."""
    tr = g.queries('train'); n = g.n
    src = np.concatenate([g.s[tr], g.s[rows]]); dst = np.concatenate([g.o[tr], g.o[rows]])
    ts = np.concatenate([g.t[tr], g.t[rows]])
    wr = Writes(src, dst, ts, n, dev)
    ttr = g.t[tr]; cur0 = ttr.max(); dur_all = ttr.max() - ttr.min()
    prev0 = ttr.min() + dur_all * (1 - 0.15); duration = cur0 - prev0
    Q = len(rows); b = np.arange(Q) // bs; nb = int(b.max()) + 1
    bmax = np.full(nb, -np.inf); np.maximum.at(bmax, b, g.t[rows].astype(np.float64))
    cur = np.maximum.accumulate(np.concatenate([[float(cur0)], bmax]))[:-1]
    prev = np.where(cur > cur0, cur - duration, prev0)
    cutoff = len(tr) + b * bs
    key = store.lookup(g.tix[rows], g.s[rows], g.r[rows])
    out = {m: np.empty(Q, np.float32) for m in ('unlimited', 'tw')}
    live_neg = {m: np.empty(Q, np.int32) for m in ('unlimited', 'tw')}
    for c0 in range(0, Q, chunk):
        sl = slice(c0, min(c0 + chunk, Q))
        negs, _ = store.negs(key[sl])
        cand = torch.as_tensor(np.concatenate([g.o[rows][sl][:, None], negs.astype(np.int64)], 1), device=dev)
        lt = wr.last_ts(torch.as_tensor(g.s[rows][sl], device=dev), cand, torch.as_tensor(cutoff[sl], device=dev))
        for m in ('unlimited', 'tw'):
            sc = (lt > float('-inf')) if m == 'unlimited' else (lt >= torch.as_tensor(prev[b[sl]], device=dev)[:, None])
            sc = sc.to(torch.float32)
            out[m][sl] = rr_from_scores(sc[:, 0], sc[:, 1:]).cpu().numpy()
            live_neg[m][sl] = sc[:, 1:].sum(1).to(torch.int32).cpu().numpy()
    return out, live_neg, dict(prev_t0=float(prev0), duration=float(duration), writes=int(wr.W), pairs=int(len(wr.U)),
                               batches=nb)


def edgebank_literal(g, store, rows, nq, bs=200):
    """Literal transcription of EdgeBankPredictor (dict memory, row order, batch updates) on the first nq queries."""
    from tgb.linkproppred.evaluate import Evaluator
    ev = Evaluator(name=NAME); tr = g.queries('train'); out = {}
    for mode in ('unlimited', 'tw'):
        ts = g.t[tr]; mem = {}
        cur_t = ts.max(); duration = ts.max() - ts.min(); prev_t = ts.min() + duration * (1 - 0.15); duration = cur_t - prev_t
        for a, b_, c in zip(g.s[tr].tolist(), g.o[tr].tolist(), ts.tolist()):
            mem[(a, b_)] = c
        rr = []
        for b0 in range(0, nq, bs):
            j = rows[b0:min(b0 + bs, nq)]
            k = store.lookup(g.tix[j], g.s[j], g.r[j]); negs, _ = store.negs(k)
            for i, row in enumerate(j):
                qs = int(g.s[row]); qd = [int(g.o[row])] + negs[i].tolist()
                pred = np.zeros(len(qd))
                for x, d in enumerate(qd):
                    if (qs, d) in mem and (mode == 'unlimited' or mem[(qs, d)] >= prev_t):
                        pred[x] = 1.0
                rr.append(float(ev.eval({'y_pred_pos': pred[:1], 'y_pred_neg': pred[1:], 'eval_metric': ['mrr']})['mrr']))
            if mode == 'tw' and g.t[j].max() > cur_t:
                cur_t = g.t[j].max(); prev_t = cur_t - duration
            for a, b_, c in zip(g.s[j].tolist(), g.o[j].tolist(), g.t[j].tolist()):
                mem[(a, b_)] = c
        out[mode] = np.array(rr)
        del mem
    return out


def query_classes(g, rows):
    """Strict causal state (events of TRAIN + VALIDATION rows in snapshots before the query's): 0 repeat (s, r, o),
    1 new object of a known (s, r), 2 new (s, r) of a known subject, 3 new subject."""
    m = g.masks['train'] | g.masks['val']; ev = np.flatnonzero(m); n, R2 = g.n, g.R2

    def first(code_ev, code_q):
        o = np.lexsort((g.tix[ev], code_ev)); c = code_ev[o]; tx = g.tix[ev][o]
        u, st = np.unique(c, return_index=True)
        i = np.minimum(np.searchsorted(u, code_q), len(u) - 1)
        return np.where(u[i] == code_q, tx[st][i], np.iinfo(np.int64).max)

    tq = g.tix[rows]
    rep = first((g.r[ev] * n + g.s[ev]) * n + g.o[ev], (g.r[rows] * n + g.s[rows]) * n + g.o[rows]) < tq
    srk = first(g.r[ev] * n + g.s[ev], g.r[rows] * n + g.s[rows]) < tq
    sk = first(g.s[ev], g.s[rows]) < tq
    return np.where(rep, 0, np.where(srk, 1, np.where(sk, 2, 3)))


CLASS_NAMES = ('repeat_sro', 'new_object_known_sr', 'new_sr_known_subject', 'new_subject')


def profile(v, cls, rows, g):
    p = {nm: dict(queries=int((cls == i).sum()), share=float((cls == i).mean()),
                  mrr=float(v[cls == i].astype(np.float64).mean()) if (cls == i).any() else None)
         for i, nm in enumerate(CLASS_NAMES)}
    inv = g.r[rows] >= g.R
    p['direction'] = dict(original=float(v[~inv].astype(np.float64).mean()), inverse=float(v[inv].astype(np.float64).mean()))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--literal-queries', type=int, default=2000)
    ap.add_argument('--skip-genops', action='store_true'); ap.add_argument('--gpu-mem-frac', type=float, default=0.25)
    a = ap.parse_args(); start = time.time()
    out = ROOT / 'experiments/results/tgb' / f'{a.tag}.json'; assert not out.exists(), 'fresh tag'
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_frac)
    res = dict(tag=a.tag, benchmark='TGB 2.0 TKG', dataset=NAME, split='val', host='local', device=str(dev),
               leaderboard_test_val=LEADERBOARD)
    g = te.TKG(NAME, tgb_root()); res['load_wall_s'] = time.time() - start; res['load_vm'] = te.vm()
    res['checks'] = g.checks(); print(json.dumps(res['checks']), flush=True)
    sdir = ns.store_dir(ROOT / 'data/tgb/root', NAME, 'val')
    store = ns.NegStore(sdir, g.n, g.R2, g.ut); res['store'] = store.receipt
    rows = g.queries('val')
    c = np.unique((g.tix[rows] * g.n + g.s[rows]) * g.R2 + g.r[rows])
    assert np.array_equal(c, store.codes), 'store keys != (t, s, r) of the VALIDATION rows'
    res['keys_equal_split'] = True; res['queries'] = int(len(rows))
    if not a.skip_genops:
        pkl = ROOT / f'data/tgb/root/tkgl_wikidata/{NAME}_val_ns.pkl'
        res['store_vs_official_pickle'] = verify_store_genops(pkl, store, report_every=250000)
        print(json.dumps(res['store_vs_official_pickle']), flush=True)
    t0 = time.time(); conf = {}
    cq = (g.tix[rows] * g.n + g.s[rows]) * g.R2 + g.r[rows]; o = np.lexsort((g.o[rows], cq))
    cs, co = cq[o], g.o[rows][o]; ki = np.searchsorted(store.codes, cs); kk = store.order[ki]
    hit = np.zeros(len(store.t), np.int64); dup = 0
    for c0 in range(0, len(store.t), 200000):
        k = np.arange(c0, min(c0 + 200000, len(store.t))); negs, _ = store.negs(k)
        srt = np.sort(negs, 1); dup += int((srt[:, 1:] == srt[:, :-1]).any(1).sum())
        m = (kk >= c0) & (kk < k[-1] + 1)
        if m.any():
            sub = negs[kk[m] - c0]; hit_m = (sub == co[m][:, None]).any(1)
            np.add.at(hit, kk[m][hit_m], 1)
    conf.update(keys=int(len(store.t)), keys_with_a_true_object_among_negatives=int((hit > 0).sum()),
                keys_with_duplicate_negatives=dup, conflict_objects=int(len(cs)), wall_s=time.time() - t0,
                note='TKGNegativeEdgeGenerator fills from all destinations (true objects not excluded) when relation r has '
                     'fewer than 1,000 other destinations; such keys can list a true object as a negative')
    res['negative_invariants'] = conf; print(json.dumps(conf), flush=True)
    t0 = time.time(); c0 = time.process_time()
    rr, live, info = edgebank(g, store, rows, dev)
    res['edgebank'] = {m: float(rr[m].astype(np.float64).mean()) for m in rr}
    res['edgebank_info'] = dict(info, wall_s=time.time() - t0, cpu_s=time.process_time() - c0,
                                mean_live_negatives={m: float(live[m].mean()) for m in live})
    print(json.dumps(res['edgebank']), flush=True)
    t0 = time.time(); lit = edgebank_literal(g, store, rows, a.literal_queries)
    res['literal_check'] = {m: dict(queries=int(len(lit[m])), max_abs_diff=float(np.abs(lit[m] - rr[m][:len(lit[m])]).max()))
                            for m in lit}
    res['literal_check']['wall_s'] = time.time() - t0
    assert all(res['literal_check'][m]['max_abs_diff'] < 1e-7 for m in lit), res['literal_check']
    cls = query_classes(g, rows)
    res['profile'] = {m: profile(rr[m], cls, rows, g) for m in rr}
    res['reproduced'] = {k: dict(ours=round(res['edgebank'][m], 4), leaderboard_val=LEADERBOARD[k][1])
                         for k, m in (('EdgeBank(tw)', 'tw'), ('EdgeBank(unlimited)', 'unlimited'))}
    res.update(status='completed', wall_s=time.time() - start, cpu_s=time.process_time(),
               max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, vm=te.vm(),
               gpu_max_allocated_gb=(torch.cuda.max_memory_allocated() / 1e9) if dev.type == 'cuda' else 0.0,
               protocol='VALIDATION only; official loader, official dst-time-filtered negatives (1,000 per (t, s, r)), '
                        'official tie rule; no TEST negatives opened',
               source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__).resolve(), HERE / 'tkg_eval.py',
                                                                         HERE / 'tkg_ns_stream.py', HERE / 'safe_tgb.py')})
    with out.open('x') as f:
        json.dump(res, f, indent=1); f.write('\n')
    print(json.dumps({k: v for k, v in res.items() if k not in ('checks', 'store')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
