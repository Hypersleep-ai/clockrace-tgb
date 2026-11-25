#!/usr/bin/env python3
"""race_affinity v3 (3 Nov 2026): the node-affinity model with a multi-clock bank and a soft per-user race over it.

v3 is race_affinity_v2_large_v2.py (itself the genre model race_affinity_v2.py, imported unchanged, plus the large-
dataset changes 1-5) plus:
  1. --extra-clocks v3: per-user edge affinities on three more clocks, lazily referenced (O(1) per clock per edge, a row
     rescale only every 20 clock-scale units; row normalisation cancels the common factor):
       time 120 days (the v2 grid ends at 30 days), count 256 and 1,024 edges of the same user.
     The readout receives their shares and log shares, and a soft per-user RACE over the whole bank of seven clocks
     (1/3/7/30/120 days, 256/1,024 edges): pi = softmax(W ctx + b) with bids from the user's context (edge volume, time
     since the last edge, revealed labels, label flag, log edge count, log edge rate; zero-initialised, so the race
     starts uniform), raced share r(u, c) = sum_m pi_m q_m(u, c), and r and log r enter the readout. r is linear in the
     clock reads, so each clock's bid receives exact responsibility credit pi_m (v_m - sum_k pi_k v_k), v_m =
     <dL/dr, q_m>, including the clocks that lose the race.
  2. Validation reporting that changes no number of the fit: per-row NDCG@10 (official tie semantics) by activity
     quartile and silence bucket, and with --baseline-checkpoint a v2 checkpoint scored on the same rows in the same
     stream (paired comparison; its overall score reproduces its recorded validation score).
With --extra-clocks none (the default) the model, features, loss, optimiser and stream are those of
race_affinity_v2_large_v2.py operation for operation.
Measured basis (cross-fitted into the sealed checkpoints): the 120-day share
adds +0.0013 NDCG@10 on genre; 256/1,024-edge shares +0.0009 on reddit, +0.0019 in its least-active quartile and +0.0023
for users silent >= 7 days.
"""
import argparse, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/pylib_tgb')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_tgb
import tgb_sparse_labels
from race_affinity_v2 import Readout, TAU_DAYS, TAU_LAB, EPS, _PF
from race_affinity_v2_large_v2 import State

DAY = 86400.0
EXTRA = {'none': (), 'v3': (('time', 120.0), ('count', 256), ('count', 1024))}
N_CTX = 6
RATE_EDGES = {'tgbn-genre': (1.3246026683141467, 1.5561537739504836, 1.731248627403268),
              'tgbn-reddit': (0.25348100937818574, 0.4724558254524506, 0.699469862549213),
              'tgbn-token': (-0.2985433813573953, -0.04401348550394996, 0.32608656027060967)}
SIL_EDGES = (1.0, 7.0)
SIL_NAMES = ('<1d', '1-7d', '>=7d', 'no_edges')
RATE_NAMES = ('rate_q1', 'rate_q2', 'rate_q3', 'rate_q4', 'no_edges')
DISC = 1.0 / np.log2(np.arange(2, 12)); DCUM = np.cumsum(DISC); KC = 64


def ndcg_rows(S, Y):
    """Per-row NDCG@10 with sklearn ndcg_score semantics (tie-averaged DCG, linear gains). Untied rows use the top-64
    candidates, exact when the 10th score exceeds the 11th and the 64th; other rows use the full row."""
    ideal = -np.sort(np.partition(-Y, 9, axis=1)[:, :10], 1) @ DISC
    idx = np.argpartition(-S, KC - 1, axis=1)[:, :KC]
    M = np.take_along_axis(S, idx, 1); bound = M.min(1)
    j = np.argpartition(-M, 10, axis=1)[:, :11]; mv = np.take_along_axis(M, j, 1)
    o = np.argsort(-mv, 1, kind='stable'); j = np.take_along_axis(j, o, 1); mv = np.take_along_axis(mv, o, 1)
    exact = (mv[:, 9] > np.maximum(mv[:, 10], bound)) & np.all(mv[:, :9] > mv[:, 1:10], 1)
    dcg = np.take_along_axis(Y, np.take_along_axis(idx, j[:, :10], 1), 1) @ DISC
    for r in np.flatnonzero(~exact):
        _, inv, counts = np.unique(-S[r], return_inverse=True, return_counts=True)
        ranked = np.zeros(len(counts)); np.add.at(ranked, inv, Y[r]); ranked /= counts
        dcg[r] = float((ranked * np.diff(np.r_[0.0, DCUM[np.minimum(np.cumsum(counts) - 1, 9)]])).sum())
    return np.where(ideal > 0, dcg / np.maximum(ideal, 1e-300), 0.0)


class ClockBank:
    """Per-user edge mass over classes on extra clocks: A[j,u,c] = sum_i w_i exp((v_i - ref[j,u]) / s_j), v_i the edge's
    clock value (seconds, or the user's edge index). Also the user's edge count and first edge time (activity)."""

    def __init__(self, U, C, kinds, bound=20.0):
        self.kinds = kinds
        self.A = np.zeros((len(kinds), U, C), np.float32); self.ref = np.zeros((len(kinds), U))
        self.scale = np.array([s * DAY if k == 'time' else float(s) for k, s in kinds])
        self.is_time = np.array([k == 'time' for k, _ in kinds], bool); self.bound = bound
        self.n = np.zeros(U, np.int64); self.first = np.full(U, np.nan); self.last = np.full(U, np.nan)

    def add_edges(self, u, c, w, t):
        if len(u) == 0:
            return
        o = np.argsort(u, kind='stable'); us, ts = u[o], t[o]
        new = np.r_[True, us[1:] != us[:-1]]
        start = np.maximum.accumulate(np.where(new, np.arange(len(us)), 0))
        rank = np.empty(len(u), np.int64); rank[o] = np.arange(len(us)) - start
        uu, inv, cnt = np.unique(u, return_inverse=True, return_counts=True)
        g = np.flatnonzero(new)
        self.first[uu] = np.where(np.isnan(self.first[uu]), np.minimum.reduceat(ts, g), self.first[uu])
        self.last[uu] = np.maximum.reduceat(ts, g)
        cval = (self.n[u] + rank + 1).astype(float)
        for j in range(len(self.kinds)):
            v = t if self.is_time[j] else cval
            vmax = np.full(len(uu), -np.inf); np.maximum.at(vmax, inv, v)
            need = (vmax - self.ref[j, uu]) / self.scale[j] > self.bound
            if need.any():
                ru = uu[need]
                f = np.exp(-(vmax[need] - self.ref[j, ru]) / self.scale[j]).astype(np.float32)
                self.A[j, ru] *= f[:, None]; self.ref[j, ru] = vmax[need]
            np.add.at(self.A[j], (u, c), (w * np.exp((v - self.ref[j, u]) / self.scale[j])).astype(np.float32))
        self.n[uu] += cnt

    def shares(self, u):
        a = self.A[:, u]; s = a.sum(-1, keepdims=True)
        return np.where(s > 0, a / np.maximum(s, 1e-30), 0.0).astype(np.float32)

    def activity(self, u, t):
        span = np.maximum((t - self.first[u]) / DAY, 1.0)
        return np.where(self.n[u] > 0, self.n[u] / span, np.nan)


def v3_inputs(x, bank, u, t):
    """x: v2 features [n, C, 27]. Returns (readout features [n, C, 27 + 2J], clock reads q [n, C, 4 + J], ctx [n, 6])."""
    S = bank.shares(u)
    Sn = np.moveaxis(S, 0, -1)
    xx = np.concatenate([x, Sn, np.log(Sn + EPS)], -1)
    q = np.concatenate([x[..., 5:5 + len(TAU_DAYS)], Sn], -1)
    rate = np.nan_to_num(bank.activity(u, t), nan=0.0)
    ctx = np.concatenate([x[:, 0, -4:], np.log1p(bank.n[u])[:, None], np.log1p(rate)[:, None]], -1)
    return xx.astype(np.float32), q.astype(np.float32), ctx.astype(np.float32)


class RaceReadout(nn.Module):
    """v2 readout over [v2 features, extra clock shares, raced share r and log r]; r = sum_m pi_m q_m with a soft per-user
    race pi = softmax(W ctx + b) over the clock bank (zero-initialised bids: uniform race at the start)."""

    def __init__(self, nf, h, n_clocks):
        super().__init__()
        self.readout = Readout(nf + 2, h)
        self.bid = nn.Linear(N_CTX, n_clocks)
        nn.init.zeros_(self.bid.weight); nn.init.zeros_(self.bid.bias)

    def race(self, ctx):
        return torch.softmax(self.bid(ctx), -1)

    def forward(self, x, q, ctx):
        pi = self.race(ctx)
        r = torch.einsum('nm,ncm->nc', pi, q)
        return self.readout(torch.cat([x, r[..., None], torch.log(r + EPS)[..., None]], -1))


def run_stream(ds, ev, model, opt, splits, C, U, train_ok, w_all, bs=200, max_label_times=0, chunk=1024, extra=(),
               dataset=None, baseline=None):
    """One streaming pass over the given splits (race_affinity_v2_large_v2.run_stream plus the clock bank and reporting).
    Returns {split: (mean NDCG over scored label times, label times, report)}."""
    d = ds.full_data; src_all = d['sources'].astype(np.int64); dst_all = d['destinations'].astype(np.int64)
    t_all = d['timestamps'].astype(float)
    st = State(U, C); ds.reset_label_time(); out = {}; fired = 0
    bank = ClockBank(U, C, extra)
    race = isinstance(model, RaceReadout); edges = RATE_EDGES.get(dataset)
    for name, mask in splits:
        idx = np.where(mask)[0]; res = []; label_t = ds.return_label_ts()
        rep = dict(model_sil=np.zeros(4), model_rate=np.zeros(5), base_sil=np.zeros(4), base_rate=np.zeros(5),
                   n_sil=np.zeros(4), n_rate=np.zeros(5), base_official=[], pi_sum=None, pi_win=None, rows=0)
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
                        x = st.features(u[s:s + chunk], float(lt)); yt = torch.from_numpy(y[s:s + chunk])
                        if race:
                            xx, q, ctx = v3_inputs(x, bank, u[s:s + chunk], float(lt))
                            sc = model(torch.from_numpy(xx), torch.from_numpy(q), torch.from_numpy(ctx))
                        else:
                            sc = model(torch.from_numpy(x))
                        loss = -(yt * torch.log_softmax(sc, -1)).sum(-1).sum() / n
                        loss.backward()
                    opt.step()
                elif name != 'train':
                    model.eval(); preds, bpreds, pis = [], [], []
                    with torch.no_grad():
                        for s in range(0, n, chunk):
                            x = st.features(u[s:s + chunk], float(lt))
                            if race:
                                xx, q, ctx = v3_inputs(x, bank, u[s:s + chunk], float(lt)); ct = torch.from_numpy(ctx)
                                preds.append(model(torch.from_numpy(xx), torch.from_numpy(q), ct).numpy())
                                pis.append(model.race(ct).numpy())
                            else:
                                preds.append(model(torch.from_numpy(x)).numpy())
                            if baseline is not None:
                                bpreds.append(baseline(torch.from_numpy(x)).numpy())
                    pred = np.concatenate(preds)
                    res.append(float(ev.eval({'y_true': y, 'y_pred': pred, 'eval_metric': ['ndcg']})['ndcg']))
                    if edges is not None:
                        el = st.elast[u]
                        sil = np.where(np.isnan(el), np.inf, (float(lt) - np.nan_to_num(el)) / DAY)
                        sb = np.digitize(sil, SIL_EDGES); sb[np.isinf(sil)] = 3
                        rate = bank.activity(u, float(lt))
                        rb = np.where(np.isnan(rate), 4, np.digitize(np.log10(np.where(np.isnan(rate), 1.0, rate)), edges))
                        yd = y.astype(np.float64)
                        nd = ndcg_rows(pred.astype(np.float64), yd)
                        np.add.at(rep['model_sil'], sb, nd); np.add.at(rep['model_rate'], rb, nd)
                        rep['n_sil'] += np.bincount(sb, minlength=4); rep['n_rate'] += np.bincount(rb, minlength=5)
                        if baseline is not None:
                            bp = np.concatenate(bpreds)
                            rep['base_official'].append(float(ev.eval({'y_true': y, 'y_pred': bp, 'eval_metric': ['ndcg']})['ndcg']))
                            nb = ndcg_rows(bp.astype(np.float64), yd)
                            np.add.at(rep['base_sil'], sb, nb); np.add.at(rep['base_rate'], rb, nb)
                    if pis:
                        pi = np.concatenate(pis)
                        rep['pi_sum'] = pi.sum(0) if rep['pi_sum'] is None else rep['pi_sum'] + pi.sum(0)
                        w = np.bincount(pi.argmax(1), minlength=pi.shape[1])
                        rep['pi_win'] = w if rep['pi_win'] is None else rep['pi_win'] + w
                        rep['rows'] += n
                st.reveal(u, y); label_t = ds.return_label_ts(); fired += 1
                if max_label_times and fired >= max_label_times:
                    break
            s = src_all[ix] - C; keep = s >= 0
            st.add_edges(s[keep], dst_all[ix][keep], w_all[ix][keep], tq)
            bank.add_edges(s[keep], dst_all[ix][keep], w_all[ix][keep], t_all[ix][keep])
        out[name] = (float(np.mean(res)) if res else None, len(res), report(rep, bank.kinds))
    return out


def report(rep, kinds):
    def means(s, n):
        return [float(a / c) if c else None for a, c in zip(s, n)]
    r = dict(silence_buckets=list(SIL_NAMES), rate_buckets=list(RATE_NAMES), rows_silence=rep['n_sil'].astype(int).tolist(),
             rows_rate=rep['n_rate'].astype(int).tolist(), model_silence=means(rep['model_sil'], rep['n_sil']),
             model_rate=means(rep['model_rate'], rep['n_rate']))
    if rep['base_official']:
        r.update(baseline_ndcg=float(np.mean(rep['base_official'])), baseline_silence=means(rep['base_sil'], rep['n_sil']),
                 baseline_rate=means(rep['base_rate'], rep['n_rate']))
    if rep['pi_sum'] is not None:
        names = [f'time_{t:g}d' for t in TAU_DAYS] + [f'{k}_{s:g}{"d" if k == "time" else "e"}' for k, s in kinds]
        r['race'] = dict(clocks=names, mean_pi=(rep['pi_sum'] / rep['rows']).tolist(),
                         win_share=(rep['pi_win'] / rep['rows']).tolist())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', required=True)
    ap.add_argument('--root', default='../../../data/tgb/root'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--max-label-times', type=int, default=0)
    ap.add_argument('--edge-weight', choices=('count', 'feat'), default='feat')
    ap.add_argument('--chunk-users', type=int, default=1024)
    ap.add_argument('--extra-clocks', choices=sorted(EXTRA), default='none')
    ap.add_argument('--baseline-checkpoint', default='', help='v2 checkpoint scored on the same validation rows (reporting)')
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
    extra = EXTRA[a.extra_clocks]
    if extra:
        model = RaceReadout(nf + 2 * len(extra), a.hidden, len(TAU_DAYS) + len(extra))
    else:
        model = Readout(nf, a.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    baseline, base_info = None, None
    if a.baseline_checkpoint:
        baseline = Readout(nf, a.hidden); bp = ROOT / a.baseline_checkpoint
        baseline.load_state_dict(torch.load(bp)); baseline.eval()
        base_info = dict(checkpoint=a.baseline_checkpoint, sha256=hashlib.sha256(bp.read_bytes()).hexdigest())
    od = ROOT / 'experiments/results/tgb'; od.mkdir(parents=True, exist_ok=True); ckpt = od / f'{a.tag}.pt'
    kw = dict(max_label_times=a.max_label_times, chunk=a.chunk_users, dataset=a.dataset)
    p0 = time.time()
    pf = run_stream(ds, ev, _PF(), None, [('train', ds.train_mask), ('val', ds.val_mask)], C, U, False, w_all, **kw)
    print('persistent forecast (our replay)', json.dumps(pf['val'][:2]), f'{time.time() - p0:.0f}s', flush=True)
    hist, best, best_rep = [], -1.0, None
    for ep in range(a.epochs):
        e0 = time.time()
        r = run_stream(ds, ev, model, opt, [('train', ds.train_mask), ('val', ds.val_mask)], C, U, True, w_all,
                       extra=extra, baseline=baseline, **kw)
        hist.append(dict(epoch=ep, val_ndcg=r['val'][0], val_label_times=r['val'][1], epoch_s=time.time() - e0,
                         vmhwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, val_report=r['val'][2]))
        print(json.dumps({k: v for k, v in hist[-1].items() if k != 'val_report'}), flush=True)
        if r['val'][0] is not None and r['val'][0] > best:
            best = r['val'][0]; best_rep = r['val'][2]; torch.save(model.state_dict(), ckpt)
    res = dict(status='completed', benchmark='TGB', tag=a.tag, dataset=a.dataset, model='race_affinity v3 (multi-clock bank, soft per-user clock race)',
               args=vars(a), classes=C, users=U, features=nf + 2 * len(extra) + (2 if extra else 0), extra_clocks=[list(k) for k in extra],
               parameters=sum(p.numel() for p in model.parameters()),
               persistent_forecast_val_ndcg=pf['val'][0], persistent_forecast_val_label_times=pf['val'][1],
               best_val_ndcg=best, best_val_report=best_rep, history=hist, checkpoint=str(ckpt.relative_to(ROOT)),
               baseline=base_info, load_s=load_s, rate_edges_log10=RATE_EDGES.get(a.dataset),
               extra_state_bytes=int(len(extra) * U * C * 4))
    if a.score_test:
        model.load_state_dict(torch.load(ckpt))
        r = run_stream(ds, ev, model, None, [('train', ds.train_mask), ('val', ds.val_mask), ('test', ds.test_mask)], C, U, False,
                       w_all, extra=extra, **kw)
        res['replay_val_ndcg'] = r['val'][0]; res['test_ndcg'] = r['test'][0]; res['test_label_times'] = r['test'][1]
        res['test_report'] = r['test'][2]
    res['wall_s'] = time.time() - t0; res['vmhwm_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    here = Path(__file__).resolve().parent
    res['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__).resolve(), here / 'race_affinity_v2.py', here / 'race_affinity_v2_large_v2.py',
                                      here / 'safe_tgb.py', here / 'tgb_sparse_labels.py')}
    (od / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('RESULT', json.dumps({k: res.get(k) for k in ('persistent_forecast_val_ndcg', 'best_val_ndcg', 'test_ndcg', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
