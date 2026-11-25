#!/usr/bin/env python3
"""tgbl-review causal driver v7 = v6 + the addressed satiation race (refractory pair clocks), and a sealed TEST path.

Failure (diagnosis 2 Nov 2026, VALIDATION only): the official v2 negatives
(tgbl-review_neg_generator.py: hist_rnd, ratio 0.5, historical_data = TRAIN) make up to 50 of each query's 100 negatives
the source's own past destinations, and a reviewer almost never reviews an item twice (0.17% of validation positives).
v2-v6 never put a source's own history into its training race, so the network learned a seen pair as POSITIVE evidence
and ranks those historical negatives above the positive.

Mechanism (both off by default; off = v6 exactly, asserted by an equivalence smoke):
  --satiation-race H   training race of each query = positive + up to floor(H*K) competitors read from the source's own
                       addressed history (its distinct past destinations, i.e. its pair slots; the positive excluded)
                       + the v6 proposal for the rest (recent-event ring + uniform, duplicates redrawn as in v2). With
                       --logq the correction uses the exact per-query mixture q(j) = w_h 1[j in hist(s)]/|hist(s)| +
                       (1 - w_h) q_v6(j), w_h = k_h / K, computed at draw time.
  --refractory         explicit refractory clock on every addressed pair slot: the candidate's race score is lowered by
                       r(s,c,t) = seen(s,c) * (softplus(rho_0) + sum_k softplus(rho_k) exp(-(t - t_last(s,c)) / tau_k)),
                       tau = 1, 7, 30, 365 days (the state's clocks), rho initialised at 0. Only touched pairs inhibit;
                       r >= 0 by construction (inhibition only), learned through the satiation race.
  --score-test         sealed path: after training (best VALIDATION epoch kept), a fresh causal state replays TRAIN and
                       VALIDATION, the best checkpoint scores the official TEST negatives once (Evaluator agreement on the
                       first 50 queries).
Retained: strict causal per-query state (ties excluded), sparse addressed pair/destination/source clocks, sampled race
readout with losing-candidate credit, v4 switches, v5 hard negatives, v6 log-Q. Evaluation and the official Evaluator are
unchanged.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import race_link_review_v6 as v6
import race_link_review_v5 as v5
import race_link_review_v4 as v4
import race_link_review_v2 as v2
import race_link_review as base

SAT = dict(on=False, H=0.5)
REFR = dict(on=False)
_v6_add = base.State.add
_v6_causal_batch = v2.causal_batch
_ModelV6 = base.Model


def add(self, s, c, t):
    """v6 add; with the satiation race also the source's distinct past destinations (its pair slots, in order)."""
    if SAT['on']:
        hist = self.__dict__.setdefault('src_hist', {})
        new = set()
        for u, p in zip(np.asarray(s).tolist(), np.asarray(c).tolist()):
            if (u, p) not in self.pidx and (u, p) not in new:
                new.add((u, p)); hist.setdefault(u, []).append(p)
    _v6_add(self, s, c, t)


def q_v6(st, j):
    """Probability that one v6 proposal draw yields j (as v6.log_q, without the log)."""
    n = st.n
    if v5.HARD['on']:
        m = max(min(getattr(st, '_ring_n', 0), v5.HARD['R']), 1)
        cnt = getattr(st, '_ring_cnt', None)
        qh = (cnt[j] / m) if cnt is not None else np.zeros(len(j))
    else:
        L = max(len(st.seen_list), 1)
        qh = st.seen_dst[j].astype(float) / L
    return 0.5 * qh + 0.5 / n


def causal_batch(st, s, c, t, candidates=None, neg=20, rng=None):
    if candidates is not None or not SAT['on']:
        return _v6_causal_batch(st, s, c, t, candidates=candidates, neg=neg, rng=rng)
    features, rows, lqs = [], [], []
    src_hist = st.__dict__.setdefault('src_hist', {})
    for q in range(len(s)):
        st.advance(float(t[q]))
        u, pos = int(s[q]), int(c[q])
        hist = src_hist.get(u, [])
        pool = [x for x in hist if x != pos] if (u, pos) in st.pidx else hist
        k_h = min(int(neg * SAT['H']), len(pool))
        h = np.asarray(pool, np.int64)[rng.choice(len(pool), k_h, replace=False)] if k_h else np.zeros(0, np.int64)
        rest = st.sample_negatives(neg - k_h, rng, 1)[0] if neg > k_h else np.zeros(0, np.int64)
        used = {pos, *h.tolist()}
        for j in range(len(rest)):
            while int(rest[j]) in used:
                rest[j] = rng.integers(st.n)
            used.add(int(rest[j]))
        cand = np.concatenate(([pos], h, rest)).astype(np.int64)
        assert len(set(cand.tolist())) == len(cand) and all((u, int(x)) in st.pidx for x in h)
        if v6.LOGQ['on']:
            w_h = k_h / neg
            in_pool = np.zeros(len(cand), bool)
            if k_h:
                ps = set(pool); in_pool = np.array([int(x) in ps for x in cand])
            qq = w_h * in_pool / max(len(pool), 1) + (1.0 - w_h) * q_v6(st, cand)
            lqs.append(np.log(qq))
        features.append(st.features(np.asarray([s[q]]), cand[None], np.asarray([t[q]]))[0])
        rows.append(cand)
        st.push(s[q], c[q], float(t[q]))
    cand, feat = np.stack(rows), np.stack(features)
    v6.LOGQ['stash'] = np.stack(lqs) if v6.LOGQ['on'] else None
    return cand, feat


TAUS = torch.tensor(base.TAUS, dtype=torch.float32)
SEEN, PAIR_DT = 5, 6


class Model(_ModelV6):
    def __init__(self, n, dim, hidden):
        super().__init__(n, dim, hidden)
        self.rho = nn.Parameter(torch.zeros(1 + len(base.TAUS))) if REFR['on'] else None

    def refractory(self, feat):
        seen = feat[..., SEEN]
        x = feat[..., PAIR_DT]
        dt = torch.sign(x) * torch.expm1(x.abs())
        clocks = torch.exp(-dt[..., None].clamp(min=0) / TAUS)
        a = F.softplus(self.rho)
        return seen * (a[0] + (clocks * a[1:]).sum(-1))

    def forward(self, s, cand, feat):
        x = super().forward(s, cand, feat)
        if self.rho is not None:
            x = x - self.refractory(feat)
        return x


def score_test(res, a):
    """Sealed TEST: best checkpoint, fresh causal state through TRAIN + VALIDATION, official TEST negatives once."""
    base.safe_tgb.patch_tgb()
    from tgb.linkproppred.dataset import LinkPropPredDataset
    from tgb.linkproppred.evaluate import Evaluator
    t0 = time.time()
    ds = LinkPropPredDataset(name=res['args']['dataset'], root=res['args']['root'], preprocess=True, download=False)
    ev = Evaluator(name=res['args']['dataset']); ds.load_test_ns(); d = ds.full_data
    n = int(max(d['sources'].max(), d['destinations'].max())) + 1
    model = base.Model(n, res['args']['dim'], res['args']['hidden'])
    model.load_state_dict(torch.load(v2.ROOT / res['checkpoint'])); model.eval()
    st = v2.CausalState(n); hist = np.flatnonzero(ds.train_mask | ds.val_mask)
    st.history(d['sources'][hist], d['destinations'][hist], d['timestamps'][hist].astype(float))
    mrr, nq, wall, profile = v2.evaluate(ds, ev, model, st, ds.test_mask, 'test')
    return dict(test_mrr=mrr, test_queries=nq, test_query_profile=profile, test_wall_s=time.time() - t0,
                test_protocol='best VALIDATION epoch checkpoint; causal state replayed through TRAIN and VALIDATION; '
                              'official v2 TEST negatives and Evaluator (agreement on the first 50 queries); scored once')


def main():
    test = '--score-test' in sys.argv
    if test:
        sys.argv.remove('--score-test')
    if '--satiation-race' in sys.argv:
        i = sys.argv.index('--satiation-race'); SAT.update(on=True, H=float(sys.argv[i + 1])); del sys.argv[i:i + 2]
        assert 0 < SAT['H'] <= 1
    if '--refractory' in sys.argv:
        sys.argv.remove('--refractory'); REFR['on'] = True
    if SAT['on']:
        base.State.add = add
        v2.causal_batch = causal_batch
    if REFR['on']:
        base.Model = Model
    tag = sys.argv[sys.argv.index('--tag') + 1]
    v6.main()
    out = v2.ROOT / 'experiments/results/tgb' / f'{tag}.json'
    res = json.loads(out.read_text())
    res['driver'] = 'race_link_review_v7.py (v6 + addressed satiation race / refractory pair clocks; sealed TEST path)'
    res['satiation_race'] = dict(SAT); res['refractory'] = dict(REFR); res['score_test'] = test
    res.setdefault('source_sha256', {})[str(Path(__file__).resolve().relative_to(v2.ROOT))] = \
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if REFR['on'] and res.get('status') == 'completed':
        sd = torch.load(v2.ROOT / res['checkpoint'])
        res['refractory_amplitudes'] = dict(zip(['seen', '1d', '7d', '30d', '365d'], F.softplus(sd['rho']).tolist()))
    out.write_text(json.dumps(res, indent=1) + '\n')
    if test and res.get('status') == 'completed':
        res.update(score_test(res, res['args']))
        res['protocol'] = res['protocol'].replace('; no TEST', '; TEST scored once with the best VALIDATION checkpoint')
        out.write_text(json.dumps(res, indent=1) + '\n')
        print('TEST', json.dumps({k: res[k] for k in ('val_mrr', 'test_mrr', 'test_queries')}), flush=True)


if __name__ == '__main__':
    main()
