#!/usr/bin/env python3
"""race_link v4 for tgbl-coin-v2 (7 Nov 2026): v3 with the amount code repaired. The official edge
weights of tgbl-coin include values <= -1, so v3's log1p amount code gave -inf / NaN and its --amounts arm stopped at the
first non-finite loss. v4 codes amounts as sign(w) log1p(|w|) with non-finite values set to 0;
everything else is v3.

v3 description follows.
race_link v3 for tgbl-coin-v2 (7 Nov 2026): v2 plus flow-through slots and amount reads.
v2 unchanged with both new flags off.
  --flow: a transfer s -> c is filed under the payer's newest incoming payer y (at a strictly earlier timestamp) in
     per-payer flow slots (s, y -> c) and global flow slots (y -> c): where money arriving from y goes next, for this
     payer and through any intermediary (deposit addresses forwarding to an exchange, routers forwarding to pools). The
     query reads the slots addressed by the payer's newest payer before t.
  --amounts: every slot event stores its amount (log1p of the transferred value); reads give the pair's newest amount,
     the payer's newest incoming and outgoing amounts and their match (a forwarder pays on what it just received).
  --novelty: the payer's number of distinct recipients before t, its novelty rate (distinct recipients per transfer)
     and the time since it last paid a new recipient; the candidate's number of distinct payers, its rate and the time
     since a new payer last paid it (exchanges and payout bots pay new addresses; hubs are paid by many). Read from the
     pair-slot directories sorted by first event.

v2 description follows.
race_link v2 for tgbl-coin-v2 (7 Nov 2026): v1 plus exact two-hop path reads through addressed
payloads and transition slots.

v1 unchanged with both new flags off. Diagnosis: the random-projection sketch added +0.0016 on
the smoke because hubs (exchanges, pools) sum thousands of codes and drown the two-hop signal in projection noise. v2
reads two-hop flows exactly instead, by pointer-chasing through addressed memory:
  --paths M: every slot event stores its counterparty (payload); the candidate's last M recipients x and last M payers y,
     and the source's last M recipients and payers, address the pair slots that close a two-step flow:
     c -> x -> s (e.g. a deposit swept into the paying exchange), s -> y -> c, s -> x -> c and c -> y -> s. Reads are
     strictly causal (events with time < t); aggregates over the M paths: max clocks, total count, newest, any.
  --transitions: the source's last recipient l (payload of its newest slot event) addresses per-source transition slots
     (s, l -> c) and global transition slots (l -> c): which recipient follows l, for this payer and for all payers (bots
     cycling recipients, router -> pool sequences). A transition event's predecessor is the payer's newest recipient at a
     strictly earlier timestamp, the same rule the query read uses.

v1 description follows.
race_link for tgbl-coin-v2 (7 Nov 2026): a race among candidate recipient addresses read from sparse addressed
temporal memory, with renewal clocks, source-event clocks, return-flow reads and a directed two-hop addressed sketch.

Lineage: race_link_flight.py (tgbl-flight-v2: exact vectorised reads of sorted slot families, hist_rnd-shaped training
race), race_link_review_v7 (tgbl-review-v2: the source's own past destinations as training competitors), race_link_v4
(tgbl-wiki-v2: per-event causal state). Retained: strict causal per-query state (every event with time < t_query, ties
excluded), sparse addressed pair / return-pair / destination / source clocks, the race readout with losing-candidate
credit, no identity tables, official negatives and Evaluator, best VALIDATION epoch, TEST only with --score-test through
the sealed path.
New for tgbl-coin, each required by a property of the data (stablecoin transfers between addresses, second timestamps,
bursty bots, exchanges and pools with millions of transfers):
  1. clocks from ten minutes to a month (600 s, 1 h, 1 day, 7 days, 30 days) on every slot family.
  2. renewal read of the pair slot: the pair's mean inter-transfer gap and the elapsed time in units of it (a regular
     payer is due when dt reaches its own gap, whatever the wall-clock scale).
  3. source-event clock: how many transfers the source made since it last paid the candidate (time measured in the
     source's own events; bursty sources make wall time incomparable across sources).
  4. daily and weekly delayed reads: pair and destination transfers within one hour of exactly 1 and 7 days earlier.
  5. return flow: the candidate's own slot toward the source (c -> s), and the source's activity as a recipient.
  6. --sketch-d > 0: a directed two-hop addressed sketch. Every address keeps decayed sums of fixed random codes of the
     addresses it paid (out) and that paid it (in), at --sketch-taus timescales; an event updates only its two endpoint
     slots. Reads are inner products: common recipients <out_s, out_c>, common payers <in_s, in_c>, forward paths
     s -> x -> c <out_s, in_c>, backward paths c -> x -> s <in_s, out_c>. Exact with one-hot codes (contract). The sketch
     is streamed in time order with per-timestamp groups, so a query reads every event strictly before it.
--diagnose runs no fit: the scorer proof (EdgeBank unlimited and time-window with TGB's batch-of-200 memory against the
leaderboard's validation 0.315 / 0.492), strict-causal heuristics and the query classes, on VALIDATION only.
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
TAUS = np.array([600.0, 3600.0, DAY, 7 * DAY, 30 * DAY])
LAGS = (1, 7)
LAG_HALF = 3600.0
NO_EVENT_DT = 1e9
CLASSES = ('repeat_pair', 'return_only', 'new_pair_known_source', 'new_source')
DATASET = 'tgbl-coin'
RESCALE = 20.0


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
    def __init__(self, keys, rank, t, nbits, clocks=True, payload=None, amount=None):
        self.u, inv = np.unique(keys, return_inverse=True); self.nbits = nbits
        comp = (inv.astype(np.int64) << nbits) | rank
        o = np.argsort(comp, kind='stable'); del inv
        self.comp = comp[o]; self.t = t[o]
        self.payload = None if payload is None else payload[o]
        self.amount = None if amount is None else amount[o]
        del comp, o
        self.start = np.searchsorted(self.comp, np.arange(len(self.u), dtype=np.int64) << nbits)
        self.A = decay_scan(self.start, np.diff(np.append(self.start, len(self.comp))), self.t) if clocks else None

    def slot(self, q):
        i = np.minimum(np.searchsorted(self.u, q), len(self.u) - 1)
        return i, self.u[i] == q

    def count(self, q, qr):
        i, found = self.slot(q)
        hi = np.searchsorted(self.comp, (i.astype(np.int64) << self.nbits) | qr)
        return np.where(found, hi - self.start[i], 0)

    def last(self, q, qr, M):
        """Payloads (counterparties) of the newest M events with rank < qr, newest first; -1 where absent."""
        i, found = self.slot(q)
        hi = np.searchsorted(self.comp, (i.astype(np.int64) << self.nbits) | qr); st = self.start[i]
        out = []
        for m in range(M):
            j = hi - 1 - m; ok = found & (j >= st)
            out.append(np.where(ok, self.payload[np.maximum(j, 0)], -1))
        return np.stack(out, -1)

    def newest_amount(self, q, qr):
        """Amount of the newest event with rank < qr and whether one exists."""
        i, found = self.slot(q)
        hi = np.searchsorted(self.comp, (i.astype(np.int64) << self.nbits) | qr)
        has = found & (hi > self.start[i])
        return np.where(has, self.amount[np.maximum(hi - 1, 0)], 0.0), has

    def read(self, q, qr, t):
        """count, dt since the last event, clocks, time of the first event (all over events with rank < qr)."""
        i, found = self.slot(q)
        hi = np.searchsorted(self.comp, (i.astype(np.int64) << self.nbits) | qr)
        cnt = np.where(found, hi - self.start[i], 0); has = cnt > 0; j = np.maximum(hi - 1, 0)
        dt = np.where(has, t - self.t[j], NO_EVENT_DT).astype(np.float64)
        first = np.where(has, self.t[self.start[i]], t).astype(np.float64)
        clk = None
        if self.A is not None:
            clk = np.where(has[..., None], self.A[j] * np.exp(-np.where(has, dt, 0.0)[..., None] / TAUS), 0.0)
        return cnt, dt, clk, first


class Memory:
    """Addressed slot families over the events at positions [0, E) of the stream (time ordered)."""
    def __init__(self, src, dst, ts, n, paths=0, transitions=False, flow=False, amt=None, novelty=False):
        self.n = n; self.paths = paths; self.transitions = transitions; self.flow = flow; self.amounts = amt is not None
        self.novelty = novelty
        self.ut = np.unique(ts); rank = np.searchsorted(self.ut, ts).astype(np.int64)
        nb = int(np.ceil(np.log2(len(self.ut) + 2)))
        self.fam = dict(pair=Slots(src * n + dst, rank, ts, nb, amount=amt), dst=Slots(dst, rank, ts, nb, payload=src, amount=amt),
                        src=Slots(src, rank, ts, nb, payload=dst, amount=amt))
        if flow:
            f, g = self.fam['src'], self.fam['dst']; mask = (1 << nb) - 1
            slot_of = np.repeat(np.arange(len(f.u)), np.diff(np.append(f.start, len(f.comp))))
            fs = f.u[slot_of]; del slot_of
            i, found = g.slot(fs)
            prev = np.searchsorted(g.comp, (i.astype(np.int64) << nb) | (f.comp & mask), 'left') - 1
            ok = found & (prev >= g.start[i])
            fs, fy, fc = fs[ok], g.payload[np.maximum(prev, 0)][ok], f.payload[ok]
            fr, ft = (f.comp & mask)[ok], f.t[ok]; del i, found, prev, ok
            self.fam['fls'] = Slots((fs * n + fy) * n + fc, fr, ft, nb)
            self.fam['flg'] = Slots(fy * n + fc, fr, ft, nb)
            del fs, fy, fc, fr, ft
        if transitions:
            f = self.fam['src']; mask = (1 << nb) - 1
            slot_of = np.repeat(np.arange(len(f.u)), np.diff(np.append(f.start, len(f.comp))))
            prev = np.searchsorted(f.comp, f.comp, 'left') - 1
            ok = prev >= f.start[slot_of]
            ps, pl, pd = f.u[slot_of][ok], f.payload[np.maximum(prev, 0)][ok], f.payload[ok]
            pr, pt = (f.comp & mask)[ok], f.t[ok]; del slot_of, prev, ok
            self.fam['trs'] = Slots((ps * n + pl) * n + pd, pr, pt, nb)
            self.fam['trg'] = Slots(pl * n + pd, pr, pt, nb)
            del ps, pl, pd, pr, pt
        p = self.fam['pair']
        self.first_t = p.t[p.start]; srcof = p.u // n
        o = np.lexsort((self.first_t, srcof))
        self.h_dst = (p.u % n)[o]; self.h_first = self.first_t[o]; self.h_src = srcof[o]
        self.h_lo = np.searchsorted(self.h_src, np.arange(n + 1))
        self.h_key = self.h_src * (len(self.ut) + 1) + self.rank(self.h_first)
        if novelty:
            dstof = p.u % n; o = np.lexsort((self.first_t, dstof))
            self.g_first = self.first_t[o]; g_dst = dstof[o]; del dstof, o
            self.g_lo = np.searchsorted(g_dst, np.arange(n + 1))
            self.g_key = g_dst * (len(self.ut) + 1) + self.rank(self.g_first); del g_dst

    def rank(self, x):
        return np.searchsorted(self.ut, x).astype(np.int64)

    def nf(self):
        k = len(TAUS)
        return (2 * (k + 3 + len(LAGS) + 3) + (k + 2 + len(LAGS)) + 3 * (k + 2) + (4 * (k + 3) if self.paths else 0)
                + (2 * (k + 2) + 1 if self.transitions else 0) + (2 * (k + 2) if self.flow else 0)
                + (5 if self.amounts else 0) + (6 if self.novelty else 0))

    def distinct(self, key, lo, first, q, qr, t, events):
        """Distinct counterparties before qr, distinct per event, time since the newest first contact."""
        nd = np.searchsorted(key, q * (len(self.ut) + 1) + qr) - lo[q]
        newest = np.where(nd > 0, first[np.maximum(lo[q] + nd - 1, 0)], 0)
        return [np.log1p(nd)[..., None], (np.log1p(nd) - np.log1p(events))[..., None],
                slog(np.where(nd > 0, t - newest, NO_EVENT_DT))[..., None]]

    def path_cols(self, X, key, QR, T):
        """Aggregate over the M two-step paths whose closing pair slots are key [Q, C, M] (X < 0: no path)."""
        M = X.shape[-1]
        q3, t3 = (np.broadcast_to(x[..., None], x.shape + (M,)) for x in (QR, T))
        cnt, dt, clk, _ = self.fam['pair'].read(np.where(X >= 0, key, -1), q3, t3)
        return [np.log1p(clk.max(-2)), np.log1p(cnt.sum(-1))[..., None], slog(dt.min(-1))[..., None],
                (cnt > 0).any(-1)[..., None]]

    def window(self, fam, key, t, L):
        lo, hi = t - L * DAY - LAG_HALF, np.minimum(t - L * DAY + LAG_HALF, t - 1.0)
        return np.log1p(self.fam[fam].count(key, self.rank(hi + 0.5)) - self.fam[fam].count(key, self.rank(lo)))[..., None]

    def pair_cols(self, key, S, QR, T):
        """Clocks, count, seen, dt, delayed reads, renewal read and the source-event clock of pair slots key = a*n + b."""
        cnt, dt, clk, first = self.fam['pair'].read(key, QR, T)
        has = cnt > 0
        gap = np.where(cnt > 1, (T - dt - first) / np.maximum(cnt - 1, 1), NO_EVENT_DT)
        rel = np.where(cnt > 1, np.log1p(dt) - np.log1p(gap), 0.0)
        last_r = self.rank(T - dt)
        since = np.where(has, self.fam['src'].count(S, QR) - self.fam['src'].count(S, last_r + 1), 0)
        cols = [np.log1p(clk), np.log1p(cnt)[..., None], has[..., None], slog(dt)[..., None]]
        cols += [self.window('pair', key, T, L) for L in LAGS]
        cols += [slog(gap)[..., None], rel[..., None], np.log1p(since)[..., None]]
        return cols

    def features(self, s, cand, t):
        """s, t [Q]; cand [Q, C] -> float32 [Q, C, NF]; reads only events with time < t."""
        n = self.n; Q, C = cand.shape
        qr = self.rank(t)
        QR, T, S = (np.broadcast_to(x[:, None], (Q, C)) for x in (qr, t, s))
        cols = self.pair_cols(S * n + cand, S, QR, T)
        cols += self.pair_cols(cand * n + S, S, QR, T)
        cnt, dt, clk, _ = self.fam['dst'].read(cand, QR, T)
        cols += [np.log1p(clk), np.log1p(cnt)[..., None], slog(dt)[..., None]]
        cols += [self.window('dst', cand, T, L) for L in LAGS]
        own = []
        for fam in ('src', 'dst'):
            cnt, dt, clk, _ = self.fam[fam].read(s, qr, t)
            own.append(np.concatenate([np.log1p(clk), np.log1p(cnt)[:, None], slog(dt)[:, None]], 1))
        cols.append(np.broadcast_to(np.concatenate(own, 1)[:, None], (Q, C, 2 * (len(TAUS) + 2))))
        cnt, dt, clk, _ = self.fam['src'].read(cand, QR, T)
        cols += [np.log1p(clk), np.log1p(cnt)[..., None], slog(dt)[..., None]]
        if self.paths:
            M = self.paths; S3 = S[..., None]; C3 = cand[..., None]
            X = self.fam['src'].last(cand, QR, M); cols += self.path_cols(X, X * n + S3, QR, T)
            Y = self.fam['dst'].last(cand, QR, M); cols += self.path_cols(Y, S3 * n + Y, QR, T)
            X = np.broadcast_to(self.fam['src'].last(s, qr, M)[:, None], (Q, C, M))
            cols += self.path_cols(X, X * n + C3, QR, T)
            Y = np.broadcast_to(self.fam['dst'].last(s, qr, M)[:, None], (Q, C, M))
            cols += self.path_cols(Y, C3 * n + Y, QR, T)
        if self.transitions:
            L = np.broadcast_to(self.fam['src'].last(s, qr, 1)[:, :1], (Q, C))
            for fam, key in (('trs', (S * n + L) * n + cand), ('trg', L * n + cand)):
                cnt, dt, clk, _ = self.fam[fam].read(np.where(L >= 0, key, -1), QR, T)
                cols += [np.log1p(clk), np.log1p(cnt)[..., None], slog(dt)[..., None]]
            cols.append(((cand == L) & (L >= 0))[..., None])
        if self.flow:
            Y = np.broadcast_to(self.fam['dst'].last(s, qr, 1)[:, :1], (Q, C))
            for fam, key in (('fls', (S * n + Y) * n + cand), ('flg', Y * n + cand)):
                cnt, dt, clk, _ = self.fam[fam].read(np.where(Y >= 0, key, -1), QR, T)
                cols += [np.log1p(clk), np.log1p(cnt)[..., None], slog(dt)[..., None]]
        if self.amounts:
            la, hl = self.fam['pair'].newest_amount(S * n + cand, QR)
            ia, hi_ = self.fam['dst'].newest_amount(s, qr); oa, _ = self.fam['src'].newest_amount(s, qr)
            IA = np.broadcast_to(ia[:, None], (Q, C)); both = hl & np.broadcast_to(hi_[:, None], (Q, C))
            cols += [la[..., None], hl[..., None], IA[..., None], np.broadcast_to(oa[:, None], (Q, C))[..., None],
                     np.where(both, -np.abs(la - IA), 0.0)[..., None]]
        if self.novelty:
            own = self.distinct(self.h_key, self.h_lo, self.h_first, s, qr, t, self.fam['src'].count(s, qr))
            cols += [np.broadcast_to(x[:, None], (Q, C, 1)) for x in own]
            cols += self.distinct(self.g_key, self.g_lo, self.g_first, cand, QR, T, self.fam['dst'].count(cand, QR))
        return np.concatenate([np.asarray(x, np.float32) for x in cols], -1)

    def query_class(self, s, c, t):
        qr = self.rank(t)
        seen = self.fam['pair'].count(s * self.n + c, qr) > 0
        ret = self.fam['pair'].count(c * self.n + s, qr) > 0
        known = self.fam['src'].count(s, qr) > 0
        return np.where(seen, 0, np.where(ret, 1, np.where(known, 2, 3)))

    def competitors(self, s, c, t, K, rng):
        """hist_rnd-shaped race: up to floor(K/2) of the source's destinations first seen before t (positive excluded),
        the rest uniform over addresses without a pair event before t (positive excluded). Draws with replacement."""
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


class Sketch:
    """Directed two-hop addressed sketch. state[v, 0, k] = sum over transfers v -> x of w_k R[x] (whom v paid),
    state[v, 1, k] = sum over x -> v of w_k R[x] (who paid v), w_k = exp(-(t - t_event) / tau_k), stored lazily as
    exp((t_event - t0_k) / tau_k) and rescaled when the exponent exceeds RESCALE. An event updates its two endpoint slots
    only. Reads at time t combine every update applied so far (callers apply events strictly before t)."""
    def __init__(self, n, d, taus, seed, codes=None):
        self.taus = np.asarray(taus, np.float64); self.K = len(self.taus); self.n, self.d = n, d
        if codes is None:
            codes = np.random.default_rng(10007 + seed).standard_normal((n, d)).astype(np.float32) / np.float32(np.sqrt(d))
        self.R = codes
        self.state = np.zeros((n, 2, self.K, d), np.float32); self.t0 = None

    def nf(self):
        return 4 * self.K

    def reset(self):
        self.state[:] = 0.0; self.t0 = None

    def _rescale(self, t):
        e = (t - self.t0) / self.taus
        for k in np.flatnonzero(e > RESCALE):
            self.state[:, :, k] *= np.float32(np.exp(-e[k])); self.t0[k] = t

    def update(self, u, v, t):
        """Apply transfers u -> v at times t (sorted). Splits at rescale points so results depend only on the events."""
        if not len(u):
            return
        if self.t0 is None:
            self.t0 = np.full(self.K, float(t[0]))
        b = 0
        while b < len(u):
            lim = self.t0 + RESCALE * self.taus
            e = b + int(np.searchsorted(t[b:], lim.min(), 'right'))
            if e == b:
                self._rescale(float(t[b])); continue
            w = np.exp((t[b:e, None] - self.t0[None, :]) / self.taus[None, :]).astype(np.float32)
            np.add.at(self.state[:, 0], u[b:e], w[:, :, None] * self.R[v[b:e]][:, None, :])
            np.add.at(self.state[:, 1], v[b:e], w[:, :, None] * self.R[u[b:e]][:, None, :])
            b = e

    def read(self, s, cand, t):
        """s, t [Q]; cand [Q, C] -> [Q, C, 4K]: common recipients, common payers, paths s->x->c, paths c->x->s."""
        Q, C = cand.shape
        if self.t0 is None:
            return np.zeros((Q, C, self.nf()), np.float32)
        a = self.state[s]; b = self.state[cand]
        f = np.stack([np.einsum('qkd,qckd->qck', a[:, 0], b[:, :, 0]), np.einsum('qkd,qckd->qck', a[:, 1], b[:, :, 1]),
                      np.einsum('qkd,qckd->qck', a[:, 0], b[:, :, 1]), np.einsum('qkd,qckd->qck', a[:, 1], b[:, :, 0])], -1)
        dec = np.exp(-2.0 * (t[:, None] - self.t0[None, :]) / self.taus[None, :])
        return (f.astype(np.float64) * dec[:, None, :, None]).reshape(Q, C, 4 * self.K)


class Stream:
    """Per-query features in time order: exact static slot reads plus the sketch. Before the queries of a timestamp are
    read, the sketch receives every event of an earlier timestamp and none of that timestamp (strict causality, ties
    excluded); events of the current timestamp stay pending until a later timestamp is read."""
    def __init__(self, mem, sketch, src, dst, ts):
        self.mem, self.sk, self.src, self.dst, self.ts = mem, sketch, src, dst, ts; self.pos = 0

    def reset(self):
        if self.sk is not None:
            self.sk.reset()
        self.pos = 0

    def before(self, t):
        """Stream position of the first event with time >= t."""
        return int(np.searchsorted(self.ts, t, 'left'))

    def advance(self, upto):
        """Apply every event at stream positions [pos, upto) to the sketch."""
        if self.sk is not None and upto > self.pos:
            for b in range(self.pos, upto, 1 << 18):
                e = min(upto, b + (1 << 18))
                self.sk.update(self.src[b:e], self.dst[b:e], self.ts[b:e].astype(np.float64))
        self.pos = max(self.pos, upto)

    def features(self, j, cand):
        """Features of the queries at stream positions j (time ordered)."""
        s, t = self.src[j], self.ts[j]
        base = self.mem.features(s, cand, t)
        if self.sk is None:
            return base
        sk = np.empty(cand.shape + (self.sk.nf(),), np.float32)
        cut = np.flatnonzero(np.diff(t)) + 1; bounds = np.concatenate([[0], cut, [len(j)]])
        for a, b in zip(bounds[:-1], bounds[1:]):
            first = self.before(t[a])
            assert self.pos <= first, 'an event at or after the query time reached the sketch'
            self.advance(first)
            sk[a:b] = slog(self.sk.read(s[a:b], cand[a:b], t[a:b].astype(np.float64)))
        return np.concatenate([base, sk], -1)


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


def amount_code(w):
    """Signed log amount: sign(w) log1p(|w|), non-finite values coded 0."""
    w = np.nan_to_num(np.asarray(w, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return slog(w).astype(np.float32)


def load(root, split):
    safe_tgb.patch_tgb()
    import tgb.linkproppred.dataset as dsm
    from tgb.linkproppred.evaluate import Evaluator
    assert dsm.load_pkl is safe_tgb.load_pkl
    ds = dsm.LinkPropPredDataset(name=DATASET, root=root, preprocess=True, download=False)
    d = ds.full_data
    amt = amount_code(d['w'])
    for k in ('edge_feat', 'w', 'edge_label', 'edge_idxs'):
        d.pop(k, None)
    ds._edge_feat = None; gc.collect(); trim()
    (ds.load_val_ns if split == 'val' else ds.load_test_ns)()
    src, dst, ts = (np.asarray(d[k], np.int64) for k in ('sources', 'destinations', 'timestamps'))
    assert np.all(np.diff(ts) >= 0), 'v2 edge list is chronological'
    n = int(max(src.max(), dst.max())) + 1
    negs = CompactNegatives(ds, split, np.unique(ts), n)
    return ds, Evaluator(name=DATASET), negs, src, dst, ts, n, amt


def evaluate(ev, model, stream, idx, split, negatives, src, dst, ts, scorer=None):
    """Scores the queries at stream positions idx (consecutive); the stream must stand at idx[0]."""
    out, cls = [], []; checked = 0; t0 = time.time(); mem = stream.mem
    if model is not None:
        model.eval()
    with torch.no_grad():
        for b in range(0, len(idx), 200):
            j = idx[b:b + 200]; s, c, t = src[j], dst[j], ts[j]
            neg = negatives(t, s, c).astype(np.int64)
            cand = np.concatenate([c[:, None], neg], 1); valid = cand >= 0; cc = np.where(valid, cand, 0)
            if scorer is None:
                sc = model(torch.from_numpy(stream.features(j, cc))).numpy().astype(np.float64)
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
    """Vectorised slot, lag, renewal and source-event reads equal direct sums over strictly earlier events; the sketch
    with one-hot codes equals the exact decayed two-hop sums; the streamed sketch equals a replay; competitors obey the
    race rules."""
    rng = np.random.default_rng(0); E, n = 6000, 25
    ts = np.sort(rng.integers(0, 40 * 24, E)).astype(np.int64) * 3600
    src, dst = rng.integers(0, n, E), rng.integers(0, n, E)
    mem = Memory(src, dst, ts, n)
    s, c = rng.integers(0, n, 300), rng.integers(0, n, (300, 6))
    t = np.sort(rng.choice(ts, 300)); t[:5] = ts[100]
    feat = mem.features(s, c, t); worst = 0.0; k = len(TAUS)
    for q in range(len(s)):
        for m_ in range(c.shape[1]):
            m = (src == s[q]) & (dst == c[q, m_]) & (ts < t[q])
            ref = np.log1p(np.exp(-(t[q] - ts[m])[:, None] / TAUS).sum(0)) if m.any() else np.zeros(k)
            worst = max(worst, float(np.abs(feat[q, m_, :k] - ref).max()))
            assert feat[q, m_, k] == np.float32(np.log1p(m.sum()))
            for li, L in enumerate(LAGS):
                lo, hi = t[q] - L * DAY - LAG_HALF, min(t[q] - L * DAY + LAG_HALF, t[q] - 1)
                lag = (m & (ts >= lo) & (ts <= hi)).sum()
                assert feat[q, m_, k + 3 + li] == np.float32(np.log1p(lag)), (q, m_, L)
            o = k + 3 + len(LAGS)
            if m.sum() > 1:
                gap = (ts[m].max() - ts[m].min()) / (m.sum() - 1)
                assert abs(feat[q, m_, o] - slog(gap)) < 1e-5
                assert abs(feat[q, m_, o + 1] - (np.log1p(t[q] - ts[m].max()) - np.log1p(gap))) < 1e-5
            since = ((src == s[q]) & (ts > ts[m].max()) & (ts < t[q])).sum() if m.any() else 0
            assert feat[q, m_, o + 2] == np.float32(np.log1p(since)), (q, m_)
            r = (src == c[q, m_]) & (dst == s[q]) & (ts < t[q])
            assert feat[q, m_, (o + 3) + k] == np.float32(np.log1p(r.sum()))
    assert worst < 1e-4, worst
    assert feat.shape[-1] == mem.nf()
    taus = (3600.0, DAY)
    sk = Sketch(n, n, taus, 0, codes=np.eye(n, dtype=np.float32))
    st = Stream(mem, sk, src, dst, ts); qs = rng.integers(0, n, (E, 4)); err = 0.0; nq = 0
    for b in range(0, E, 200):
        j = np.arange(b, min(E, b + 200)); f = st.features(j, qs[j])[..., mem.nf():]
        for q in rng.choice(len(j), 3, replace=False):
            tq = ts[j[q]]; pe = ts < tq; sq = src[j[q]]
            for kk, tau in enumerate(taus):
                w = np.where(pe, np.exp(-np.maximum(tq - ts, 0) / tau), 0.0)
                Wout = np.zeros((n, n)); np.add.at(Wout, (src, dst), w)
                for ci, cc in enumerate(qs[j[q]]):
                    ref = [Wout[sq] @ Wout[cc], Wout[:, sq] @ Wout[:, cc], Wout[sq] @ Wout[:, cc], Wout[:, sq] @ Wout[cc]]
                    got = f[q, ci, 4 * kk:4 * kk + 4]
                    d_ = float(np.max(np.abs(got - slog(np.array(ref))))); assert np.isfinite(d_); err = max(err, d_)
                    nq += 1
    assert err < 2e-3, err
    rs = Sketch(n, 16, taus, 1); st1 = Stream(mem, rs, src, dst, ts)
    for b in range(0, 3000, 200):
        st1.features(np.arange(b, b + 200), qs[b:b + 200])
    st1.advance(3000)
    rep = Sketch(n, 16, taus, 1); Stream(mem, rep, src, dst, ts).advance(3000)
    assert np.array_equal(rs.state, rep.state) and np.array_equal(rs.t0, rep.t0), 'streamed sketch != replay'
    M = 3; mv = Memory(src, dst, ts, n, paths=M, transitions=True); fv = mv.features(s, c, t); b0 = mem.nf()
    assert fv.shape[-1] == mv.nf() and np.array_equal(fv[..., :b0], feat), 'v2 with new reads keeps the v1 columns'
    pos = np.arange(E)

    def newest(mask, M_):
        ix = pos[mask]; return ix[np.lexsort((ix, ts[ix]))][::-1][:M_]

    def prev_rec(e):
        ix = newest((src == src[e]) & (ts < ts[e]), 1); return int(dst[ix[0]]) if len(ix) else -1
    prv = np.array([prev_rec(e) for e in range(E)])
    npath = ntr = 0
    for q in range(0, len(s), 7):
        tq, sq = t[q], s[q]
        for m_ in range(c.shape[1]):
            cq = c[q, m_]; got = fv[q, m_]; o = b0
            sides = [(dst[newest((src == cq) & (ts < tq), M)], lambda x: (src == x) & (dst == sq)),
                     (src[newest((dst == cq) & (ts < tq), M)], lambda y: (src == sq) & (dst == y)),
                     (dst[newest((src == sq) & (ts < tq), M)], lambda x: (src == x) & (dst == cq)),
                     (src[newest((dst == sq) & (ts < tq), M)], lambda y: (src == cq) & (dst == y))]
            for mids, close in sides:
                clk = np.zeros(k); tot = 0; mindt = NO_EVENT_DT
                for x in mids:
                    mm = close(x) & (ts < tq)
                    if mm.any():
                        clk = np.maximum(clk, np.exp(-(tq - ts[mm])[:, None] / TAUS).sum(0)); tot += mm.sum()
                        mindt = min(mindt, float(tq - ts[mm].max()))
                ref = np.concatenate([np.log1p(clk), [np.log1p(tot), slog(mindt), float(tot > 0)]])
                assert np.abs(got[o:o + k + 3] - ref).max() < 1e-4, (q, m_, got[o:o + k + 3], ref); o += k + 3; npath += 1
            ln = newest((src == sq) & (ts < tq), 1); L = int(dst[ln[0]]) if len(ln) else -1
            for mm in (((src == sq) & (dst == cq) & (prv == L) & (ts < tq)), ((dst == cq) & (prv == L) & (ts < tq))):
                mm = mm & (L >= 0)
                ref = np.log1p(np.exp(-(tq - ts[mm])[:, None] / TAUS).sum(0)) if mm.any() else np.zeros(k)
                assert np.abs(got[o:o + k] - ref).max() < 1e-4 and got[o + k] == np.float32(np.log1p(mm.sum())), (q, m_)
                o += k + 2; ntr += 1
            assert got[o] == float(cq == L and L >= 0)
    code = amount_code(np.array([-5.0, -1.0, -0.5, 0.0, 2.0, np.nan, np.inf]))
    assert np.all(np.isfinite(code)) and code[1] == np.float32(-np.log(2)) and code[3] == 0, code
    amt = amount_code(rng.normal(0, 50, E))
    m3 = Memory(src, dst, ts, n, paths=M, transitions=True, flow=True, amt=amt, novelty=True); f3 = m3.features(s, c, t)
    assert f3.shape[-1] == m3.nf() and np.array_equal(f3[..., :mv.nf()], fv), 'v3 keeps the v2 columns'

    def prev_payer(e):
        ix = newest((dst == src[e]) & (ts < ts[e]), 1); return int(src[ix[0]]) if len(ix) else -1
    pyr = np.array([prev_payer(e) for e in range(E)]); nflow = namt = nnov = 0
    for q in range(0, len(s), 7):
        tq, sq = t[q], s[q]
        yn = newest((dst == sq) & (ts < tq), 1); Y = int(src[yn[0]]) if len(yn) else -1
        on = newest((src == sq) & (ts < tq), 1)
        for m_ in range(c.shape[1]):
            cq = c[q, m_]; got = f3[q, m_]; o = mv.nf()
            for mm in (((src == sq) & (dst == cq) & (pyr == Y) & (ts < tq)), ((dst == cq) & (pyr == Y) & (ts < tq))):
                mm = mm & (Y >= 0)
                ref = np.log1p(np.exp(-(tq - ts[mm])[:, None] / TAUS).sum(0)) if mm.any() else np.zeros(k)
                assert np.abs(got[o:o + k] - ref).max() < 1e-4 and got[o + k] == np.float32(np.log1p(mm.sum())), (q, m_)
                o += k + 2; nflow += 1
            pn = newest((src == sq) & (dst == cq) & (ts < tq), 1)
            la = amt[pn[0]] if len(pn) else 0.0; ia = amt[yn[0]] if len(yn) else 0.0; oa = amt[on[0]] if len(on) else 0.0
            ref = [la, float(len(pn) > 0), ia, oa, -abs(la - ia) if (len(pn) and len(yn)) else 0.0]
            assert np.allclose(got[o:o + 5], ref, atol=1e-6), (q, m_, got[o:o + 5], ref); namt += 1; o += 5
            for side, ent in (((src == sq) & (ts < tq), 'src'), ((dst == cq) & (ts < tq), 'dst')):
                other = dst if ent == 'src' else src
                firsts = {}
                for e in np.flatnonzero(side):
                    firsts.setdefault(int(other[e]), ts[e])
                nd = len(firsts); ev_ = side.sum()
                ref = [np.log1p(nd), np.log1p(nd) - np.log1p(ev_), slog(tq - max(firsts.values())) if nd else slog(NO_EVENT_DT)]
                assert np.allclose(got[o:o + 3], ref, atol=1e-5), (q, m_, ent, got[o:o + 3], ref); o += 3; nnov += 1
    big = Memory(src, dst, ts, 40 * n)
    cs = mem.competitors(s, c[:, 0], t, 20, rng); cb = big.competitors(s, c[:, 0], t, 20, rng)
    assert not (cs == c[:, :1]).any() and not (cb == c[:, :1]).any(), 'the positive is never a competitor'
    hist_ok, hist_n, rnd_seen = 0, 0, 0
    for q in range(len(s)):
        hist = set(dst[(src == s[q]) & (ts < t[q])].tolist()) - {int(c[q, 0])}
        kq = min(10, len(hist)); hist_n += kq
        hist_ok += sum(int(x) in hist for x in cb[q, :kq])
        rnd_seen += sum(int(x) in hist for x in cb[q, kq:])
    assert hist_ok == hist_n and rnd_seen == 0, (hist_ok, hist_n, rnd_seen)
    return dict(max_abs_clock_error=worst, queries=int(len(s) * c.shape[1]), ties_excluded=True, lag_reads_exact=True,
                renewal_and_source_event_clock_exact=True, sketch_onehot_max_abs_error=err, sketch_reads_checked=nq,
                streamed_sketch_equals_replay=True, path_reads_checked=npath, transition_reads_checked=ntr,
                v1_columns_unchanged=True, flow_reads_checked=nflow, amount_reads_checked=namt, novelty_reads_checked=nnov, v2_columns_unchanged=True,
                positive_never_competitor=True,
                history_competitors_from_prior_history=hist_n, random_competitors_outside_history=True)


def diagnose(a, start):
    """Scorer proof and query anatomy on VALIDATION (no fit)."""
    ds, ev, negs, src, dst, ts, n, _ = load(a.root, 'val')
    tr, va = np.flatnonzero(ds.train_mask), np.flatnonzero(ds.val_mask)
    assert tr[0] == 0 and va[0] == tr[-1] + 1 and np.all(np.diff(va) == 1)
    t_load = time.time() - start
    E = va[-1] + 1
    mem = Memory(src[:E], dst[:E], ts[:E], n); t_mem = time.time() - start - t_load
    pos = np.arange(E, dtype=np.int64)
    bank = Slots(src[:E] * n + dst[:E], pos, ts[:E], int(np.ceil(np.log2(E + 2))), clocks=False)
    tmin, tmax = float(ts[tr].min()), float(ts[tr].max()); dur = (tmax - tmin) * 0.15
    if a.max_val:
        va = va[:a.max_val]
    sk = Sketch(n, a.sketch_d or 64, (DAY,), a.seed); stream = Stream(mem, sk, src, dst, ts)
    T = lambda t, cc: np.broadcast_to(t[:, None], cc.shape).astype(np.float64)
    R = lambda t, cc: np.broadcast_to(mem.rank(t)[:, None], cc.shape)

    def edgebank(window):
        def score(s, cc, t, j):
            b0 = (j[0] - va[0]) // 200 * 200 + va[0]
            qr = np.full(cc.shape, b0, np.int64); key = s[:, None] * n + cc
            cnt, _, _, _ = bank.read(key, qr, T(t, cc))
            if not window:
                return (cnt > 0).astype(np.float64)
            i, found = bank.slot(key); hi = np.searchsorted(bank.comp, (i.astype(np.int64) << bank.nbits) | qr)
            last = bank.t[np.maximum(hi - 1, 0)]; cur = max(tmax, float(ts[b0 - 1]))
            return ((cnt > 0) & (last >= cur - dur)).astype(np.float64)
        return score

    def pair(s, cc, t, back=False):
        key = (cc * n + s[:, None]) if back else (s[:, None] * n + cc)
        return mem.fam['pair'].read(key, R(t, cc), T(t, cc))

    def pop(cc, t, k=2):
        return mem.fam['dst'].read(cc, R(t, cc), T(t, cc))[2][..., k]

    def recency_pop(s, cc, t, j):
        cnt, dt, _, _ = pair(s, cc, t)
        return np.where(cnt > 0, 1e6 - dt / DAY, 0.0) + 1e-3 * np.log1p(pop(cc, t))

    def lrlp(s, cc, t, j):
        cnt, dt, _, _ = pair(s, cc, t)
        return np.where(cnt > 0, 1e6 - dt / DAY + 1e-6 * np.log1p(cnt), 0.0)

    def freq_rec(s, cc, t, j):
        cnt, dt, _, _ = pair(s, cc, t)
        return np.where(cnt > 0, 1e3 * np.log1p(cnt) - dt / (DAY * 1e3), 0.0) + 1e-6 * np.log1p(pop(cc, t))

    def clock1d(s, cc, t, j):
        _, _, clk, _ = pair(s, cc, t)
        return 1e3 * clk[..., 2] + 1e-3 * np.log1p(pop(cc, t))

    def rec_ret_pop(s, cc, t, j):
        cnt, dt, _, _ = pair(s, cc, t); rc, rdt, _, _ = pair(s, cc, t, back=True)
        return (np.where(cnt > 0, 2e6 - dt / DAY, np.where(rc > 0, 1e6 - rdt / DAY, 0.0)) + 1e-3 * np.log1p(pop(cc, t)))

    def rec_ret_2hop_pop(s, cc, t, j):
        base = rec_ret_pop(s, cc, t, j)
        f = stream.features(j, cc)[..., mem.nf():]
        two = np.log1p(np.maximum(f[..., 0], 0) + np.maximum(f[..., 1], 0) + np.maximum(f[..., 2], 0) + np.maximum(f[..., 3], 0))
        return np.where(base >= 5e5, base, 1e3 * two + 1e-3 * np.log1p(pop(cc, t)))

    res = dict(load_wall_s=t_load, memory_build_wall_s=t_mem, memory_vm=vm(), nodes=n, events=int(len(src)),
               train=int(len(tr)), val=int(np.sum(ds.val_mask)), test=int(np.sum(ds.test_mask)), scored_val=int(len(va)),
               negatives_per_query=int(negs.K), negatives_equal_to_query_batch=negs.checked,
               unique_timestamps_train_val=int(len(mem.ut)), span_days=float((ts[-1] - ts[0]) / DAY),
               val_start_utc=time.strftime('%Y-%m-%d %H:%M', time.gmtime(float(ts[va[0]]))))
    scorers = (('edgebank_unlimited_bs200', edgebank(False)), ('edgebank_tw015_bs200', edgebank(True)),
               ('recency_then_popularity_strict', recency_pop), ('local_recency_local_popularity_strict', lrlp),
               ('frequency_then_recency_strict', freq_rec), ('pair_clock_1day_strict', clock1d),
               ('recency_return_popularity_strict', rec_ret_pop))
    for name, sc in scorers:
        mrr, nq, w, prof = evaluate(ev, None, stream, va, 'val', negs, src, dst, ts, scorer=sc)
        res[name] = dict(val_mrr=mrr, queries=nq, wall_s=w, query_profile=prof); print(json.dumps({name: res[name]}), flush=True)
    stream.reset(); t0 = time.time(); stream.advance(int(va[0])); res['sketch_replay_train_wall_s'] = time.time() - t0
    mrr, nq, w, prof = evaluate(ev, None, stream, va, 'val', negs, src, dst, ts, scorer=rec_ret_2hop_pop)
    res['recency_return_2hop_popularity_strict'] = dict(val_mrr=mrr, queries=nq, wall_s=w, query_profile=prof, sketch_d=sk.d)
    print(json.dumps({'recency_return_2hop_popularity_strict': res['recency_return_2hop_popularity_strict']}), flush=True)
    j = va[:min(len(va), 200000)]; neg = negs(ts[j], src[j], dst[j]).astype(np.int64)
    hist = mem.fam['pair'].count(src[j][:, None] * n + np.maximum(neg, 0), np.broadcast_to(mem.rank(ts[j])[:, None], neg.shape)) > 0
    res['negatives_seen_pair_share_first_200k'] = float(hist[neg >= 0].mean())
    res['negatives_valid_per_query_first_200k'] = float((neg >= 0).sum(1).mean())
    res['leaderboard_validation'] = dict(TPNet=0.796, HyperEvent=0.750, LocalRecencyLocalPopularity=0.743, TNCN=0.740,
                                         DyGFormer=0.730, EdgeBank_tw=0.492, EdgeBank_unlimited=0.315)
    res['diagnosis_vm'] = vm()
    return res


def score_sealed(a, ckpt, split):
    t0 = time.time(); ds, ev, negs, src, dst, ts, n, amt = load(a.root, split)
    mask = ds.test_mask if split == 'test' else ds.val_mask
    idx = np.flatnonzero(mask); upto = int(idx[-1]) + 1
    if split == 'val' and a.max_val:
        idx = idx[:a.max_val]
    assert split != 'test' or upto == len(src)
    mem = Memory(src[:upto], dst[:upto], ts[:upto], n, a.paths, a.transitions, a.flow, amt[:upto] if a.amounts else None,
                 a.novelty)
    sk = Sketch(n, a.sketch_d, a.sketch_taus, a.seed) if a.sketch_d else None
    stream = Stream(mem, sk, src, dst, ts); stream.advance(int(idx[0]))
    model = Model(mem.nf() + (sk.nf() if sk else 0), a.hidden); model.load_state_dict(torch.load(ckpt))
    mrr, nq, wall, prof = evaluate(ev, model, stream, idx, split, negs, src, dst, ts)
    p = 'test' if split == 'test' else 'sealed_path_smoke_val'
    return {f'{p}_mrr': mrr, f'{p}_queries': nq, f'{p}_query_profile': prof, f'{p}_wall_s': time.time() - t0,
            f'{p}_eval_wall_s': wall, f'{p}_vm': vm(), f'{p}_negatives_equal_to_query_batch': negs.checked,
            f'{p}_protocol': 'best VALIDATION epoch checkpoint; fresh load; slots rebuilt over every event up to the end '
                             f'of {split.upper()} and read strictly causally (time < t_query); sketch replayed over every '
                             f'event before {split.upper()} and streamed through it; official {split.upper()} '
                             'negatives and Evaluator (agreement on the first 50 queries); scored once'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--root', default='../../../data/tgb/root')
    ap.add_argument('--contract', action='store_true'); ap.add_argument('--diagnose', action='store_true')
    for k, v in (('seed', 0), ('epochs', 3), ('neg', 20), ('hidden', 64), ('train_last', 0), ('max_val', 0),
                 ('sketch_d', 0), ('paths', 0)):
        ap.add_argument('--' + k.replace('_', '-'), type=int, default=v)
    ap.add_argument('--sketch-taus', type=float, nargs='+', default=[3600.0, DAY, 30 * DAY])
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--transitions', action='store_true'); ap.add_argument('--flow', action='store_true')
    ap.add_argument('--amounts', action='store_true'); ap.add_argument('--novelty', action='store_true')
    ap.add_argument('--score-test', action='store_true'); ap.add_argument('--sealed-path-smoke', action='store_true')
    a = ap.parse_args(); start = time.time(); torch.set_num_threads(1)
    assert not (a.score_test and (a.sealed_path_smoke or a.max_val)), 'sealed runs score the full TEST'
    out = ROOT / 'experiments/results/tgb' / f'{a.tag}.json'; assert not out.exists(), 'fresh tag'
    if a.contract:
        res = dict(status='completed', checks=contracts(),
                   scope='vectorised slot, lag, renewal and source-event reads vs direct sums; one-hot sketch vs exact '
                         'decayed two-hop sums; streamed sketch vs replay; v2 two-hop path reads and per-source / global '
                         'transition slots vs brute force; v3 flow slots, amount and novelty reads vs brute force; competitor rules; no data, no fit')
    elif a.diagnose:
        res = dict(status='completed', diagnosis=diagnose(a, start))
    else:
        torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
        ds, ev, negs, src, dst, ts, n, amt = load(a.root, 'val')
        tr, va = np.flatnonzero(ds.train_mask), np.flatnonzero(ds.val_mask)
        assert tr[0] == 0 and va[0] == tr[-1] + 1 and np.all(np.diff(va) == 1)
        t_load = time.time() - start
        E = va[-1] + 1
        mem = Memory(src[:E], dst[:E], ts[:E], n, a.paths, a.transitions, a.flow, amt[:E] if a.amounts else None, a.novelty)
        sk = Sketch(n, a.sketch_d, a.sketch_taus, a.seed) if a.sketch_d else None
        stream = Stream(mem, sk, src, dst, ts)
        t_mem = time.time() - start - t_load; mem_vm = vm()
        nf = mem.nf() + (sk.nf() if sk else 0)
        model = Model(nf, a.hidden)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        ckpt = out.with_suffix('.pt'); best, best_ep, history = -1.0, -1, []
        lim = tr[-a.train_last:] if a.train_last else tr
        vq = va[:a.max_val] if a.max_val else va
        for ep in range(a.epochs):
            model.train(); t0 = time.time(); total = 0.0; nb = 0
            stream.reset(); stream.advance(stream.before(ts[lim[0]]))
            for b in range(0, len(lim), 200):
                j = lim[b:b + 200]; s, c, t = src[j], dst[j], ts[j]
                cand = np.concatenate([c[:, None], mem.competitors(s, c, t, a.neg, rng)], 1)
                sc = model(torch.from_numpy(stream.features(j, cand)))
                loss = F.cross_entropy(sc, torch.zeros(len(s), dtype=torch.long)); assert torch.isfinite(loss)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                total += float(loss.detach()); nb += 1
            tw = time.time() - t0
            stream.advance(int(va[0]))
            mrr, nq, vw, prof = evaluate(ev, model, stream, vq, 'val', negs, src, dst, ts)
            row = dict(epoch=ep, train_loss=total / max(nb, 1), val_mrr=mrr, val_queries=nq, train_wall_s=tw,
                       val_wall_s=vw, query_profile=prof, vm=vm())
            history.append(row); print(json.dumps(row), flush=True)
            if mrr > best:
                best, best_ep = mrr, ep; torch.save(model.state_dict(), ckpt)
        res = dict(status='completed', args=vars(a), dataset=DATASET, nodes=n, nf=nf, nf_slots=mem.nf(),
                   nf_sketch=sk.nf() if sk else 0,
                   parameters=sum(p.numel() for p in model.parameters()), val_mrr=best, best_epoch=best_ep,
                   history=history, checkpoint=str(ckpt.relative_to(ROOT)), trained_queries=len(lim) * a.epochs,
                   load_wall_s=t_load, memory_build_wall_s=t_mem, memory_vm=mem_vm,
                   negatives_equal_to_query_batch=negs.checked,
                   protocol='Per-query features from events strictly earlier than the query timestamp (ties excluded); '
                            'hist_rnd-shaped training race; official hist_rnd validation negatives and Evaluator; no TEST')
        if a.score_test or a.sealed_path_smoke:
            del mem, ds, negs, src, dst, ts, stream, sk, amt; gc.collect(); trim()
            res.update(score_sealed(a, ckpt, 'test' if a.score_test else 'val'))
            if a.score_test:
                res['protocol'] = res['protocol'].replace('; no TEST', '; TEST scored once with the best VALIDATION checkpoint')
    res.update(tag=a.tag, benchmark='TGB tgbl-coin-v2', driver='race_link_coin_v4.py', score_test=a.score_test,
               wall_s=time.time() - start, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__).resolve(), HERE / 'safe_tgb.py')})
    with out.open('x') as f:
        json.dump(res, f, indent=1); f.write('\n')
    print(json.dumps({k: v for k, v in res.items() if k not in ('history', 'args')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
