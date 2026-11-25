#!/usr/bin/env python3
"""TGB 2.0 TKG negatives without the dict: stream an official `<name>_<split>_ns.pkl` into a memory-mapped store.

The official files are pickles of a dict {(t, s, r): numpy int array} (py-tgb 2.3.0 TKGNegativeEdgeGenerator; for
tkgl-wikidata strategy 'dst-time-filtered', 1,000 sampled negatives per (t, s, r) key). Loading them with py-tgb's
load_pkl (pickle.load) holds every array plus the unpickler memo (every in-band buffer is memoized): about 24 GB of
arrays per tkgl-wikidata split before dict overhead.

Here a pure-Python unpickler (pickle._Unpickler, the reference implementation of the same format) reads the stream once:
  * globals are restricted to the allow-list of safe_tgb (numpy dtype, _frombuffer, scalar); anything else raises;
  * each array reconstruction (_frombuffer) is written straight to a flat int32 file (values asserted to fit int32, so
    the copy is exact) and replaced by its integer token;
  * numpy scalar keys become Python ints;
  * memo entries of large buffers and of argument tuples holding them are not retained (each in-band buffer
    is referenced once by the pickle).
The dict that remains maps (t, s, r) -> token; it is converted to sorted int64 arrays (t, s, r, offset, length) and freed.
Reads are per batch from np.memmap. Nothing here reads a split the caller does not name; TEST is converted only by the
sealed path.

Store layout (directory <data dir>/ns_store/<split>/): negs.i32 (flat int32), index.npz (t, s, r, off, len in pickle
order), receipt.json (source size/sha256, keys, objects, dtype, value range, wall, peak RSS).
"""
import hashlib, io, json, pickle, resource, time
from pathlib import Path
import numpy as np


def _sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


class _Sink:
    def __init__(self, fh):
        self.fh, self.lens, self.dtypes, self.vmin, self.vmax = fh, [], set(), None, None

    def add(self, a):
        a = np.asarray(a).ravel()
        self.dtypes.add(str(a.dtype))
        if len(a):
            lo, hi = int(a.min()), int(a.max())
            assert lo >= 0 and hi < 2 ** 31, 'negative ids do not fit int32'
            self.vmin = lo if self.vmin is None else min(self.vmin, lo)
            self.vmax = hi if self.vmax is None else max(self.vmax, hi)
        self.fh.write(a.astype(np.int32).tobytes())
        self.lens.append(len(a))
        return len(self.lens) - 1


class _StreamUnpickler(pickle._Unpickler):
    BIG = 256

    def __init__(self, fh, sink):
        super().__init__(fh)
        self.sink = sink
        d = dict(pickle._Unpickler.dispatch)
        d[pickle.MEMOIZE[0]] = _StreamUnpickler._memoize
        d[pickle.BINPUT[0]] = _StreamUnpickler._binput
        d[pickle.LONG_BINPUT[0]] = _StreamUnpickler._long_binput
        self.dispatch = d

    def _big(self, obj):
        return isinstance(obj, (bytes, bytearray, memoryview)) and len(obj) > self.BIG

    def _keep(self, obj):
        """Memo entry: large buffers, and argument tuples holding one (the (buf, dtype, shape, order) of _frombuffer),
        are referenced once by the stream and are not retained."""
        if self._big(obj) or (isinstance(obj, tuple) and any(self._big(x) for x in obj)):
            return None
        return obj

    def _memoize(self):
        self.memo[len(self.memo)] = self._keep(self.stack[-1])

    def _binput(self):
        i = self.read(1)[0]; self.memo[i] = self._keep(self.stack[-1])

    def _long_binput(self):
        i, = __import__('struct').unpack('<I', self.read(4)); self.memo[i] = self._keep(self.stack[-1])

    def find_class(self, module, name):
        sink = self.sink
        if (module, name) == ('numpy', 'dtype'):
            return np.dtype
        if module in ('numpy.core.numeric', 'numpy._core.numeric') and name == '_frombuffer':
            def _frombuffer(buf, dtype, shape, order='C'):
                return sink.add(np.frombuffer(buf, dtype=dtype).reshape(shape, order=order))
            return _frombuffer
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name == 'scalar':
            def _scalar(dtype, data=None):
                return np._core.multiarray.scalar(dtype, data).item()
            return _scalar
        raise pickle.UnpicklingError(f'blocked global {module}.{name}')


def convert(pkl, out_dir):
    """Stream `pkl` into out_dir (negs.i32, index.npz, receipt.json). Returns the receipt."""
    pkl, out_dir = Path(pkl), Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); c0 = time.process_time()
    tmp = out_dir / 'negs.i32.partial'
    with open(pkl, 'rb', buffering=1 << 24) as fh, open(tmp, 'wb', buffering=1 << 24) as oh:
        sink = _Sink(oh); d = _StreamUnpickler(fh, sink).load()
        trailing = fh.read(1)
    assert trailing == b'', 'bytes after STOP'
    assert isinstance(d, dict)
    K = len(d); tok = np.fromiter(d.values(), np.int64, K)
    keys = np.array(list(d.keys()), dtype=np.int64).reshape(K, 3); del d
    assert np.array_equal(np.sort(tok), np.arange(len(sink.lens))), 'every array is one dict value'
    lens = np.asarray(sink.lens, np.int64); off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    T, S, R = np.empty(K, np.int64), np.empty(K, np.int64), np.empty(K, np.int64)
    T[tok], S[tok], R[tok] = keys[:, 0], keys[:, 1], keys[:, 2]
    tmp.rename(out_dir / 'negs.i32')
    np.savez(out_dir / 'index.npz', t=T, s=S, r=R, off=off, len=lens)
    rec = dict(source=str(pkl.name), source_bytes=pkl.stat().st_size, source_sha256=_sha256(pkl), keys=int(K),
               objects=int(lens.sum()), len_min=int(lens.min()), len_max=int(lens.max()), dtypes=sorted(sink.dtypes),
               value_min=sink.vmin, value_max=sink.vmax, store_bytes=int((out_dir / 'negs.i32').stat().st_size),
               wall_s=time.time() - t0, cpu_s=time.process_time() - c0,
               max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               method='pickle._Unpickler stream; allow-listed globals; arrays -> flat int32 file; large buffers not '
                      'memoized; dict of (t, s, r) -> token only')
    (out_dir / 'receipt.json').write_text(json.dumps(rec, indent=1) + '\n')
    return rec


class NegStore:
    """Memory-mapped official negatives of one split: rows(t, s, r) -> key index; negs(k) -> int32 [len(k), L]."""
    def __init__(self, out_dir, n, R2, ut):
        out_dir = Path(out_dir); z = np.load(out_dir / 'index.npz')
        self.t, self.s, self.r, self.off, self.len = (z[k] for k in ('t', 's', 'r', 'off', 'len'))
        self.receipt = json.loads((out_dir / 'receipt.json').read_text())
        self.flat = np.memmap(out_dir / 'negs.i32', dtype=np.int32, mode='r')
        self.n, self.R2, self.ut = n, R2, ut
        tix = np.searchsorted(ut, self.t); assert np.all(ut[np.minimum(tix, len(ut) - 1)] == self.t)
        code = (tix * n + self.s) * R2 + self.r; o = np.argsort(code)
        self.codes, self.order = code[o], o
        assert np.all(np.diff(self.codes) > 0), 'duplicate keys'
        self.L = int(self.len.max()); self.uniform = bool((self.len == self.L).all())

    def lookup(self, tix, s, r):
        c = (np.asarray(tix, np.int64) * self.n + s) * self.R2 + r
        i = np.minimum(np.searchsorted(self.codes, c), len(self.codes) - 1)
        assert np.all(self.codes[i] == c), 'query not in the official evaluation set'
        return self.order[i]

    def negs(self, k):
        """[len(k), L] int32 (padded with -1 when lengths differ) and the lengths."""
        k = np.asarray(k); L = self.len[k]
        if self.uniform:
            idx = self.off[k][:, None] + np.arange(self.L)[None, :]
            return np.asarray(self.flat[idx.ravel()]).reshape(len(k), self.L), L
        out = np.full((len(k), int(L.max()) if len(k) else 0), -1, np.int32)
        for j, (o, l) in enumerate(zip(self.off[k], L)):
            out[j, :l] = self.flat[o:o + l]
        return out, L


def store_dir(root, name, split):
    return Path(root) / name.replace('-', '_') / 'ns_store' / split


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='tkgl-wikidata'); ap.add_argument('--split', required=True, choices=('val', 'test'))
    ap.add_argument('--root', default=str(Path(__file__).resolve().parents[2] / 'data/tgb/root'))
    ap.add_argument('--sealed-test', action='store_true', help='required to convert TEST (sealed path only)')
    a = ap.parse_args()
    assert a.split == 'val' or a.sealed_test, 'TEST negatives are converted only by the sealed path'
    d = Path(a.root) / a.dataset.replace('-', '_'); out = store_dir(a.root, a.dataset, a.split)
    if (out / 'receipt.json').exists():
        print(json.dumps(json.loads((out / 'receipt.json').read_text()), indent=1)); return
    print(json.dumps(convert(d / f'{a.dataset}_{a.split}_ns.pkl', out), indent=1), flush=True)


if __name__ == '__main__':
    main()
