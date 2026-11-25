#!/usr/bin/env python3
"""Exact sparse stand-in for py-tgb's `load_label_dict` (tgbn-reddit, tgbn-token), 2 Nov 2026.

The official loader re-parses `<name>_node_labels.csv` on every load into {ts: {node: np.zeros(C) float64}}: 4.4M
(reddit) / 3.3M (token) dense vectors, 25-26 GB, the dominant memory of a run (VmHWM 27.9 GB for the reddit load).
Here the same rows, parsed by the same expressions (int(ts), node_ids[user], int(rd_dict[class]), float(weight); a
repeated (ts, node, class) row overwrites, as there) fill the same nested dicts in the same insertion order, but each
label is a `SparseLabel` that densifies on np.array() to the float64 vector the official dict holds. The official
`find_next_labels_batch` builds its batch with np.array(label_dict[ts][node]), so the loop receives identical arrays.
`patch()` installs it in tgb.nodeproppred.dataset (only the reddit/token branches call load_label_dict).

  python tgb_sparse_labels.py --dataset tgbn-reddit --out <json>
      loads the official dense dict and this one and compares every label time: key order, node order and the
      densified vectors (np.array_equal, dtype float64). Only equality counts are written; no label statistic.
"""
import argparse, csv, json, resource, sys, time
from pathlib import Path
import numpy as np


class SparseLabel:
    __slots__ = ('idx', 'val', 'n')

    def __init__(self, idx, val, n):
        self.idx, self.val, self.n = idx, val, n

    def __array__(self, dtype=None, copy=None):
        y = np.zeros(self.n)
        y[self.idx] = self.val
        return y if dtype is None else y.astype(dtype)

    def __len__(self):
        return self.n


def load_label_dict(fname, node_ids, rd_dict):
    label_size = len(rd_dict); out = {}
    with open(fname, 'r') as fh:
        r = csv.reader(fh, delimiter=','); next(r)
        for row in r:
            u = node_ids[row[1]]; ts = int(row[0]); v = int(rd_dict[row[2]]); weight = float(row[3])
            d = out.get(ts)
            if d is None:
                d = out[ts] = {}
            e = d.get(u)
            if e is None:
                e = d[u] = {}
            e[v] = weight
    for d in out.values():
        for u, e in d.items():
            d[u] = SparseLabel(np.fromiter(e.keys(), np.int64, len(e)), np.fromiter(e.values(), np.float64, len(e)), label_size)
    return out


def patch():
    import tgb.nodeproppred.dataset as dsmod
    dsmod.load_label_dict = load_label_dict


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / '.cache/pylib_tgb')); sys.path.insert(0, str(Path(__file__).resolve().parent))
    import safe_tgb
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); t0 = time.time()
    safe_tgb.patch_tgb()
    from tgb.nodeproppred.dataset import NodePropPredDataset
    import tgb.utils.utils as tu
    ds = NodePropPredDataset(name=a.dataset, root='../../../data/tgb/root', preprocess=True)
    ddir = Path(ds.root); dense = ds.label_dict
    node_ids = tu.load_pkl(str(ddir / f'ml_{a.dataset}_node.pkl')); rd = tu.load_pkl(str(ddir / f'ml_{a.dataset}_label.pkl'))
    t1 = time.time(); sp = load_label_dict(ds.meta_dict['nodefile'], node_ids, rd); sparse_s = time.time() - t1
    ts_order = list(dense.keys()) == list(sp.keys())
    n_ts = n_lab = bad_order = bad_vec = 0
    for ts in dense:
        n_ts += 1; dk = list(dense[ts].keys())
        if dk != list(sp[ts].keys()):
            bad_order += 1
        for u in dk:
            n_lab += 1; y = np.array(sp[ts][u])
            bad_vec += not (y.dtype == np.float64 and np.array_equal(y, np.asarray(dense[ts][u])))
    res = dict(schema='sparse_labels_equivalence_v1', dataset=a.dataset, label_times=n_ts, labels=n_lab,
               label_time_order_equal=ts_order, node_order_mismatches=bad_order, vector_mismatches=bad_vec,
               equal=bool(ts_order and bad_order == 0 and bad_vec == 0), sparse_parse_s=sparse_s, wall_s=time.time() - t0,
               vmhwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               note='official NodePropPredDataset dense label dict vs tgb_sparse_labels.load_label_dict on every label time; '
                    'only equality counts are recorded (no label statistic).')
    (root / a.out).write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
