#!/bin/sh
# tgbn-token: the five seeds of the submitted entry (NDCG@10). Run from anywhere; results are written to
# src/experiments/results/tgb/<tag>.json. py-tgb prepends its own package directory to the dataset root, so the root is
# passed relative to that directory; the default places every dataset under src/data/tgb/root (the directory name is
# fixed by the tkgl-wikidata driver, and all entries share it).
set -e
cd "$(dirname "$0")/../../src"
export PYTHONPATH="$PWD/.cache/pylib_tgb${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TGB_ROOT="${TGB_ROOT:-$(python -c "import os, tgb.utils.info as i; print(os.path.relpath('$PWD/data/tgb/root', i.PROJ_DIR))")}"
mkdir -p experiments/results/tgb

python experiments/tgb/race_affinity_v2_large_v2.py --tag tgbn-token_s0 --dataset tgbn-token --epochs 4 --hidden 32 --lr 0.003 --edge-weight feat --chunk-users 1024 --seed 0 --score-test --root "$TGB_ROOT"
python experiments/tgb/race_affinity_v2_large_v2.py --tag tgbn-token_s1 --dataset tgbn-token --epochs 4 --hidden 32 --lr 0.003 --edge-weight feat --chunk-users 1024 --seed 1 --score-test --root "$TGB_ROOT"
python experiments/tgb/race_affinity_v2_large_v2.py --tag tgbn-token_s2 --dataset tgbn-token --epochs 4 --hidden 32 --lr 0.003 --edge-weight feat --chunk-users 1024 --seed 2 --score-test --root "$TGB_ROOT"
python experiments/tgb/race_affinity_v2_large_v2.py --tag tgbn-token_s3 --dataset tgbn-token --epochs 4 --hidden 32 --lr 0.003 --edge-weight feat --chunk-users 1024 --seed 3 --score-test --root "$TGB_ROOT"
python experiments/tgb/race_affinity_v2_large_v2.py --tag tgbn-token_s4 --dataset tgbn-token --epochs 4 --hidden 32 --lr 0.003 --edge-weight feat --chunk-users 1024 --seed 4 --score-test --root "$TGB_ROOT"
