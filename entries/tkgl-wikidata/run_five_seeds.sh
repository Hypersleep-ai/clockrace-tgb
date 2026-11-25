#!/bin/sh
# tkgl-wikidata: the five seeds of the submitted entry (MRR). Run from anywhere; results are written to
# src/experiments/results/tgb/<tag>.json. py-tgb prepends its own package directory to the dataset root, so the root is
# passed relative to that directory; the default places every dataset under src/data/tgb/root (the directory name is
# fixed by the tkgl-wikidata driver, and all entries share it).
set -e
cd "$(dirname "$0")/../../src"
export PYTHONPATH="$PWD/.cache/pylib_tgb${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TGB_ROOT="${TGB_ROOT:-$(python -c "import os, tgb.utils.info as i; print(os.path.relpath('$PWD/data/tgb/root', i.PROJ_DIR))")}"
mkdir -p experiments/results/tgb

python experiments/tgb/tkg_ns_stream.py --split val   # validation negatives into the memory-mapped store (once)
python experiments/tgb/race_tkg_wd_v2.py --tag tkgl-wikidata_s0 --hidden 64 --k 32 --train-from 1950 --gpu-mem-frac 0.12 --score-test --lr-decay cos
python experiments/tgb/race_tkg_wd_v2.py --tag tkgl-wikidata_s1 --seed 1 --hidden 64 --k 32 --train-from 1950 --gpu-mem-frac 0.12 --score-test --lr-decay cos
python experiments/tgb/race_tkg_wd_v2.py --tag tkgl-wikidata_s2 --seed 2 --hidden 64 --k 32 --train-from 1950 --gpu-mem-frac 0.12 --score-test --lr-decay cos
python experiments/tgb/race_tkg_wd_v2.py --tag tkgl-wikidata_s3 --seed 3 --hidden 64 --k 32 --train-from 1950 --gpu-mem-frac 0.12 --score-test --lr-decay cos
python experiments/tgb/race_tkg_wd_v2.py --tag tkgl-wikidata_s4 --seed 4 --hidden 64 --k 32 --train-from 1950 --gpu-mem-frac 0.12 --score-test --lr-decay cos
