#!/bin/sh
# tkgl-polecat: the five seeds of the submitted entry (MRR). Run from anywhere; results are written to
# src/experiments/results/tgb/<tag>.json. py-tgb prepends its own package directory to the dataset root, so the root is
# passed relative to that directory; the default places every dataset under src/data/tgb/root (the directory name is
# fixed by the tkgl-wikidata driver, and all entries share it).
set -e
cd "$(dirname "$0")/../../src"
export PYTHONPATH="$PWD/.cache/pylib_tgb${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TGB_ROOT="${TGB_ROOT:-$(python -c "import os, tgb.utils.info as i; print(os.path.relpath('$PWD/data/tgb/root', i.PROJ_DIR))")}"
mkdir -p experiments/results/tgb

python experiments/tgb/race_tkg_path_v1.py --tag tkgl-polecat_s0 --dataset tkgl-polecat --root "$TGB_ROOT" --seed 0 --epochs 2 --window-rec 365 --window-tr 30 --window-trans 30 --window-pop 30 --path-window 30 --hidden 64 --score-test
python experiments/tgb/race_tkg_path_v1.py --tag tkgl-polecat_s1 --dataset tkgl-polecat --root "$TGB_ROOT" --seed 1 --epochs 2 --window-rec 365 --window-tr 30 --window-trans 30 --window-pop 30 --path-window 30 --hidden 64 --score-test
python experiments/tgb/race_tkg_path_v1.py --tag tkgl-polecat_s2 --dataset tkgl-polecat --root "$TGB_ROOT" --seed 2 --epochs 2 --window-rec 365 --window-tr 30 --window-trans 30 --window-pop 30 --path-window 30 --hidden 64 --score-test
python experiments/tgb/race_tkg_path_v1.py --tag tkgl-polecat_s3 --dataset tkgl-polecat --root "$TGB_ROOT" --seed 3 --epochs 2 --window-rec 365 --window-tr 30 --window-trans 30 --window-pop 30 --path-window 30 --hidden 64 --score-test
python experiments/tgb/race_tkg_path_v1.py --tag tkgl-polecat_s4 --dataset tkgl-polecat --root "$TGB_ROOT" --seed 4 --epochs 2 --window-rec 365 --window-tr 30 --window-trans 30 --window-pop 30 --path-window 30 --hidden 64 --score-test
