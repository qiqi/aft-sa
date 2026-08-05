#!/bin/bash
# Post-stall extension. usage: run_highaoa.sh <gpu> <level> <alpha>...
# At these angles the flap is expected to separate; a steady RANS may not reach
# the 1e-9 tolerance at all, so the step ceiling is the fallback and any case
# that hits it is reported as NOT converged rather than quietly used.
set -u
D=/home/qiqi/flexcompute/sa-ai/twoelement
cd "$D" || exit 1
G=$1; L=$2; shift 2
export YPLUS_SCALE=0.5 STEP_SCALE=4.0 GPU=$G
PY=/home/qiqi/flexcompute/compute/.venv/bin/python
echo "START gpu=$G level=$L alphas=$* $(date -u +%H:%M:%S)"
for a in "$@"; do
  echo "######## $L alpha=$a  $(date -u +%H:%M:%S)"
  ALPHA=$a "$PY" -u run_ladder_v2.py "$L" 2>&1 \
    | grep -E "^=== |done in|forces:|FAILED" | head -4
done
echo "DONE gpu=$G $(date -u +%H:%M:%S)"
