#!/bin/bash
# One GPU's share of the L2 ladder.  usage: run_l2_group.sh <gpu> <alpha>...
set -u
D=/home/qiqi/flexcompute/sa-ai/twoelement
cd "$D" || exit 1
G=$1; shift
export YPLUS_SCALE=0.5 STEP_SCALE=4.0 GPU=$G
PY=/home/qiqi/flexcompute/compute/.venv/bin/python
echo "START gpu=$G alphas=$* $(date -u +%H:%M:%S)"
for a in "$@"; do
  echo "######## L2 alpha=$a  $(date -u +%H:%M:%S)"
  ALPHA=$a "$PY" -u run_ladder_v2.py L2 2>&1 \
    | grep -E "^=== |done in|forces:|FAILED" | head -4
done
echo "DONE gpu=$G $(date -u +%H:%M:%S)"
