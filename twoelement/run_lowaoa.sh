#!/bin/bash
# Deep-negative AoA, on a separate GPU so it runs alongside the converged sweep.
# At these angles the fore element's lower surface is strongly amplified and its
# discharge should blanket the flap's leading edge.
set -u
D=/home/qiqi/flexcompute/sa-ai/twoelement
cd "$D" || exit 1
export YPLUS_SCALE=0.5 STEP_SCALE=4.0 GPU=1
PY=/home/qiqi/flexcompute/compute/.venv/bin/python
echo "START $(date -u +%H:%M:%S) on GPU $GPU"
for a in -7.0 -9.0; do
  for lvl in L0 L1; do
    echo "######## $lvl alpha=$a  $(date -u +%H:%M:%S)"
    ALPHA=$a "$PY" -u run_ladder_v2.py $lvl 2>&1 \
      | grep -E "^=== |done in|forces:|FAILED" | head -4
  done
done
echo "DONE $(date -u +%H:%M:%S)"
