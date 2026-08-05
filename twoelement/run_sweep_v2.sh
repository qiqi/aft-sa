#!/bin/bash
# L2 at the design point, then an upward alpha sweep on L0 and L1.
# All at YPLUS_SCALE=0.5 (the adopted wall setting). Must cd: ssh lands in $HOME.
set -u
cd /home/qiqi/flexcompute/sa-ai/twoelement || exit 1
echo "cwd: $(pwd)"
PY=/home/qiqi/flexcompute/compute/.venv/bin/python
export YPLUS_SCALE=0.5

echo "################ L2 at alpha=-1"
ALPHA=-1.0 "$PY" -u run_ladder_v2.py L2 2>&1 | grep -E "^=== |done in|forces:|mesh:|FAILED" | head -6

for a in 1.0 3.0 5.0 7.0; do
  for lvl in L0 L1; do
    echo "################ $lvl at alpha=$a"
    ALPHA=$a "$PY" -u run_ladder_v2.py $lvl 2>&1 | grep -E "^=== |done in|forces:|FAILED" | head -4
  done
done
echo "################ sweep complete"
