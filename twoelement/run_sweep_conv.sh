#!/bin/bash
# Converged sweep: every case runs until the solver's own 1e-9 momentum
# tolerance, not until max_steps. STEP_SCALE=4 is headroom -- the alpha=-3 L1
# probe exited at 19,640 of 42,000 -- so the cost is set by convergence, not by
# the ceiling. Case dirs carry an _s4 suffix, leaving the 12k runs intact for
# comparison.
set -u
cd /home/qiqi/flexcompute/sa-ai/twoelement || exit 1
export YPLUS_SCALE=0.5 STEP_SCALE=4.0
PY=/home/qiqi/flexcompute/compute/.venv/bin/python
echo "START $(date -u +%H:%M:%S)  cwd=$(pwd)"
for a in -5.0 -3.0 -1.0 1.0 3.0 5.0 7.0; do
  for lvl in L0 L1; do
    echo "######## $lvl alpha=$a  $(date -u +%H:%M:%S)"
    ALPHA=$a "$PY" -u run_ladder_v2.py $lvl 2>&1 \
      | grep -E "^=== |done in|forces:|FAILED" | head -4
  done
done
echo "######## L2 alpha=-1  $(date -u +%H:%M:%S)"
ALPHA=-1.0 "$PY" -u run_ladder_v2.py L2 2>&1 \
  | grep -E "^=== |done in|forces:|FAILED" | head -4
echo "DONE $(date -u +%H:%M:%S)"
