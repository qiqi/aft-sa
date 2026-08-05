#!/bin/bash
# Negative-alpha extension: at negative incidence the fore element's LOWER
# surface becomes the suction side, so this is where turbulence could fill the
# slot from underneath. cd first: ssh lands in $HOME.
set -u
cd /home/qiqi/flexcompute/sa-ai/twoelement || exit 1
echo "cwd: $(pwd)"
PY=/home/qiqi/flexcompute/compute/.venv/bin/python
export YPLUS_SCALE=0.5
for a in -3.0 -5.0; do
  for lvl in L0 L1; do
    echo "######## $lvl alpha=$a"
    ALPHA=$a "$PY" -u run_ladder_v2.py $lvl 2>&1 \
      | grep -E "^=== |done in|forces:|FAILED" | head -4
  done
done
echo "######## negative sweep complete"
