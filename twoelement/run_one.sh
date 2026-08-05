#!/bin/bash
# One (level, alpha) at the adopted wall setting. cd first: ssh lands in $HOME.
set -u
cd /home/qiqi/flexcompute/sa-ai/twoelement || exit 1
echo "cwd: $(pwd)  ALPHA=${ALPHA:-?}  YPLUS_SCALE=${YPLUS_SCALE:-?}"
/home/qiqi/flexcompute/compute/.venv/bin/python -u run_ladder_v2.py "$@" 2>&1 \
  | grep -E "^=== |done in|forces:|FAILED" | head -5
