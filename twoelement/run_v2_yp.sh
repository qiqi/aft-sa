#!/bin/bash
set -u
cd /home/qiqi/flexcompute/sa-ai/twoelement || exit 1
echo "cwd: $(pwd)  YPLUS_SCALE=${YPLUS_SCALE:-1.0}"
YPLUS_SCALE=0.5 /home/qiqi/flexcompute/compute/.venv/bin/python -u run_ladder_v2.py "$@" 2>&1 | tail -40
