#!/bin/bash
# L0/L1 on the adopted geometry. Runs on 017; must cd first (ssh lands in $HOME).
set -u
cd /home/qiqi/flexcompute/sa-ai/twoelement || exit 1
echo "cwd: $(pwd)"
/home/qiqi/flexcompute/compute/.venv/bin/python -u run_ladder_v2.py "$@" 2>&1 | tail -40
