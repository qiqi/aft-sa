#!/bin/bash
# Flap-alone SA-AI at several anchoring incidences, for the streamline trace.
# Runs on 017; must cd first because ssh lands in $HOME.
set -u
cd /home/qiqi/flexcompute/sa-ai/twoelement || exit 1
echo "cwd: $(pwd)"
PY=/home/qiqi/flexcompute/compute/.venv/bin/python
for a in "$@"; do
    echo "########## alpha=$a"
    "$PY" -u step1v_flap_alone_rans.py "$a" 2>&1 \
        | grep -E "^M = |done in|forces:|FAILED" | head -5
done
echo "########## sweep complete"
