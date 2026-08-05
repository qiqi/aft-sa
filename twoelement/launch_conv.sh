#!/bin/bash
# Detach the converged sweep so it survives the ssh session. Absolute paths
# throughout: ssh lands in $HOME, and every relative-path attempt at this has
# silently run in the wrong directory.
D=/home/qiqi/flexcompute/sa-ai/twoelement
cd "$D" || exit 1
setsid nohup bash "$D/run_sweep_conv.sh" > "$D/sweep_conv.log" 2>&1 < /dev/null &
sleep 8
echo "launched; log head:"
head -4 "$D/sweep_conv.log"
