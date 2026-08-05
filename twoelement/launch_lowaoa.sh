#!/bin/bash
D=/home/qiqi/flexcompute/sa-ai/twoelement
cd "$D" || exit 1
setsid nohup bash "$D/run_lowaoa.sh" > "$D/lowaoa.log" 2>&1 < /dev/null &
sleep 8; echo "launched; log head:"; head -3 "$D/lowaoa.log"
