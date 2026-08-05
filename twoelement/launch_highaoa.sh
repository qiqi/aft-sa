#!/bin/bash
D=/home/qiqi/flexcompute/sa-ai/twoelement
cd "$D" || exit 1
setsid nohup bash "$D/run_highaoa.sh" 4 L0 9.0 11.0 13.0 \
  > "$D/high_L0.log" 2>&1 < /dev/null &
setsid nohup bash "$D/run_highaoa.sh" 5 L1 9.0 11.0 13.0 \
  > "$D/high_L1.log" 2>&1 < /dev/null &
sleep 10
for f in high_L0 high_L1; do echo "-- $f: $(head -2 "$D/$f.log" | tr '\n' ' ')"; done
