#!/bin/bash
# Fan the L2 ladder across 4 GPUs, leaving 4 free for other users.
D=/home/qiqi/flexcompute/sa-ai/twoelement
cd "$D" || exit 1
launch () {  # $1 = gpu, rest = alphas
  local g=$1; shift
  setsid nohup bash "$D/run_l2_group.sh" "$g" "$@" \
    > "$D/l2_g${g}.log" 2>&1 < /dev/null &
}
launch 0 -9.0 -7.0
launch 1 -5.0 -3.0
launch 2  1.0  3.0
launch 3  5.0  7.0
sleep 10
for g in 0 1 2 3; do echo "-- gpu$g: $(head -2 "$D/l2_g${g}.log" | tr '\n' ' ')"; done
