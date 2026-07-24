#!/bin/bash
# Daedalus canon campaign resume after the 2026-07-24 08:12 server reboot.
# All Phase-1 L1 runs completed except cavity a6 (killed mid-solve at 08:07).
# Re-run it first (GPUs 3,4), then WAIT for the 2D Eppler chains on GPUs 6/7
# to finish before starting the L2 phase, so the box never carries the L2
# load and the 2D chains at once.
set -u
cd /home/qiqi/flexcompute/sa-ai/daedalus
RL=/home/qiqi/flexcompute/sa-ai/runlogs

run() {  # case gpus
  echo "=== START $1 (gpus $2) $(date +%H:%M)"
  python3 run_solution.py "$1" "$2" saai > "$1/run_solution.log" 2>&1
  echo "=== DONE  $1 rc=$? $(date +%H:%M)"
}

run case_cavity_L1_saai_a6 3,4

echo "== waiting for 2D chains to release GPUs 6/7 =="
until grep -q "CHAIN-STR-EXIT" $RL/rerun_str.log 2>/dev/null && \
      grep -q "CHAIN-CAV-EXIT" $RL/rerun_cav.log 2>/dev/null; do
  sleep 60
done

echo "== PHASE 2: L2 =="
for a in 4 5 6; do
  run case_ogrid_L2_saai_a$a 0,1,2,3,4,5
done
for a in 4 5 6; do
  run case_cavity_L2_saai_a$a 0,1,2,3,4,5,6,7
done
echo DAEDALUS-CANON-DONE
