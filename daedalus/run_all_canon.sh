#!/bin/bash
# Daedalus wing sweep at the whole-equation canon (2026-07-24).
# Phase 1: L1 cases packed across GPUs; Phase 2: L2 cases sequential.
set -u
cd /home/qiqi/flexcompute/sa-ai/daedalus

# hardlink the family/level mesh into the a4/a6 case dirs
for fl in cavity_L1 cavity_L2 ogrid_L1 ogrid_L2; do
  for a in 4 6; do
    [ -f case_${fl}_saai_a$a/mesh.cgns ] || \
      ln case_${fl}_saai_a5/mesh.cgns case_${fl}_saai_a$a/mesh.cgns
  done
done

run() {  # case gpus
  echo "=== START $1 (gpus $2) $(date +%H:%M)"
  python3 run_solution.py "$1" "$2" saai > "$1/run_solution.log" 2>&1
  echo "=== DONE  $1 rc=$? $(date +%H:%M)"
}

echo "== PHASE 1: L1 =="
run case_ogrid_L1_saai_a4 0 &
run case_ogrid_L1_saai_a5 1 &
run case_ogrid_L1_saai_a6 2 &
run case_cavity_L1_saai_a5 5,6 &
run case_cavity_L1_saai_a4 3,4 &
P_CAV4=$!
wait $P_CAV4   # cavity a4's GPU pair frees first
run case_cavity_L1_saai_a6 3,4 &
wait
echo "== PHASE 2: L2 =="
for a in 4 5 6; do
  run case_ogrid_L2_saai_a$a 0,1,2,3,4,5
done
for a in 4 5 6; do
  run case_cavity_L2_saai_a$a 0,1,2,3,4,5,6,7
done
echo DAEDALUS-CANON-DONE
