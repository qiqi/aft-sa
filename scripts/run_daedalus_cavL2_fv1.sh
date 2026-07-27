#!/bin/bash
# the held cavity-L2 recomputation, unblocked by user order 2026-07-27:
# three cases sequentially on all 8 GPUs
SA=/home/qiqi/flexcompute/sa-ai
cd $SA/daedalus_fv1
for a in 4 5 6; do
  echo "=== START case_cavity_L2_saai_a$a (8 gpus) $(date +%H:%M)"
  python3 $SA/daedalus/run_solution.py case_cavity_L2_saai_a$a 0,1,2,3,4,5,6,7 saai > case_cavity_L2_saai_a$a/run_solution.log 2>&1 \
    && echo "=== DONE case_cavity_L2_saai_a$a rc=0 $(date +%H:%M)" \
    || echo "=== FAIL case_cavity_L2_saai_a$a $(date +%H:%M)"
done
echo DAEDALUS-CAVL2-ALL-DONE
