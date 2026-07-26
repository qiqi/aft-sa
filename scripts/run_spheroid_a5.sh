#!/bin/bash
# spheroid Re=1.5e6 alpha=5 ladder, sequential on GPUs 6,7 (day shift)
cd /home/qiqi/flexcompute/sa-ai
for lev in L0 L1 L2; do
  echo "=== START spheroid a5 $lev $(date +%H:%M)"
  python3 daedalus/run_solution.py spheroid_fv1/case_ogrid_${lev}_saai_a5 6,7 saai \
    && echo "=== DONE spheroid a5 $lev rc=0 $(date +%H:%M)" \
    || echo "=== FAIL spheroid a5 $lev $(date +%H:%M)"
done
echo SPHEROID-A5-LADDER-DONE
