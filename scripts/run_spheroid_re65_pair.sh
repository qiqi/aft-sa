#!/bin/bash
# workshop pair alpha=5,10 at Re_L=6.5e6, GPUs 6,7; waits for the re72a0
# ladder to release the GPUs
cd /home/qiqi/flexcompute/sa-ai
until grep -q "SPHEROID-RE72A0-LADDER-DONE" runlogs/spheroid_re72a0.log 2>/dev/null; do sleep 300; done
for a in a5 a10; do
  for lev in L0 L1 L2; do
    echo "=== START re65$a $lev $(date +%H:%M)"
    python3 daedalus/run_solution.py spheroid_fv1/case_ogrid_${lev}_saai_re65${a} 6,7 saai \
      && echo "=== DONE re65$a $lev rc=0 $(date +%H:%M)" \
      || echo "=== FAIL re65$a $lev $(date +%H:%M)"
  done
done
echo SPHEROID-RE65-PAIR-DONE
