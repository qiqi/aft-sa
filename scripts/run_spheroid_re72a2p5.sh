#!/bin/bash
# alpha=2.5, Re_L=7.2e6 ladder (Stock Fig 14b pure-TS); runs after the
# workshop pair releases GPUs 6,7
cd /home/qiqi/flexcompute/sa-ai
until grep -q "SPHEROID-RE65-PAIR-DONE" runlogs/spheroid_re65_pair.log 2>/dev/null; do sleep 300; done
for lev in L0 L1 L2; do
  echo "=== START re72a2p5 $lev $(date +%H:%M)"
  python3 daedalus/run_solution.py spheroid_fv1/case_ogrid_${lev}_saai_re72a2p5 6,7 saai \
    && echo "=== DONE re72a2p5 $lev rc=0 $(date +%H:%M)" \
    || echo "=== FAIL re72a2p5 $lev $(date +%H:%M)"
done
echo SPHEROID-RE72A2P5-LADDER-DONE
