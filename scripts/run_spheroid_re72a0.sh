#!/bin/bash
# alpha=0, Re_L=7.2e6 ladder (Stock Fig 14a pure-TS anchor), GPUs 6,7;
# waits for each re65 mesh to finish building before its level runs
cd /home/qiqi/flexcompute/sa-ai
for lev in L0 L1 L2; do
  M=/local_data/qiqi/sa-ai/spheroid_meshes/mesh_re65_${lev}.cgns
  until [ -s "$M" ] && ! lsof "$M" > /dev/null 2>&1; do sleep 120; done
  echo "=== START re72a0 $lev $(date +%H:%M)"
  python3 daedalus/run_solution.py spheroid_fv1/case_ogrid_${lev}_saai_re72a0 6,7 saai \
    && echo "=== DONE re72a0 $lev rc=0 $(date +%H:%M)" \
    || echo "=== FAIL re72a0 $lev $(date +%H:%M)"
done
echo SPHEROID-RE72A0-LADDER-DONE
