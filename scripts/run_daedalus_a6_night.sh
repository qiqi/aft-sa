#!/bin/bash
# night shift: the held-back ogrid L2 a6 (final Daedalus new-canon case)
SA=/home/qiqi/flexcompute/sa-ai
cd $SA/daedalus_fv1
echo "=== START case_ogrid_L2_saai_a6 (gpus 0,1,2,3,4,5) $(date +%H:%M)"
python3 $SA/daedalus/run_solution.py case_ogrid_L2_saai_a6 0,1,2,3,4,5 saai > case_ogrid_L2_saai_a6/run_solution.log 2>&1
echo "=== DONE case_ogrid_L2_saai_a6 rc=$? $(date +%H:%M)"
echo DAEDALUS-A6-DONE
