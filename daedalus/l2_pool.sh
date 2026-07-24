#!/bin/bash
# O-grid L2 alpha sweep, sequential, 6 ranks on GPUs 0,1,2,3,5,6
cd /home/qiqi/flexcompute/sa-ai/scripts/daedalus
for a in 4 5 6; do
  python3 run_solution.py case_ogrid_L2_saai_a$a 0,1,2,3,5,6 saai
done
echo L2_POOL_DONE
