#!/bin/bash
# L1 remaining work on GPUs 4,7: ogrid a5/a6 in parallel, then cavity a5, a6 (2-rank)
cd /home/qiqi/flexcompute/sa-ai/scripts/daedalus
python3 run_solution.py case_ogrid_L1_saai_a5 4 saai &
python3 run_solution.py case_ogrid_L1_saai_a6 7 saai &
wait
python3 run_solution.py case_cavity_L1_saai_a5 4,7 saai
python3 run_solution.py case_cavity_L1_saai_a6 4,7 saai
echo L1_POOL_DONE
