#!/bin/bash
# alpha=4 reruns with nuHat volume output on GPUs 4,7
cd /home/qiqi/flexcompute/sa-ai/scripts/daedalus
python3 run_solution.py case_cavity_L1_saai_a4chi 4,7 saai
python3 run_solution.py case_ogrid_L1_saai_a4chi 4 saai
echo A4CHI_POOL_DONE
