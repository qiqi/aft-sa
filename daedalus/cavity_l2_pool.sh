#!/bin/bash
# Cavity L2 (111M elem) alpha sweep, 8 ranks on all GPUs.
# a4 prep runs separately; this waits for GPU 4 to be truly free, then solves.
cd /home/qiqi/flexcompute/sa-ai/scripts/daedalus
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4)" -gt 1000 ]; do
  sleep 60
done
python3 run_solution.py case_cavity_L2_saai_a4 0,1,2,3,4,5,6,7 saai solve
for a in 5 6; do
  python3 run_solution.py case_cavity_L2_saai_a$a 0,1,2,3,4,5,6,7 saai
done
echo CAVITY_L2_POOL_DONE
