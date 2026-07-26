#!/bin/bash
# NLF negative pair on L0/L1, both families: 8 cases over GPUs 0-5.
# L1 cases get a dedicated GPU; the two L0 streams run both alphas.
cd /home/qiqi/flexcompute/sa-ai
run() { python3 scripts/run_nlfneg_case.py "$@"; }
(run str L1 -8 0) &
(run cav L1 -8 1) &
(run str L1 -4 2) &
(run cav L1 -4 3) &
(run str L0 -8 4; run str L0 -4 4) &
(run cav L0 -8 5; run cav L0 -4 5) &
wait
echo NLFNEG-L0L1-ALL-DONE
