#!/bin/bash
# New-canon reruns of the Sec-VI special studies (forks, ladders, extensions,
# tau=1) on GPUs 6,7 while Daedalus Phase 3 holds 0-5. All scripts source
# from and write into flow360_fv1 (FR/LOCAL_DATA flipped); canonical_ai_env
# carries the bypass. Sequential per GPU; str chain on 6, cav chain on 7.
set -u
cd /home/qiqi/flexcompute/sa-ai/paper/repro/cfd
run() { echo "=== START $* $(date -u +%H:%M)"; python3 "$@"; echo "=== DONE $* rc=$? $(date -u +%H:%M)"; }
(
  run run_continuation_ladders.py str 6
  run run_bistability_forks.py str 6
  run run_fork_extensions.py str 6
  run run_dn_extensions.py str 6
  run run_tau_test.py 6
  echo "STR-CHAIN-DONE"
) &
(
  run run_continuation_ladders.py cav 7
  run run_bistability_forks.py cav 7
  run run_fork_extensions.py cav 7
  run run_dn_extensions.py cav 7
  echo "CAV-CHAIN-DONE"
) &
wait
echo "SPECIAL-STUDIES-FV1-DONE"
