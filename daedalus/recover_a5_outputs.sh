#!/bin/bash
# Regenerate cavity-L2 a5's shutdown outputs (volume.pvtu, slicing CSVs)
# that were zeroed by the 2026-07-25 disk-full: restart from the 8-rank
# dumps, march 100 extra steps, let the solver rewrite its final outputs.
# Force means move < the history's 4e-4 std; the canon CL/CD numbers are
# from the pre-recovery total_forces_v2.csv, backed up here first.
#   ./recover_a5_outputs.sh <gpu-list, e.g. 0,1,2,3,4,5,6,7>
set -eu
cd "$(dirname "$0")"
C=case_cavity_L2_saai_a5
GPUS=${1:?gpu list}
for f in total_forces_v2.csv surface_forces_v2.csv nonlinear_residual_v2.csv; do
    [ -f "$C/$f.pre_recovery" ] || cp "$C/$f" "$C/$f.pre_recovery"
done
python3 - <<'EOF'
import json
p = 'case_cavity_L2_saai_a5/Flow360.json'
j = json.load(open(p))
j['runControl']['restart'] = True
j['timeStepping']['maxPseudoSteps'] = 20100
json.dump(j, open(p, 'w'), indent=4)
print('Flow360.json: restart=True, maxPseudoSteps=20100')
EOF
python3 run_solution.py $C "$GPUS" saai solve
for f in volume.pvtu Y_slicing_forceDistribution.csv X_slicing_forceDistribution.csv; do
    [ -s "$C/$f" ] && echo "OK   $C/$f ($(stat -c%s $C/$f) bytes)" || echo "FAIL $C/$f still empty"
done
echo A5-RECOVERY-DONE
