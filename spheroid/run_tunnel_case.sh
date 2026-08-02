#!/bin/bash
# Launch ONE staged spheroid tunnel case on exactly 2 GPUs, detached.
#
#   usage: run_tunnel_case.sh <case_dir> [gpu_a,gpu_b]
#
# Two GPUs because the meshes are partitioned npart=2 (see the template
# cases' mesh.cgns.partitionerData.npart.2*), so the solver wants 2 ranks.
# gpubind.sh maps GPU_LIST entries onto MPI local ranks.
#
# The SA-AI constants come from paper/repro/driver/saai_env.canonical_ai_env()
# rather than being transcribed, and are echoed into ai_constants.note in the
# case dir for provenance (the mpirun chain does not capture the solver's own
# SUPPORT-level constants echo).
set -euo pipefail

CASE="${1:?usage: run_tunnel_case.sh <case_dir> [gpus]}"
GPUS="${2:-0,1}"
REPO="${SAAI_REPO:-$HOME/flexcompute/sa-ai}"
SOLVER="${FLOW360_SOLVER:-$HOME/flexcompute/compute/install/release/bin/Flow360Solver}"

[ -f "$CASE/Flow360.json" ] || { echo "no Flow360.json in $CASE" >&2; exit 1; }
[ -x "$SOLVER" ] || { echo "solver not executable: $SOLVER" >&2; exit 1; }

cd "$CASE"

eval "$(cd "$REPO" && python3 -c 'import sys; sys.path.insert(0, "paper/repro/driver")
from saai_env import canonical_ai_env
for k, v in canonical_ai_env().items():
    print("export %s=%s" % (k, v))')"

(cd "$REPO" && python3 -c 'import sys, json; sys.path.insert(0, "paper/repro/driver")
from saai_env import canonical_ai_env
json.dump(canonical_ai_env(), open(sys.argv[1], "w"), indent=1)' "$CASE/ai_constants.note")

export GPU_LIST="$GPUS" OMP_NUM_THREADS=1
nohup setsid /usr/bin/mpirun -np 2 ./gpubind.sh "$SOLVER" > solver_stdout.log 2>&1 &
echo "launched pid $! on GPUs $GPUS in $CASE"
