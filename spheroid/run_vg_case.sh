#!/bin/bash
# Clone a staged tunnel case and run it with the env-gated two-branch "vg"
# rate+gate ON.  The kernel switch is env-only -- Flow360.json is byte
# identical to the canon case -- so the comparison isolates the kernel.
#
#   usage: run_vg_case.sh <canon_case_dir> [gpus]
#
# vg constants (whitepaper two-source appendix; compute 29726cb5f9):
#   AI_A_VISC   = 0.0276   viscous/curvature rate floor  a_visc
#   AI_REOMC_BC = 130      second gate branch  A + B_c/P_curv^2
# Both default to 0 = canon, which is why the canon runs need no flag.
set -euo pipefail
SRC="${1:?usage: run_vg_case.sh <canon_case_dir> [gpus]}"
GPUS="${2:-0,1}"
REPO="${SAAI_REPO:-$HOME/flexcompute/sa-ai}"
SOLVER="${FLOW360_SOLVER:-$HOME/flexcompute/compute/install/release/bin/Flow360Solver}"
DST="${SRC%/}_vg"

mkdir -p "$DST"
for f in Flow360.json Flow360Mesh.json gpubind.sh; do cp -f "$SRC/$f" "$DST/"; done
for f in mesh.cgns mesh.cgns.json mesh.cgns.partitionerData.npart.2.json \
         mesh.cgns.partitionerData.npart.2_rank_1_of_1.dmp \
         mesh.cgns_rank_1_of_2.dmp mesh.cgns_rank_2_of_2.dmp; do
  [ -e "$SRC/$f" ] && ln -sf "$(readlink -f "$SRC/$f")" "$DST/$f"
done
chmod +x "$DST/gpubind.sh"
cd "$DST"

eval "$(cd "$REPO" && python3 -c 'import sys; sys.path.insert(0,"paper/repro/driver")
from saai_env import canonical_ai_env
for k,v in canonical_ai_env().items(): print("export %s=%s"%(k,v))')"
export AI_A_VISC=0.0276 AI_REOMC_BC=130

python3 - "$DST/ai_constants.note" <<'PY'
import json, os, sys
env = {k: v for k, v in os.environ.items() if k.startswith('AI_')}
json.dump(env, open(sys.argv[1], 'w'), indent=1, sort_keys=True)
PY

export GPU_LIST="$GPUS" OMP_NUM_THREADS=1
nohup setsid /usr/bin/mpirun -np 2 ./gpubind.sh "$SOLVER" > solver_stdout.log 2>&1 &
echo "vg launched pid $! on GPUs $GPUS in $DST"
