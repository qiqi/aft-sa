#!/bin/bash
# Sequential spheroid tunnel-case queue: ONE case at a time on exactly 2 GPUs.
#
#   usage: queue_tunnel_cases.sh <gpu_a,gpu_b> <cond_seed> [cond_seed ...]
#
# The meshes are partitioned npart=2, so a case needs 2 ranks / 2 GPUs; with a
# 2-GPU budget that means strictly sequential.  The queue waits for those GPUs
# to fall idle before starting each case, so it is safe to launch while an
# earlier case is still running.
#
# Per case it: stages via build_tunnel_cases.py, ensures the MeshProcessor
# native dumps exist (shared per mesh, symlinked -- they are 1.4 GB/mesh and
# identical for every case on that mesh), launches, and waits.
set -uo pipefail

GPUS="${1:?usage: queue_tunnel_cases.sh <gpus> <cond_seed>...}"; shift
REPO="${SAAI_REPO:-$HOME/flexcompute/sa-ai}"
ROOT=/local_data/qiqi/sa-ai/spheroid_fv1
MESHDIR=/local_data/qiqi/sa-ai/spheroid_meshes
BIN=$HOME/flexcompute/compute/install/release/bin
LOG=/local_data/qiqi/sa-ai/runlogs/tunnel_queue.log
mkdir -p "$(dirname "$LOG")"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

gpus_busy() {
  local n=0
  for g in ${GPUS//,/ }; do
    local used
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
    [ "${used:-0}" -gt 500 ] && n=$((n+1))
  done
  [ "$n" -gt 0 ]
}

ensure_dumps() {   # $1 = case dir
  local C="$1" real key P
  real=$(readlink -f "$C/mesh.cgns")
  key=$(basename "$real" .cgns)
  P="$MESHDIR/processed_${key}_npart2"
  if [ ! -f "$P/mesh.cgns_rank_1_of_2.dmp" ]; then
    say "  MeshProcessor for $key (one-off, ~1.4 GB shared)"
    mkdir -p "$P"
    ( cd "$C" && /usr/bin/mpirun -np 2 "$BIN/MeshProcessor" mesh.cgns \
        > MeshProcessor.log 2>&1 ) || { say "  MeshProcessor FAILED"; return 1; }
    mv "$C"/mesh.cgns_rank_1_of_2.dmp "$C"/mesh.cgns_rank_2_of_2.dmp "$P/" || return 1
  fi
  ln -sf "$P/mesh.cgns_rank_1_of_2.dmp" "$C/mesh.cgns_rank_1_of_2.dmp"
  ln -sf "$P/mesh.cgns_rank_2_of_2.dmp" "$C/mesh.cgns_rank_2_of_2.dmp"
}

say "queue start on GPUs $GPUS : $*"
for spec in "$@"; do
  C="$ROOT/case_ogrid_L1_tun_$spec"
  if [ -f "$C/total_forces_v2.csv" ] && \
     [ "$(wc -l < "$C/total_forces_v2.csv")" -gt 4000 ]; then
    say "SKIP $spec (already has $(wc -l < "$C/total_forces_v2.csv") force rows)"
    continue
  fi
  ( cd "$REPO" && python3 spheroid/build_tunnel_cases.py --stage "$spec" ) >>"$LOG" 2>&1 \
    || { say "STAGE FAILED $spec"; continue; }
  ensure_dumps "$C" || { say "DUMPS FAILED $spec"; continue; }

  while gpus_busy; do sleep 60; done
  df_avail=$(df --output=avail -BG /local_data | tail -1 | tr -dc 0-9)
  if [ "${df_avail:-0}" -lt 40 ]; then say "ABORT QUEUE: /local_data below 40G"; break; fi

  say "START $spec"
  "$REPO/spheroid/run_tunnel_case.sh" "$C" "$GPUS" >>"$LOG" 2>&1
  sleep 90
  while gpus_busy; do sleep 60; done
  rows=$( [ -f "$C/total_forces_v2.csv" ] && wc -l < "$C/total_forces_v2.csv" || echo 0 )
  say "DONE  $spec (force rows $rows, disk $(du -sh "$C" 2>/dev/null | cut -f1))"
done
say "QUEUE COMPLETE"
