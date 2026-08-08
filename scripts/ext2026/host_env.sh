#!/bin/bash
# Host-portable loader setup for the canonical Flow360 solver binary.
# Source this (do not execute) before running any compute/ tool.
#
# The canon binary is 014's Jul-30 build -- the one that produced the paper's
# Aug-3 cases -- installed identically on 014/017/019 so all three run the same
# physics. Its RUNPATH assumes 014's Debian MPI layout and 014's /shared_data
# AMGX path. 019 lays both out differently but carries the SAME
# AMGx_Flow360-2.4.0 and an OpenMPI 4.1.x, so only the search paths differ.
# Paths are added ONLY when the loader actually fails, leaving 014/017 on their
# own resolution.
_SOLVER=/home/qiqi/flexcompute/compute/install/release/bin/Flow360Solver

if ldd "$_SOLVER" 2>/dev/null | grep -q "not found"; then
  for _d in /local_data_2/shared_data/third_party/AMGx_Flow360-2.4.0/lib \
            /shared_data/third_party/AMGx_Flow360-2.4.0/lib \
            /usr/mpi/gcc/openmpi-4.1.9a1/lib \
            /usr/lib/x86_64-linux-gnu/openmpi/lib; do
    [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}"
  done
  # NOTE: a relocated OpenMPI whose build prefix no longer exists (e.g. the
  # /local_data_2 copy of openmpi-4.1.4, built for /shared_data/...) fails in
  # MPI_Init because it looks for its plugins at the old prefix. Prefer a
  # properly-installed MPI; that is why /usr/mpi/... is listed above.
  if ldd "$_SOLVER" 2>/dev/null | grep -q "not found"; then
    echo "FATAL: unresolved solver libraries on $(hostname -s)" >&2
    ldd "$_SOLVER" | grep "not found" >&2
    return 1 2>/dev/null || exit 1
  fi
fi
export OMP_NUM_THREADS=1
