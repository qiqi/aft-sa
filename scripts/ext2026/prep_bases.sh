#!/bin/bash
# Ensure every clone-source base case carries its partitioned mesh dump.
#
# clone() hardlinks any file >10MB, so generating mesh.cgns_rank_1_of_1.dmp ONCE
# per base costs 400MB per family instead of 400MB per case. Some bases (the NLF
# pair) had theirs cleaned up after the original campaign; MeshProcessor
# regenerates it from mesh.cgns in ~20 s. MeshPartitioner must run first -- it
# writes the partitionerData that MeshProcessor consumes.
set -u
D=/home/qiqi/flexcompute/sa-ai/scripts/ext2026
. "$D/host_env.sh" || exit 1
BIN=/home/qiqi/flexcompute/compute/install/release/bin
export OMPI_COMM_WORLD_LOCAL_RANK=0 OMPI_COMM_WORLD_RANK=0 OMPI_COMM_WORLD_SIZE=1
FR=/home/qiqi/flexcompute/sa-ai/flow360_fv1

rc=0
for b in strL2prop_nlf0416_Re4M_a4 cavL2prop_nlf0416_Re4M_a4 \
         strL2prop_eppler387_Re200k_a5 cavL2prop_eppler387_Re200k_a5; do
  wd="$FR/$b"
  if [ ! -d "$wd" ]; then echo "MISSING base $b on $(hostname -s)"; rc=1; continue; fi
  if [ -s "$wd/mesh.cgns_rank_1_of_1.dmp" ]; then
    echo "OK   $b (dump present, $(stat -Lc%s "$wd/mesh.cgns_rank_1_of_1.dmp") bytes)"
    continue
  fi
  echo "PREP $b ..."
  ( cd "$wd" && "$BIN/MeshPartitioner" --meshfile mesh.cgns --partitions 1 \
      && "$BIN/MeshProcessor" --threads 1 mesh.cgns ) > "$wd/meshprep.log" 2>&1
  if [ -s "$wd/mesh.cgns_rank_1_of_1.dmp" ]; then
    echo "DONE $b ($(stat -Lc%s "$wd/mesh.cgns_rank_1_of_1.dmp") bytes)"
  else
    echo "FAIL $b -- see $wd/meshprep.log"; tail -3 "$wd/meshprep.log"; rc=1
  fi
done
exit $rc
