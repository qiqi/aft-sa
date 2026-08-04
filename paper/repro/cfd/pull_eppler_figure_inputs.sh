#!/bin/bash
# Pull the E387 figure inputs from the compute host to this host.
#
# The E387 cases live on 014-v100-dev (/local_data/qiqi/sa-ai/flow360_fv1); the
# authoritative paper tree is here. Rather than regenerate figures on the compute
# host against a tree that has drifted (whitepaper.tex there is days behind), copy
# only what the figure scripts read -- slices, surface data, force/x_tr history,
# and the resolved-kernel echo -- and plot here.
#
# Meshes are deliberately NOT copied: they are the bulk (46 MB cgns + ~400 MB
# processed dump per L2 case) and no figure script reads them.
#
# Bulk lands on the data disk; symlinks into flow360_fv1 keep the default
# SAAI_CFD_ROOT working, so no script needs an env var.
#
#   ./pull_eppler_figure_inputs.sh [host]
set -u
HOST=${1:-014-v100-dev}
REMOTE=/local_data/qiqi/sa-ai/flow360_fv1
STORE=/local_data_2/qiqi/sa-ai/flow360_fv1_cases
LINKDIR=/home/qiqi/flexcompute/sa-ai/flow360_fv1

FAMS="cav str"
LEVELS="L0 L1 L2"
LADDER="a0 a2 a5 a7"
EXT_L2="am2 a1 a3 a4x a6 a8p5"
EXT_L01="am2 a8p5"

cases=""
for f in $FAMS; do
  for L in $LEVELS; do
    for a in $LADDER; do cases="$cases ${f}${L}prop_eppler387_Re200k_${a}"; done
  done
  for a in $EXT_L2; do cases="$cases ${f}L2prop_eppler387_Re200k_${a}"; done
  for L in L0 L1; do
    for a in $EXT_L01; do cases="$cases ${f}${L}prop_eppler387_Re200k_${a}"; done
  done
done

mkdir -p "$STORE"
n=0
for c in $cases; do
  mkdir -p "$STORE/$c"
  rsync -a --info=none \
    --include='slice_centerSpan*' --include='slice_with_derived*' \
    --include='surface_fluid_eppler387*' --include='total_forces_v2.csv' \
    --include='xtr_history.csv' --include='ai_constants.log' \
    --include='contours.txt' --include='Flow360.json' \
    --include='columnar.json' --include='boundaryNames.txt' \
    --exclude='*' \
    "$HOST:$REMOTE/$c/" "$STORE/$c/" 2>/dev/null
  # symlink so the default case root resolves without an env var
  [ -e "$LINKDIR/$c" ] || ln -s "$STORE/$c" "$LINKDIR/$c"
  got=$(ls "$STORE/$c" 2>/dev/null | wc -l)
  printf '%-44s %2d files\n' "$c" "$got"
  [ "$got" -gt 0 ] && n=$((n+1))
done
echo "pulled $n/$(echo $cases | wc -w) cases -> $STORE ($(du -sh $STORE | cut -f1))"
echo "symlinked into $LINKDIR"
