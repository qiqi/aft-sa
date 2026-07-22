#!/bin/bash
# Regenerate the SELF-CONTAINED model figures (fast; no CFD tree needed) and
# recompile main.tex. The CFD figures (nlf_*/eppler_*/flat_plate_*/mesh_*)
# need the flow360/ case tree and are regenerated manually with their
# regen_*.py scripts (see ONBOARDING.md Sec. 5 for the figure->script map).
set -euo pipefail
cd /home/qiqi/flexcompute/sa-ai/paper
VENV_PY=/home/qiqi/flexcompute/compute/.venv/bin/python

echo "=== regen self-contained model figures ==="
$VENV_PY regen_kernel_maps.py 2>&1 | tail -1
$VENV_PY regen_indicator_plane.py 2>&1 | tail -1
$VENV_PY regen_klambda.py 2>&1 | tail -1
$VENV_PY regen_shapefactor_transport.py 2>&1 | tail -1
$VENV_PY regen_wall_layer.py 2>&1 | tail -1
$VENV_PY regen_fs_transport_rows.py 2>&1 | tail -1

echo "=== figure timestamps ==="
ls -la --time-style=+%m-%d_%H:%M figs/*.pdf | head
echo "=== recompile pdflatex (3 passes + bibtex) ==="
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 | tail -3
echo "=== final PDF ==="
ls -la --time-style=+%m-%d_%H:%M main.pdf
echo "=== any LaTeX errors? ==="
grep -E "^!" main.log | head -5 || echo "(none)"
