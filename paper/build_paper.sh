#!/bin/bash
# Regen figures (using compute .venv python which has vtk/matplotlib/numpy) + recompile main.tex.
set -euo pipefail
cd /home/qiqi/flexcompute/aft-sa/paper
VENV_PY=/home/qiqi/flexcompute/compute/.venv/bin/python
echo "=== regen NACA (Fig 5 + 6 + cpcf) ==="
$VENV_PY regen_naca.py 2>&1 | tail -5
echo "=== regen final (polar, xtr band, NLF cpcf) ==="
$VENV_PY regen_final.py 2>&1 | tail -5
echo "=== regen proper-refinement (drag + transition band) ==="
$VENV_PY /home/qiqi/flexcompute/aft-sa/flow360/plot_proper_grid_convergence.py 2>&1 | tail -5
echo "=== regen TE mesh viz ==="
$VENV_PY /home/qiqi/flexcompute/aft-sa/flow360/plot_TE_mesh_zoom_L3.py 2>&1 | tail -3
# regen_refine.py is OBSOLETE (produced the old 5-level improper refinement series);
# the new proper-refinement L0-L3 series is built by plot_proper_grid_convergence.py.
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
