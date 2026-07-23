#!/bin/bash
# Regenerate the SELF-CONTAINED model figures (fast; no CFD tree needed) and
# recompile sa-ai.tex. The CFD figures (nlf_*/eppler_*/flat_plate_*/mesh_*/
# daedalus_*/chi_sheet_*) need the flow360/ case tree and are regenerated
# manually with their regen_*.py scripts (see ONBOARDING.md Sec. 5).
#
# SPHERE-KERNEL NOTE (2026-07-23): the self-contained model figures that
# sa-ai.tex references are indicator_sphere, onset_graze, model_calibrate,
# fs_nuHat_rows, and shapefactor_amplification (the pre-sphere kernel_maps /
# indicator_plane / klambda* are no longer referenced). The canonical kernel
# constants live in repro/analytic/fig04_shapefactor.py (c_nu,ai = 1/6,
# onset k*softmin_2(2600, 175 + 2/(Sg)^2) with the WHOLE-EQUATION k = 0.712),
# pinned against the solver by tests/test_constants_consistency.py.
# tab_frozen_slope.py regenerates the Sec.-II.C tables and ASSERTS the
# quoted numbers -- it doubles as a consistency check.
set -euo pipefail
cd /home/qiqi/flexcompute/sa-ai/paper
# the model-figure scripts need jax -> system python3 (the compute venv lacks jax).
PY=python3

echo "=== regen self-contained model figures ==="
$PY repro/analytic/fig01_indicator_sphere.py 2>&1 | tail -1
$PY repro/analytic/fig02_onset_graze.py 2>&1 | tail -1
$PY repro/analytic/fig02_model_calibrate.py 2>&1 | tail -1
$PY repro/analytic/fig03_fs_transport_rows.py 2>&1 | tail -1
$PY repro/analytic/fig04_shapefactor.py 2>&1 | tail -1
$PY repro/analytic/tab_frozen_slope.py 2>&1 | tail -3

echo "=== figure timestamps ==="
ls -la --time-style=+%m-%d_%H:%M figs/indicator_sphere.pdf figs/onset_graze.pdf \
    figs/model_calibrate.pdf figs/fs_nuHat_rows.pdf \
    figs/shapefactor_amplification.pdf 2>/dev/null || true
echo "=== recompile pdflatex (3 passes + bibtex) ==="
pdflatex -interaction=nonstopmode sa-ai.tex >/dev/null || true
bibtex sa-ai >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode sa-ai.tex >/dev/null || true
pdflatex -interaction=nonstopmode sa-ai.tex >/dev/null 2>&1 || true
echo "=== final PDF ==="
ls -la --time-style=+%m-%d_%H:%M sa-ai.pdf
echo "=== any LaTeX errors? ==="
grep -E "^!" sa-ai.log | head -5 || echo "(none)"
echo "=== undefined refs/citations? ==="
grep -ci undefined sa-ai.log || echo 0
