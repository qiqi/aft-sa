"""Shared path + canonical-constant plumbing for the repro/ figure scripts.

Puts the sa-ai package root AND the paper/ dir on sys.path, cd's to paper/ so
figures land in paper/figs/, and re-exports every canonical constant IMPORTED
from its src/ home (never restated here). Import this first from every script.
"""
import os
import sys

REPO = "/home/qiqi/flexcompute"
SAAI = os.path.join(REPO, "sa-ai")
PAPER = os.path.join(SAAI, "paper")

for p in (REPO, SAAI, PAPER):
    if p not in sys.path:
        sys.path.insert(0, p)

# Write figures to paper/figs/ regardless of the caller's cwd.
os.makedirs(os.path.join(PAPER, "figs"), exist_ok=True)
os.chdir(PAPER)

# --- canonical kernel constants (single source of truth, imported) ----------
from src.numerics.aft_sources import (          # noqa: E402
    AFT_RATE_SCALE   as A_MAX,        # a_max
    AFT_SIGMOID_CENTER as G_C,        # g_c
    AFT_SIGMOID_SLOPE  as S_SLOPE,    # s
    AFT_RE_OMEGA_FLOOR as FLOOR,      # ReOmega floor
    AFT_BARRIER_POWER  as P,          # p
    AFT_Q4_CA          as C_A,        # c_A
    AFT_CLIFF_LAMBDA_SLOPE as K_LAMBDA,
)
from scripts.calibrate_kernel import A_TU, B_TU, C_V1, chi_inf_from_Tu_pct  # noqa: E402
from src.solvers.boundary_layer_solvers import NuHatBlasiusSolver  # noqa: E402

# c_nu,ai (aft_nuLamScale) has no module constant; read it off the solver default.
C_NU_AI = NuHatBlasiusSolver().aft_nuLamScale
SIGMA_SA = 2.0 / 3.0

# tau / tau_D (production/destruction gate widths): their single source is the
# canonical wall-layer module, whose BVP driver is now guarded under main() so
# the import is cheap.
from regen_wall_layer import CNU as _CNU_wl, TAU, TAU_D  # noqa: E402
