"""Shared path + canonical-constant plumbing for the paper/repro figure
scripts. SELF-CONTAINED: everything imports from paper/repro/lib (a verbatim,
consistency-tested copy of the project kernel modules); no code outside
paper/repro is referenced. Figures land in paper/figs/."""
import os
import sys

REPRO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # paper/repro
PAPER = os.path.dirname(REPRO)                                        # paper/
for p in (REPRO, os.path.join(REPRO, "analytic")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.makedirs(os.path.join(PAPER, "figs"), exist_ok=True)
os.chdir(PAPER)

# --- canonical kernel constants (from the repro-local copy of the kernel) ---
from lib.aft_sources import (          # noqa: E402
    AFT_RATE_SCALE   as A_MAX,        # a_max
    AFT_SIGMOID_CENTER as G_C,        # g_c
    AFT_SIGMOID_SLOPE  as S_SLOPE,    # s
    AFT_RE_OMEGA_FLOOR as FLOOR,      # ReOmega floor
    AFT_BARRIER_POWER  as P,          # p
    AFT_Q4_CA          as C_A,        # c_A (legacy Q4 gate, modes <= 4)
    AFT_LV_CV          as C_V,        # c_V: Lambda_v weight (FINAL gate)
    AFT_Q2_C2          as C_2,        # c_2: Q2 release softness (FINAL gate)
    AFT_CLIFF_LAMBDA_SLOPE as K_LAMBDA,
    AFT_FPG_RATE_SLOPE as K_R,
)
from lib.calibrate_kernel import A_TU, B_TU, C_V1, chi_inf_from_Tu_pct  # noqa: E402
from lib.boundary_layer_solvers import NuHatBlasiusSolver  # noqa: E402

C_NU_AI = NuHatBlasiusSolver().aft_nuLamScale
SIGMA_SA = 2.0 / 3.0

from lib.wall_layer import CNU as _CNU_wl, TAU, R_TIE  # noqa: E402
assert abs(_CNU_wl - C_NU_AI) < 1e-12, "wall-layer c_nu,ai != solver default"
