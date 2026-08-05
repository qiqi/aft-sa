"""Shared path + canonical-constant plumbing for the paper/repro figure
scripts. SELF-CONTAINED: everything imports from paper/repro/lib (a verbatim,
consistency-tested copy of the project kernel modules); no code outside
paper/repro is referenced. Figures land in paper/figs/.

The canonical amplification kernel is lib/sphere_kernel.py
(rate = a_max*clip(Shat*g), soft-min onset threshold, tanh ramp). The retired
Gamma-sigmoid + Q4/composite-gate kernel and its constants (g_c, s,
reOmegaFloor, p, c_A, c_V, c_2, K_lambda, K_r) are gone; ModelConstants.h keeps
their fields only for layout stability and old-run replay."""
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
from lib.sphere_kernel import (        # noqa: E402
    A_MAX,        # a_max
    C_NU_AI,      # c_nu,ai in the SA diffusion only
    SIGMA_SA,     # SA sigma
    REOM_CEIL,    # onset threshold: softmin_n(CEIL, A + B*P^-2) shape ...
    REOM_A,
    REOM_B,
    REOM_N,       # ... and its soft-min sharpness
    RAMP_W,       # onset tanh ramp width
    K_ANCHOR,     # whole-equation drain-compensation scale
)
from lib.calibrate_kernel import A_TU, B_TU, C_V1, chi_inf_from_Tu_pct  # noqa: E402

from lib.wall_layer import CNU as _CNU_wl, TAU, R_TIE  # noqa: E402
assert abs(_CNU_wl - C_NU_AI) < 1e-12, "wall-layer c_nu,ai != sphere-kernel c_nu,ai"
