"""Frozen-profile R_x for the ADOPTED composite gate (lv2, cV=4,
floor/gc/s = 243.7/0.9874/10.68): the LSB over-amplification check."""
import sys
import numpy as np
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa: F401
from _saai import C_NU_AI, SIGMA_SA
from lib.aft_sources import compute_aft_amplification_rate
from explore_lambda_v_anchors import gate_composite, gate_lambda_v
from explore_lsb_frozen_profile import build_profile, drela_spatial, PROFILES

NEW = dict(gw=4.0, gc=0.9874, s=10.68, floor=243.7)
OLD = dict(gw=1.0, gc=0.9676, s=13.62, floor=177.5)


def sigma_x(pr, Re_th, cnst, composite, ufrac=0.03):
    Th = pr['Theta']
    yh = pr['eta'][1:-1]/Th
    u = pr['u'][1:-1]
    w = Th*pr['up'][1:-1]
    dwdy = Th*Th*pr['upp'][1:-1]
    Gam = 2*(w*yh)**2/((w*yh)**2 + u**2 + 1e-300)
    rate = np.asarray(compute_aft_amplification_rate(
        np.abs(w)*yh**2*Re_th, Gam, lambda_p=0.0, sigmoid_center=cnst['gc'],
        sigmoid_slope=cnst['s'], re_omega_floor=cnst['floor'],
        barrier_power=4.0))
    if composite:
        q = gate_composite(w, dwdy, u, yh, cnst['gw'])
    else:
        q = gate_lambda_v(w, dwdy, u, yh, cnst['gw'])
    b = rate*q*np.abs(w)
    h = yh[1] - yh[0]
    k = (C_NU_AI/SIGMA_SA)/Re_th
    uf = np.maximum(u, ufrac)
    lam = eigh_tridiagonal((b - 2*k/h**2)/uf,
                           (k/h**2)/np.sqrt(uf[:-1]*uf[1:]), select='i',
                           select_range=(len(yh)-1, len(yh)-1),
                           eigvals_only=True)
    return float(lam[0])


print(f"{'H':>7} {'Re_th':>6} | {'lv (old)':>9} {'COMPOSITE':>10}   "
      "(R_x = dN/dx vs Drela, ufrac=0.03)")
for beta, guess in PROFILES:
    pr = build_profile(beta, guess)
    dS = drela_spatial(pr['H'])
    for Re_th in (200.0, 400.0):
        r0 = sigma_x(pr, Re_th, OLD, False)/dS
        r1 = sigma_x(pr, Re_th, NEW, True)/dS
        print(f"{pr['H']:7.3f} {Re_th:6.0f} | {r0:9.2f} {r1:10.2f}",
              flush=True)
