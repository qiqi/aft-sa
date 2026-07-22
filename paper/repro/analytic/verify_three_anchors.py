"""Tier-2 verification: the canonical triple satisfies the three envelope
conditions (paper Sec. III.A).

Marches the disturbance transport (eq:transport) at the canonical constants
and checks, with the residuals PRINTED (they are the paper's quoted
"by construction" tolerance, not zero -- see weakness O1):

  1. Blasius envelope meets Drela-Giles at N=1  (Re_theta = 338):  within 3%
  2. Blasius envelope meets Drela-Giles at N=9  (Re_theta = 1108): within 1%
  3. separation-limit profile (beta = -0.1988, H = 3.98):
     onset-to-transition mean rate / Drela's rate = 1 within 1%
  4. Blasius interior deviation between the anchors: <= 5% (the quoted
     residual of the fully-determined kernel)

Note (weakness O1, documented): the three-anchor system has a shallow valley.
The canonical triple satisfies the conditions at (+2.2%, +0.2%, 1.000); a
tighter re-solve lands at (~252, ~1.020, ~9.9) with all residuals < 0.5% and
a 5-8% different unfitted mid-adverse response. This script VERIFIES the
canonical point rather than re-deriving it, so the reproduction is exact.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import _saai  # noqa: F401
from _saai import C_NU_AI, SIGMA_SA, A_MAX, G_C, S_SLOPE, FLOOR, P, C_A
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0
from lib.aft_sources import (compute_aft_amplification_rate,
                             compute_composite_gate)

BETA_SEP = -0.1988


def march(fs, x_max, nx=800, ny=600):
    eta99 = np.interp(0.99, fs.u, fs.eta)
    y_top = 8.0*eta99*np.sqrt(x_max/fs.inviscid_at(x_max))
    dy = y_top/ny; yc = (np.arange(ny) + 0.5)*dy; dx = x_max/nx
    nu = np.ones(ny); N = [0.0]; xs = [0.0]
    k = (C_NU_AI/SIGMA_SA)/dy**2
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, np.arange(ny + 1)*dy, cellCentered=True)
        u = np.maximum(u, 1e-12)
        vp = np.clip(v, 0, None)/dy; vm = np.clip(-v, 0, None)/dy
        di = vp + vm + 2*k; lo = -(vp[1:] + k); up = -(vm[:-1] + k)
        di[0] += k; di[-1] -= k
        rate = np.asarray(compute_aft_amplification_rate(
            yc**2*np.abs(dudy), 2*(dudy*yc)**2/(u**2 + (dudy*yc)**2)))
        q4 = compute_composite_gate(dudy, np.gradient(dudy, yc), u, yc)
        b = rate*q4*np.abs(dudy)
        main = u/dx + di; rhs = u/dx*nu + b*nu; rhs[-1] += vm[-1]
        A = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        xs.append((i + 1)*dx); N.append(float(np.log(max(nu.max(), 1e-300))))
    return np.array(xs), np.array(N)


def envelope(beta, x_max0):
    fs = FalknerSkanWedge(beta)
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    H = np.trapezoid(1 - fs.u, fs.eta)/I_th
    x_max = x_max0
    for _ in range(14):
        xs, N = march(fs, x_max)
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.2
            continue
        if N[-1] > 9.5:
            break
        x_max *= 2.5
    Ue = fs.inviscid_at(np.maximum(xs, 1e-12))
    return I_th*np.sqrt(xs*Ue), N, H


def main():
    print(f"canonical constants: a_max={A_MAX} g_c={G_C} s={S_SLOPE} "
          f"floor={FLOOR} p={P} c_A={C_A} c_nu_ai={C_NU_AI:.6f}")
    # Blasius
    Rt, N, Hb = envelope(0.0, 4e6)
    Rtc = float(Re_theta0(Hb)); slope = float(dN_dRe_theta(Hb))
    t1, t9 = Rtc + 1.0/slope, Rtc + 9.0/slope
    Rt1 = float(np.interp(1.0, N, Rt)); Rt9 = float(np.interp(9.0, N, Rt))
    r1, r9 = Rt1/t1 - 1.0, Rt9/t9 - 1.0
    m = (N >= 1.0) & (N <= 9.0)
    intdev = float(np.max(np.abs(N[m] - slope*(Rt[m] - Rtc)))/9.0)
    print(f"  Blasius N=1 anchor: Rt={Rt1:.0f} vs target {t1:.0f}  ({r1:+.1%})")
    print(f"  Blasius N=9 anchor: Rt={Rt9:.0f} vs target {t9:.0f}  ({r9:+.1%})")
    print(f"  Blasius interior deviation (N in [1,9]): {intdev:.1%}")
    # separation-limit profile
    Rts, Ns, Hs = envelope(BETA_SEP, 1.2e6)
    d_sep = float(dN_dRe_theta(Hs))
    R1 = float(np.interp(1.0, Ns, Rts)); R9 = float(np.interp(9.0, Ns, Rts))
    ratio = (8.0/(R9 - R1))/d_sep
    print(f"  separation-limit (H={Hs:.2f}) mean/Drela: {ratio:.4f}")
    assert abs(r1) < 0.03, f"N=1 anchor residual {r1:+.1%} exceeds 3%"
    assert abs(r9) < 0.01, f"N=9 anchor residual {r9:+.1%} exceeds 1%"
    assert abs(ratio - 1.0) < 0.01, f"separation anchor {ratio:.4f} off by >1%"
    assert intdev <= 0.055, f"interior deviation {intdev:.1%} exceeds 5.5%"
    print("  OK: the canonical triple satisfies the three envelope conditions "
          "(residuals as quoted).")


if __name__ == '__main__':
    main()
