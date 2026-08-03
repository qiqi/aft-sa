"""SA-AI kernel on fully-developed pipe and channel flow.

Question: the parabola great circle Ihat = 0 is the model's neutral locus, and
Hagen-Poiseuille / plane Poiseuille are EXACTLY parabolic in wall distance.
So what does the kernel read there?

Two realizations of the curvature indicator Z matter and they DIFFER in a pipe:

  planar     Z = 1/2 d^2 * d2u/dy^2                (the RP2 derivation, Sec. sphere)
  solver     Z = 1/2 d^2 * (lap u) . uhat          (Eq. gram; lap u = -curl omega)

For a planar parabola the two coincide and Ihat == 0 identically.
For a pipe the axisymmetric Laplacian carries the transverse-curvature term
  lap u_z = (1/r) d/dr (r du_z/dr) = -4 Uc/R^2      vs   d2u/dy^2 = -2 Uc/R^2
so the solver realization sees TWICE the curvature and Ihat departs zero with
no inflection point present.

Non-dimensionalisation: R = 1, Uc (centreline) = 1.
  Re_D = U_bulk D / nu = (Uc/2)(2R)/nu = Uc R / nu = 1/nu     ->  nu = 1/Re_D

Outputs
  1. Ihat, Omegahat, rate coordinate P = Omegahat*Ihat across the radius,
     both realizations, pipe and channel.
  2. Onset gate Re_Omega(r) vs Re_Omega_crit(P); the Re_D at which it opens.
  3. Frozen-profile eigenvalue (paper Eq. frozeneig, axisymmetric form)
        [ a(P) S omega + (c_nu_ai nu / sigma) lap ] v = s u v
     symmetric-tridiagonal, as in the paper's Table frozeneig.
     -> s = e-folds per unit R; N banked over L/D.

Run: python3 -u blindspots/pipe_kernel_analysis.py
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'paper', 'repro'))
from lib.sphere_kernel import (A_MAX, C_NU_AI, C_V1, K_ANCHOR, RAMP_W, REOM_A,
                               REOM_B, REOM_CEIL, SIGMA_SA, reom_crit)


# ---------------------------------------------------------------- profiles --
def pipe_profile(r):
    """Hagen-Poiseuille. Returns u, |du/dr|, lap_axisym, n.grad(omega), d.

    THREE candidate curvature realizations, all equal on a plane parallel
    layer, NOT equal on a curved wall:

      planar        d2u/dy2                        = -2 Uc/R^2
      n.grad(omega) n = grad(d) = -e_r; omega = |du/dr| = 2 Uc r/R^2
                    -> n.grad(omega) = -domega/dr  = -2 Uc/R^2   (== planar)
      Laplacian     (1/r) d/dr(r du/dr)            = -4 Uc/R^2   (factor 2)

    The repro post-processor (paper/repro/cfd/add_derived_to_slice.py:57) uses
    n.grad(omega); the paper's Appendix E writes lap(u).uhat. On a pipe these
    give OPPOSITE answers -- see the .md.
    """
    u = 1.0 - r**2
    dudr = -2.0*r                      # omega = |du/dr| = 2r  (with Uc=R=1)
    lap = np.full_like(r, -4.0)        # (1/r) d/dr (r du/dr), exact for HP
    # n.grad(omega) computed the solver's way: n = grad(d), d = 1-r  ->  n = -e_r
    omega = np.abs(dudr)
    domega_dr = np.full_like(r, 2.0)
    n_grad_omega = -domega_dr          # = -2, i.e. exactly d2u/dy2
    d = 1.0 - r
    return u, dudr, lap, n_grad_omega, d


def channel_profile(y):
    """Plane Poiseuille on y in (-1,1), wall distance d = 1-|y|."""
    u = 1.0 - y**2
    dudy = -2.0*y
    lap = np.full_like(y, -2.0)        # planar Laplacian == d2u/dy2
    d = 1.0 - np.abs(y)
    return u, dudy, lap, lap.copy(), d


# ------------------------------------------------------------- indicators --
def indicators(u, shear, curv, d):
    """Solver ratio realization (X,Y,Z) = (|u|, d|omega|, 1/2 d^2 curv.uhat)."""
    X = np.abs(u)
    Y = d*np.abs(shear)
    Z = 0.5*d*d*curv
    R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-300
    Ohat = Y/np.sqrt(X*X + Y*Y + 1e-300)
    Ihat = (Y - X - Z)/R
    return Ohat, Ihat, Ohat*Ihat


def gate_of(P, d, shear, nu):
    ReOm = d*d*np.abs(shear)/nu
    rc = reom_crit(np.maximum(P, 1e-12))
    return ReOm, rc, 0.5*(1.0 + np.tanh((ReOm/rc - 1.0)/RAMP_W))


# ------------------------------------------------------- frozen eigenvalue --
def pipe_eigen(Re_D, n=3000):
    """Leading s of [a S omega + (c/sigma) nu lap] v = s u v, axisymmetric.

    r-weighted finite volume -> symmetric tridiagonal generalized problem
    A v = s B v with B diagonal SPD; solved via C = B^-1/2 A B^-1/2.
    Returns (s in units of 1/R, max P, max gate).
    """
    nu = 1.0/Re_D
    dr = 1.0/n
    r = (np.arange(n) + 0.5)*dr
    u, dudr, lap, _, d = pipe_profile(r)
    _, _, P = indicators(u, dudr, lap, d)
    _, _, gate = gate_of(P, d, dudr, nu)
    src = A_MAX*np.clip(P, 0.0, 1.0)*gate*np.abs(dudr)      # a(P) S omega

    c = C_NU_AI*nu/SIGMA_SA
    rp, rm = r + 0.5*dr, r - 0.5*dr
    off = c*rp[:-1]/dr                                  # A[k,k+1] = A[k+1,k]
    diag = np.empty(n)
    diag[0] = -c*rp[0]/dr                               # symmetry at the axis
    diag[1:] = -c*(rp[1:] + rm[1:])/dr
    diag[-1] -= c*rp[-1]/dr                             # wall Dirichlet ghost
    diag += src*r*dr                                    # source, r-weighted

    B = u*r*dr
    isq = 1.0/np.sqrt(B)
    s = eigh_tridiagonal(diag*isq*isq, off*isq[:-1]*isq[1:],
                         select='i', select_range=(n-1, n-1),
                         eigvals_only=True)[0]
    return s, P.max(), gate.max()


def critical_ReD(pred, lo=1e2, hi=1e6, iters=80):
    for _ in range(iters):
        mid = np.sqrt(lo*hi)
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------------------------- main --
if __name__ == "__main__":
    print("SA-AI constants: a_max=%.2f  k=%.3f  Re_Om_ceil=%.1f  Re_Om_A=%.1f  "
          "Re_Om_B=%.3f  c_nu_ai=%.4f  ramp_w=%.2f  c_v1=%.1f"
          % (A_MAX, K_ANCHOR, REOM_CEIL, REOM_A, REOM_B, C_NU_AI, RAMP_W, C_V1))

    # ---- 1. plane Poiseuille -------------------------------------------
    print("\n=== 1. PLANE POISEUILLE (channel) ===")
    y = np.linspace(-0.9999, 0.9999, 20001)
    u, dudy, lap, _, d = channel_profile(y)
    _, Ihat, P = indicators(u, dudy, lap, d)
    print("  max |Ihat| over the section = %.3e" % np.abs(Ihat).max())
    print("  max |P|    over the section = %.3e" % np.abs(P).max())
    print("  -> the channel sits EXACTLY on the neutral circle at every Re.")
    print("     Truth: plane Poiseuille is linearly UNSTABLE (TS) at "
          "Re_c = 5772 (Orszag 1971). The model cannot see it.")

    # ---- 2. pipe indicators --------------------------------------------
    print("\n=== 2. HAGEN-POISEUILLE: indicators (Re-independent) ===")
    r = np.linspace(0.0, 0.99999, 200001)
    u, dudr, lap, ngo, d = pipe_profile(r)
    print("  curvature realizations at d/R=0.5:  n.grad(omega) = %.4f   "
          "lap(u) = %.4f   (ratio %.2f)"
          % (ngo[0], lap[0], lap[0]/ngo[0]))
    for tag, curv in (("REPRO   Z = 1/2 d^2 n.grad(w)", ngo),
                      ("APP-E   Z = 1/2 d^2 lap(u)   ", lap)):
        Oh, Ih, P = indicators(u, dudr, curv, d)
        i = int(np.argmax(P))
        print("  %s max P = %+.5f at d/R = %.4f  (Ohat %.4f, Ihat %+.4f)"
              % (tag, P[i], d[i], Oh[i], Ih[i]))
    Oh, Ih, P = indicators(u, dudr, lap, d)          # solver realization
    print("\n  solver-realization profile (P = Omegahat*Ihat):")
    print("    d/R      u       Ohat     Ihat       P")
    for dd in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        k = int(np.argmin(np.abs(d - dd)))
        print("   %5.2f  %6.3f  %7.4f  %+7.4f  %+7.4f"
              % (d[k], u[k], Oh[k], Ih[k], P[k]))
    print("  (Blasius peak for reference: max_y Omegahat*Ihat = 0.078)")

    # ---- 3. onset gate --------------------------------------------------
    print("\n=== 3. onset gate (solver realization) ===")
    for ReD in (1e3, 1.5e3, 2.04e3, 3e3, 1e4, 1e5):
        ReOm, rc, g = gate_of(P, d, dudr, 1.0/ReD)
        k = int(np.argmax(A_MAX*np.clip(P, 0, 1)*g))
        print("  Re_D=%8.4g  peak source at d/R=%.3f: P=%.4f  Re_Om=%8.4g  "
              "Re_Om_c=%7.4g  gate=%.4f  a*S=%.4g"
              % (ReD, d[k], P[k], ReOm[k], rc[k], g[k],
                 A_MAX*np.clip(P[k], 0, 1)*g[k]))
    f_open = lambda R: gate_of(P, d, dudr, 1.0/R)[2].max() >= 0.5
    print("  gate first reaches 0.5 anywhere in the section at Re_D = %.4g"
          % critical_ReD(f_open))

    # ---- 4. frozen eigenvalue -------------------------------------------
    print("\n=== 4. frozen-profile eigenvalue (net growth incl. diffusion) ===")
    print("   Re_D        s*R        s*D     L/D for N=9   L/D for N=6   gate")
    for ReD in (1e3, 1.5e3, 2.04e3, 3e3, 5e3, 1e4, 3e4, 1e5, 1e6):
        s, Pm, gm = pipe_eigen(ReD)
        sD = 2.0*s
        l9 = 9.0/sD if sD > 1e-12 else np.inf
        l6 = 6.0/sD if sD > 1e-12 else np.inf
        print("  %8.4g  %+10.4g  %+10.4g  %12.4g  %12.4g   %.3f"
              % (ReD, s, sD, l9, l6, gm))
    f_grow = lambda R: pipe_eigen(R, n=1500)[0] > 0.0
    print("  net growth (s>0) first at Re_D = %.4g" % critical_ReD(f_grow))

    # grid convergence of one entry
    print("  grid check at Re_D=1e4:", ", ".join(
        "n=%d s*D=%.5g" % (n, 2.0*pipe_eigen(1e4, n=n)[0])
        for n in (750, 1500, 3000, 6000)))

    # ---- 5. entrance-region competition ---------------------------------
    print("\n=== 5. entrance region vs fully developed ===")
    print("  laminar entrance length x_e/D ~ 0.05 Re_D; the wall BL there is")
    print("  Blasius-like, which the model DOES amplify in the normal way.")
    for Ncrit in (9.0, 7.0, 5.0):
        Re_theta = 1108.0 - (9.0 - Ncrit)/1.0e-2      # DG Blasius envelope
        Re_x = (Re_theta/0.664)**2
        ReD_cross = np.sqrt(Re_x/0.05)
        print("   N_crit=%4.1f: Re_theta_tr=%6.0f  Re_x_tr=%9.3g   "
              "entrance-BL route wins above Re_D = %.4g"
              % (Ncrit, Re_theta, Re_x, ReD_cross))
