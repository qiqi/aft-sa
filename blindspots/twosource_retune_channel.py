"""Retune the viscous branch on PLANE CHANNEL, and compare coupled vs decoupled.

Two questions (user, 2026-08-02):

  Q1. Can the viscous branch be anchored on plane Poiseuille (Re_c = 5772)
      instead of the low-H Falkner-Skan family, and does that land near the
      FS-tuned (a_visc, B_c) = (0.0276, 130)?

  Q2. Does it make more sense to DECOUPLE the two mechanisms,
        source = softmax_2( a_inv P_I S_inv ,  a_visc P_curv S_visc )
      each rate carrying its OWN onset gate, instead of the current
        source = a_max softmax_2(P_I, eps_r P_curv) * S(softmin(thr_inv, thr_visc))
      which soft-maxes the rates and soft-mins the thresholds?

Coordinates (both structures):
    P_I    = Ohat <Ihat>_+                    inflectional (inviscid)
    P_curv = Ohat <-Zhat>_+                   curvature    (viscous)
    thr_inv  = A   + B  /P_I^2                 no ceiling
    thr_visc = A_c + B_c/P_curv^2 ,  A_c = A   (net-zero: C is dropped)

Constant count vs canon: drop C (1851.2), add B_c -> net 0 on the gate;
a_visc is the single new constant.

Run: python3 -u blindspots/twosource_retune_channel.py
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh_tridiagonal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'paper', 'repro'))
from lib.boundary_layer import FalknerSkanWedge
from lib.sphere_kernel import (A_MAX, C_NU_AI, RAMP_W, REOM_A, REOM_B, REOM_N,
                               SIGMA_SA)

RE_C_CHANNEL_TRUE = 5772.22          # Orszag 1971, U_c h / nu
FS_TUNED = (0.0276, 130.0)           # (a_visc, B_c) from commit 2010535 / Part VIII


# ------------------------------------------------------------- coordinates --
def coords(u, om, upp, d):
    """P_I, P_curv from (u, |omega|, wall-normal u'', wall distance)."""
    X, Y, Z = np.abs(u), d*np.abs(om), 0.5*d*d*upp
    R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-300
    Ohat = Y/(np.sqrt(X*X + Y*Y) + 1e-300)
    Ihat = (Y - X - Z)/R
    return Ohat*np.clip(Ihat, 0, None), Ohat*np.clip(-Z/R, 0, None)


def thresholds(P_I, P_curv, B_c, A_c=REOM_A):
    ti = REOM_A + REOM_B/np.maximum(P_I, 1e-300)**2
    tv = A_c + B_c/np.maximum(P_curv, 1e-300)**2
    return ti, tv


def S(ReOm, thr):
    return 0.5*(1.0 + np.tanh((ReOm/thr - 1.0)/RAMP_W))


def source(P_I, P_curv, ReOm, a_visc, B_c, mode):
    """Dimensionless a*S (multiply by omega for the production rate)."""
    ti, tv = thresholds(P_I, P_curv, B_c)
    if mode == 'coupled':                      # current form
        thr = (ti**(-REOM_N) + tv**(-REOM_N))**(-1.0/REOM_N)      # softmin
        a = np.sqrt((A_MAX*P_I)**2 + (a_visc*P_curv)**2)          # softmax rates
        return a*S(ReOm, thr)
    if mode == 'decoupled':                    # proposed form, C dropped
        s_i = A_MAX*P_I*S(ReOm, ti)
        s_v = a_visc*P_curv*S(ReOm, tv)
        return np.sqrt(s_i**2 + s_v**2)                            # softmax sources
    if mode == 'decoupled_ceil':               # proposed form, canon C retained
        C = 2600.0*0.712
        ti_c = (ti**(-REOM_N) + C**(-REOM_N))**(-1.0/REOM_N)
        s_i = A_MAX*P_I*S(ReOm, ti_c)          # == canon inviscid source exactly
        s_v = a_visc*P_curv*S(ReOm, tv)
        return np.sqrt(s_i**2 + s_v**2)
    if mode == 'canon':
        return A_MAX*P_I*S(ReOm, (( (REOM_A+REOM_B/np.maximum(P_I,1e-300)**2)**(-REOM_N)
                                    + (2600*0.712)**(-REOM_N))**(-1.0/REOM_N)))
    raise ValueError(mode)


# ------------------------------------------------- channel frozen eigenvalue --
def channel_growth(Re, a_visc, B_c, mode='decoupled', n=1500):
    """e-folds per unit h. Profile u = 2s - s^2, s = wall distance / h."""
    ds = 1.0/n
    s = (np.arange(n) + 0.5)*ds
    u, om, upp = 2*s - s*s, 2*(1 - s), np.full(n, -2.0)
    P_I, P_curv = coords(u, om, upp, s)
    ReOm = s*s*om*Re
    src = source(P_I, P_curv, ReOm, a_visc, B_c, mode)*om
    c = C_NU_AI/(Re*SIGMA_SA)
    off = np.full(n-1, c/ds)
    diag = np.full(n, -2.0*c/ds)
    diag[-1] = -c/ds                       # symmetry at the centreline
    diag += src*ds
    B = u*ds
    isq = 1.0/np.sqrt(B)
    return eigh_tridiagonal(diag*isq*isq, off*isq[:-1]*isq[1:],
                            select='i', select_range=(n-1, n-1),
                            eigvals_only=True)[0]


def channel_Rec(a_visc, B_c, mode='decoupled', lo=1e3, hi=1e6):
    for _ in range(60):
        m = np.sqrt(lo*hi)
        if channel_growth(m, a_visc, B_c, mode) > 0:
            hi = m
        else:
            lo = m
    return hi


def Bc_for_target(a_visc, mode='decoupled', target=RE_C_CHANNEL_TRUE):
    """B_c that puts the channel's neutral point on `target`."""
    lo, hi = 1.0, 1e5                      # larger B_c -> higher threshold -> later
    for _ in range(50):
        m = np.sqrt(lo*hi)
        if channel_Rec(a_visc, m, mode) > target:
            hi = m
        else:
            lo = m
    return hi


# ------------------------------------------------------------ FS diagnostics --
def fs_profile(beta, Re_theta, ny=4000, ymax=12.0):
    """Return (u, omega, u'', d) on a wedge at the given Re_theta, edge-normalised."""
    w = FalknerSkanWedge(beta)
    # march Rex until momentum thickness matches Re_theta (edge units)
    def theta_of(Rex):
        yg = np.linspace(0, ymax*np.sqrt(Rex)/np.sqrt(w.inviscid_at(Rex)), ny+1)
        y, u, dudy, _ = w.at(Rex, yg)
        ue = u[-1]
        return np.trapezoid(u/ue*(1 - u/ue), y)*ue, y, u, dudy, ue
    lo, hi = 1e2, 1e12
    for _ in range(80):
        m = np.sqrt(lo*hi)
        if theta_of(m)[0] > Re_theta:
            hi = m
        else:
            lo = m
    _, y, u, dudy, ue = theta_of(hi)
    upp = np.gradient(dudy, y)
    return u, np.abs(dudy), upp, y, ue


if __name__ == "__main__":
    print("channel target Re_c = %.0f (Orszag 1971);  FS-tuned (a_visc,B_c) = "
          "(%.4f, %.0f)\n" % (RE_C_CHANNEL_TRUE, *FS_TUNED))

    print("=== Q1: channel-anchored locus  (a_visc, B_c) giving Re_c = 5772 ===")
    print("  NOTE: on a parabola P_I == 0 exactly, so the inviscid branch is")
    print("        inert and COUPLED and DECOUPLED coincide here -> the channel")
    print("        anchors the viscous branch with zero contamination.\n")
    print("   a_visc    B_c(channel-anchored)     Re_c at FS-tuned B_c=130")
    for av in (0.015, 0.020, 0.0276, 0.035, 0.045, 0.060, 0.19):
        bc = Bc_for_target(av)
        rc130 = channel_Rec(av, 130.0)
        print("   %6.4f   %14.1f        %14.0f" % (av, bc, rc130))

    bc_star = Bc_for_target(FS_TUNED[0])
    print("\n  At the FS-tuned a_visc = %.4f the channel wants B_c = %.1f"
          "  (FS gave %.0f, ratio %.2f)"
          % (FS_TUNED[0], bc_star, FS_TUNED[1], bc_star/FS_TUNED[1]))
    print("  Conversely, at the FS-tuned B_c = 130 the channel neutral point is"
          " Re_c = %.0f (%.2fx the true 5772)"
          % (channel_Rec(*FS_TUNED), channel_Rec(*FS_TUNED)/RE_C_CHANNEL_TRUE))

    print("\n=== Q2: coupled vs decoupled on Falkner-Skan ===")
    print("  How much does each structure perturb the EXISTING calibration?")
    print("  (canon = single-source kernel actually in the paper)\n")
    print("  beta   Re_th   maxP_I  maxP_cv  thr_inv  thr_visc |  canon    coupled  "
          "decoupled |  cpl/canon  dec/canon")
    for beta, Re_th in ((0.0, 500.0), (0.0, 1000.0), (-0.10, 400.0),
                        (0.10, 1000.0), (0.20, 2000.0), (0.35, 4000.0)):
        try:
            u, om, upp, y, ue = fs_profile(beta, Re_th)
        except Exception as e:
            print("  beta=%+.2f  FS solve failed: %s" % (beta, e)); continue
        d = y
        P_I, P_curv = coords(u, om, upp, d)
        # Re_Omega in edge units: nu = 1 in the marched variables
        ReOm = d*d*om
        r = {m: source(P_I, P_curv, ReOm, *FS_TUNED, m).max()
             for m in ('canon', 'coupled', 'decoupled')}
        i, j = int(np.argmax(P_I)), int(np.argmax(P_curv))
        ti, tv = thresholds(P_I, P_curv, FS_TUNED[1])
        print("  %+.2f  %6.0f  %6.4f  %6.4f  %8.1f %9.1f | %8.5f %8.5f %8.5f |"
              "  %7.3f   %7.3f"
              % (beta, Re_th, P_I[i], P_curv[j], ti[i], tv[j],
                 r['canon'], r['coupled'], r['decoupled'],
                 r['coupled']/max(r['canon'], 1e-12),
                 r['decoupled']/max(r['canon'], 1e-12)))

    print("\n  Blasius is the k anchor: the closer dec/canon is to 1.0 there,")
    print("  the less the existing k = 0.712 has to move when the branch is added.")
