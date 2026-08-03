"""The retired TWO-SOURCE kernel evaluated on plane Poiseuille and Hagen-Poiseuille.

Kernel (git 2010535 'FPG floor Part III' + 4e8961c Part V + Part VIII 'vg'),
the form proposed just before the 2026-07-29 Drela meeting and parked after it:

  RATE (form 'zc'/'zb'):
      P_r = Ω̂ · sqrt( ⟨Î⟩₊²  +  (ε_r ⟨−Ẑ⟩₊)² ),      Ẑ = Z/R
      a   = a_max · clip(P_r, 0, 1)
      equivalently  a = sqrt( (a_inv Ω̂⟨Î⟩₊)² + (a_visc Ω̂⟨−Ẑ⟩₊)² )
      with a_inv = a_max = 0.19,  a_visc = ε_r a_max = 0.0276  (ε_r = 0.1455)

  GATE (form 'vg', two-branch, no constant ceiling C):
      Re_Ω^c = softmin_n( A + B/P_I² ,  A_c + B_c/P_curv² )
      P_I = Ω̂⟨Î⟩₊ ,  P_curv = Ω̂⟨−Ẑ⟩₊ ,  A_c = A = 124.6 , B_c = 130 (k-carrying)

WHY THIS MATTERS HERE: on an exactly parabolic profile Î ≡ 0, so the canonical
single-coordinate kernel returns exactly zero (see 01-*.md). But the parabola has
u'' < 0, hence −Z > 0, so the VISCOUS branch is alive. The two-source kernel is
therefore the one form that has anything to say about channel and pipe flow.

Both flows reduce to the SAME profile in wall units:
    u = 2s − s² ,  ω = |u'| = 2(1−s) ,  u''_wall-normal = −2 ,  s = d/h or d/R
(the pipe using the repro realization Z = ½d² n̂·∇ω, which equals ½d² u''.)
So the indicator profiles are identical; only the flow-Reynolds bookkeeping and
the diffusion operator differ.

Run: python3 -u blindspots/twosource_pipe_channel.py
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'paper', 'repro'))
from lib.sphere_kernel import (A_MAX, C_NU_AI, K_ANCHOR, RAMP_W, REOM_A,
                               REOM_B, REOM_N, SIGMA_SA)

EPS_R = 0.1455                 # ε_r  (a_visc = ε_r a_max = 0.0276)
A_VISC = EPS_R*A_MAX
# BC = 130.0 in fpg_recalibration_study.py is ALREADY k-carrying
# ("AC/BC are k-carrying (same scale as REOM_A/REOM_B)"), so do NOT rescale.
B_C = 130.0
A_C = REOM_A                   # net-zero option: share the floor


def profile(s):
    """u = 2s − s² in wall distance s; identical for channel and pipe."""
    u = 2.0*s - s*s
    om = 2.0*(1.0 - s)                 # |du/ds|
    upp = np.full_like(s, -2.0)        # wall-normal second derivative
    X = u
    Y = s*om
    Z = 0.5*s*s*upp                    # = −s²  (negative: favorable curvature)
    R = np.sqrt(X*X + Y*Y + Z*Z)
    Ohat = Y/np.sqrt(X*X + Y*Y)
    Ihat = (Y - X - Z)/R
    Zhat = Z/R
    return u, om, X, Y, Z, R, Ohat, Ihat, Zhat


def two_source(s):
    u, om, X, Y, Z, R, Ohat, Ihat, Zhat = profile(s)
    P_I = Ohat*np.clip(Ihat, 0.0, None)              # inflectional coordinate
    P_curv = Ohat*np.clip(-Zhat, 0.0, None)          # viscous coordinate
    P_r = Ohat*np.sqrt(np.clip(Ihat, 0, None)**2 + (EPS_R*np.clip(-Zhat, 0, None))**2)
    a = A_MAX*np.clip(P_r, 0.0, 1.0)
    b1 = REOM_A + REOM_B/np.maximum(P_I, 1e-300)**2
    b2 = A_C + B_C/np.maximum(P_curv, 1e-300)**2
    reomc = (b1**(-REOM_N) + b2**(-REOM_N))**(-1.0/REOM_N)
    return dict(u=u, om=om, P_I=P_I, P_curv=P_curv, P_r=P_r, a=a, reomc=reomc)


def gate(s, Re):
    """Re = U_c h/ν (channel) or U_bulk D/ν (pipe) — both equal 1/ν here."""
    k = two_source(s)
    ReOm = s*s*k['om']*Re
    return 0.5*(1.0 + np.tanh((ReOm/k['reomc'] - 1.0)/RAMP_W)), ReOm, k


def eigen(s_n, Re, axisym):
    """Leading s of [a S ω + (c/σ)ν ∇²] v = s u v. Returns e-folds per unit h (or R)."""
    nu = 1.0/Re
    ds = 1.0/s_n
    s = (np.arange(s_n) + 0.5)*ds
    g, _, k = gate(s, Re)
    src = k['a']*g*k['om']
    c = C_NU_AI*nu/SIGMA_SA
    # weight w = r for axisymmetric (r = 1 − s), 1 for planar
    w = (1.0 - s) if axisym else np.ones_like(s)
    wp = (1.0 - (s + 0.5*ds)) if axisym else np.ones_like(s)
    wm = (1.0 - (s - 0.5*ds)) if axisym else np.ones_like(s)
    wp = np.maximum(wp, 0.0)
    off = c*wp[:-1]/ds
    diag = np.empty(s_n)
    diag[0] = -c*wp[0]/ds                       # wall side: Dirichlet handled below
    diag[1:] = -c*(wp[1:] + wm[1:])/ds
    diag[0] -= c*wm[0]/ds                       # v = 0 at the wall (s = 0)
    diag += src*w*ds
    B = k['u']*w*ds
    isq = 1.0/np.sqrt(B)
    return eigh_tridiagonal(diag*isq*isq, off*isq[:-1]*isq[1:],
                            select='i', select_range=(s_n-1, s_n-1),
                            eigvals_only=True)[0]


def bisect(pred, lo=1e2, hi=1e7, it=90):
    for _ in range(it):
        m = np.sqrt(lo*hi)
        if pred(m):
            hi = m
        else:
            lo = m
    return hi


if __name__ == "__main__":
    print("two-source kernel: a_inv=%.3f  a_visc=%.4f (eps_r=%.4f)  "
          "A=%.1f B=%.3f  A_c=%.1f B_c=%.1f" %
          (A_MAX, A_VISC, EPS_R, REOM_A, REOM_B, A_C, B_C))
    print("profile u = 2s - s^2  (identical for channel and pipe in wall units)\n")

    s = np.linspace(1e-6, 1.0 - 1e-9, 400001)
    k = two_source(s)
    i = int(np.argmax(k['P_curv']))
    j = int(np.argmax(k['a']))
    print("=== 1. coordinates ===")
    print("  max P_I    (inflectional) = %.3e   <- exactly zero on a parabola"
          % k['P_I'].max())
    print("  max P_curv (viscous)      = %.5f at d/h = %.4f" % (k['P_curv'][i], s[i]))
    print("  max rate a                = %.6f at d/h = %.4f  (Blasius canon: %.5f)"
          % (k['a'][j], s[j], A_MAX*0.078))
    print("  ratio to Blasius canon rate = %.3f" % (k['a'][j]/(A_MAX*0.078)))
    print("  Re_Omega_c at the P_curv peak = %.1f" % k['reomc'][i])

    print("\n=== 2. profile table ===")
    print("    d/h      u       Ohat    P_curv    a        Re_Om_c")
    for dd in (0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = int(np.argmin(np.abs(s - dd)))
        print("   %5.2f  %6.3f  %7.4f  %7.4f  %8.5f  %9.1f"
              % (s[m], k['u'][m], k['P_curv'][m]/max(k['P_curv'][m], 1e-30)
                 * k['P_curv'][m], k['P_curv'][m], k['a'][m], k['reomc'][m]))

    print("\n=== 3. gate opening (Re = U_c h/nu = U_bulk D/nu = 1/nu) ===")
    for Re in (3e3, 5.772e3, 1e4, 1.35e4, 2e4, 5e4):
        g, ReOm, kk = gate(s, Re)
        m = int(np.argmax(kk['a']*g))
        print("  Re=%9.4g   peak a*S at d/h=%.3f : a=%.5f  Re_Om=%8.4g  "
              "Re_Om_c=%7.1f  gate=%.4f  a*S=%.6f"
              % (Re, s[m], kk['a'][m], ReOm[m], kk['reomc'][m], g[m],
                 (kk['a']*g)[m]))
    print("  gate reaches 0.5 anywhere at Re = %.4g"
          % bisect(lambda R: gate(s, R)[0].max() >= 0.5))

    print("\n=== 4. net growth (frozen eigenvalue, diffusion retained) ===")
    print("   Re          channel s*h    pipe s*R      pipe s*D")
    for Re in (5e3, 1e4, 1.35e4, 2e4, 5e4, 1e5, 1e6):
        sc = eigen(3000, Re, axisym=False)
        sp = eigen(3000, Re, axisym=True)
        print("  %9.4g   %+11.5g   %+11.5g   %+11.5g" % (Re, sc, sp, 2.0*sp))
    print("  channel net growth first at Re_c = %.4g"
          % bisect(lambda R: eigen(1500, R, axisym=False) > 0.0))
    print("  pipe    net growth first at Re_D = %.4g"
          % bisect(lambda R: eigen(1500, R, axisym=True) > 0.0))
    print("  grid check Re=2e4 channel:", ", ".join(
        "n=%d s=%.5g" % (n, eigen(n, 2e4, False)) for n in (750, 1500, 3000, 6000)))

    print("\n=== 5. e-folds banked over a development length ===")
    print("  channel: N over L/h ;  pipe: N over L/D")
    for Re in (1e4, 2e4, 5e4, 1e5):
        sc = eigen(3000, Re, False)
        sp = 2.0*eigen(3000, Re, True)
        print("   Re=%7.4g  channel s*h=%.5f -> L/h for N=9: %8.1f    "
              "pipe s*D=%.5f -> L/D for N=9: %8.1f"
              % (Re, sc, 9.0/sc if sc > 0 else np.inf,
                 sp, 9.0/sp if sp > 0 else np.inf))

    print("\n=== 6. reference points ===")
    print("  plane Poiseuille TRUE linear critical Re_c = 5772 (Orszag 1971)")
    print("  Hagen-Poiseuille TRUE: linearly STABLE at all Re")
    print("  pipe observed natural transition: Reynolds 1883 ~1.3e4;"
          " Ekman ~4.4e4; Pfenniger ~1e5")
