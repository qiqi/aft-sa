"""Prototype test of the corridor damping factor (user-proposed window:
Gamma slightly above 1, Lambda_v negative; Gamma=2 explicitly PROTECTED
because a free shear layer spans the whole Lambda_v axis at Gamma ~ 2):

    Q_mod = Q_v * [1 - A * s(Gamma) * f((-Lambda_v)_+)],
    s(Gamma) = [4 (Gamma-1)(2-Gamma)]_+^p     (zero at Gamma<=1 and Gamma=2)
    f(x)     = x^2 / (x^2 + x0^2)

Measured on the frozen-profile eigen instruments (explore_lsb_frozen_profile):
  (a) does it kill the spurious spatial mode of the convective separated
      profiles (R_x = dN/dx vs Drela)?
  (b) what does it cost the attached calibration profiles (Blasius is
      sign-protected: Lambda_v >= 0 everywhere; the separation-limit and
      mid-adverse profiles have sub-inflection Lambda_v < 0 excursions
      inside the corridor) -- raw march mean-rate ratios, NO re-anchoring.
"""
import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa: F401
from _saai import C_NU_AI, SIGMA_SA
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0
from lib.aft_sources import compute_aft_amplification_rate
from explore_lsb_frozen_profile import (build_profile, drela_spatial,
                                        PROFILES, GATES)

# Lambda_v-branch kernel constants (anchors re-solved, phase A/B)
LV = dict(gw=1.0, gc=0.9676, s=13.62, floor=177.5, KL=7.44, KR=10.0)
P_BUMP, X0, AMP = 1.0, 0.3, 1.0
MODE = 'corridor'      # set by main() from argv
Q2_L0, Q2_C2 = -2.0, 4.0   # 'q2' mode: second pinch center and width


def damp(Gam, Lv):
    x = np.maximum(-Lv, 0.0)
    if MODE == 'q2':
        # SECOND pinch (user design): Q = Q1 * Q2. Q1 (the existing gate)
        # protects the wall zero-advection locus (Gamma, Lambda_v) = (1, 0);
        # Q2 protects the recirculation's interior zero-advection locus at
        # ~ (2, -2):  Q2 = 1 - (Gamma-1)_+^2 / (1 + c2 |Lambda_v - L0|^2).
        # Exactly 1 for Gamma <= 1 (anchors structurally untouched); a free
        # shear layer meets the dip only on a slab of thickness ~ w/d.
        s = np.clip(Gam - 1.0, 0.0, None)**2
        return 1.0 - s/(1.0 + Q2_C2*np.abs(Lv - Q2_L0)**2)
    if MODE == 'corridor':
        # corridor bump: vanishes at Gamma<=1 AND Gamma=2 (hard Gamma=2
        # protection); f is a one-sided sigmoid in the negative-Lv depth.
        s = np.clip(4.0*(Gam - 1.0)*(2.0 - Gam), 0.0, None)**P_BUMP
        s = np.where(Gam > 1.0, s, 0.0)
        f = x*x/(x*x + X0*X0)
    else:
        # 'bandpass': s rises above Gamma ~ 1.65 up to Gamma=2 (sparing the
        # separation-limit anchor whose excursion tops at Gamma=1.58); free
        # shear is protected on the Lambda_v AXIS instead: far from walls
        # |Lambda_v| ~ |u''|d/u' is either ~0 (layer center sliver) or LARGE
        # (d -> inf), so a band-pass in -Lv (damp 0.3 <~ -Lv <~ 4, reopen
        # beyond) leaves it open, while the bubble's u=0 slab (finite d,
        # -Lv ~ 0.8-2.3 observed) is squarely inside the band. Absolutely
        # unstable profiles reopen by SIGN (Lv >= 0 at the crossing).
        s = np.clip((Gam - 1.65)/0.35, 0.0, 1.0)**2
        f = x*x/(x*x + X0*X0)/(1.0 + (x/4.0)**4)
    return 1.0 - AMP*s*f


def kernel_b_mod(pr, Re_th, damped):
    Th = pr['Theta']
    yh = pr['eta'][1:-1]/Th
    u = pr['u'][1:-1]
    w = Th*pr['up'][1:-1]
    dwdy = Th*Th*pr['upp'][1:-1]
    absw = np.abs(w)
    den = (w*yh)**2 + u**2 + 1e-300
    Gam = 2*(w*yh)**2/den
    Lv_ = -w*dwdy*yh**3/den                      # signed, flipped convention
    rate = np.asarray(compute_aft_amplification_rate(
        absw*yh**2*Re_th, Gam, lambda_p=0.0, sigmoid_center=LV['gc'],
        sigmoid_slope=LV['s'], re_omega_floor=LV['floor'], barrier_power=4.0))
    if damped and MODE == 'plus':
        # one-sided gate: |Lambda_v| -> max(0, Lambda_v) in the denominator
        term = LV['gw']*np.maximum(-w*dwdy, 0.0)*yh**3
    else:
        term = LV['gw']*np.abs(w*dwdy)*yh**3
    Q = 1.0 - 2.0*absw*yh*np.abs(u)/(term + (w*yh)**2 + u**2 + 1e-300)
    if damped and MODE != 'plus':
        Q = Q*damp(Gam, Lv_)
    return yh, u, rate*Q*absw


def sigma_spatial_mod(pr, Re_th, damped, ufrac=0.03):
    yh, u, b = kernel_b_mod(pr, Re_th, damped)
    h = yh[1] - yh[0]
    k = (C_NU_AI/SIGMA_SA)/Re_th
    uf = np.maximum(u, ufrac)
    d = (b - 2.0*k/h**2)/uf
    e = (k/h**2)/np.sqrt(uf[:-1]*uf[1:])
    lam = eigh_tridiagonal(d, e, select='i',
                           select_range=(len(yh)-1, len(yh)-1),
                           eigvals_only=True)
    return float(lam[0])


def march(fs, x_max, beta, damped, nx=800, ny=600, seed=1.0, ufrac=0.0):
    m = beta/(2.0 - beta)
    eta99 = np.interp(0.99, np.maximum.accumulate(fs.u), fs.eta)
    y_top = 8.0*eta99*np.sqrt(x_max/fs.inviscid_at(x_max))
    dy = y_top/ny
    yc = (np.arange(ny) + 0.5)*dy
    dx = x_max/nx
    nu = np.ones(ny)*seed
    N = [0.0]
    xs = [0.0]
    k = (C_NU_AI/SIGMA_SA)/dy**2
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, np.arange(ny + 1)*dy, cellCentered=True)
        u = np.maximum(u, max(1e-12, ufrac*fs.inviscid_at(x)))
        vp = np.clip(v, 0, None)/dy
        vm = np.clip(-v, 0, None)/dy
        di = vp + vm + 2*k
        lo = -(vp[1:] + k)
        up_ = -(vm[:-1] + k)
        di[0] += k
        di[-1] -= k
        lam = m*yc**2*fs.inviscid_at(x)**2/(x*u)
        den = (dudy*yc)**2 + u**2 + 1e-300
        Gam = 2*(dudy*yc)**2/den
        rate = np.asarray(compute_aft_amplification_rate(
            yc**2*np.abs(dudy), Gam, lambda_p=lam,
            sigmoid_center=LV['gc'], sigmoid_slope=LV['s'],
            re_omega_floor=LV['floor'], barrier_power=4.0,
            cliff_lambda_slope=LV['KL'], fpg_rate_slope=LV['KR']))
        upp = np.gradient(dudy, yc)
        Lv_ = -dudy*upp*yc**3/den
        if damped and MODE == 'plus':
            term = LV['gw']*np.maximum(-dudy*upp, 0.0)*yc**3
        else:
            term = LV['gw']*np.abs(dudy*upp)*yc**3
        q = 1.0 - 2.0*np.abs(dudy)*yc*np.abs(u)/(
            term + (dudy*yc)**2 + u**2 + 1e-300)
        if damped and MODE != 'plus':
            q = q*damp(Gam, Lv_)
        b = rate*q*np.abs(dudy)
        main = u/dx + di
        rhs = u/dx*nu + b*nu
        rhs[-1] += vm[-1]*seed
        A = sp.diags([lo, main, up_], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        xs.append((i + 1)*dx)
        N.append(float(np.log(max(nu.max()/seed, 1e-300))))
    return np.array(xs), np.array(N)


def mean_ratio(beta, damped, guess=None, ufrac=0.0):
    fs = FalknerSkanWedge(beta, guess=guess)
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    H = np.trapezoid(1 - fs.u, fs.eta)/I_th
    x_max = 4e6 if beta == 0.0 else (3e5 if beta > 0 else 1.2e6)
    for _ in range(12):
        xs, N = march(fs, x_max, beta, damped, ufrac=ufrac)
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.15
            continue
        if N[-1] > 14.0:
            x_max = 1.1*float(np.interp(14.0, N, xs))
            xs, N = march(fs, x_max, beta, damped, ufrac=ufrac)
            break
        x_max *= 3.0
    Ue = fs.inviscid_at(np.maximum(xs, 1e-12))
    Rt = I_th*np.sqrt(xs*Ue)
    Rt1 = float(np.interp(1.0, N, Rt))
    Rt9 = float(np.interp(9.0, N, Rt))
    return H, (8.0/(Rt9 - Rt1))/float(dN_dRe_theta(H))


def main():
    global MODE, Q2_L0, Q2_C2
    args = [a for a in sys.argv[1:] if a != '--frozen-only']
    frozen_only = '--frozen-only' in sys.argv
    if args:
        MODE = args[0]
    if len(args) > 2:
        Q2_L0, Q2_C2 = float(args[1]), float(args[2])
    print(f"damping mode: {MODE}"
          + (f"  (L0={Q2_L0}, c2={Q2_C2})" if MODE == 'q2' else "") + "\n")
    print("(a) frozen separated profiles: R_x = dN/dx / Drela (ufrac=0.03)")
    print(f"{'H':>7} {'Re_th':>6} | {'undamped':>9} {'damped':>8}")
    for beta, guess in PROFILES:
        pr = build_profile(beta, guess)
        dS = drela_spatial(pr['H'])
        for Re_th in (200.0, 400.0):
            r0 = sigma_spatial_mod(pr, Re_th, False)/dS
            r1 = sigma_spatial_mod(pr, Re_th, True)/dS
            print(f"{pr['H']:7.3f} {Re_th:6.0f} | {r0:9.2f} {r1:8.2f}",
                  flush=True)

    if frozen_only:
        return
    print("\n(b) attached calibration marches, mean rate / Drela "
          "(RAW, no re-anchor):")
    print(f"{'beta':>8} | {'undamped':>9} {'damped':>8}")
    for beta, guess, uf in ((0.0, None, 0.0), (-0.09, None, 0.0),
                            (-0.15, None, 0.0), (-0.1988, None, 0.0),
                            (-0.17, -0.06, 0.03)):
        H0, m0 = mean_ratio(beta, False, guess=guess, ufrac=uf)
        H1, m1 = mean_ratio(beta, True, guess=guess, ufrac=uf)
        tag = ' (rev, H=%.2f)' % H0 if guess else ''
        print(f"{beta:+8.4f} | {m0:9.2f} {m1:8.2f}{tag}", flush=True)


if __name__ == '__main__':
    main()
