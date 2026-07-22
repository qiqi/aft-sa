"""Briggs-Bers pinch-point instrument, SHOOTING formulation (task #59).

The FD-eigenvalue approach failed structurally: the discretized Rayleigh
operator carries a cloud of spurious eigenvalues (the finite-grid stand-in
for the continuous spectrum) and eigenvalue tracking hops branches as
complex alpha moves, so omega(alpha) was effectively non-smooth. Here the
dispersion relation is a SHOOTING function instead:

    integrate the Rayleigh ODE  phi'' = [alpha^2 + U''/(U - c)] phi
    from the far field (decaying branch phi ~ e^{-alpha y}) down to the
    wall;  D(alpha, c) = phi(wall).

D is analytic in BOTH arguments by construction (ODE solutions depend
analytically on parameters), so:
  * c(alpha) from Newton on D = 0 (warm-started) is smooth,
  * omega(alpha) = alpha c(alpha) admits honest FD derivatives,
  * the saddle Newton d(omega)/d(alpha) = 0 converges,
  * Im(omega_0) at the saddle is the absolute growth rate.

Numerical care:
  * the integration path dips BELOW the real axis through the shear layer
    (Lin's rule for U' > 0 critical layers), keeping near-neutral cases
    regular; the path returns to the real axis at both ends;
  * tabulated profiles (reversed Falkner-Skan) are Chebyshev-continued so
    U, U'' are evaluable at complex y (analytic profiles use their formula);
  * free shear layers (no wall) use two-sided shooting and a Wronskian:
    D = phi+ phi-' - phi+' phi-  at the match point.

Validation anchor: the Huerre-Monkewitz (1985) counterflow mixing layer,
threshold velocity ratio R = 1.315 (13.6% counterflow).
Primary use: verify the Avanci geometric labels on the exact profile
families behind the Q2 pocket calibration (explore_q2_systematic.py).
"""
import os
import sys
import numpy as np
from numpy.polynomial import chebyshev as C

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa: F401

FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_explore')
NSTEP = 2400
DIP = 0.35              # imaginary depth of the contour through the layer


# ---------------------------------------------------------------- profiles
class AnalyticProfile:
    """U given by a formula valid at complex y. wall=True: domain [0, ymax];
    wall=False: free layer on [-ymax, ymax]."""

    def __init__(self, U, Upp, ymax, wall=True):
        self.U, self.Upp, self.ymax, self.wall = U, Upp, ymax, wall


def tanh_wall(h, ur, dw=1.0, ymax=None):
    ymax = ymax or max(h + 14.0, 24.0)

    def U(y):
        return (0.5*(1 - ur) + 0.5*(1 + ur)*np.tanh(y - h)) \
            * (1.0 - np.exp(-(y/dw)**2))

    def Upp(y):
        t = np.tanh(y - h)
        w = 1.0 - np.exp(-(y/dw)**2)
        core = 0.5*(1 - ur) + 0.5*(1 + ur)*t
        d1c = 0.5*(1 + ur)*(1 - t*t)
        d2c = -(1 + ur)*t*(1 - t*t)
        d1w = (2*y/dw**2)*np.exp(-(y/dw)**2)
        d2w = (2/dw**2 - 4*y*y/dw**4)*np.exp(-(y/dw)**2)
        return d2c*w + 2*d1c*d1w + core*d2w
    return AnalyticProfile(U, Upp, ymax, wall=True)


def hm85_layer(Umean, ymax=18.0):
    """u = Umean + tanh(y): velocity ratio R = 1/Umean; HM85 threshold
    R = 1.315 <=> Umean = 0.7605."""
    return AnalyticProfile(lambda y: Umean + np.tanh(y),
                           lambda y: -2*np.tanh(y)/np.cosh(y)**2,
                           ymax, wall=False)


def cheb_profile(y_tab, u_tab, upp_tab, ymax):
    """Chebyshev continuation of a TABULATED profile (analytic in reality,
    e.g. Falkner-Skan): fit U and U'' separately, evaluable at complex y
    within the fit's Bernstein ellipse (dip DIP << domain size)."""
    fU = C.Chebyshev.fit(y_tab, u_tab, 140, domain=[0, ymax])
    fUpp = C.Chebyshev.fit(y_tab, upp_tab, 140, domain=[0, ymax])
    return AnalyticProfile(lambda y: fU(y), lambda y: fUpp(y), ymax,
                           wall=True)


# ---------------------------------------------------------------- shooting
def _path(a, b, n, dip):
    """Complex contour from y=a to y=b (real endpoints) dipping -i*dip in
    the middle; returns nodes and dy of a straight-segment polyline."""
    s = np.linspace(0.0, 1.0, n + 1)
    y = a + (b - a)*s
    bump = np.sin(np.pi*s)**2
    y = y - 1j*dip*bump
    return y


def _integrate(pr, alpha, c, a, b, phi0, dphi0, n=NSTEP, dip=DIP):
    """RK4 for phi'' = [alpha^2 + U''/(U-c)] phi along the dipped contour."""
    y = _path(a, b, n, dip)
    phi, dphi = complex(phi0), complex(dphi0)

    def q(yy):
        return alpha*alpha + pr.Upp(yy)/(pr.U(yy) - c)

    for k in range(n):
        h = y[k+1] - y[k]
        y0, y1 = y[k], y[k] + 0.5*h
        q0, q1, q2 = q(y0), q(y1), q(y[k+1])
        k1p, k1d = dphi, q0*phi
        k2p, k2d = dphi + 0.5*h*k1d, q1*(phi + 0.5*h*k1p)
        k3p, k3d = dphi + 0.5*h*k2d, q1*(phi + 0.5*h*k2p)
        k4p, k4d = dphi + h*k3d, q2*(phi + h*k3p)
        phi = phi + h/6*(k1p + 2*k2p + 2*k3p + k4p)
        dphi = dphi + h/6*(k1d + 2*k2d + 2*k3d + k4d)
    return phi, dphi


def dispersion(pr, alpha, c):
    """D(alpha, c): wall -> phi(0) from downward shooting; free layer ->
    Wronskian of the two decaying branches at the center."""
    if pr.wall:
        phi, dphi = _integrate(pr, alpha, c, pr.ymax, 0.0,
                               1.0, -alpha)          # decaying branch
        return phi
    phiR, dphiR = _integrate(pr, alpha, c, pr.ymax, 0.0, 1.0, -alpha)
    phiL, dphiL = _integrate(pr, alpha, c, -pr.ymax, 0.0, 1.0, +alpha,
                             dip=-DIP)
    return phiR*dphiL - dphiR*phiL


def c_solve(pr, alpha, c0, tol=1e-11, itmax=25):
    """Newton on D(alpha, c) = 0 in c, warm-started."""
    c = complex(c0)
    for _ in range(itmax):
        D = dispersion(pr, alpha, c)
        dc = 1e-5*max(abs(c.imag), 1e-2)
        Dp = dispersion(pr, alpha, c + dc)
        g = (Dp - D)/dc
        if abs(g) < 1e-300:
            break
        step = -D/g
        if abs(step) > 0.2:
            step *= 0.2/abs(step)
        c += step
        if abs(step) < tol:
            break
    return c


def omega(pr, alpha, c0):
    c = c_solve(pr, alpha, c0)
    return alpha*c, c


def saddle(pr, a0, c0, itmax=20, d1=2e-3):
    """Newton on d(omega)/d(alpha) = 0; omega(alpha) is analytic (shooting),
    so plain FD derivatives are trustworthy."""
    a, c = complex(a0), complex(c0)
    for _ in range(itmax):
        wp, cp = omega(pr, a + d1, c)
        wm, _ = omega(pr, a - d1, c)
        w0, c = omega(pr, a, c)
        F = (wp - wm)/(2*d1)
        Fpp = (wp - 2*w0 + wm)/d1**2
        if abs(Fpp) < 1e-12:
            break
        step = -F/Fpp
        if abs(step) > 0.15:
            step *= 0.15/abs(step)
        a += step
        if abs(step) < 1e-8:
            break
    w0, c = omega(pr, a, c)
    return a, w0, c


def seed_c(pr, alpha, ngrid=260):
    """FD eigensolve ONCE (real alpha) to seed the c-Newton: coarse grid is
    fine, shooting refines."""
    from scipy.linalg import eig
    if pr.wall:
        y = np.linspace(0.0, pr.ymax, ngrid)
    else:
        y = np.linspace(-pr.ymax, pr.ymax, ngrid)
    h = y[1] - y[0]
    U = pr.U(y).real
    Upp = pr.Upp(y).real
    n = ngrid
    D2 = (np.diag(np.ones(n-1), -1) - 2*np.eye(n)
          + np.diag(np.ones(n-1), 1))/h**2
    B = D2 - alpha**2*np.eye(n)
    A = np.diag(U) @ B - np.diag(Upp)
    cs = eig(A[1:-1, 1:-1], B[1:-1, 1:-1], right=False)
    cs = cs[np.isfinite(cs)]
    lo, hi = U.min(), U.max()
    phys = cs[(np.real(cs) > lo + 0.02) & (np.real(cs) < hi - 0.02)]
    if len(phys) == 0:
        phys = cs
    return complex(phys[np.argmax(np.imag(phys))])


def absolute_growth(pr, a_start=None, c_start=None, agrid=None):
    """Find the KH pinch: seed on the real axis at the most-unstable alpha,
    walk into the lower half plane, Newton to the saddle. Returns
    (alpha0, omega0, c0)."""
    agrid = agrid if agrid is not None else np.geomspace(0.08, 1.0, 10)
    if a_start is None:
        best = None
        for a in agrid:
            try:
                cg = seed_c(pr, a)
                w, c = omega(pr, a, cg)
            except Exception:
                continue
            if best is None or w.imag > best[1].imag:
                best = (a, w, c)
        a_start, _, c_start = best
        # descend into the complex plane in small continuation steps
        for ai in (-0.04, -0.08, -0.12):
            _, c_start = omega(pr, a_start + 1j*ai, c_start)
        a_start = a_start + 1j*(-0.12)
    return saddle(pr, a_start, c_start)


def main():
    os.makedirs(FIGD, exist_ok=True)
    # ---- validation anchor: HM85 counterflow layer, R_crit = 1.315 ----
    print("HM85 anchor (u = Um + tanh y; absolute for R = 1/Um > 1.315):")
    lo, hi = 0.60, 0.95                    # Um bracket around 0.7605
    a0, c0 = None, None
    vals = {}
    for Um in (0.95, 0.85, 0.76, 0.70, 0.62):
        pr = hm85_layer(Um)
        a0, w0, c0 = absolute_growth(pr, a0, c0)
        vals[Um] = w0.imag
        print(f"  Um={Um:.3f} (R={1/Um:.3f}): alpha0={a0.real:+.4f}"
              f"{a0.imag:+.4f}i  w0={w0.real:+.5f}{w0.imag:+.5f}i  "
              f"{'ABS' if w0.imag > 0 else 'conv'}", flush=True)
    # bisect the threshold
    lo_um, hi_um = 0.70, 0.85              # brackets ABS / conv from above
    a0, c0 = None, None
    for _ in range(18):
        Um = 0.5*(lo_um + hi_um)
        pr = hm85_layer(Um)
        a0, w0, c0 = absolute_growth(pr, a0, c0)
        if w0.imag > 0:
            lo_um = Um
        else:
            hi_um = Um
    Um_c = 0.5*(lo_um + hi_um)
    print(f"  threshold: Um = {Um_c:.4f}  ->  R_crit = {1/Um_c:.4f} "
          f"(HM85: 1.315)", flush=True)
    err = abs(1/Um_c - 1.315)/1.315
    assert err < 0.03, f"HM85 anchor off by {err*100:.1f}%"
    print(f"  OK ({err*100:.2f}% from HM85)\n", flush=True)

    # ---- primary use: verify Avanci labels on the Q2 calibration family --
    from explore_q2_systematic import (tanh_profile, crossing_and_label,
                                       TANH_H, TANH_UR)
    print("tanh-wall family: Briggs-Bers vs Avanci label")
    print(f"{'h':>5} {'u_r':>5} | {'u_rev%':>6} {'margin':>7} "
          f"{'Avanci':>7} | {'Im w0':>8} {'BB':>5} | agree")
    n_agree = n_tot = 0
    for h in TANH_H:
        a0 = c0 = None
        for ur in TANH_UR:
            p = tanh_profile(h, ur)
            r = crossing_and_label(p['y']/p['theta'], p['u'],
                                   p['theta']*p['up'],
                                   p['theta']**2*p['upp'], 1.0)
            if r is None:
                continue
            urev, _, marg = r
            pr = tanh_wall(h, ur)
            try:
                if a0 is None:
                    a0, w0, c0 = absolute_growth(pr)
                else:
                    a0, w0, c0 = saddle(pr, a0, c0)
            except Exception as e:
                print(f"{h:5.1f} {ur:5.2f} | saddle failed: {e}")
                a0 = c0 = None
                continue
            av = 'ABS' if marg > 0 else 'conv'
            bb = 'ABS' if w0.imag > 0 else 'conv'
            ok = av == bb
            n_agree += ok
            n_tot += 1
            print(f"{h:5.1f} {ur:5.2f} | {urev*100:6.1f} {marg:+7.2f} "
                  f"{av:>7} | {w0.imag:+8.4f} {bb:>5} | "
                  f"{'yes' if ok else 'NO'}", flush=True)
    print(f"\nagreement: {n_agree}/{n_tot}")

    # ---- reversed FS profiles (Chebyshev continuation): expect all conv --
    from explore_lsb_frozen_profile import build_profile
    print("\nreversed Falkner-Skan (Stewartson branch):")
    for beta, guess in ((-0.19, -0.03), (-0.15, -0.08), (-0.12, -0.10)):
        prf = build_profile(beta, guess)
        Th = prf['Theta']
        ymax = float(prf['eta'][-1]/Th)
        pr = cheb_profile(prf['eta']/Th, prf['u'], Th*Th*prf['upp'], ymax)
        try:
            a0, w0, c0 = absolute_growth(pr)
            print(f"  beta={beta:+.2f} (H={prf['H']:.2f}): "
                  f"w0={w0.real:+.5f}{w0.imag:+.5f}i  "
                  f"{'ABS' if w0.imag > 0 else 'conv'}", flush=True)
        except Exception as e:
            print(f"  beta={beta:+.2f}: failed ({e})", flush=True)


if __name__ == '__main__':
    main()
