"""tab:ratelimit + tab:frozeneig -> printed LaTeX rows (Sec. II.C).

The rate kernel's growth against the Drela-Giles correlation, analytically,
on frozen parallel Falkner-Skan profiles -- no marching, no onset gate.

INVISCID (tab:ratelimit): a disturbance grows at a(S_hat*g)*|omega| while
advecting at the local u, so the most growth per unit length the rate can
sustain is the pointwise supremum max_y a*|omega|/u. Compared to Drela's
explicit spatial envelope rate s_DG/theta,
s_DG = (m(H)+1)/2 * l(H) * dN/dRe_theta, with no wave-speed convention on
either side. Independent of Reynolds number, diffusion, and c_nu,ai.

VISCOUS (tab:frozeneig): growth, diffusion, and advection compose the
generalized eigenproblem

    [ a(S_hat*g)*|omega| + (c_nu,ai*nu/sigma) d^2/dy^2 ] v = s u v,

whose leading eigenvalue s is the realizable dN/dx at that Re_theta. The
problem symmetrizes under u^(1/2) similarity (one symmetric tridiagonal
eigenvalue per entry); reversed-flow advection is floored at 0.02 U_e
(floors 0.01-0.05 move no printed digit). s -> the inviscid supremum as
Re_theta -> infinity, approached ~Re_theta^(-1/2).

Also printed: the c_nu,ai sweep of the Blasius/family decade geometric
means quoted in the text, and the ungated neutral point (growth first
beats drain, Re_theta ~ 20 on Blasius) that motivates the onset gate.

Optional --rayleigh: each inflectional profile's inviscid temporal ceiling
(wall-bounded Rayleigh eigenvalue, max over wavenumber).

Run from paper/: python3 repro/analytic/tab_frozen_slope.py [--rayleigh]
"""
import sys
import numpy as np
from scipy.linalg import eigh_tridiagonal

import _saai  # noqa: F401
from _saai import A_MAX
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0

SIGMA_SA = 2.0/3.0
C_CANON = 1.0/6.0
UFLOOR = 0.02
RTS = (200.0, 400.0, 800.0, 1600.0)

# (beta, lower-branch guess or None, row label)
PROFILES = [
    (0.10,   None,  'moderate favorable'),
    (0.0,    None,  'Blasius'),
    (-0.10,  None,  'moderate adverse'),
    (-0.1988, None, 'separation limit'),
    (-0.19,  -0.03, 'separated, shallow'),
]
# Shape factors past H~5 are excluded: Drela's Orr-Sommerfeld data stop at
# H=5.00 (1986 thesis Fig. 6.7) and no similarity profile is validated beyond.


def build(beta, guess, N=6000):
    fs = FalknerSkanWedge(beta, guess=guess) if guess is not None \
        else FalknerSkanWedge(beta)
    y0, u0, up0 = (np.asarray(v) for v in (fs.eta, fs.u, fs.dudeta))
    y = np.linspace(0.0, float(y0[-1]), N)
    u = np.interp(y, y0, u0)
    up = np.interp(y, y0, up0)
    upp = np.gradient(up, y)
    I_th = float(np.trapezoid(u*(1.0 - u), y))
    H = float(np.trapezoid(1.0 - u, y))/I_th
    X, Y, Z = u, y*up, 0.5*y**2*upp
    R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
    P = (Y/np.sqrt(X*X + Y*Y + 1e-30))*(Y - X - Z)/R
    b = A_MAX*np.clip(P, 0.0, 1.0)*np.abs(up)
    ell = (6.54*H - 14.07)/H**2
    m = (0.058*(H - 4.0)**2/(H - 1.0) - 0.068)/ell
    sDG = float(dN_dRe_theta(H))*0.5*(m + 1.0)*ell
    return dict(y=y, h=y[1] - y[0], u=u, up=up, b=b, I_th=I_th, H=H, sDG=sDG,
                Rt0=float(np.asarray(Re_theta0(H))))


def s_ratio(pr, Re_theta, cnu):
    """Leading eigenvalue of [diag(b) + D d2] v = s diag(u) v over Drela."""
    D = cnu*pr['I_th']/(SIGMA_SA*Re_theta)
    h = pr['h']
    uf = np.maximum(pr['u'], UFLOOR)[1:-1]
    d = (pr['b'][1:-1] - 2.0*D/h**2)/uf
    e = (D/h**2)/np.sqrt(uf[:-1]*uf[1:])
    w = eigh_tridiagonal(d, e, select='i',
                         select_range=(len(d) - 1, len(d) - 1))[0]
    return float(w[0])*pr['I_th']/pr['sDG']


def s_limit_ratio(pr):
    """Re -> infinity: pointwise sup of growth over advection, over Drela."""
    return float(np.max(pr['b']/np.maximum(pr['u'], UFLOOR)))*pr['I_th']/pr['sDG']


def neutral_ungated(pr, cnu):
    """Re_theta where the leading eigenvalue crosses zero (drain balance)."""
    lo, hi = 5.0, 1e5
    if s_ratio(pr, lo, cnu) > 0:
        return lo
    for _ in range(60):
        mid = np.sqrt(lo*hi)
        if s_ratio(pr, mid, cnu) > 0:
            hi = mid
        else:
            lo = mid
    return np.sqrt(lo*hi)


def rayleigh_ceiling(pr, N=300):
    """Wall-bounded temporal Rayleigh eigenvalue / peak vorticity."""
    from scipy.linalg import eig
    y = np.linspace(0.0, float(pr['y'][-1]), N)
    U = np.interp(y, pr['y'], pr['u'])
    Upp = np.interp(y, pr['y'], np.gradient(pr['up'], pr['y']))
    h = y[1] - y[0]
    D2 = (np.diag(np.ones(N - 1), -1) - 2*np.eye(N)
          + np.diag(np.ones(N - 1), 1))/h**2
    I = np.eye(N)

    def wi(al):
        B = (D2 - al**2*I)[1:-1, 1:-1]
        A = (np.diag(U) @ (D2 - al**2*I) - np.diag(Upp))[1:-1, 1:-1]
        return al*float(np.max(np.imag(eig(A, B, right=False))))

    ag = np.geomspace(0.08, 1.5, 12)
    wg = [wi(a) for a in ag]
    j = int(np.argmax(wg))
    best = wg[j]
    for a in np.linspace(ag[max(j - 1, 0)], ag[min(j + 1, len(ag) - 1)], 7)[1:-1]:
        best = max(best, wi(a))
    return best/float(np.max(np.abs(pr['up'])))


def main():
    prs = [(build(b, g), b, lab) for b, g, lab in PROFILES]

    print("tab:ratelimit (inviscid limit, any c_nu,ai):")
    print(f"{'profile':>20} {'beta':>8} {'H':>6} {'limit':>7}")
    for pr, beta, lab in prs:
        print(f"{lab:>20} {beta:+8.4f} {pr['H']:6.2f} {s_limit_ratio(pr):7.2f}")

    print("\ntab:frozeneig (ratio to Drela; -- where the profile is STABLE,")
    print("Rt < Re_theta0(H), and the correlation has no growth to compare):")
    print(f"{'profile':>20} {'Rt0':>5} | {'c_nu,ai = 1':>27} | {'c_nu,ai = 1/6':>27}")
    blas = fav = None
    for pr, beta, lab in prs:
        def cell(r, c):
            return f"{s_ratio(pr, r, c):+6.2f}" if r > pr['Rt0'] else "    --"
        r1 = " ".join(cell(r, 1.0) for r in RTS)
        rc = " ".join(cell(r, C_CANON) for r in RTS)
        if lab == 'Blasius':
            blas = pr
        if lab == 'moderate favorable':
            fav = pr
        print(f"{lab:>20} {pr['Rt0']:5.0f} | {r1} | {rc}")
    assert abs(s_ratio(blas, 400.0, C_CANON) - 1.02) < 0.015
    assert abs(s_ratio(fav, 1600.0, C_CANON) - 1.02) < 0.015

    print("\nc_nu,ai ladder at the just-past-critical Blasius station Rt=400")
    print("(text quotes 0.49/0.75/0.87/1.02/1.13) and unstable-station means")
    print("(text quotes Blasius 0.70/0.91/1.00/1.12/1.20):")
    rts_u = [r for r in RTS if r > blas['Rt0']]
    for c in (1.0, 0.5, 1.0/3.0, C_CANON, 1.0/12.0):
        v400 = s_ratio(blas, 400.0, c)
        row = [s_ratio(blas, r, c) for r in rts_u]
        gm = float(np.exp(np.mean(np.log(np.maximum(row, 1e-9)))))
        print(f"  c = {c:6.4f}: Rt400 {v400:5.2f}, unstable-mean {gm:5.2f}")

    print("\nungated neutral point at c=1/6 (text: ~20 on Blasius):")
    for pr, beta, lab in prs[:3]:
        print(f"  {lab:>20}: Re_theta = {neutral_ungated(pr, C_CANON):5.0f}")

    if '--rayleigh' in sys.argv[1:]:
        print("\ninviscid Rayleigh ceilings (temporal, /omega_peak):")
        for pr, beta, lab in prs[3:]:
            a_ray = rayleigh_ceiling(pr)
            a_D = pr['sDG']*0.5/(float(np.max(np.abs(pr['up'])))*pr['I_th'])
            print(f"  {lab:>20}: a_ray = {a_ray:.4f}, Drela(temporal) = "
                  f"{a_D:.4f}  ({a_D/a_ray:.1f}x ceiling)")


if __name__ == '__main__':
    main()
