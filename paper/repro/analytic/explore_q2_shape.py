"""Q2 SHAPE scan with the center pinned at the ANALYTIC point (2, -2)
(user request): can adjusting shape/aspect instead of the center match the
calibrated (-1.8) pocket?

Shapes (all pinch exactly at (2,-2); all reuse the universal parabola):
  'off'    : Q2 = 1                                          (reference)
  'center' : current fleet form, 1 - (G-1)_+^2/(1+2(Lv+1.8)^2)
  'band'   : loop-interior form; P = (Lv+G)^2 - G(2-G) (P<=0 = inside the
             parabola loop):  Q2 = 1 - (G-1)_+^q/(1 + c2 max(P,0))
  'branch' : distance to the UPPER branch xi = Lv + G - sqrt(G(2-G)):
             Q2 = 1 - (G-1)_+^q/(1 + c2 xi^2)

Aspect ratio = q (Gamma-direction sharpness) vs c2 (off-curve width).
Metrics: frozen R_x (5 profiles x Re 200/400) + raw attached marches for
the two exposed anchors-region profiles (sep-limit beta=-0.1988 whose
near-wall parabola point (8/5,-4/5) is the t->inf ENDPOINT of the same
curve, and beta=-0.15).
"""
import sys
import numpy as np
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa: F401
from _saai import C_NU_AI, SIGMA_SA
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta
from lib.aft_sources import compute_aft_amplification_rate
from explore_lsb_frozen_profile import build_profile, drela_spatial, PROFILES

K = dict(gw=4.0, gc=0.9874, s=10.68, floor=243.7, KL=6.20, KR=5.80)


def q2_factor(Gam, Lv, shape, q=2.0, c2=2.0):
    if shape == 'off':
        return np.ones_like(Gam)
    g = np.clip(Gam - 1.0, 0.0, None)**q
    if shape == 'center':
        return 1.0 - np.clip(Gam - 1.0, 0.0, None)**2/(1.0 + 2.0*(Lv + 1.8)**2)
    if shape == 'band':
        P = (Lv + Gam)**2 - Gam*(2.0 - Gam)
        return 1.0 - g/(1.0 + c2*np.clip(P, 0.0, None))
    if shape == 'branch':
        xi = Lv + Gam - np.sqrt(np.clip(Gam*(2.0 - Gam), 0.0, None))
        return 1.0 - g/(1.0 + c2*xi*xi)
    raise ValueError(shape)


def gate(dudy, upp, u, yc, shape, q, c2):
    den_core = (dudy*yc)**2 + u**2 + 1e-300
    num = 2.0*np.abs(dudy)*yc*np.abs(u)
    q1 = 1.0 - num/np.sqrt(den_core**2 + (K['gw']*dudy*upp*yc**3)**2)
    Gam = 2.0*(dudy*yc)**2/den_core
    Lv = -dudy*upp*yc**3/den_core
    return q1*q2_factor(Gam, Lv, shape, q, c2)


def sigma_x(pr, Re_th, shape, q, c2, ufrac=0.03):
    Th = pr['Theta']
    yh = pr['eta'][1:-1]/Th
    u = pr['u'][1:-1]
    w = Th*pr['up'][1:-1]
    dwdy = Th*Th*pr['upp'][1:-1]
    Gam = 2*(w*yh)**2/((w*yh)**2 + u**2 + 1e-300)
    rate = np.asarray(compute_aft_amplification_rate(
        np.abs(w)*yh**2*Re_th, Gam, lambda_p=0.0, sigmoid_center=K['gc'],
        sigmoid_slope=K['s'], re_omega_floor=K['floor'], barrier_power=4.0))
    b = rate*gate(w, dwdy, u, yh, shape, q, c2)*np.abs(w)
    h = yh[1] - yh[0]
    kd = (C_NU_AI/SIGMA_SA)/Re_th
    uf = np.maximum(u, ufrac)
    lam = eigh_tridiagonal((b - 2*kd/h**2)/uf,
                           (kd/h**2)/np.sqrt(uf[:-1]*uf[1:]), select='i',
                           select_range=(len(yh)-1, len(yh)-1),
                           eigvals_only=True)
    return float(lam[0])


def march_mean(beta, shape, q, c2, nx=800, ny=600):
    fs = FalknerSkanWedge(beta)
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    H = np.trapezoid(1 - fs.u, fs.eta)/I_th
    m = beta/(2.0 - beta)
    x_max = 1.2e6
    for _ in range(12):
        eta99 = np.interp(0.99, np.maximum.accumulate(fs.u), fs.eta)
        y_top = 8.0*eta99*np.sqrt(x_max/fs.inviscid_at(x_max))
        dy = y_top/ny
        yc = (np.arange(ny) + 0.5)*dy
        dx = x_max/nx
        nu = np.ones(ny)
        N = [0.0]
        xs = [0.0]
        kdif = (C_NU_AI/SIGMA_SA)/dy**2
        for i in range(nx):
            x = (i + 0.5)*dx
            _, u, dudy, v = fs.at(x, np.arange(ny + 1)*dy, cellCentered=True)
            u = np.maximum(u, 1e-12)
            vp = np.clip(v, 0, None)/dy
            vm = np.clip(-v, 0, None)/dy
            di = vp + vm + 2*kdif
            lo = -(vp[1:] + kdif)
            up_ = -(vm[:-1] + kdif)
            di[0] += kdif
            di[-1] -= kdif
            lam = m*yc**2*fs.inviscid_at(x)**2/(x*u)
            rate = np.asarray(compute_aft_amplification_rate(
                yc**2*np.abs(dudy),
                2*(dudy*yc)**2/(u**2 + (dudy*yc)**2 + 1e-300),
                lambda_p=lam, sigmoid_center=K['gc'], sigmoid_slope=K['s'],
                re_omega_floor=K['floor'], barrier_power=4.0,
                cliff_lambda_slope=K['KL'], fpg_rate_slope=K['KR']))
            upp = np.gradient(dudy, yc)
            b = rate*gate(dudy, upp, u, yc, shape, q, c2)*np.abs(dudy)
            main = u/dx + di
            rhs = u/dx*nu + b*nu
            rhs[-1] += vm[-1]
            A = sp.diags([lo, main, up_], [-1, 0, 1], format='csc')
            nu = spla.spsolve(A, rhs)
            xs.append((i + 1)*dx)
            N.append(float(np.log(max(nu.max(), 1e-300))))
        xs, N = np.array(xs), np.array(N)
        if not np.isfinite(N).all() or N[-1] > 60.0:
            x_max *= 0.15
            continue
        if N[-1] > 14.0:
            break
        x_max *= 3.0
    Ue = fs.inviscid_at(np.maximum(xs, 1e-12))
    Rt = I_th*np.sqrt(xs*Ue)
    Rt1 = float(np.interp(1.0, N, Rt))
    Rt9 = float(np.interp(9.0, N, Rt))
    return (8.0/(Rt9 - Rt1))/float(dN_dRe_theta(H))


def main():
    variants = [('center', 2, 2.0)] + \
        [('band', q, c2) for q in (2, 4) for c2 in (2.0, 8.0)] + \
        [('branch', q, c2) for q in (2, 4) for c2 in (2.0, 8.0)]
    prs = [(build_profile(b, g)) for b, g in PROFILES]
    print("(a) frozen R_x (Re_th=200/400):")
    hdr = "  ".join(f"H={p['H']:.1f}" for p in prs)
    print(f"{'shape':>8} {'q':>2} {'c2':>4} | {hdr}")
    for shape, q, c2 in variants:
        cells = []
        for p in prs:
            dS = drela_spatial(p['H'])
            r2 = sigma_x(p, 200.0, shape, q, c2)/dS
            r4 = sigma_x(p, 400.0, shape, q, c2)/dS
            cells.append(f"{r2:.2f}/{r4:.2f}")
        print(f"{shape:>8} {q:>2} {c2:4.1f} | " + "  ".join(cells),
              flush=True)

    print("\n(b) attached raw marches, mean/Drela (anchors watch):")
    print(f"{'shape':>8} {'q':>2} {'c2':>4} | {'sep-limit':>9} {'b=-0.15':>8}")
    for shape, q, c2 in [('off', 2, 2.0)] + variants:
        m1 = march_mean(-0.1988, shape, q, c2)
        m2 = march_mean(-0.15, shape, q, c2)
        print(f"{shape:>8} {q:>2} {c2:4.1f} | {m1:9.2f} {m2:8.2f}",
              flush=True)


if __name__ == '__main__':
    main()
