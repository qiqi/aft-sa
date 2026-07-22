"""Tier-3 determination of K_r, the favorable-rate factor slope (paper
Sec. III.C, Eq. fpgrate).

The factor multiplies the nondimensional amplification rate a:
    f_lambda = 1 / (1 + (K_r * max(0, lambda_p))^2)
It is exactly 1 for lambda_p <= 0 (adverse branch and Blasius untouched, so
the three anchors and K_lambda are unaffected -- pure Tier 3) and C^1 at
lambda_p = 0. K_r is fit at ONE point: the largest overshoot of the
un-factored shape-factor family (beta = 0.35, H = 2.34, where the un-factored branch
runs 2.0x hot in the mean measure), by requiring
    onset-to-transition mean rate / Drela's rate = 1.
The fit gives K_r = 5.47; the canonical value is the rounded 5.5.

Form selection (recorded; rerun with --forms): three one-constant candidates
were fit the same way and judged on the flatness of the re-fitted family
(mean-ratio range over beta in [0.05, 1]) and smoothness at lambda_p = 0:
    E : exp(-K lam)          K=1.51  means 0.77-1.13x  (C^0 kink at lam=0)
    R1: 1/(1+K lam)          K=2.94  means 0.80-1.08x  (C^0 kink at lam=0)
    R2: 1/(1+(K lam)^2)      K=5.47  means 0.84-1.00x  (C^1, flattest)  <- chosen
(lam = max(0, lambda_p); "means" spans beta = 0.05 ... 1.)

Protocol: marches the disturbance transport (eq:transport) on Falkner-Skan
wedges WITH each wedge's own lambda_p feeding BOTH the onset cliff and the
factor (the physical protocol; same as fig04_shapefactor.py).

Verifies:
  1. the one-point fit at beta = 0.35 lands within 2% of the canonical 5.5;
  2. at the canonical K_r = 5.5 the favorable-family mean measure is within
     [0.80, 1.05] x Drela for beta in [0.05, 1] (vs 1.2-2.0x un-factored);
  3. Blasius is bit-identical with the factor on and off (lambda_p = 0).
"""
import argparse

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import _saai  # noqa: F401
from _saai import C_NU_AI, SIGMA_SA, K_R
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0
from lib.aft_sources import (compute_aft_amplification_rate,
                             compute_composite_gate)

BETA_FIT = 0.35            # largest overshoot of the un-factored family
FORMS = {
    'E':  lambda lam, K: np.exp(-K*np.maximum(lam, 0.0)),
    'R1': lambda lam, K: 1.0/(1.0 + K*np.maximum(lam, 0.0)),
    'R2': lambda lam, K: 1.0/(1.0 + (K*np.maximum(lam, 0.0))**2),
}


def march(fs, x_max, beta, k_r, form=None, K=0.0, nx=800, ny=600):
    """March eq:transport with wedge lambda_p feeding cliff + factor.

    k_r is passed to the kernel (its built-in R2 factor); form/K instead
    apply an EXTERNAL candidate factor with the kernel's own disabled
    (k_r=0) -- used only by the --forms study.
    """
    m = beta/(2.0 - beta)
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
        lam = m*yc**2*fs.inviscid_at(x)**2/(x*u)
        rate = np.asarray(compute_aft_amplification_rate(
            yc**2*np.abs(dudy), 2*(dudy*yc)**2/(u**2 + (dudy*yc)**2),
            lambda_p=lam, fpg_rate_slope=k_r))
        if form is not None:
            rate = rate*FORMS[form](lam, K)
        q4 = compute_composite_gate(dudy, np.gradient(dudy, yc), u, yc)
        b = rate*q4*np.abs(dudy)
        main = u/dx + di; rhs = u/dx*nu + b*nu; rhs[-1] += vm[-1]
        A = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        xs.append((i + 1)*dx); N.append(float(np.log(max(nu.max(), 1e-300))))
    return np.array(xs), np.array(N)


def measures(beta, k_r, form=None, K=0.0, x0=3e5):
    """Standard family measures at one wedge: mean/late secants vs Drela and
    the N=1 onset vs Drela's N=1 station (the O6 tracking measure)."""
    fs = FalknerSkanWedge(beta)
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    H = np.trapezoid(1 - fs.u, fs.eta)/I_th
    x_max = x0
    for _ in range(14):
        xs, N = march(fs, x_max, beta, k_r, form, K)
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.2; continue
        if N[-1] > 9.5:
            break
        x_max *= 2.5
    Ue = fs.inviscid_at(np.maximum(xs, 1e-12)); Rt = I_th*np.sqrt(xs*Ue)
    d = float(dN_dRe_theta(H)); Rtc = float(Re_theta0(H))
    R1 = float(np.interp(1.0, N, Rt)); R5 = float(np.interp(5.0, N, Rt))
    R9 = float(np.interp(9.0, N, Rt))
    return dict(H=H, mean=(8.0/(R9 - R1))/d, late=(4.0/(R9 - R5))/d,
                Rt1=R1, Rt1_D=Rtc + 1.0/d)


def fit_k(mean_at_k, k0=3.0, k1=8.0, tol=0.005):
    """Secant solve mean_at_k(K) = 1."""
    f0, f1 = mean_at_k(k0) - 1.0, mean_at_k(k1) - 1.0
    for _ in range(10):
        if abs(f1 - f0) < 1e-9:
            break
        k2 = min(max(k1 - f1*(k1 - k0)/(f1 - f0), 0.01), 20.0)
        k0, f0 = k1, f1
        k1, f1 = k2, mean_at_k(k2) - 1.0
        if abs(f1) < tol:
            break
    return k1, f1


def family(k_r, betas=(0.05, 0.10, 0.20, 0.35, 0.55, 1.00), form=None, K=0.0):
    out = []
    for b in betas:
        o = measures(b, k_r, form, K, x0=6e5 if b < 0.3 else 3e5)
        print(f"  beta={b:+.2f} H={o['H']:.3f} mean={o['mean']:.2f}x "
              f"late={o['late']:.2f}x N1@{o['Rt1']:.0f} "
              f"({o['Rt1']/o['Rt1_D']:.2f}x Drela N=1)", flush=True)
        out.append(o)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--forms', action='store_true',
                    help='rerun the E/R1/R2 form-selection study (slow)')
    args = ap.parse_args()

    print("un-factored favorable family (k_r = 0; the paper's admitted hot branch):")
    base = family(0.0)

    print(f"\none-point fit of K_r at beta = {BETA_FIT} (mean ratio -> 1):")
    kfit, res = fit_k(lambda K: measures(BETA_FIT, K)['mean'])
    print(f"  K_r(fit) = {kfit:.3f}  (residual {res:+.3f}; canonical = {K_R})")

    print(f"\nfamily at the canonical K_r = {K_R}:")
    fam = family(K_R)

    if args.forms:
        for form in ('E', 'R1', 'R2'):
            kf, r = fit_k(lambda K: measures(BETA_FIT, 0.0, form, K)['mean'],
                          k0=0.3, k1=1.0)
            print(f"\nform {form}: K = {kf:.3f} (residual {r:+.3f})")
            family(0.0, form=form, K=kf)

    # 3. Blasius invariance: lambda_p = 0 -> the factor is exactly 1.
    b_on = measures(0.0, K_R, x0=4e6); b_off = measures(0.0, 0.0, x0=4e6)
    assert b_on['Rt1'] == b_off['Rt1'] and b_on['mean'] == b_off['mean'], \
        "Blasius must be bit-identical with the factor on/off"
    print(f"\nBlasius invariance: Rt1 = {b_on['Rt1']:.1f} identical with "
          f"factor on/off (lambda_p = 0). OK")

    assert abs(kfit - K_R) < 0.02*K_R, \
        f"fit K_r = {kfit:.3f} not within 2% of canonical {K_R}"
    means = [o['mean'] for o in fam]
    assert 0.80 <= min(means) and max(means) <= 1.05, \
        f"factored family means {min(means):.2f}-{max(means):.2f} out of [0.80, 1.05]"
    hot = [o['mean'] for o in base]
    print(f"OK: fit K_r = {kfit:.3f} -> canonical {K_R}; family means "
          f"{min(means):.2f}-{max(means):.2f}x (un-factored {min(hot):.2f}-{max(hot):.2f}x).")


if __name__ == '__main__':
    main()
