"""Destruction-floor back-reaction on the laminar branch, quantified on
the parabolized (frozen-profile march) solutions of Sec. II (annotated-pdf
request, 2026-07-24): the ratio of the tied floor destruction
sigma_D_floor * D_SA = 0.751 c_w1 f_w (nuHat/d)^2 to the amplification
production P_AI = a*omega*nuHat, evaluated at the marched nuHat peak, at
the stations where chi crosses 0.1, 1, and c_v1 -- for three
Falkner-Skan wedges (favorable beta=+0.10, Blasius, the separation limit
beta=-0.1988) and three seeds N_crit = 7, 9, 11.  At chi = c_v1 the
floor is also compared with the SA production it is tied against,
sigma_P * P_SA (sigma_t at tau=4).

The march transport is linear in nuHat, so one march per beta suffices:
the physical chi at a station is chi_inf * e^N, and a chi-level crossing
maps to the station where N = N_crit - ln(c_v1/level).

  python3 floor_backreaction_table.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import _saai
from _saai import SIGMA_SA, K_R, C_NU_AI, A_MAX
from fig04_shapefactor import sphere_rate
from lib.boundary_layer import FalknerSkanWedge
from lib.spalart_allmaras import CB1 as cb1, CW1 as cw1, CW2 as cw2, \
    CW3 as cw3, CV1 as cv1, KAPPA as kap, CB2 as cb2

FLOOR = (1.0 + cb2) / (SIGMA_SA * cw1)          # 0.751
TAU = 4.0
BETAS = [("+0.10", 0.10), ("0 (Blasius)", 0.0), ("-0.1988 (sep)", -0.1988)]
NCRITS = [7.0, 9.0, 11.0]
LEVELS = [0.1, 1.0, cv1]


def fw(r):
    g = r + cw2 * (r**6 - r)
    return g * ((1 + cw3**6) / (g**6 + cw3**6))**(1.0 / 6.0)


def fv1(c):
    return c**3 / (c**3 + cv1**3)


def fv2(c):
    return 1.0 - c / (1.0 + c * fv1(c))


def march_with_peak(fs, x_max, nx=1600, ny=1200):
    eta99 = np.interp(0.99, np.maximum.accumulate(fs.u), fs.eta)
    y_top = 8.0 * eta99 * np.sqrt(x_max / fs.inviscid_at(x_max))
    dy = y_top / ny
    yc = (np.arange(ny) + 0.5) * dy
    dx = x_max / nx
    nu = np.ones(ny)
    k = (C_NU_AI / SIGMA_SA) / dy**2
    rec = []                                     # per station: N, peak state
    for i in range(nx):
        x = (i + 0.5) * dx
        _, u, dudy, v = fs.at(x, np.arange(ny + 1) * dy, cellCentered=True)
        u = np.maximum(u, 1e-12)
        vp = np.clip(v, 0, None) / dy
        vm = np.clip(-v, 0, None) / dy
        di = vp + vm + 2 * k
        lo = -(vp[1:] + k)
        up = -(vm[:-1] + k)
        di[0] += k
        di[-1] -= k
        b = sphere_rate(u, dudy, yc) * np.abs(dudy)
        main = u / dx + di
        rhs = u / dx * nu + b * nu
        rhs[-1] += vm[-1] * 1.0
        A = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        j = int(np.argmax(nu))
        rec.append(dict(N=float(np.log(max(nu.max(), 1e-300))),
                        d=float(yc[j]), om=float(abs(dudy[j])),
                        b=float(b[j])))
    return rec


def ratios_at(rec, N_target, chi):
    Ns = np.array([r['N'] for r in rec])
    if N_target > Ns.max():
        return None
    i = int(np.argmax(Ns >= N_target))
    r = rec[i]
    d, om, b = r['d'], r['om'], r['b']
    # physical nuHat = chi * nu_mol (nu_mol = 1 in march units)
    St = om + chi * fv2(chi) / (kap**2 * d**2)
    rr = min(chi / (max(St, 1e-30) * kap**2 * d**2), 10.0)
    D_floor = FLOOR * cw1 * fw(rr) * (chi / d)**2
    P_ai = b * chi
    out = dict(D_over_Pai=D_floor / max(P_ai, 1e-300))
    if chi >= cv1 * 0.99:
        sig_p = max(1.0 - np.exp(-(chi - 1.0) / TAU), 0.0)
        P_sa = sig_p * cb1 * St * chi
        out['D_over_sPPsa'] = D_floor / max(P_sa, 1e-300)
    return out


if __name__ == '__main__':
    print(f"{'beta':>14} {'N_crit':>6} | {'D/P_AI @chi=0.1':>15} "
          f"{'@chi=1':>10} {'@chi=cv1':>10} | {'D/(sP*P_SA) @cv1':>16}")
    for tag, beta in BETAS:
        fs = FalknerSkanWedge(beta=beta)
        # auto-scale x_max until the envelope clears the largest target
        # (N = 11 at chi = c_v1), same convention as fig04's instrument
        x_max = 4e6 if beta == 0.0 else (3e5 if beta > 0 else 1.2e6)
        for _ in range(12):
            rec = march_with_peak(fs, x_max)
            Nend = rec[-1]['N']
            if not np.isfinite(Nend) or Nend > 60.0:
                x_max *= 0.15
                continue
            if Nend > 12.5:
                break
            x_max *= 3.0
        for nc in NCRITS:
            chi_inf = cv1 * np.exp(-nc)
            row = [f"{tag:>14} {nc:>6.0f} |"]
            vals = []
            for lev in LEVELS:
                N_t = nc - np.log(cv1 / lev)
                res = ratios_at(rec, N_t, lev)
                vals.append(res)
                row.append(f"{res['D_over_Pai']:15.2e}" if res else f"{'--':>15}")
            last = vals[-1]
            row.append(f"{last['D_over_sPPsa']:16.2e}"
                       if last and 'D_over_sPPsa' in last else f"{'--':>16}")
            print(' '.join(row))
