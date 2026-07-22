"""explore-lambda-v, step A: bracket the Lambda_v gate weight c_V with the
three anchors RE-SOLVED per candidate (the Sec-III.A quotient-space protocol,
cloned from scan_background_constants.py with the band gate swapped).

Gate under test (compact form, exact analog of Q4):
    Q_v = 1 - 2|w| d |u| / (c_V |u' u''| d^3 + (w d)^2 + u^2)
      == 1 - sqrt(Gamma(2-Gamma)) / (1 + c_V |Lambda_v|),
    Lambda_v = -u' u'' d^3 / ((u' d)^2 + u^2)
             = (d2u/dn2 . (n x omega)) d^3 / (...)  [3D],
sign convention chosen so Blasius (and every favorable profile) is all
POSITIVE (Lambda_v ~ -d(u'^2)/dn); the NEGATIVE lobe is the sub-inflection
shear-growth layer unique to adverse profiles. The gate itself is
sign-blind (|Lambda_v|), so the convention does not affect any number here.

Everything else canonical: a_max=0.19, c_nu_ai=1/12, p=4, lambda_p terms
inert on the zero/adverse anchors. Candidates c_V in {1,2,4,8,16} plus the
committed Q4 (cA=4) reference row.
"""
import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from multiprocessing import Pool

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa: F401
from _saai import SIGMA_SA
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0
from lib.aft_sources import compute_aft_amplification_rate, compute_q4_gate

BETA_SEP = -0.1988
CNU, PBAR = 1.0/12.0, 4.0


def gate_lambda_v(dudy, upp, u, yc, cV):
    num = 2.0*np.abs(dudy)*yc*np.abs(u)
    den = cV*np.abs(dudy*upp)*yc**3 + (dudy*yc)**2 + u**2 + 1e-300
    return 1.0 - num/den


def gate_lambda_v_smooth(dudy, upp, u, yc, cV):
    """Smooth variant: 1 + cV|Lambda_v|  ->  sqrt(1 + (cV Lambda_v)^2).
    Same value and curvature-free opening at the axis, same linear opening
    at large |Lambda_v|, but C-infinity through Lambda_v = 0 (the |.| kink
    made ugly pointed contours). In raw variables the denominator is just
    hypot(den_core, cV u'u'' d^3)."""
    num = 2.0*np.abs(dudy)*yc*np.abs(u)
    den_core = (dudy*yc)**2 + u**2 + 1e-300
    den = np.sqrt(den_core**2 + (cV*dudy*upp*yc**3)**2)
    return 1.0 - num/den


Q2_L0, Q2_C2 = -1.8, 2.0


def gate_composite(dudy, upp, u, yc, cV, band_pow=0.5):
    """CANDIDATE (2026-07-17): smooth Q1 x Q2 two-pinch product.
    Q1 pins the wall zero-advection locus (Gamma, Lambda_v) = (1, 0) with the
    smooth hypot denominator; Q2 pins the recirculation's interior
    zero-advection locus at (2, L0). Both pinches are the same family:
    places where the transported scalar has no advection to bound it.
    band_pow = p in B^p, B = Gamma(2-Gamma): p=1/2 is the classical band
    factor (slow mid-band opening); p=1 opens quadratically away from
    Gamma=1 (fully open at Gamma = 0, 2 for any p; the pinch is unchanged)."""
    den_core = (dudy*yc)**2 + u**2 + 1e-300
    num = 2.0*np.abs(dudy)*yc*np.abs(u)
    band = np.clip(num/den_core, 0.0, 1.0)          # = sqrt(Gamma(2-Gamma))
    q1 = 1.0 - band**(2.0*band_pow)/np.sqrt(1.0 + (cV*dudy*upp*yc**3
                                                   / den_core)**2)
    Gam = 2.0*(dudy*yc)**2/den_core
    Lv = -dudy*upp*yc**3/den_core
    q2 = 1.0 - np.clip(Gam - 1.0, 0.0, None)**2/(1.0 + Q2_C2*(Lv - Q2_L0)**2)
    return q1*q2


def gate_composite_band(dudy, upp, u, yc, cV, q=2.0, c2=8.0):
    """ADOPTED (2026-07-18, supersedes the center-offset form): smooth Q1 x
    band-form Q2. The Q2 pocket is the INTERIOR OF THE UNIVERSAL THIN-BUBBLE
    PARABOLA LOOP near its zero crossing -- P = (Lv+G)^2 - G(2-G) <= 0 --
    pinned exactly at the analytic point (2,-2), with a single softness
    constant c2 for the release outside the loop. Zero fitted location
    parameters (user derivation: any quadratic profile with a wall zero
    maps onto one universal curve whose stagnation points are the two gate
    pinches)."""
    den_core = (dudy*yc)**2 + u**2 + 1e-300
    num = 2.0*np.abs(dudy)*yc*np.abs(u)
    q1 = 1.0 - num/np.sqrt(den_core**2 + (cV*dudy*upp*yc**3)**2)
    Gam = 2.0*(dudy*yc)**2/den_core
    Lv = -dudy*upp*yc**3/den_core
    P = (Lv + Gam)**2 - Gam*(2.0 - Gam)
    q2 = 1.0 - np.clip(Gam - 1.0, 0.0, None)**q/(1.0 + c2*np.clip(P, 0.0,
                                                                  None))
    return q1*q2


def gate_lambda_v_plus(dudy, upp, u, yc, cV):
    """One-sided variant: |Lambda_v| -> max(0, Lambda_v) (flipped-sign
    convention: Lambda_v ~ -u'u''), so only shear-DECAY regions open the
    gate; the sub-inflection shear-growth layer gets the bare band form."""
    num = 2.0*np.abs(dudy)*yc*np.abs(u)
    den = cV*np.maximum(-dudy*upp, 0.0)*yc**3 + (dudy*yc)**2 + u**2 + 1e-300
    return 1.0 - num/den


def march(fs, x_max, floor, gc, s, gate, gw, nx=800, ny=600, seed=1.0):
    eta99 = np.interp(0.99, fs.u, fs.eta)
    y_top = 8.0*eta99*np.sqrt(x_max/fs.inviscid_at(x_max))
    dy = y_top/ny; yc = (np.arange(ny) + 0.5)*dy; dx = x_max/nx
    nu = np.ones(ny)*seed; N = [0.0]; xs = [0.0]
    k = (CNU/SIGMA_SA)/dy**2
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, np.arange(ny + 1)*dy, cellCentered=True)
        u = np.maximum(u, 1e-12)
        vp = np.clip(v, 0, None)/dy; vm = np.clip(-v, 0, None)/dy
        di = vp + vm + 2*k; lo = -(vp[1:] + k); up = -(vm[:-1] + k)
        di[0] += k; di[-1] -= k
        rate = np.asarray(compute_aft_amplification_rate(
            yc**2*np.abs(dudy), 2*(dudy*yc)**2/(u**2 + (dudy*yc)**2),
            sigmoid_center=gc, sigmoid_slope=s, re_omega_floor=floor,
            barrier_power=PBAR))
        upp = np.gradient(dudy, yc)
        if gate == 'q4':
            q = compute_q4_gate(upp, np.abs(dudy), u, yc, cA=gw)
        elif gate == 'lvp':
            q = gate_lambda_v_plus(dudy, upp, u, yc, gw)
        elif gate == 'lvs':
            q = gate_lambda_v_smooth(dudy, upp, u, yc, gw)
        elif gate == 'lv2':
            q = gate_composite(dudy, upp, u, yc, gw)
        elif gate == 'lv2b':
            q = gate_composite(dudy, upp, u, yc, gw, band_pow=1.0)
        elif gate == 'lv3':
            q = gate_composite_band(dudy, upp, u, yc, gw)
        else:
            q = gate_lambda_v(dudy, upp, u, yc, gw)
        b = rate*q*np.abs(dudy)
        main = u/dx + di; rhs = u/dx*nu + b*nu; rhs[-1] += vm[-1]*seed
        A = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        xs.append((i + 1)*dx); N.append(float(np.log(max(nu.max()/seed, 1e-300))))
    return np.array(xs), np.array(N)


def envelope(beta, x_max0, floor, gc, s, gate, gw):
    fs = FalknerSkanWedge(beta)
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    H = np.trapezoid(1 - fs.u, fs.eta)/I_th
    x_max = x_max0
    for _ in range(14):
        xs, N = march(fs, x_max, floor, gc, s, gate, gw)
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.2; continue
        if N[-1] > 9.5:
            break
        x_max *= 2.5
    Ue = fs.inviscid_at(np.maximum(xs, 1e-12))
    Rt = I_th*np.sqrt(xs*Ue)
    return Rt, N, H


def blasius_residuals(floor, gc, s, gate, gw, targets):
    Rt, N, H = envelope(0.0, 4e6, floor, gc, s, gate, gw)
    if N[-1] < 9.0:
        return None, None
    Rt1 = float(np.interp(1.0, N, Rt)); Rt9 = float(np.interp(9.0, N, Rt))
    return Rt1 - targets[0], Rt9 - targets[1]


def solve_triple(gate, gw, x0=(300.0, 1.0, 10.0), tag=""):
    fs = FalknerSkanWedge(0.0)
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    Hb = np.trapezoid(1 - fs.u, fs.eta)/I_th
    Rtc = float(Re_theta0(Hb)); slope = float(dN_dRe_theta(Hb))
    targets = (Rtc + 1.0/slope, Rtc + 9.0/slope)
    fs_sep = FalknerSkanWedge(BETA_SEP)
    I_ts = np.trapezoid(fs_sep.u*(1 - fs_sep.u), fs_sep.eta)
    Hs = np.trapezoid(1 - fs_sep.u, fs_sep.eta)/I_ts
    d_sep = float(dN_dRe_theta(Hs))

    def inner(s_val, floor0, gc0):
        floor, gc = floor0, gc0
        for it in range(10):
            r = blasius_residuals(floor, gc, s_val, gate, gw, targets)
            if r[0] is None:
                floor *= 0.7; continue
            r1, r9 = r
            if abs(r1) < 1.0 and abs(r9) < 3.0:
                return floor, gc, True
            dfl = max(4.0, 0.05*floor)
            rb = blasius_residuals(floor + dfl, gc, s_val, gate, gw, targets)
            if rb[0] is not None and abs(rb[0] - r1) > 1e-9:
                floor = min(max(floor - r1*dfl/(rb[0] - r1), 30.0), 1500.0)
            rc = blasius_residuals(floor, gc, s_val, gate, gw, targets)
            if rc[0] is None:
                floor = min(max(floor*1.2, 30.0), 1500.0); continue
            dgc = 0.01
            rd = blasius_residuals(floor, gc + dgc, s_val, gate, gw, targets)
            if rd[1] is not None and abs(rd[1] - rc[1]) > 1e-9:
                gc = min(max(gc - rc[1]*dgc/(rd[1] - rc[1]), 0.80), 1.40)
        return floor, gc, False

    def sep_ratio(s_val, floor, gc):
        Rt, N, _ = envelope(BETA_SEP, 1.2e6, floor, gc, s_val, gate, gw)
        if N[-1] < 9.0:
            return None
        Rt1 = float(np.interp(1.0, N, Rt)); Rt9 = float(np.interp(9.0, N, Rt))
        return (8.0/(Rt9 - Rt1))/d_sep

    floor, gc, s = x0
    hist = []
    for it in range(8):
        floor, gc, ok = inner(s, floor, gc)
        r = sep_ratio(s, floor, gc)
        hist.append((s, floor, gc, r, ok))
        print(f"[{tag}] it{it}: s={s:.2f} floor={floor:.1f} gc={gc:.4f} "
              f"sep={r if r else float('nan'):.4f} ok={ok}", flush=True)
        if r is not None and abs(r - 1.0) < 0.005:
            break
        if len(hist) >= 2 and hist[-2][3] is not None and r is not None \
                and abs(r - hist[-2][3]) > 1e-6:
            s = min(max(s - (r - 1.0)*(s - hist[-2][0])/(r - hist[-2][3]), 2.0), 40.0)
        else:
            s = s*(1.25 if (r or 0) < 1.0 else 0.8)
    return floor, gc, s


def family_observables(floor, gc, s, gate, gw):
    out = {}
    fs = FalknerSkanWedge(0.0)
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    Hb = np.trapezoid(1 - fs.u, fs.eta)/I_th
    Rtc = float(Re_theta0(Hb)); slope = float(dN_dRe_theta(Hb))
    Rt, N, _ = envelope(0.0, 4e6, floor, gc, s, gate, gw)
    m = (N >= 1.0) & (N <= 9.0)
    out['bl_intdev'] = float(np.max(np.abs(N[m] - slope*(Rt[m] - Rtc)))/9.0)
    Rt5 = float(np.interp(5.0, N, Rt)); Rt9 = float(np.interp(9.0, N, Rt))
    out['bl_late'] = (4.0/(Rt9 - Rt5))/slope
    for beta in (-0.09, -0.15):
        Rt, N, H = envelope(beta, 1.2e6, floor, gc, s, gate, gw)
        d = float(dN_dRe_theta(H))
        Rt1 = float(np.interp(1.0, N, Rt)); Rt9 = float(np.interp(9.0, N, Rt))
        out[f'mean{beta}'] = (8.0/(Rt9 - Rt1))/d
    return out


def one(args):
    name, gate, gw, x0 = args
    floor, gc, s = solve_triple(gate, gw, x0=x0, tag=name)
    return name, gate, gw, floor, gc, s, family_observables(floor, gc, s, gate, gw)


if __name__ == '__main__':
    CANDS = [
        ("q4_ref", 'q4', 4.0, (254.0, 1.005, 11.0)),
        ("cV1",  'lv', 1.0,  (300.0, 1.0, 10.0)),
        ("cV2",  'lv', 2.0,  (300.0, 1.0, 10.0)),
        ("cV4",  'lv', 4.0,  (300.0, 1.0, 10.0)),
        ("cV8",  'lv', 8.0,  (300.0, 1.0, 10.0)),
        ("cV16", 'lv', 16.0, (300.0, 1.0, 10.0)),
    ]
    with Pool(6) as pool:
        results = pool.map(one, CANDS)
    print("\n===== LAMBDA_V GATE SCAN (anchors re-solved per candidate) =====")
    print(f"{'cand':>7} {'gate':>5} {'w':>5} | {'floor':>7} {'g_c':>7} {'s':>6} | "
          f"{'bl_int':>7} {'bl_late':>8} {'m(-.09)':>8} {'m(-.15)':>8}")
    for name, gate, gw, floor, gc, s, o in results:
        print(f"{name:>7} {gate:>5} {gw:5.1f} | {floor:7.1f} {gc:7.4f} {s:6.2f} | "
              f"{o['bl_intdev']:7.3f} {o['bl_late']:8.3f} {o['mean-0.09']:8.3f} "
              f"{o['mean-0.15']:8.3f}")
