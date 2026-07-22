"""Sensitivity of the three-anchor calibration to the departure-anchor level
(the paper's Sec. III.A sensitivity paragraph; weakness O2, "why N=1?").

Result (2026-07-12): predictions are pinned by the N=9 anchor (operational
onset +-0.5%, floor <3% across N_low in [0.5, 2]) while (g_c, s) slide along
the system's soft direction ((0.99,12.9) at 0.5 -> (1.10,6.6) at 2) and the
unfitted mid-adverse family responds ~+20%/-25% -- the family selects
N_low ~ 1. A control with the separation window held at [1,9] (only the
Blasius anchor moved) gives nearly the same swings, so the Blasius departure
level itself, not the window convention, carries the sensitivity.

For N_low in {0.5, 1, 2}, re-solve the triple (floor, g_c, s) with the anchors
  Blasius meets Drela at N = N_low   and at N = 9,
  separation-limit mean over [N_low, 9] = Drela,
then report, per case:
  - the solved digits,
  - the family response in the FIXED standard measures (mean = N in [1,9]
    secant, late = [5,9] secant) at beta = -0.09, -0.15, 0 for comparability,
  - the Blasius operational-onset proxy: Re_theta at N = ln(c_v1/chi_inf)
    = 7.04 (chi = 1 on the coupled flat plate at N_crit = 9 seeding), and at
    N = 9 (pinned by construction).
"""
import sys
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from scan_background_constants import (envelope, FalknerSkanWedge, dN_dRe_theta,
                                       Re_theta0, BETA_SEP)

CNU, CA, PP = 1/12, 4.0, 4.0
N_OP = 7.04                       # ln(c_v1) below N_crit=9: chi=1 station


def solve_triple_nlow(n_low, x0, tag=""):
    fs = FalknerSkanWedge(0.0)
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    Hb = np.trapezoid(1 - fs.u, fs.eta)/I_th
    Rtc = float(Re_theta0(Hb)); slope = float(dN_dRe_theta(Hb))
    t_lo, t_hi = Rtc + n_low/slope, Rtc + 9.0/slope

    fs_sep = FalknerSkanWedge(BETA_SEP)
    I_ts = np.trapezoid(fs_sep.u*(1 - fs_sep.u), fs_sep.eta)
    Hs = np.trapezoid(1 - fs_sep.u, fs_sep.eta)/I_ts
    d_sep = float(dN_dRe_theta(Hs))

    def bl_res(floor, gc, s):
        Rt, N, _ = envelope(0.0, 4e6, floor, gc, s, CNU, CA, PP)
        if N[-1] < 9.0:
            return None, None
        Rlo = float(np.interp(n_low, N, Rt)); Rhi = float(np.interp(9.0, N, Rt))
        return Rlo - t_lo, Rhi - t_hi

    def inner(s_val, floor0, gc0):
        floor, gc = floor0, gc0
        for _ in range(10):
            r1, r9 = bl_res(floor, gc, s_val)
            if r1 is None:
                floor *= 0.7; continue
            if abs(r1) < 1.0 and abs(r9) < 3.0:
                return floor, gc, True
            dfl = max(4.0, 0.05*floor)
            r1b, _ = bl_res(floor + dfl, gc, s_val)
            if r1b is not None and abs(r1b - r1) > 1e-9:
                floor = min(max(floor - r1*dfl/(r1b - r1), 30.0), 1500.0)
            r1c, r9c = bl_res(floor, gc, s_val)
            if r1c is None:
                floor = min(max(floor*1.2, 30.0), 1500.0); continue
            dgc = 0.01
            _, r9d = bl_res(floor, gc + dgc, s_val)
            if r9d is not None and abs(r9d - r9c) > 1e-9:
                gc = min(max(gc - r9c*dgc/(r9d - r9c), 0.80), 1.40)
        return floor, gc, False

    def sep_ratio(s_val, floor, gc):
        Rt, N, _ = envelope(BETA_SEP, 1.2e6, floor, gc, s_val, CNU, CA, PP)
        if N[-1] < 9.0:
            return None
        Rlo = float(np.interp(n_low, N, Rt)); Rhi = float(np.interp(9.0, N, Rt))
        return ((9.0 - n_low)/(Rhi - Rlo))/d_sep

    floor, gc, s = x0
    hist = []
    for it in range(8):
        floor, gc, ok = inner(s, floor, gc)
        r = sep_ratio(s, floor, gc)
        hist.append((s, r))
        print(f"[{tag}] it{it}: s={s:.2f} floor={floor:.1f} gc={gc:.4f} "
              f"sep={r if r else float('nan'):.4f} ok={ok}", flush=True)
        if r is not None and abs(r - 1.0) < 0.005:
            break
        if len(hist) >= 2 and hist[-2][1] is not None and r is not None \
                and abs(r - hist[-2][1]) > 1e-6:
            s = min(max(s - (r - 1.0)*(s - hist[-2][0])/(r - hist[-2][1]), 2.0), 40.0)
        else:
            s = s*(1.25 if (r or 0) < 1.0 else 0.8)
    return floor, gc, s


def observables(floor, gc, s):
    out = {}
    fs = FalknerSkanWedge(0.0)
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    Hb = np.trapezoid(1 - fs.u, fs.eta)/I_th
    Rtc = float(Re_theta0(Hb)); slope = float(dN_dRe_theta(Hb))
    Rt, N, _ = envelope(0.0, 4e6, floor, gc, s, CNU, CA, PP)
    out['Rt_op'] = float(np.interp(N_OP, N, Rt))           # chi=1 proxy station
    out['Rt9'] = float(np.interp(9.0, N, Rt))
    out['bl_mean'] = (8.0/(out['Rt9'] - float(np.interp(1.0, N, Rt))))/slope
    Rt5 = float(np.interp(5.0, N, Rt))
    out['bl_late'] = (4.0/(out['Rt9'] - Rt5))/slope
    for beta in (-0.09, -0.15):
        Rt, N, H = envelope(beta, 1.2e6, floor, gc, s, CNU, CA, PP)
        d = float(dN_dRe_theta(H))
        R1 = float(np.interp(1.0, N, Rt)); R9 = float(np.interp(9.0, N, Rt))
        out[f'mean{beta}'] = (8.0/(R9 - R1))/d
    return out


def one(args):
    tag, n_low, x0 = args
    floor, gc, s = solve_triple_nlow(n_low, x0, tag=tag)
    return tag, n_low, floor, gc, s, observables(floor, gc, s)


if __name__ == '__main__':
    # Neutral per-level seeds (NOT the canonical triple -- this sweep is a
    # sensitivity study of the Tier-2 solve and must not consume its digits).
    CASES = [("Nlow0.5", 0.5, (230.0, 1.01, 10.5)),
             ("Nlow1.0", 1.0, (300.0, 1.00, 10.0)),
             ("Nlow2.0", 2.0, (290.0, 1.03, 9.5))]
    with Pool(3) as pool:
        res = pool.map(one, CASES)
    print("\n===== DEPARTURE-ANCHOR LEVEL SWEEP (triple re-solved per level) =====")
    print(f"{'case':>8} | {'floor':>7} {'g_c':>7} {'s':>6} | {'Rt@N7.04':>9} {'Rt@N9':>7} | "
          f"{'bl_mean':>8} {'bl_late':>8} {'m(-.09)':>8} {'m(-.15)':>8}")
    for tag, n_low, floor, gc, s, o in res:
        print(f"{tag:>8} | {floor:7.1f} {gc:7.4f} {s:6.2f} | {o['Rt_op']:9.0f} "
              f"{o['Rt9']:7.0f} | {o['bl_mean']:8.3f} {o['bl_late']:8.3f} "
              f"{o['mean-0.09']:8.3f} {o['mean-0.15']:8.3f}")
