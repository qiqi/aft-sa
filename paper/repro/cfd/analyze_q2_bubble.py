"""How much did the second gate pinch (Q2) reduce in-bubble over-amplification?

Controlled measurement on the Eppler 387: same kernel, Q2-ON (flow360_v2)
vs Q2-OFF (env AI_VG_GATE_C2=-1), tracking the amplifying cell through the
bubble (regen_eppler_v2.trajectory: argmax of a*|omega|*nuHat per x-bin).
Reports, over the PRE-TRANSITION laminar growth window (x_sep -> where the
implied N first reaches N_crit=9):
  * dN_impl/dx, Q2-on vs Q2-off vs the e^9 reference dN/dx,
  * the transition station x_tr (implied N = 9), on/off/e9,
  * the profile-fullness Gamma at which the amplifying disturbance rides
    (the diagnostic for WHETHER Q2's pocket, centered at the u=0 crossing
    Gamma~2, even overlaps the transitioning mode).
"""
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regen_eppler_v2 as R

CHI_INF = R.C_V1*np.exp(-9.0)
ON = "/home/qiqi/flexcompute/sa-ai/flow360_v2"
OFF = "/home/qiqi/flexcompute/sa-ai/flow360_q2off"
AI = "/home/qiqi/flexcompute/sa-ai/flow360_ai"

CASES = [
    ("a5 Re200k cav", "cavL1prop_eppler387_Re200k_a5", 200, ("mfoil", 5.0)),
    ("a7 Re200k cav", "cavL1prop_eppler387_Re200k_a7", 200, ("mfoil", 7.0)),
    ("a5 Re60k  cav", "sweep_Re60k_a5",                 60, ("flex", 60)),
]

_mf = pickle.load(open(f"{AI}/mfoil_eppler387_Re200k.pkl", "rb"))
_ff = pickle.load(open(f"{AI}/flexfoil_eppler387_sweep_a5.pkl", "rb"))


def model_curve(case_dir):
    """(x, N_impl, Gamma) on the upper surface, tracking the amplifying cell."""
    tr = R.trajectory(case_dir, side='upper')
    if tr is None:
        return None
    xc, ReO, Gam, chi = tr
    with np.errstate(divide='ignore', invalid='ignore'):
        N = np.log(chi/CHI_INF)
    return xc, N, Gam


def e9_curve(src):
    kind, key = src
    st = (_mf.get(key) if kind == "mfoil" else _ff.get(key))
    up = st["upper"]
    x = np.asarray(up["x"], float)
    N = np.asarray(up["n"], float)
    m = np.isfinite(N)
    return x[m], N[m]


def x_at_N(x, N, Nt=9.0):
    m = np.isfinite(N)
    x, N = x[m], N[m]
    i = np.argmax(N >= Nt)
    if not (N[i] >= Nt) or i == 0:
        return np.nan
    f = (Nt - N[i-1])/(N[i] - N[i-1] + 1e-30)
    return float(x[i-1] + f*(x[i]-x[i-1]))


def growth_rate(x, N, x0, x1):
    """mean dN/dx over [x0, x1] on the rising (N in [1,9]) part."""
    m = np.isfinite(N) & (x >= x0) & (x <= x1) & (N >= 1.0) & (N <= 9.0)
    if m.sum() < 2:
        return np.nan
    xm, Nm = x[m], N[m]
    return float((Nm[-1]-Nm[0])/(xm[-1]-xm[0]))


def bubble_sep(case_dir):
    (xu, cfu, _), _ = R.airfoil_walk_contour(case_dir)
    o = np.argsort(xu); xu, cfu = xu[o], cfu[o]
    m = (xu > 0.1) & (xu < 0.99); xu, cfu = xu[m], cfu[m]
    neg = cfu < 0
    return float(xu[np.argmax(neg)]) if neg.any() else np.nan


def main():
    print(f"{'case':>15} | {'x_sep':>6} | {'x_tr (N=9) on/off/e9':>22} | "
          f"{'dN/dx[x_sep..x_tr] on/off/e9':>30} | {'Gamma@tr on/off':>16}")
    for tag, name, Rk, src in CASES:
        don, doff = f"{ON}/{name}", f"{OFF}/{name}"
        mc_on, mc_off = model_curve(don), model_curve(doff)
        if mc_on is None or mc_off is None:
            print(f"{tag:>15} | slice missing"); continue
        xon, Non, Gon = mc_on
        xof, Nof, Gof = mc_off
        xe, Ne = e9_curve(src)
        xsep = bubble_sep(don)
        tr_on, tr_off, tr_e = (x_at_N(xon, Non), x_at_N(xof, Nof),
                               x_at_N(xe, Ne))
        # growth window: separation to the model's own transition
        r_on = growth_rate(xon, Non, xsep, tr_on if np.isfinite(tr_on) else 1)
        r_off = growth_rate(xof, Nof, xsep,
                            tr_off if np.isfinite(tr_off) else 1)
        r_e = growth_rate(xe, Ne, xsep, tr_e if np.isfinite(tr_e) else 1)
        # Gamma at the tracked cell nearest each transition station
        g_on = float(Gon[np.nanargmin(np.abs(xon - tr_on))]) \
            if np.isfinite(tr_on) else np.nan
        g_off = float(Gof[np.nanargmin(np.abs(xof - tr_off))]) \
            if np.isfinite(tr_off) else np.nan
        print(f"{tag:>15} | {xsep:6.3f} | "
              f"{tr_on:6.3f}/{tr_off:6.3f}/{tr_e:6.3f}   | "
              f"{r_on:8.1f}/{r_off:8.1f}/{r_e:8.1f}       | "
              f"{g_on:5.2f}/{g_off:5.2f}", flush=True)


if __name__ == "__main__":
    main()
