"""FPG (favorable-pressure-gradient) rate & onset audit of the sphere kernel.

Tests the 2026-07-28 user hypothesis on the high-Re cylinder miss: "our
favorable pressure gradient model isn't carefully tuned -- the rate is
abysmally low compared to Drela, and onset might also be high."  Two parts,
CPU only, NO model-constant changes (diagnosis, not tuning):

PART A -- Falkner-Skan FPG ladder, beta in {0, +0.05, +0.10, +0.20, +0.50,
+1.0 (plane stagnation)}:
  1. transported rate: the paper's calibration instrument (fig04_shapefactor
     march of Eq. (transport), canonical (c_nu_ai, k) = (1/6, 0.712),
     nx=1600 x ny=1200) -> dN/dRe_theta secants s_mean (N in [1,5]) and
     s_late (N in [5,9]), and the N=1 crossing Re_theta;
  2. frozen-profile quantities on the exact FS profile (tab_frozen_slope
     build): max_y Omega_hat*I_hat (cross-checked with the savgol W=61
     estimator calibrated in spheroid_a0_physics.py), the gated frozen
     eigenvalue ratio s*I_th/s_DG at Re_theta in {400, 1000, 2000, 4000},
     and the ONSET-GATE opening Re_theta: min over the P>0 band of
     Re_Omega_c(P(y)) * I_th / (y^2 |u'|), with the P-independent saturated
     ceiling Re_theta(Re_Omega=1851.2) as the secondary check (the softmin
     SATURATES as P->0: the gate opens in any thick-enough FPG layer;
     the starvation is the RATE, not the gate);
  3. Drela references: (i) the repo's Drela-Giles 1987 Eq.29/30 fit
     (paper/repro/lib/correlations.py -- the fit the model was calibrated
     against), and (ii) the mfoil envelope (src/validation/mfoil.py
     get_damp, Fidkowski's port of XFOIL: dN/dRet
     da = 0.028(Hk-1) - 0.0345 exp(-(3.87/(Hk-1) - 2.52)^2), critical
     log10 Ret lrc = 2.492/(Hk-1)^0.43 + 0.7(tanh(14/(Hk-1) - 9.24) + 1),
     spatial factor af -- reproduced here and cross-checked by direct call).

PART B -- high-Re cylinder nose map at Re_D in {2e6, 7e6, 2e7} (Tu = 0.2%
up-branch cases of the drag-crisis campaign, records 2026-07-28-0010/0412):
  1. u_e(theta) measured from the converged surface Cp
     (u_e/U_inf = sqrt(1 - Cp), M = 0.1 incompressible surrogate; potential
     2 sin(theta) overlaid for reference);
  2. PLANAR laminar BL march: bl_march of spheroid_a0_meanflow.py
     (mode='planar', geom='flat' -- the 0105 instrument with the Mangler
     terms off), validated in-script against Blasius (H = 2.5905,
     theta = 0.664 sqrt(nu x/ue)) and plane stagnation FS beta=1
     (H = 2.216) -> Re_theta(theta), H(theta);
  3. three-way N(theta): (a) Drela-Giles chain-rule envelope (+ mfoil
     variant), (b) the model's frozen-profile rate integrated along the
     march -- both the pointwise kernel bound max_y(a S omega / u) and the
     gated c_nu_ai=1/6 eigenvalue of eq:frozeneig on the marched profiles
     (savgol W=61 estimator), (c) the solver's chi=1 front
     (paper/data/dragcrisis_matrix_summary.jsonl);
  4. seed thresholds from Mack's map (eq:tumap, run_dragcrisis_matrix.py
     SEEDS): chi_inf = 1.0835e-2 at Tu 0.2% -> N(chi=1) = ln(1/chi_inf) =
     4.525, N(handover chi=c_v1) = 6.485; Tu 0.05%: 7.852/9.812;
     Tu 0.7%: 1.518/3.478.

Outputs (exploratory, NOT paper figures; house style, captions in the
companion record agent-paper-review/2026-07-28-*-fpg-rate-audit.md):
  paper/repro/cfd/figs_explore/fpg_rate_audit_ladder.png
  paper/repro/cfd/figs_explore/fpg_rate_audit_cylinder.png
  paper/repro/cfd/figs_explore/fpg_rate_audit.json      (all numbers)

Data dependencies (regen paths): /local_data/qiqi/sa-ai/dragcrisis_matrix/
case dirs (surface_fluid_cylinder.pvtu + Flow360.json), committed
paper/data/dragcrisis_matrix_summary.jsonl.

Run:  python3 -u paper/repro/cfd/fpg_rate_audit.py [--skip-ladder]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figs_explore")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (HERE, os.path.join(REPO, "paper", "repro"),
          os.path.join(REPO, "paper", "repro", "analytic"),
          os.path.join(REPO, "spheroid")):
    sys.path.insert(0, p)

import _saai                                          # noqa: E402 (chdir paper/)
from fig04_shapefactor import (                       # noqa: E402
    march, profile_ints, drela, sphere_rate, C_NU_AI,
    A_MAX, REOM_CEIL, REOM_A, REOM_B, REOM_N, RAMP_W, K_ANCHOR)
from _saai import SIGMA_SA                            # noqa: E402
from lib.boundary_layer import FalknerSkanWedge       # noqa: E402
from lib.correlations import dN_dRe_theta, Re_theta0  # noqa: E402
from tab_frozen_slope import build, UFLOOR            # noqa: E402

# canon onset constants must match the solver env (saai_env.py / canon)
assert abs(REOM_CEIL - 1851.2) < 1e-9 and abs(REOM_A - 124.6) < 1e-9 \
    and abs(REOM_B - 1.424) < 1e-9, "onset constants drifted from canon"

MATRIX_ROOT = os.environ.get("SAAI_DRAGCRISIS_ROOT",
                             "/local_data/qiqi/sa-ai/dragcrisis_matrix")
SUMMARY_JSONL = os.path.join(REPO, "paper", "data",
                             "dragcrisis_matrix_summary.jsonl")
BETAS = (0.0, 0.05, 0.10, 0.20, 0.50, 1.0)
RTS_EIG = (400.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)
CASES = [  # (Re_D, case dir, mesh family)
    (2e6, "cyl_Re2000000_Tu0.2_up", "pilot"),
    (7e6, "cyl_Re7000000_Tu0.2_up_highre", "highre"),
    (2e7, "cyl_Re20000000_Tu0.2_up_highre", "highre"),
]
# Mack-map seeds, VERIFIED against run_dragcrisis_matrix.SEEDS and eq:tumap
# (chi_inf = c_v1 exp(-N_crit), N_crit = -8.43 - 2.4 ln Tu_frac, c_v1 = 7.1)
SEEDS = {"0.05": 3.8895e-4, "0.2": 1.0835e-2, "0.7": 2.1908e-1}
C_V1 = 7.1
N_CHI1 = {k: float(np.log(1.0 / v)) for k, v in SEEDS.items()}
N_HAND = {k: v + float(np.log(C_V1)) for k, v in N_CHI1.items()}

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.fontsize": 9.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "lines.linewidth": 1.8, "figure.dpi": 110, "savefig.dpi": 150})
C = dict(model="#3b6bb5", modelb="#7aa0d4", drela="#b0483a",
         mfoil="#e0821f", eig="#7a5aa8", gate="#3e8f5c", gray="0.45")


# ------------------------------------------------ mfoil envelope (reference)
def mfoil_envelope(Hk):
    """mfoil/XFOIL envelope pieces (src/validation/mfoil.py get_damp,
    Fidkowski's mfoil port; same fit as XFOIL 6.9x dampl): returns
    (dN/dRe_theta 'da', spatial factor 'af', critical Re_theta 10^lrc).
    Cross-checked against a direct get_damp call in mfoil_crosscheck()."""
    Hk = np.maximum(np.asarray(Hk, float), 1.05)
    Hmi = 1.0 / (Hk - 1.0)
    lrc = 2.492 * Hmi**0.43 + 0.7 * (np.tanh(14 * Hmi - 9.24) + 1.0)
    ar = 3.87 * Hmi - 2.52
    da = 0.028 * (Hk - 1.0) - 0.0345 * np.exp(-ar**2)
    af = (-0.05 + 2.7 * Hmi - 5.5 * Hmi**2 + 3 * Hmi**3
          + 0.1 * np.exp(-20 * Hmi))
    return da, af, 10.0**lrc


def mfoil_crosscheck():
    """Verify the standalone formula against src/validation/mfoil.get_damp
    (theta=1, ue such that Ret is far above critical -> damp = af*da/th)."""
    sys.path.insert(0, os.path.join(REPO, "src", "validation"))
    import mfoil as MF
    param = MF.Param()
    param.mu0, param.rho0 = 1.0e-5, 1.0     # only their ratio enters Ret
    param.ncrit = 9.0
    ok = True
    for Hk, Ret in ((2.3, 2e4), (2.59, 1e4), (3.2, 5e3)):
        th, ue = 1.0, Ret * param.mu0 / param.rho0
        ds = Hk * th   # get_Hk reads Hk from ds/th (Ma=0)
        U = np.array([th, ds, 0.0, ue])
        damp, _ = MF.get_damp(U, param)
        da, af, _ = mfoil_envelope(Hk)
        ref = af * da / th
        ok &= abs(damp - ref) < 1e-10 + 1e-6 * abs(ref)
        print(f"  mfoil cross-check Hk={Hk}: get_damp={damp:.6e} "
              f"formula={ref:.6e}", flush=True)
    return bool(ok)


# ------------------------------------------------------- sphere-kernel bits
def onset_reomc(P):
    """Canonical onset threshold Re_Omega_c(P) (softmin, saturates at
    REOM_CEIL=1851.2 as P->0)."""
    pw = REOM_A + REOM_B * np.maximum(P, 1e-12)**(-2.0)
    return (REOM_CEIL**(-REOM_N) + pw**(-REOM_N))**(-1.0 / REOM_N)


def kernel_on_profile(y, u, up, upp):
    """Sphere-kernel coordinates on a wall-normal profile (planar/FS
    convention: X=|u|, Y=y u', Z=1/2 y^2 u'' sign(u))."""
    X = np.abs(u)
    Y = y * np.abs(up)
    Z = 0.5 * y * y * upp * np.sign(u)
    R = np.sqrt(X * X + Y * Y + Z * Z) + 1e-30
    Shat = Y / (np.hypot(X, Y) + 1e-30)
    P = Shat * (Y - X - Z) / R
    return P


def gate_open_Rt(pr):
    """Gate-opening Re_theta on a frozen FS profile: smallest Rt such that
    Re_Omega(y) = y^2 |u'| Rt / I_th reaches Re_Omega_c(P(y)) somewhere in
    the P>0 band (gate center S=0.5).  Also the P-independent saturated
    ceiling Rt(Re_Omega = REOM_CEIL) and the amplifying-band peak values."""
    y, u, up = pr["y"], pr["u"], pr["up"]
    upp = np.gradient(up, y)
    P = kernel_on_profile(y, u, up, upp)
    reom_per_Rt = y * y * np.abs(up) / pr["I_th"]     # Re_Omega / Re_theta
    band = (P > 1e-9) & (reom_per_Rt > 1e-12)
    out = dict(maxP=float(np.max(P)),
               reom_ratio_max=float(np.max(reom_per_Rt)),
               Rt_gate_sat=float(REOM_CEIL / max(np.max(reom_per_Rt), 1e-30)))
    if band.any():
        need = onset_reomc(P[band]) / reom_per_Rt[band]
        j = int(np.argmin(need))
        jj = np.flatnonzero(band)[j]
        out.update(Rt_gate=float(need[j]), y_gate=float(y[jj]),
                   P_at_gate=float(P[jj]),
                   reomc_at_gate=float(onset_reomc(P[jj])))
    else:
        out.update(Rt_gate=float("inf"), y_gate=float("nan"),
                   P_at_gate=0.0, reomc_at_gate=float(REOM_CEIL))
    return out


def gated_eig_ratio(pr, Rt, cnu=C_NU_AI):
    """GATED frozen-profile eigenvalue of eq:frozeneig at Re_theta = Rt
    (tab_frozen_slope.s_ratio machinery + the canonical onset gate in b),
    as a ratio to Drela s_DG."""
    y, u, up = pr["y"], pr["u"], pr["up"]
    upp = np.gradient(up, y)
    P = kernel_on_profile(y, u, up, upp)
    reom = y * y * np.abs(up) * Rt / pr["I_th"]
    gate = 0.5 * (1.0 + np.tanh((reom / onset_reomc(np.maximum(P, 0.0))
                                 - 1.0) / RAMP_W))
    b = A_MAX * np.clip(P, 0.0, 1.0) * gate * np.abs(up)
    D = cnu * pr["I_th"] / (SIGMA_SA * Rt)
    h = pr["h"]
    uf = np.maximum(u, UFLOOR)[1:-1]
    d = (b[1:-1] - 2.0 * D / h**2) / uf
    e = (D / h**2) / np.sqrt(uf[:-1] * uf[1:])
    w = eigh_tridiagonal(d, e, select="i",
                         select_range=(len(d) - 1, len(d) - 1))[0]
    return float(w[0]) * pr["I_th"] / pr["sDG"]


def savgol_maxP_check(fs, n=600, W=61):
    """The 0040 thread's savgol estimator (calibrated on exact FS profiles,
    <1% at W=61) applied to this profile: cross-check of the direct maxP."""
    eta99 = np.interp(0.99, np.maximum.accumulate(fs.u), fs.eta)
    yg = np.linspace(0.0, 3.0 * eta99, n)
    h = yg[1] - yg[0]
    ug = np.interp(yg, fs.eta, fs.u)
    u_s = savgol_filter(ug, W, 4)
    du = savgol_filter(ug, W, 4, deriv=1, delta=h)
    d2u = savgol_filter(ug, W, 4, deriv=2, delta=h)
    return float(np.max(kernel_on_profile(yg, u_s, du, d2u)))


# --------------------------------------------------- PART A: the FS ladder
def marched_secants(beta):
    """fig04_shapefactor march (canonical instrument, nx=1600 x ny=1200)
    with the domain-adaptation loop of measures_for_beta; returns the
    dN/dRe_theta secants and N-crossings."""
    fs = FalknerSkanWedge(beta)
    I_th, H = profile_ints(fs)
    x_max = 4e6 if beta == 0.0 else 3e5
    for _ in range(14):
        xs, N = march(fs, x_max, nx=400, ny=400)      # coarse domain sizing
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.15
            continue
        if N[-1] > 14.0:
            x_max = 1.1 * float(np.interp(14.0, N, xs))
            break
        x_max *= 3.0
    xs, N = march(fs, x_max)                          # canonical resolution
    Ue = fs.inviscid_at(np.maximum(xs, 1e-12))
    Rt = I_th * np.sqrt(xs * Ue)
    cross = {f"Rt{n:g}": (float(np.interp(n, N, Rt)) if N[-1] >= n
                          else float("nan")) for n in (1.0, 5.0, 9.0)}
    s_mean = (4.0 / (cross["Rt5"] - cross["Rt1"])
              if np.isfinite(cross["Rt5"]) else float("nan"))
    s_late = (4.0 / (cross["Rt9"] - cross["Rt5"])
              if np.isfinite(cross["Rt9"]) else float("nan"))
    return dict(H=float(H), I_th=float(I_th), x_max=float(x_max),
                N_end=float(N[-1]), s_mean=s_mean, s_late=s_late, **cross)


def run_ladder():
    rows = []
    for beta in BETAS:
        fs = FalknerSkanWedge(beta)
        pr = build(beta, None)
        g = gate_open_Rt(pr)
        m = marched_secants(beta)
        H = pr["H"]
        dDG = float(np.asarray(dN_dRe_theta(H)))
        Rt0 = float(np.asarray(Re_theta0(H)))
        da, af, Rtc_mf = mfoil_envelope(H)
        eig = {f"{int(r)}": gated_eig_ratio(pr, r) for r in RTS_EIG}
        row = dict(beta=beta, H=H, maxP=g["maxP"],
                   maxP_savgol=savgol_maxP_check(fs),
                   drela_dNdRt=dDG, drela_Rt0=Rt0,
                   mfoil_dNdRt=float(da), mfoil_Rtcrit=float(Rtc_mf),
                   s_mean=m["s_mean"], s_late=m["s_late"],
                   ratio_mean=m["s_mean"] / dDG, ratio_late=m["s_late"] / dDG,
                   Rt1=m["Rt1"], Rt5=m["Rt5"], Rt9=m["Rt9"],
                   Rt1_drela=Rt0 + 1.0 / dDG,
                   Rt_gate=g["Rt_gate"], Rt_gate_sat=g["Rt_gate_sat"],
                   y_gate=g["y_gate"], P_at_gate=g["P_at_gate"],
                   reomc_at_gate=g["reomc_at_gate"],
                   onset_ratio=g["Rt_gate"] / Rt0,
                   eig_ratio_gated=eig, N_end=m["N_end"], x_max=m["x_max"])
        rows.append(row)
        print(f"beta={beta:+.2f} H={H:.3f} maxP={g['maxP']:.4f} "
              f"(savgol {row['maxP_savgol']:.4f}) | "
              f"s_mean={m['s_mean']:.3e} s_late={m['s_late']:.3e} "
              f"Drela={dDG:.3e} mfoil={da:.3e} | ratio mean/late "
              f"{row['ratio_mean']:.3f}/{row['ratio_late']:.3f} | "
              f"Rt1={m['Rt1']:.0f} (DG N1 {row['Rt1_drela']:.0f}) "
              f"gate@{g['Rt_gate']:.0f} sat@{g['Rt_gate_sat']:.0f} "
              f"Rt0={Rt0:.0f}", flush=True)
    return rows


# ------------------------------------------------- PART B: cylinder nose map
def surface_ue(case_dir):
    """theta (deg from forward stagnation) and u_e/U_inf from surface Cp.
    Both spanwise rows and both sides pooled (steady symmetric branch)."""
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(os.path.join(case_dir, "surface_fluid_cylinder.pvtu"))
    r.Update()
    g = r.GetOutput()
    P3 = vtk_to_numpy(g.GetPoints().GetData())
    cp = vtk_to_numpy(g.GetPointData().GetArray("Cp"))
    th = 180.0 - np.abs(np.degrees(np.arctan2(P3[:, 2], P3[:, 0] - 0.5)))
    bins = np.arange(0.0, 180.1, 0.25)
    tb, cb = [], []
    idx = np.digitize(th, bins)
    for b in np.unique(idx):
        m = idx == b
        tb.append(float(th[m].mean()))
        cb.append(float(cp[m].mean()))
    tb, cb = np.array(tb), np.array(cb)
    o = np.argsort(tb)
    return tb[o], cb[o]


def cp_to_ue(cp, mach=0.1):
    """Exact isentropic Cp -> u_e/U_inf (handles the compressible
    stagnation Cp = 1.004 > 1 at M = 0.1 that breaks sqrt(1-Cp))."""
    g = 1.4
    p_rat = 1.0 + 0.5 * g * mach**2 * np.asarray(cp)   # p/p_inf
    p0 = (1.0 + 0.2 * mach**2)**3.5                    # p0/p_inf
    r = np.clip(p0 / np.maximum(p_rat, 1e-9), 1.0, None)
    Me2 = (r**(1.0 / 3.5) - 1.0) / 0.2
    Te = (1.0 + 0.2 * mach**2) / (1.0 + 0.2 * Me2)     # T_e/T_inf
    return np.sqrt(Me2 * Te) / mach


def cylinder_march(theta_deg, cp, nu, th_max=88.0, mach=0.1):
    """Planar laminar BL march (bl_march, mode='planar', geom='flat') on the
    measured u_e(theta).  Returns march dict + profile stations each 1 deg.
    Stagnation stub: odd polynomial ue = k s + c s^3 fitted over
    theta in [3, 20] deg, scale-matched to the measured curve at the
    6-deg joint (the Cp -> ue map is ill-conditioned below ~3 deg)."""
    from spheroid_a0_meanflow import bl_march, ygrid
    ue_raw = cp_to_ue(cp, mach)
    m = theta_deg <= th_max + 4.0
    ue_s = savgol_filter(ue_raw[m], 31, 3)
    s_raw = 0.5 * np.radians(theta_deg[m])            # R = 0.5 D
    itp = PchipInterpolator(s_raw, ue_s)
    fit = (theta_deg[m] >= 3.0) & (theta_deg[m] <= 20.0)
    A = np.stack([s_raw[fit], s_raw[fit]**3], 1)
    kfit, c3 = np.linalg.lstsq(A, ue_s[fit], rcond=None)[0]
    kfit = float(kfit)
    s_jn = 0.5 * np.radians(6.0)
    scale = float(itp(s_jn) / (kfit * s_jn + c3 * s_jn**3))
    s_max = 0.5 * np.radians(th_max)
    sm = np.linspace(2e-4, s_max, 4000)
    ue_m = np.where(sm >= s_jn, itp(np.clip(sm, s_jn, s_raw[-1])),
                    scale * (kfit * sm + c3 * sm**3))
    # wall-normal grid sized to the stagnation momentum thickness
    th_stag = 0.29 * np.sqrt(nu / kfit)
    y = ygrid(h1=max(th_stag / 60.0, 1e-8), ytop=0.02, n=460)
    st_theta = np.arange(4.0, th_max + 0.001, 1.0)
    st_s = list(0.5 * np.radians(st_theta))
    theta0 = float(np.sqrt(0.075 * nu / kfit))        # planar Thwaites, ue=ks
    out = bl_march(sm, ue_m, nu, "planar", x_profiles=st_s,
                   theta0=theta0, y=y, geom="flat")
    out["ue"] = np.interp(out["x"], sm, ue_m)
    # truncate past laminar separation (planar march invalid there)
    bad = np.where((out["cf_theta"] <= 1e-4) | (out["H"] >= 4.5))[0]
    if len(bad):
        j = int(bad[0])
        s_cut = float(out["x"][j])
        for kkey in ("x", "theta", "dstar", "H", "cf_theta", "ue"):
            out[kkey] = out[kkey][:j]
        out["profiles"] = {k: v for k, v in out["profiles"].items()
                           if k < s_cut}
        st_theta = st_theta[0.5 * np.radians(st_theta) < s_cut]
        print(f"  march truncated at laminar separation theta="
              f"{np.degrees(2 * s_cut):.1f} deg", flush=True)
    out["theta_deg"] = np.degrees(2.0 * out["x"])
    out["st_theta"] = st_theta
    out["k_stag"] = kfit
    return out


def profile_kernel(y, u, ue, nu, W=61, n=600):
    """Savgol-estimator kernel on a marched profile (PLANAR_W convention of
    spheroid_a0_physics.py) + the two model rates:
    sup: max_y a S omega / u (u > 0.3 ue), and the gated c=1/6 eigenvalue."""
    j99 = int(np.argmax(u >= 0.99 * ue))
    d99 = float(y[max(j99, 1)])
    yg = np.linspace(0.0, 3.0 * d99, n)
    h = yg[1] - yg[0]
    ug = np.interp(yg, y, u)
    u_s = savgol_filter(ug, W, 4)
    du = savgol_filter(ug, W, 4, deriv=1, delta=h)
    d2u = savgol_filter(ug, W, 4, deriv=2, delta=h)
    P = kernel_on_profile(yg, u_s, du, d2u)
    reom = yg * yg * np.abs(du) / nu
    gate = 0.5 * (1.0 + np.tanh((reom / onset_reomc(np.maximum(P, 0.0))
                                 - 1.0) / RAMP_W))
    rate = A_MAX * np.clip(P, 0.0, 1.0) * gate
    sel = (u_s > 0.3 * ue) & (yg > 0)
    sup = float(np.max(np.where(sel, rate * np.abs(du) /
                                np.maximum(u_s, 1e-30), 0.0)))
    # gated eigenvalue (eq:frozeneig) on the same profile
    b = rate * np.abs(du)
    uf = np.maximum(u_s, 0.02 * ue)[1:-1]
    D = C_NU_AI * nu / SIGMA_SA
    d = (b[1:-1] - 2.0 * D / h**2) / uf
    e = (D / h**2) / np.sqrt(uf[:-1] * uf[1:])
    w = eigh_tridiagonal(d, e, select="i",
                         select_range=(len(d) - 1, len(d) - 1))[0]
    jP = int(np.argmax(P))
    jR = int(np.argmax(reom))
    return dict(maxP=float(P[jP]), reom_max=float(reom[jR]),
                reomc_at_maxP=float(onset_reomc(max(P[jP], 1e-12))),
                gate_max=float(np.max(gate)), sup=sup,
                s_eig=float(w[0]), d99=d99)


def marcher_validation_checks(nu=1e-7):
    """(1) Blasius flat plate; (2) plane-stagnation FS beta=1 (ue = k s):
    the planar marcher must reproduce H = 2.5905 and 2.216."""
    from spheroid_a0_meanflow import bl_march, ygrid
    checks = {}
    x = np.concatenate([np.linspace(0.02, 0.06, 200, endpoint=False),
                        np.arange(0.06, 1.2001, 5e-4)])
    m = bl_march(x, np.full_like(x, 0.1), nu, "planar", geom="flat",
                 y=ygrid(2e-6, 0.03, 420))
    j = int(np.argmin(np.abs(m["x"] - 1.0)))
    th_bl = 0.664 * np.sqrt(nu * 1.0 / 0.1)
    checks["blasius"] = dict(H=float(m["H"][j]),
                             theta_ratio=float(m["theta"][j] / th_bl))
    k = 4.0
    x = np.linspace(1e-3, 0.4, 3000)
    m = bl_march(x, k * x, nu, "planar", geom="flat",
                 theta0=float(np.sqrt(0.075 * nu / k)),
                 y=ygrid(2e-7, 0.004, 420))
    fs1 = FalknerSkanWedge(1.0)
    I_th, H1 = profile_ints(fs1)
    j = int(np.argmin(np.abs(m["x"] - 0.35)))
    # FS beta=1 exact: theta = I_th * sqrt(nu x / ue) with Ue = k x
    th_fs = I_th * np.sqrt(nu * 0.35 / (k * 0.35))
    checks["stagnation"] = dict(H=float(m["H"][j]), H_exact=float(H1),
                                theta_ratio=float(m["theta"][j] / th_fs))
    print(f"  marcher validation: Blasius H={checks['blasius']['H']:.4f} "
          f"(2.5905), theta/exact={checks['blasius']['theta_ratio']:.4f}; "
          f"FS beta=1 H={checks['stagnation']['H']:.4f} "
          f"({H1:.4f}), theta/exact="
          f"{checks['stagnation']['theta_ratio']:.4f}", flush=True)
    return checks


def solver_fronts():
    fronts = {}
    for line in open(SUMMARY_JSONL):
        r = json.loads(line)
        for re_d, case, _fam in CASES:
            if r["case"] == case:
                fronts[case] = dict(
                    chi1_front=float(min(r["chi1_front_upper"],
                                         r["chi1_front_lower"])),
                    Cd=r["Cd"], knee_upper=r.get("knee_upper"))
    return fronts


def run_cylinder():
    checks = marcher_validation_checks()
    fronts = solver_fronts()
    out = []
    for re_d, case, fam in CASES:
        cdir = os.path.join(MATRIX_ROOT, case)
        fj = json.load(open(os.path.join(cdir, "Flow360.json")))
        mach = fj["freestream"]["Mach"]
        nu = fj["freestream"]["muRef"] / mach          # units U_inf = D = 1
        assert abs(1.0 / nu - re_d) / re_d < 1e-6
        th, cp = surface_ue(cdir)
        mres = cylinder_march(th, cp, nu, th_max=92.0)
        # kernel + rates on each stored profile
        st_theta = mres["st_theta"]
        rows = []
        for tq in st_theta:
            sq = 0.5 * np.radians(tq)
            key = min(mres["profiles"], key=lambda s: abs(s - sq))
            y, u, ue = mres["profiles"][key]
            kk = profile_kernel(y, u, ue, nu)
            kk["theta"] = float(tq)
            kk["Rt"] = float(np.interp(sq, mres["x"], mres["theta"]) * ue / nu)
            kk["H"] = float(np.interp(sq, mres["x"], mres["H"]))
            rows.append(kk)
        # three-way N(theta) on the march grid
        s = mres["x"]
        ue_m = mres["ue"]
        Rt = ue_m * mres["theta"] / nu
        H = mres["H"]
        Rt0 = np.asarray(Re_theta0(H))
        dDG = np.asarray(dN_dRe_theta(H))
        dRt_ds = np.gradient(Rt, s)
        dN = np.where(Rt > Rt0, np.maximum(dDG * dRt_ds, 0.0), 0.0)
        N_dg = np.concatenate([[0.0], np.cumsum(
            0.5 * (dN[1:] + dN[:-1]) * np.diff(s))])
        da, af, Rtc_mf = mfoil_envelope(H)
        dNm = np.where(Rt > Rtc_mf, np.maximum(da * dRt_ds, 0.0), 0.0)
        N_mf = np.concatenate([[0.0], np.cumsum(
            0.5 * (dNm[1:] + dNm[:-1]) * np.diff(s))])
        # model: integrate sup and eig rates over s from the station table
        st_s = 0.5 * np.radians(st_theta)
        sup = np.array([r["sup"] for r in rows])
        eig = np.clip(np.array([r["s_eig"] for r in rows]), 0.0, None)
        N_sup = np.concatenate([[0.0], np.cumsum(
            0.5 * (sup[1:] + sup[:-1]) * np.diff(st_s))])
        N_eig = np.concatenate([[0.0], np.cumsum(
            0.5 * (eig[1:] + eig[:-1]) * np.diff(st_s))])
        th_g = mres["theta_deg"]

        def crossing(xv, Nv, level):
            hit = np.where(Nv >= level)[0]
            if len(hit) and hit[0] > 0:
                j = hit[0]
                return float(np.interp(level, Nv[j - 1:j + 1],
                                       xv[j - 1:j + 1]))
            return float("nan")

        lv1, lvh = N_CHI1["0.2"], N_HAND["0.2"]
        rec = dict(
            re=re_d, case=case, mesh=fam, nu=nu,
            chi1_front_solver=fronts[case]["chi1_front"],
            Cd=fronts[case]["Cd"],
            k_stag=mres["k_stag"],
            theta_deg=[float(v) for v in th_g[::40]],
            Rt=[float(v) for v in Rt[::40]],
            H=[float(v) for v in H[::40]],
            N_drela=[float(v) for v in N_dg[::40]],
            N_mfoil=[float(v) for v in N_mf[::40]],
            st_theta=[float(v) for v in st_theta],
            N_model_sup=[float(v) for v in N_sup],
            N_model_eig=[float(v) for v in N_eig],
            station_rows=rows,
            cross=dict(
                drela_chi1=crossing(th_g, N_dg, lv1),
                drela_hand=crossing(th_g, N_dg, lvh),
                mfoil_chi1=crossing(th_g, N_mf, lv1),
                mfoil_hand=crossing(th_g, N_mf, lvh),
                model_sup_chi1=crossing(st_theta, N_sup, lv1),
                model_eig_chi1=crossing(st_theta, N_eig, lv1)),
            N_at_75=dict(drela=float(np.interp(75.0, th_g, N_dg)),
                         mfoil=float(np.interp(75.0, th_g, N_mf)),
                         model_sup=float(np.interp(75.0, st_theta, N_sup)),
                         model_eig=float(np.interp(75.0, st_theta, N_eig))),
            N_at_end=dict(theta=float(th_g[-1]), drela=float(N_dg[-1]),
                          mfoil=float(N_mf[-1]),
                          model_sup=float(N_sup[-1]),
                          model_eig=float(N_eig[-1])),
            onset_theta=dict(
                drela=crossing(th_g, (Rt > Rt0).astype(float), 0.5),
                mfoil=crossing(th_g, (Rt > Rtc_mf).astype(float), 0.5),
                gate_sat=crossing(
                    [r["theta"] for r in rows],
                    np.array([r["reom_max"] for r in rows]) / REOM_CEIL, 1.0)),
            ue_vs_potential={f"{t:g}": [
                float(np.interp(0.5 * np.radians(t), s, ue_m)),
                float(2.0 * np.sin(np.radians(t)))] for t in (30, 60, 75)},
            march=dict(theta_deg=[float(v) for v in th_g[::40]],
                       ue=[float(v) for v in ue_m[::40]]))
        out.append(rec)
        print(f"Re={re_d:.0e} [{fam}]: solver chi1 front "
              f"{rec['chi1_front_solver']:.0f} deg | N@75deg: "
              f"DG {rec['N_at_75']['drela']:.2f}, mfoil "
              f"{rec['N_at_75']['mfoil']:.2f}, model sup "
              f"{rec['N_at_75']['model_sup']:.2f}, eig "
              f"{rec['N_at_75']['model_eig']:.2f} | need {lv1:.2f} (chi=1) "
              f"/ {lvh:.2f} (handover) | DG onset "
              f"{rec['onset_theta']['drela']:.1f} deg, gate-sat "
              f"{rec['onset_theta']['gate_sat']:.1f} deg | at march end "
              f"{rec['N_at_end']['theta']:.1f} deg: DG "
              f"{rec['N_at_end']['drela']:.2f}, sup "
              f"{rec['N_at_end']['model_sup']:.2f}, eig "
              f"{rec['N_at_end']['model_eig']:.2f} | DG chi1-crossing "
              f"{rec['cross']['drela_chi1']:.1f} deg", flush=True)
    return out, checks


# ------------------------------------------------------------------ figures
def fig_ladder(rows, fp):
    H = np.array([r["H"] for r in rows])
    beta = np.array([r["beta"] for r in rows])
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.6))
    a1, a2, a3, a4 = axs.ravel()
    a1.semilogy(H, [r["drela_dNdRt"] for r in rows], "o--", color=C["drela"],
                label="Drela-Giles Eq.29 (repo fit)")
    a1.semilogy(H, [r["mfoil_dNdRt"] for r in rows], "s:", color=C["mfoil"],
                label="mfoil/XFOIL envelope")
    a1.semilogy(H, [r["s_mean"] for r in rows], "o-", color=C["model"],
                label="model marched $N\\in[1,5]$ secant")
    a1.semilogy(H, [r["s_late"] for r in rows], "s-", color=C["modelb"],
                label="model marched $N\\in[5,9]$ secant")
    for r in rows:
        a1.annotate(f"$\\beta$={r['beta']:g}", (r["H"], r["drela_dNdRt"]),
                    textcoords="offset points", xytext=(4, 5), fontsize=8.5,
                    color=C["gray"])
    a1.set_xlabel("H")
    a1.set_ylabel(r"$dN/dRe_\theta$")
    a1.legend(loc="lower right")
    a2.semilogy(H, [r["ratio_mean"] for r in rows], "o-", color=C["model"],
                label="mean-secant / Drela")
    a2.semilogy(H, [r["ratio_late"] for r in rows], "s-", color=C["modelb"],
                label="late-secant / Drela")
    a2.axhline(1.0, color="k", lw=0.8)
    a2.set_xlabel("H")
    a2.set_ylabel("model / Drela rate ratio")
    a2.legend(loc="lower right")
    a3.semilogy(H, [r["drela_Rt0"] for r in rows], "o--", color=C["drela"],
                label=r"Drela critical $Re_{\theta 0}(H)$")
    a3.semilogy(H, [r["mfoil_Rtcrit"] for r in rows], "s:", color=C["mfoil"],
                label="mfoil critical")
    a3.semilogy(H, [r["Rt_gate"] for r in rows], "o-", color=C["gate"],
                label="model gate opens (frozen profile)")
    a3.semilogy(H, [r["Rt_gate_sat"] for r in rows], "^-.", color=C["gate"],
                alpha=0.6, label=r"saturated ceiling $Re_\Omega{=}1851$")
    a3.semilogy(H, [r["Rt1"] for r in rows], "d-", color=C["model"],
                label="model marched $N{=}1$")
    a3.semilogy(H, [r["Rt1_drela"] for r in rows], "d--", color=C["drela"],
                alpha=0.7, label="Drela implied $N{=}1$")
    a3.set_xlabel("H")
    a3.set_ylabel(r"onset $Re_\theta$")
    a3.legend(loc="upper right", fontsize=8.5)
    a4.semilogy(beta, [r["maxP"] for r in rows], "o-", color=C["model"],
                label=r"$\max_y \hat\Omega\hat I$ (exact profile)")
    a4.semilogy(beta, [r["maxP_savgol"] for r in rows], "x", ms=8,
                color=C["eig"], label="savgol W=61 estimator")
    a4.set_xlabel(r"wedge parameter $\beta$")
    a4.set_ylabel(r"$\max_y \hat\Omega\hat I$")
    a4.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def fig_cylinder(recs, fp):
    fig, axs = plt.subplots(len(recs), 3, figsize=(15.0, 4.0 * len(recs)),
                            sharex="col")
    lv1, lvh = N_CHI1["0.2"], N_HAND["0.2"]
    for i, rec in enumerate(recs):
        aH, aL, aR = axs[i]
        th = np.array(rec["theta_deg"])
        H = np.array(rec["H"])
        aH.plot(th, H, color=C["model"])
        aH.axhline(2.591, color=C["gray"], lw=0.9, ls=":")
        if i == 0:
            aH.text(4, 2.60, "Blasius 2.59", fontsize=8.5, color=C["gray"])
        aH.set_ylabel(f"Re={rec['re']:.0e}\nH")
        aH.set_ylim(2.0, 3.6)
        Rt = np.array(rec["Rt"])
        aL.semilogy(th, Rt, color=C["model"],
                    label=r"$Re_\theta$ (planar march)")
        aL.semilogy(th, np.asarray(Re_theta0(H)), "--", color=C["drela"],
                    label=r"Drela critical $Re_{\theta 0}(H)$")
        aL.semilogy(th, mfoil_envelope(H)[2], ":", color=C["mfoil"],
                    label="mfoil critical")
        aL.set_ylabel(r"$Re_\theta$")
        if i == 0:
            aL.legend(loc="lower right")
        aR.plot(th, rec["N_drela"], color=C["drela"],
                label="Drela-Giles envelope")
        aR.plot(th, rec["N_mfoil"], ":", color=C["mfoil"], label="mfoil fit")
        aR.plot(rec["st_theta"], rec["N_model_sup"], color=C["model"],
                label="model kernel bound (sup)")
        aR.plot(rec["st_theta"], rec["N_model_eig"], "--", color=C["eig"],
                label=r"model gated eigenvalue ($c_{\nu}{=}1/6$)")
        aR.axhline(lv1, color=C["gray"], lw=0.9, ls=":")
        aR.axhline(lvh, color=C["gray"], lw=0.9, ls="-.")
        aR.text(2, lv1 + 0.15, r"$N(\chi{=}1)=4.53$ (Tu 0.2%)", fontsize=8.5,
                color=C["gray"])
        aR.text(2, lvh + 0.15, r"$N$(handover)$=6.49$", fontsize=8.5,
                color=C["gray"])
        aR.axvline(rec["chi1_front_solver"], color="k", lw=1.4, ls="-.")
        aR.text(rec["chi1_front_solver"] - 1.5, 0.4,
                f"solver $\\chi{{=}}1$ front {rec['chi1_front_solver']:.0f}"
                r"$^\circ$", rotation=90, fontsize=9, ha="right")
        aR.set_ylabel("N")
        aR.set_ylim(0, 14)
        aR.set_xlim(0, 100)
        if i == 0:
            aR.legend(loc="upper left")
        if i == len(recs) - 1:
            for a in (aH, aL, aR):
                a.set_xlabel(r"$\theta$ [deg from forward stagnation]")
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ladder", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIGD, exist_ok=True)
    print(f"canon: a_max={A_MAX}, c_nu_ai={C_NU_AI:.4f}, k={K_ANCHOR}, "
          f"ReOmc=(ceil {REOM_CEIL}, A {REOM_A}, B {REOM_B}), "
          f"ramp {RAMP_W}", flush=True)
    print(f"seed thresholds (Mack eq:tumap): N(chi=1)={N_CHI1}, "
          f"N(handover)={N_HAND}", flush=True)
    mf_ok = mfoil_crosscheck()

    out = dict(constants=dict(a_max=A_MAX, c_nu_ai=C_NU_AI, k=K_ANCHOR,
                              reomc=[REOM_CEIL, REOM_A, REOM_B],
                              ramp=RAMP_W, seeds=SEEDS, N_chi1=N_CHI1,
                              N_handover=N_HAND),
               mfoil_crosscheck_ok=mf_ok)
    if not args.skip_ladder:
        print("\n=== PART A: Falkner-Skan FPG ladder ===", flush=True)
        out["ladder"] = run_ladder()
        fig_ladder(out["ladder"],
                   os.path.join(FIGD, "fpg_rate_audit_ladder.png"))
    print("\n=== PART B: cylinder nose map ===", flush=True)
    recs, checks = run_cylinder()
    out["cylinder"] = recs
    out["marcher_validation"] = checks
    fig_cylinder(recs, os.path.join(FIGD, "fpg_rate_audit_cylinder.png"))
    fj = os.path.join(FIGD, "fpg_rate_audit.json")
    json.dump(out, open(fj, "w"), indent=1)
    print("\nwrote", fj, flush=True)


if __name__ == "__main__":
    main()
