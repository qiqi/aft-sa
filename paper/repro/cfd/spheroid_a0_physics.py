"""Physics deep-dive on the alpha=0, Re_L=7.2e6 spheroid miss (re72a0 L2):
what overwhelms the amplifying band mid-body, given c_nu_ai = 1/6 was chosen
(Sec II.D frozen-profile eigenvalue) precisely so laminar diffusion would NOT
overwhelm it?

Follow-on to the kill-chain audit (agent-paper-review/
2026-07-27-2142-spheroid-flank-kernel-audit.md, Part 1.2): gate OPEN, P small
(0.02-0.06) and decaying, kernel bound 8.2 e-folds by the measured front but
the transported chi realizes 4.4 and then flat-lines (5.0 -> 5.2 over
x/L = 0.55-0.85).  This script quantifies WHY the realization stalls:

  1. BL character along the phi=90 meridian: Re_theta(x/L), H(x/L) computed
     properly (displacement/momentum thickness from the wall-normal meridional
     velocity profiles), fronts marked (measured 0.438 / Stock e^N 0.425 from
     paper/data/stock2006_fig14a_digitized.json; our chi=1 front re-extracted
     from this sweep).
  2. Laminar-envelope e^N on OUR field's edge conditions: u_e(x) extracted
     from the L2 field, axisymmetric (Mangler-weighted) Thwaites march for
     laminar theta/H, Drela-Giles critical Re_theta0(H) onset + envelope
     dN/dRe_theta integrated to N(x); crossings vs N = 6 / 8 / 11.65.
  3. The chi field: log10(chi) in the (x/L, wall-distance) plane with the
     P=0 amplifying-band boundary, the S=0.5 onset-gate contour, and
     delta99(x)/theta(x) overlaid -- the sliver and the stall, visually.
  4. Band-width / confinement-penalty quantification: at six stations,
     amplifying-band width w in theta/delta99 units; the frozen-profile
     generalized eigenvalue of Sec II.D (eq:frozeneig) SOLVED ON THE EXTRACTED
     PROFILES,  [a_max clip(P) S_gate |omega| + (c_nu_ai nu / sigma) d2/dy2] v
     = s u v,  across the c_nu_ai ladder 1, 1/3, 1/6, 1/12, ->0; heuristic
     penalty a_max P_max omega - D (pi/w)^2 alongside; Falkner-Skan
     calibration-geometry reference (tab_frozen_slope machinery) at matched
     Re_theta.
  5. chi (nuHat) transport balance at the stalled station x/L = 0.70 from the
     frozen field: streamwise/wall-normal convection vs AI production vs the
     laminar-branch diffusion (c_nu_ai nu + nuHat)/sigma + c_b2 term vs the
     sigma_D-tied destruction floor (SpalartAllmaras.h / SASourceTerms.h
     forms, sigma_D = 1 - R = 0.7511 in the laminar range).

Laminar-branch equation audited (chi << 1, is_turb = 0, fSlow = 1 for this
campaign -- ai_constants verified in the case's solver_stdout.log):
  u.grad(nuHat) = a_max clip<P> S(Re_Om/Re_Om_c(P)) |om| nuHat
                + (1/sigma) div[(c_nu_ai nu + nuHat) grad nuHat]
                + (c_b2/sigma) |grad nuHat|^2
                - sigma_D c_w1 f_w (nuHat/d)^2,   sigma_D = 1 - 0.2489.

Extraction machinery reused from spheroid_flank_kernel_audit.py (VTK chained
gradients on `velocity`, analytic-normal rays, canon kernel constants).
Cross-checks added after the first pass exposed estimator issues:
  - planar (FS-convention) profile kernel needs SMOOTHING: raw double-
    np.gradient second derivatives of the probed profile are noise-dominated
    (see PLANAR_W calibration note; the flank-audit's raw P_ii values are
    noise-inflated by the same effect);
  - the same BL-integral operator is run on the verified Sec-III flat plate
    (H must read the laminar 2.59) and on re72a0 L0/L1 (grid trend of the
    anomalously full mean profile), reference_checks().

Outputs (exploratory, NOT paper figures; house style: no in-figure titles,
captions in the companion md):
  paper/repro/cfd/figs_explore/spheroid_a0_physics_blchar.png
  paper/repro/cfd/figs_explore/spheroid_a0_physics_eN.png
  paper/repro/cfd/figs_explore/spheroid_a0_physics_chifield.png
  paper/repro/cfd/figs_explore/spheroid_a0_physics_band.png
  paper/repro/cfd/figs_explore/spheroid_a0_physics_balance.png
  paper/repro/cfd/figs_explore/spheroid_a0_physics_captions.md
  paper/repro/cfd/figs_explore/spheroid_a0_physics.json      (all numbers)

Run:  python3 -u paper/repro/cfd/spheroid_a0_physics.py
"""
import json
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.linalg import eigh_tridiagonal
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figs_explore")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "paper", "repro"))
sys.path.insert(0, os.path.join(REPO, "paper", "repro", "analytic"))

# reuse the audit's extraction machinery verbatim
from spheroid_flank_kernel_audit import (            # noqa: E402
    kernel_from_xyz, load_case_with_derived, probe, dyy, RATESCALE)
from surface_map import surface_frame                # noqa: E402
from lib.correlations import (                       # noqa: E402
    dN_dRe_theta, Re_theta0, compute_nondimensional_spatial_rate)

CASE = os.path.join(os.environ.get("SAAI_SPH_ROOT",
                                   os.path.join(REPO, "spheroid_fv1")),
                    "case_ogrid_L2_saai_re72a0")
STOCK = os.path.join(REPO, "paper", "data", "stock2006_fig14a_digitized.json")

# laminar-branch constants (ModelConstants.h, verified in solver_stdout.log)
SIGMA, CB2, CB1, KAPPA, CW2, CW3, CV1 = 2/3, 0.622, 0.1355, 0.41, 0.3, 2.0, 7.1
CW1 = CB1 / KAPPA**2 + (1 + CB2) / SIGMA
CNU = 1.0 / 6.0
SIGMA_D_LAM = 1.0 - 0.2489          # sigma-d-tie floor, is_turb = 0
A_MAX = RATESCALE                    # 0.19
SEED = 8.76e-6                       # physical seed (fSlow = 1, campaign)

PHI = 90.0                           # equator azimuth (axisymmetric case)
RAY = np.geomspace(2.5e-6, 0.08, 240)
YBAND = 0.05
XS = np.arange(0.02, 0.9651, 0.004)
STATIONS = (0.20, 0.30, 0.42, 0.55, 0.70, 0.85)
X_BAL, DX_BAL = 0.70, 0.01
CNU_LADDER = (1.0, 1/3, 1/6, 1/12, 0.0)

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 15, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.fontsize": 11.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "lines.linewidth": 2.0, "figure.dpi": 110, "savefig.dpi": 150})
C = dict(field="#3b6bb5", lam="#b0483a", kernel="#3e8f5c", eig="#7a5aa8",
         meas="#c02942", stock="#e0821f", model="#3b6bb5", gray="0.45")


# --------------------------------------------------------------- extraction
def edge_and_integrals(st):
    """BL edge + integrals from the stored profiles.  Edge = FIRST wall-normal
    local maximum of the speed (natural on a convex body, where the potential
    speed decays along the outward normal beyond the BL edge); stations with
    no local max in the band (the nose stagnation region, where speed rises
    monotonically along the ray) fall back to the band max and are FLAGGED
    (edge_ok=False) -- they are excluded from the e^N march."""
    y, U, us = st["y"], st["U"], st["us"]
    band = y <= YBAND
    bmax = float(np.max(np.where(band, U, -np.inf)))
    i_e, edge_ok = None, True
    for i in range(3, len(y) - 2):
        if not band[i] or y[i] < 5e-5:
            continue
        if (U[i] >= U[i + 1] and U[i] >= U[i + 2] and U[i] >= 0.97 * bmax):
            i_e = i
            break
    if i_e is None:
        i_e = int(np.argmax(np.where(band, U, -np.inf)))
        edge_ok = False
    u_e = us[i_e]
    f = np.clip(us / max(u_e, 1e-30), 0.0, 1.2)
    dstar = float(np.trapz((1 - f)[:i_e + 1], y[:i_e + 1]))
    theta = float(np.trapz((f * (1 - f))[:i_e + 1], y[:i_e + 1]))
    j99 = np.where(us[:i_e + 1] >= 0.99 * u_e)[0]
    d99 = np.nan
    if len(j99) and j99[0] > 0:
        j = j99[0]
        fint = (0.99 * u_e - us[j - 1]) / (us[j] - us[j - 1])
        d99 = float(y[j - 1] + fint * (y[j] - y[j - 1]))
    st.update(i_e=i_e, u_e=float(u_e), edge_ok=edge_ok, dstar=dstar,
              theta=theta, H=dstar / max(theta, 1e-30), d99=d99)
    return st


PLANAR_W = 61   # savgol window (of 600 pts over 3*d99).  Calibration on exact
# FS profiles sampled onto the same ray: W=31/61/91/121 recover Blasius
# maxP=0.0782 as 0.0781/0.0780/0.0775/0.0764 and beta=+0.10 maxP=0.0370 as
# 0.0369/0.0368/0.0366/0.0359 (<1% at W=61).  On the FIELD profiles the raw
# double-np.gradient variant is NOISE-DOMINATED (probe piecewise-linear kinks:
# maxP 0.28/0.13/0.056/0.050/0.048 at W=15/31/61/91/121 at x/L=0.42) and
# converges for W>=61 to the solver's compact-Laplacian value -- the audit's
# raw P_ii=0.075 there was noise-inflated; the converged planar P ~ 0.048.


def planar_kernel_smooth(st):
    """FS-table-convention kernel on the smoothed streamwise profile
    (uniform resample to 3*d99 + savgol, window per the calibration above).
    Returns the uniform grid and the kernel dict on it."""
    d99 = st["d99"]
    nu = float(np.median(st["nu"]))
    yg = np.linspace(0.0, 3 * d99, 600)
    h = yg[1] - yg[0]
    ug = np.interp(yg, st["y"], st["us"])
    u_s = savgol_filter(ug, PLANAR_W, 4)
    du = savgol_filter(ug, PLANAR_W, 4, deriv=1, delta=h)
    d2u = savgol_filter(ug, PLANAR_W, 4, deriv=2, delta=h)
    k = kernel_from_xyz(np.abs(u_s), yg * np.abs(du),
                        0.5 * yg * yg * d2u * np.sign(u_s),
                        yg * yg * np.abs(du) / nu)
    k["y"], k["om"] = yg, np.abs(du)
    return k


def extract_rays(grid, nu_ref, specs):
    """Rays at (point, normal, tangent) specs; kernel + profiles per ray."""
    pts = np.concatenate([p0 + RAY[:, None] * n for p0, n, _ in specs])
    res, ok = probe(grid, pts)
    ny = len(RAY)
    nuhat_name = ("solutionTurbulence" if "solutionTurbulence" in res
                  else "nuHat")
    out = []
    for k, (p0, n, ts) in enumerate(specs):
        sl = slice(k * ny, (k + 1) * ny)
        u = res["velocity"][sl].astype(float)
        vort = res["vort_vec"][sl].astype(float)
        lap = res["lap_u"][sl].astype(float)
        nuhat = res[nuhat_name][sl].astype(float)
        rho = res.get("rho", np.ones(len(pts)))[sl].astype(float)
        nu = nu_ref / np.maximum(rho, 1e-6)
        y = RAY
        U = np.linalg.norm(u, axis=1)
        uh = u / (U[:, None] + 1e-30)
        om = np.linalg.norm(vort, axis=1)
        chi = rho * nuhat / nu_ref
        us = u @ ts
        un = u @ n
        upp = np.einsum("ij,ij->i", lap, uh)
        re_om = y * y * om / nu
        ki = kernel_from_xyz(U, om * y, 0.5 * y * y * upp, re_om)
        st = dict(y=y, U=U, us=us, un=un, om=om, chi=chi, nuhat=nuhat, nu=nu,
                  P=ki["P"], gate=ki["gate"], rate=ki["rate"], re_om=re_om,
                  valid=ok[sl])
        out.append(edge_and_integrals(st))
    return out


def extract_meridian(grid, nu_ref, xs):
    """Rays along the phi=PHI meridian of the spheroid."""
    P, n3, t_s, _ = surface_frame(xs, np.full_like(xs, np.radians(PHI)))
    specs = [(P[k], n3[k], t_s[k]) for k in range(len(xs))]
    out = extract_rays(grid, nu_ref, specs)
    for st, x in zip(out, xs):
        st["x"] = float(x)
    return out


def front_crossing(xs, vals, level):
    """First sub-cell x where vals crosses level from below."""
    v = np.asarray(vals)
    hits = np.where(np.isfinite(v) & (v > level))[0]
    if len(hits) and hits[0] > 0:
        j = hits[0]
        f = (level - v[j - 1]) / (v[j] - v[j - 1])
        return float(xs[j - 1] + f * (xs[j] - xs[j - 1]))
    return np.nan


# ------------------------------------------------- Thwaites / e^N machinery
def spheroid_arc(x):
    """Arc length from the nose along a meridian of the 6:1 spheroid
    (A = 1/2, B = 1/12), plus body radius r0(x); x = x/L from nose."""
    A, B = 0.5, 1.0 / 12.0
    t = np.arccos(np.clip(1.0 - x / A, -1.0, 1.0))     # x = A(1 - cos t)
    tt = np.linspace(0.0, np.pi, 4001)
    dsdt = np.sqrt((A * np.sin(tt))**2 + (B * np.cos(tt))**2)
    s_of_t = np.concatenate([[0.0], np.cumsum(0.5 * (dsdt[1:] + dsdt[:-1])
                                              * np.diff(tt))])
    return np.interp(t, tt, s_of_t), B * np.sin(t)


def thwaites_eN(x_f, ue_f, nu, n_targets):
    """Axisymmetric (Mangler-weighted) Thwaites march on the FIELD edge
    velocity + Drela-Giles envelope e^N.  Returns dict of arrays on x_f."""
    s_f, r0_f = spheroid_arc(x_f)
    ue = savgol_filter(ue_f, 15, 3)
    # nose stub 0 <= x < x_f[0]: u_e linear in s (weight ue^5 r0^2 -> tiny)
    x_stub = np.linspace(1e-4, x_f[0], 60, endpoint=False)
    s_st, r0_st = spheroid_arc(x_stub)
    ue_st = ue[0] * s_st / s_f[0]
    s = np.concatenate([s_st, s_f])
    r0 = np.concatenate([r0_st, r0_f])
    uef = np.concatenate([ue_st, ue])
    w = uef**5 * r0**2
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1])
                                             * np.diff(s))])
    th2 = 0.45 * nu * integ / np.maximum(uef**6 * r0**2, 1e-30)
    theta = np.sqrt(th2)
    due_ds = np.gradient(uef, s)
    lam = np.clip(th2 / nu * due_ds, -0.12, 0.25)
    H = np.where(lam >= 0, 2.61 - 3.75 * lam + 5.24 * lam**2,
                 2.088 + 0.0731 / (lam + 0.14))          # Cebeci-Bradshaw
    Rt = uef * theta / nu
    Rt0 = np.asarray(Re_theta0(H))
    dNdRt = np.asarray(dN_dRe_theta(H))
    grow = Rt > Rt0
    # primary: chain rule on the marched Re_theta(s)
    dRt_ds = np.gradient(Rt, s)
    dN_ds = np.where(grow, np.maximum(dNdRt * dRt_ds, 0.0), 0.0)
    N = np.concatenate([[0.0], np.cumsum(0.5 * (dN_ds[1:] + dN_ds[:-1])
                                         * np.diff(s))])
    # secondary: Drela's Falkner-Skan spatial conversion s_DG/theta
    dN_ds2 = np.where(grow, np.asarray(compute_nondimensional_spatial_rate(H))
                      / np.maximum(theta, 1e-30), 0.0)
    N2 = np.concatenate([[0.0], np.cumsum(0.5 * (dN_ds2[1:] + dN_ds2[:-1])
                                          * np.diff(s))])
    m = len(x_stub)
    x_all = np.concatenate([x_stub, x_f])
    onset = front_crossing(x_all, Rt - Rt0, 0.0)
    cross = {f"N{nt:g}": front_crossing(x_all, N, nt) for nt in n_targets}
    cross2 = {f"N{nt:g}": front_crossing(x_all, N2, nt) for nt in n_targets}
    return dict(x=x_all[m:], theta=theta[m:], H=H[m:], Rt=Rt[m:],
                Rt0=Rt0[m:], N=N[m:], N2=N2[m:], onset_x=onset,
                lam=lam[m:], crossings=cross, crossings2=cross2)


def field_H_envelope(x, Rt_f, H_f, n_targets, xmax=0.88):
    """Drela-Giles envelope integrated on the FIELD's own extracted shape:
    dN/ds = dN/dRe_theta(H_field) * dRe_theta_field/ds where Re_theta and H
    come from the RANS profiles themselves (laminar up to the late front).
    Restricted to x <= xmax (upstream of the front/handover)."""
    m = x <= xmax
    xs = x[m]
    s, _ = spheroid_arc(xs)
    Rt = savgol_filter(Rt_f[m], 11, 3)
    H = savgol_filter(H_f[m], 11, 3)
    Rt0 = np.asarray(Re_theta0(H))
    dN_ds = np.where(Rt > Rt0,
                     np.maximum(np.asarray(dN_dRe_theta(H))
                                * np.gradient(Rt, s), 0.0), 0.0)
    N = np.concatenate([[0.0], np.cumsum(0.5 * (dN_ds[1:] + dN_ds[:-1])
                                         * np.diff(s))])
    onset = front_crossing(xs, Rt - Rt0, 0.0)
    cross = {f"N{nt:g}": front_crossing(xs, N, nt) for nt in n_targets}
    return dict(x=xs, N=N, Rt=Rt, Rt0=Rt0, onset_x=onset, crossings=cross)


# ------------------------------------------------ frozen-profile eigenvalue
def band_width(y, P, ymin=1e-5):
    """Contiguous P>0 interval containing the in-layer max (y >= ymin)."""
    Pm = np.where(y >= ymin, P, -1.0)
    j = int(np.argmax(np.where(y <= YBAND, Pm, -np.inf)))
    if Pm[j] <= 0:
        return np.nan, np.nan, j
    lo = j
    while lo > 0 and Pm[lo - 1] > 0:
        lo -= 1
    hi = j
    while hi < len(y) - 1 and Pm[hi + 1] > 0:
        hi += 1
    y_lo = y[lo] if lo == 0 else np.interp(0.0, [Pm[lo - 1], Pm[lo]],
                                           [y[lo - 1], y[lo]])
    y_hi = y[hi] if hi == len(y) - 1 else np.interp(
        0.0, [Pm[hi + 1], Pm[hi]], [y[hi + 1], y[hi]])
    return float(y_hi - y_lo), float(y_lo), j


def eig_station(st, cnu, n=2400, ytop_fac=4.0, planar=False):
    """Leading eigenvalue s [1/L] of the gated frozen-profile problem
    [rate(y)*om(y) + (cnu*nu/sigma) d2/dy2] v = s u(y) v on the EXTRACTED
    profiles (u floored at 0.02 u_e; v = 0 at wall and y_top).  planar=True
    books the FS-table-convention variant (smoothed planar kernel)."""
    ytop = min(ytop_fac * st["d99"], YBAND)
    yg = np.linspace(0.0, ytop, n)
    h = yg[1] - yg[0]
    if planar:
        kp = planar_kernel_smooth(st)
        b = np.interp(yg, kp["y"], kp["rate"] * kp["om"], left=0.0, right=0.0)
    else:
        b = np.interp(yg, st["y"], st["rate"] * st["om"], left=0.0)
    uu = np.maximum(np.interp(yg, st["y"], st["us"], left=0.0),
                    0.02 * st["u_e"])
    if cnu <= 0.0:
        return float(np.max(b / uu)), None
    D = cnu * float(np.median(st["nu"])) / SIGMA
    d = (b[1:-1] - 2.0 * D / h**2) / uu[1:-1]
    e = (D / h**2) / np.sqrt(uu[1:-2] * uu[2:-1])
    w, v = eigh_tridiagonal(d, e, select="i",
                            select_range=(len(d) - 1, len(d) - 1))
    return float(w[0]), (yg[1:-1], v[:, 0])


def fs_reference(rts, cnu_list):
    """Falkner-Skan calibration geometry + eigen retention at matched
    Re_theta (tab_frozen_slope machinery; ungated, as in Sec II.D)."""
    from tab_frozen_slope import build, UFLOOR
    out = {}
    for beta, lab in ((0.10, "FS favorable b=+0.10"), (0.0, "Blasius")):
        pr = build(beta, None)
        y, u, b = pr["y"], pr["u"], pr["b"]
        P = b / (A_MAX * np.maximum(np.abs(pr["up"]), 1e-30))
        w_eta, ylo, j = band_width(y, np.where(b > 0, 1.0, -1.0) *
                                   np.maximum(P, 1e-12), ymin=0.0)
        d99 = float(np.interp(0.99, u[:np.argmax(u) + 1],
                              y[:np.argmax(u) + 1]))
        sup = float(np.max(b / np.maximum(u, UFLOOR)))
        rows = {}
        for Rt in rts:
            row = {}
            for cnu in cnu_list:
                if cnu <= 0:
                    row["0"] = sup
                    continue
                Dd = cnu * pr["I_th"] / (SIGMA * Rt)
                h = pr["h"]
                uf = np.maximum(u, UFLOOR)[1:-1]
                dm = (b[1:-1] - 2.0 * Dd / h**2) / uf
                em = (Dd / h**2) / np.sqrt(uf[:-1] * uf[1:])
                row[f"{cnu:g}"] = float(eigh_tridiagonal(
                    dm, em, select="i",
                    select_range=(len(dm) - 1, len(dm) - 1))[0][0])
            rows[f"{Rt:g}"] = row
        out[lab] = dict(H=pr["H"], Pmax=float(np.max(P)),
                        w_over_theta=w_eta / pr["I_th"],
                        w_over_d99=w_eta / d99, sup=sup, eig=rows)
    return out


# -------------------------------------------------------- transport balance
def transport_balance(sts, x0, dx):
    """All laminar-branch nuHat budget terms along the x0 ray, streamwise
    derivatives from x0 +/- dx rays at constant wall height."""
    sm, s0, sp = sts
    s_arc, _ = spheroid_arc(np.array([x0 - dx, x0, x0 + dx]))
    ds = 0.5 * (s_arc[2] - s_arc[0])
    y, nu = s0["y"], s0["nu"]
    nh = s0["nuhat"]
    dnh_dy = np.gradient(nh, y)
    dnh_ds = (sp["nuhat"] - sm["nuhat"]) / (2 * ds)
    d2nh_ds2 = (sp["nuhat"] - 2 * nh + sm["nuhat"]) / ds**2
    conv_s = s0["us"] * dnh_ds
    conv_n = s0["un"] * dnh_dy
    prod = s0["rate"] * s0["om"] * nh
    dcoef = CNU * nu + nh
    diff_n = np.gradient(dcoef * dnh_dy, y) / SIGMA
    cb2_n = (CB2 / SIGMA) * dnh_dy**2
    diff_s = dcoef * d2nh_ds2 / SIGMA + (CB2 / SIGMA) * dnh_ds**2
    # destruction: sigma_D-tied floor, standard f_w with SA modified S
    chi = s0["chi"]
    fv1 = chi**3 / (chi**3 + CV1**3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    Shat = s0["om"] + nh / (KAPPA**2 * y**2 + 1e-30) * fv2
    nuRefw = Shat * (KAPPA * y)**2
    r = np.where((nuRefw <= 0) | (nh > 10 * nuRefw), 10.0,
                 nh / np.maximum(nuRefw, 1e-30))
    g = r + CW2 * (r**6 - r)
    fw = g * ((1 + CW3**6) / (g**6 + CW3**6))**(1.0 / 6.0)
    destr = SIGMA_D_LAM * CW1 * fw * (nh / np.maximum(y, 1e-30))**2
    resid = prod + diff_n + cb2_n + diff_s - destr - conv_s - conv_n
    return dict(st=s0, y=y, conv_s=conv_s, conv_n=conv_n, prod=prod,
                diff_n=diff_n, cb2_n=cb2_n, diff_s=diff_s, destr=destr,
                resid=resid, ds=float(ds))


# ------------------------------------------------------------------ figures
def fig_blchar(sw, th, fronts, fp):
    x = np.array([s["x"] for s in sw])
    Rt = np.array([s["u_e"] * s["theta"] / np.median(s["nu"]) for s in sw])
    H = np.array([s["H"] for s in sw])
    m = (x >= 0.10) & (x <= 0.95)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 8.4), sharex=True)
    a1.plot(x[m], Rt[m], color=C["field"], label="L2 field (extracted)")
    a1.plot(th["x"], th["Rt"], "--", color=C["lam"],
            label="laminar Thwaites-Mangler on field $u_e$")
    a1.plot(th["x"], th["Rt0"], ":", color=C["gray"], lw=1.8,
            label=r"Drela-Giles critical $Re_{\theta 0}(H)$")
    a1.set_ylabel(r"$Re_\theta$")
    a1.legend(loc="upper left")
    a2.plot(x[m], H[m], color=C["field"], label="L2 field (extracted)")
    a2.plot(th["x"], th["H"], "--", color=C["lam"],
            label="laminar Thwaites-Mangler")
    a2.axhline(2.59, color=C["gray"], lw=1.0, ls=":")
    a2.text(0.115, 2.60, "Blasius H = 2.59", fontsize=11, color=C["gray"])
    a2.set_ylabel(r"shape factor  $H=\delta^*/\theta$")
    a2.set_xlabel(r"$x/L$")
    a2.set_ylim(2.0, 3.4)
    a2.legend(loc="upper left")
    for ax in (a1, a2):
        for key, col, lab in (("meas", C["meas"], "measured 0.438"),
                              ("stock", C["stock"], "Stock $e^N$ 0.425"),
                              ("chi1", C["model"], None)):
            ax.axvline(fronts[key], color=col, lw=1.4, ls="-.")
        ax.set_xlim(0.08, 0.97)
    a1.text(fronts["meas"] + 0.008, a1.get_ylim()[1] * 0.42,
            f"measured front {fronts['meas']:.3f}", rotation=90, fontsize=11,
            color=C["meas"], ha="left")
    a1.text(fronts["stock"] - 0.008, a1.get_ylim()[1] * 0.42,
            f"Stock $e^N$(8) {fronts['stock']:.3f}", rotation=90, fontsize=11,
            color=C["stock"], ha="right")
    a1.text(fronts["chi1"] - 0.015, a1.get_ylim()[1] * 0.10,
            f"model $\\chi$=1 front {fronts['chi1']:.3f}", rotation=90,
            fontsize=11, color=C["model"], ha="right")
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def fig_eN(sw, th, fh, fronts, kernN, actN, fp):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 8.8), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1.6]))
    a1.plot(th["x"], th["N"], color=C["lam"],
            label=r"$e^N$ envelope on field $u_e$ (Thwaites shapes)")
    a1.plot(th["x"], th["N2"], "--", color=C["lam"], lw=1.4, alpha=0.75,
            label=r"$e^N$, Drela FS spatial conversion")
    a1.plot(fh["x"], fh["N"], color="#8a4a3a", ls=(0, (4, 1.5, 1, 1.5)),
            lw=2.0, label=r"$e^N$ on the field's OWN $H$, $Re_\theta$")
    a1.plot(kernN[0], kernN[1], color=C["kernel"],
            label="model kernel bound $\\int \\max_y(a\\,\\omega/u)\\,dx$")
    a1.plot(actN[0], actN[1], ":", color=C["model"], lw=2.4,
            label=r"model realized  $\ln(\chi_{max}/\chi_\infty)$")
    for nt, lab in ((6.0, "N=6 (tunnel-seed class)"),
                    (8.0, "N=8 (Stock's calibration)"),
                    (11.65, "N=11.65 (flight-quiet seed, as run)")):
        a1.axhline(nt, color=C["gray"], lw=0.9, ls=":")
        a1.text(0.685, nt + 0.15, lab, fontsize=10.5, color=C["gray"])
    for key, col in (("meas", C["meas"]), ("stock", C["stock"]),
                     ("chi1", C["model"])):
        a1.axvline(fronts[key], color=col, lw=1.4, ls="-.")
        a2.axvline(fronts[key], color=col, lw=1.4, ls="-.")
    a1.set_ylabel("amplification factor  $N$")
    a1.set_ylim(0, 16)
    a1.legend(loc="upper left")
    a2.plot(th["x"], th["Rt"] / th["Rt0"], color=C["lam"])
    a2.axhline(1.0, color=C["gray"], lw=1.0)
    a2.plot([th["onset_x"]], [1.0], "o", ms=9, color=C["lam"], zorder=5)
    a2.text(th["onset_x"] + 0.012, 1.12,
            f"$e^N$ amplification onset  x/L = {th['onset_x']:.3f}",
            fontsize=11.5, color=C["lam"])
    a2.set_ylabel(r"$Re_\theta / Re_{\theta 0}(H)$")
    a2.set_xlabel(r"$x/L$")
    a2.set_xlim(0.08, 0.97)
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def fig_chifield(sw, fronts, fp):
    x = np.array([s["x"] for s in sw])
    chi = np.log10(np.maximum(np.stack([s["chi"] for s in sw]), 1e-12))
    P2 = np.stack([s["P"] for s in sw])
    G2 = np.stack([s["gate"] for s in sw])
    d99 = np.array([s["d99"] for s in sw])
    th = np.array([s["theta"] for s in sw])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 9.6))
    norm = Normalize(vmin=-5.2, vmax=0.5)
    pm = a1.pcolormesh(x, RAY, chi.T, cmap="viridis", norm=norm,
                       shading="nearest", rasterized=True)
    a1.contour(x, RAY, P2.T, levels=[0.0], colors="w",
               linewidths=1.6, linestyles="--")
    a1.contourf(x, RAY, (P2 > 0).T.astype(float), levels=[0.5, 1.5],
                colors="none", hatches=["///"])
    a1.contour(x, RAY, G2.T, levels=[0.5], colors="#e0821f", linewidths=1.6)
    a1.plot(x, d99, color="w", lw=2.2)
    a1.plot(x, d99, color="k", lw=1.0)
    a1.plot(x, th, color="w", lw=2.2, ls=":")
    a1.plot(x, th, color="k", lw=1.0, ls=":")
    a1.set_yscale("log")
    a1.set_ylim(3e-6, 0.03)
    a1.set_ylabel(r"wall distance $y/L$")
    a1.text(0.12, 1.6e-3, r"$\delta_{99}$", color="w", fontsize=13)
    a1.text(0.12, 1.1e-4, r"$\theta$", color="w", fontsize=13)
    a1.text(0.60, 6e-3, "hatched: P > 0 amplifying band\nwhite dashed: P = 0"
            "\norange: onset gate S = 0.5", color="w", fontsize=11.5)
    # normalized coordinates
    eta = np.linspace(0.0, 1.6, 200)
    chiN = np.full((len(x), len(eta)), np.nan)
    PN = np.full_like(chiN, np.nan)
    GN = np.full_like(chiN, np.nan)
    for i in range(len(x)):
        if np.isfinite(d99[i]):
            chiN[i] = np.interp(eta * d99[i], RAY, chi[i])
            PN[i] = np.interp(eta * d99[i], RAY, P2[i])
            GN[i] = np.interp(eta * d99[i], RAY, G2[i])
    pm2 = a2.pcolormesh(x, eta, chiN.T, cmap="viridis", norm=norm,
                        shading="nearest", rasterized=True)
    a2.contour(x, eta, PN.T, levels=[0.0], colors="w", linewidths=1.6,
               linestyles="--")
    a2.contourf(x, eta, (PN > 0).T.astype(float), levels=[0.5, 1.5],
                colors="none", hatches=["///"])
    a2.contour(x, eta, GN.T, levels=[0.5], colors="#e0821f", linewidths=1.6)
    a2.set_ylabel(r"$y/\delta_{99}$")
    a2.set_xlabel(r"$x/L$")
    for ax in (a1, a2):
        for key, col in (("meas", C["meas"]), ("stock", C["stock"]),
                         ("chi1", "w")):
            ax.axvline(fronts[key], color=col, lw=1.6, ls="-.")
        ax.set_xlim(0.08, 0.965)
    cb = fig.colorbar(pm, ax=(a1, a2), pad=0.015, aspect=35)
    cb.set_label(r"$\log_{10}\chi$   ($\chi = \tilde\nu/\nu$; seed"
                 r" $8.76\times10^{-6}$, $\chi=1$ = handover)")
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)


def fig_band(x_sw, tab, eig_sw, fs, fp):
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(9.5, 12.6), sharex=True)
    xs = [t["x"] for t in tab]
    # amplifying coordinate vs the FS calibration class (planar convention)
    a0.plot(xs, [t["Pmax_planar"] for t in tab], "o-", color=C["field"],
            label=r"spheroid $\max_y\hat\Omega\hat I$ (planar profile"
                  r" convention, as FS tables)")
    a0.plot(xs, [t["Pmax"] for t in tab], "s--", color="#7aa0d4",
            label=r"spheroid $\max_y P$ (solver magnitude-triple)")
    for lab, col, txt in (("FS favorable b=+0.10", C["lam"],
                           r"FS $\beta$=+0.10 (marcher-verified)"),
                          ("Blasius", "0.4", "Blasius")):
        a0.axhline(fs[lab]["Pmax"], color=col, lw=1.4, ls=":")
        a0.text(0.205, fs[lab]["Pmax"] - 0.0065,
                txt + f": {fs[lab]['Pmax']:.3f} (H = {fs[lab]['H']:.2f})",
                fontsize=11, color=col)
    a0.set_ylabel(r"$\max_y\,\hat\Omega\hat I$")
    a0.set_ylim(0, 0.11)
    a0.legend(loc="upper right")
    a1.plot(xs, [t["w_over_theta"] for t in tab], "o-", color=C["field"],
            label=r"$w/\theta$ (field, P>0 band)")
    a1.plot(xs, [t["w_over_d99"] * 10 for t in tab], "s--", color=C["eig"],
            label=r"$w/\delta_{99}\times 10$ (field)")
    a1.axhline(tab[0]["fs_fav_w_over_theta"], color=C["lam"], lw=1.4, ls=":")
    a1.text(0.55, tab[0]["fs_fav_w_over_theta"] - 0.22,
            r"Falkner-Skan $\beta$=+0.10 band $w/\theta$",
            fontsize=11, color=C["lam"])
    a1.axhline(tab[0]["fs_blasius_w_over_theta"], color=C["lam"], lw=1.4,
               ls="--")
    a1.text(0.55, tab[0]["fs_blasius_w_over_theta"] + 0.08,
            r"Blasius band $w/\theta$", fontsize=11,
            color=C["lam"])
    a1.set_ylabel(r"amplifying-band width")
    a1.legend(loc="center right")
    for cnu, col, lab in ((1.0, "0.75", r"$c_{\nu,ai}=1$"),
                          (1/3, "#9a8ec9", r"$c_{\nu,ai}=1/3$"),
                          (1/6, C["eig"], r"$c_{\nu,ai}=1/6$ (model)"),
                          (1/12, "#4a3d78", r"$c_{\nu,ai}=1/12$"),
                          (0.0, "k", r"$c_{\nu,ai}\to 0$ (inviscid sup)")):
        a2.plot(x_sw, eig_sw[f"{cnu:g}"], "-", color=col, label=lab)
    a2.plot(x_sw, eig_sw["realized"], ":", color=C["model"], lw=2.6,
            label=r"realized  $d\ln\chi_{max}/dx$ (field)")
    a2.axhline(0, color="k", lw=0.8)
    a2.set_ylabel(r"frozen-profile growth $s$  [e-folds / $L$]")
    a2.set_xlabel(r"$x/L$")
    a2.set_ylim(-3, 32)
    a2.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def fig_balance(bal, fp):
    st = bal["st"]
    yy = bal["y"] / st["d99"]
    sc = float(np.nanmax(bal["prod"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 6.4), sharey=True)
    for a in (a1, a2):
        a.axvline(0, color="k", lw=0.8)
        a.set_ylim(0, 1.6)
    a1.plot(bal["prod"] / sc, yy, color=C["kernel"], label="AI production")
    a1.plot(bal["diff_n"] / sc, yy, color=C["eig"],
            label=r"wall-normal diffusion $\partial_y[(c_\nu\nu+\tilde\nu)"
                  r"\partial_y\tilde\nu]/\sigma$")
    a1.plot(bal["cb2_n"] / sc, yy, "--", color=C["eig"], lw=1.6,
            label=r"$c_{b2}|\partial_y\tilde\nu|^2/\sigma$")
    a1.plot(-bal["destr"] / sc, yy, color="#8a8a8a", lw=1.6,
            label=r"$-$destruction ($\sigma_D$ floor)")
    a1.plot(bal["diff_s"] / sc, yy, ":", color="#4a3d78", lw=1.6,
            label="streamwise diffusion")
    a1.set_xlabel("source terms / max production")
    a1.set_ylabel(r"$y/\delta_{99}$")
    a1.set_xlim(-1.65, 1.25)
    a1.legend(loc="upper left", fontsize=10.5)
    a2.plot(bal["conv_s"] / sc, yy, color=C["lam"],
            label=r"streamwise convection $u_s\,\partial_s\tilde\nu$")
    a2.plot(bal["conv_n"] / sc, yy, "--", color=C["lam"], lw=1.8,
            label=r"wall-normal convection $u_n\,\partial_y\tilde\nu$")
    a2.plot((bal["prod"] + bal["diff_n"] + bal["cb2_n"] + bal["diff_s"]
             - bal["destr"]) / sc, yy, color="0.3", lw=1.4,
            label="sum of sources (= required convection)")
    a2.plot(bal["resid"] / sc, yy, ":", color="k", lw=1.4,
            label="residual (imbalance)")
    a2.set_xlabel("terms / max production")
    a2.set_xlim(-1.65, 1.45)
    a2.legend(loc="upper left", fontsize=10.5)
    ax1t = a1.twinx()
    ax1t.set_yticks([])
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


# --------------------------------------------------------------------- main
CACHED = os.environ.get(
    "SAAI_A0PHYS_CACHE_DIR",
    "/tmp/claude-1007/-home-qiqi-flexcompute/"
    "52a71f92-8d18-4ada-8b4a-3a1e02709572/scratchpad")


def get_data():
    """Grid load + all probing, cached (the 12.6M-pt VTK gradient chain takes
    ~15 min; cache in the session scratchpad, safe to delete any time)."""
    cache = os.path.join(CACHED, "a0phys_cache.pkl")
    if os.path.exists(cache):
        print(f"using cached extraction {cache}", flush=True)
        sw, bal_sts, nu_ref = pickle.load(open(cache, "rb"))
    else:
        grid, nu_ref, mach = load_case_with_derived(CASE)
        sw = extract_meridian(grid, nu_ref, XS)
        bal_sts = extract_meridian(
            grid, nu_ref, np.array([X_BAL - DX_BAL, X_BAL, X_BAL + DX_BAL]))
        try:
            pickle.dump((sw, bal_sts, nu_ref), open(cache, "wb"))
        except OSError:
            pass
    for st in sw + bal_sts:      # refresh edge/integral fields on cache reuse
        edge_and_integrals(st)
    return sw, bal_sts, nu_ref


def reference_checks():
    """Operator/grid cross-checks for the H and P readings: the SAME
    extraction on (a) the verified Sec-III flat plate (H should read the
    laminar ~2.59 where chi<1 -- validates the BL-integral operator), and
    (b) the re72a0 L0/L1 grids at matched stations (is the anomalously full
    mean profile a grid trend?)."""
    cache = os.path.join(CACHED, "a0phys_refs_cache.pkl")
    if os.path.exists(cache):
        print(f"using cached references {cache}", flush=True)
        return pickle.load(open(cache, "rb"))
    refs = {}
    fp_root = os.environ.get("SAAI_CFD_ROOT", os.path.join(REPO,
                                                           "flow360_fv1"))
    cases = [("plate", os.path.join(fp_root, "flatplate_sphere_Tu0040"),
              [1.0, 2.0, 3.0]),
             ("L0", os.path.join(os.path.dirname(CASE),
                                 "case_ogrid_L0_saai_re72a0"),
              [0.20, 0.42, 0.70]),
             ("L1", os.path.join(os.path.dirname(CASE),
                                 "case_ogrid_L1_saai_re72a0"),
              [0.20, 0.42, 0.70])]
    for name, cdir, xs in cases:
        grid, nu_ref, mach = load_case_with_derived(cdir)
        if name == "plate":
            specs = [(np.array([x, -0.05, 0.0]), np.array([0.0, 0.0, 1.0]),
                      np.array([1.0, 0.0, 0.0])) for x in xs]
        else:
            P, n3, t_s, _ = surface_frame(
                np.asarray(xs), np.full(len(xs), np.radians(PHI)))
            specs = [(P[k], n3[k], t_s[k]) for k in range(len(xs))]
        rows = []
        for st, x in zip(extract_rays(grid, nu_ref, specs), xs):
            nu = float(np.median(st["nu"]))
            kp = planar_kernel_smooth(st)
            bnd = st["y"] >= 1e-5
            j = int(np.argmax(np.where(bnd & (st["y"] <= YBAND),
                                       st["P"], -np.inf)))
            jp = int(np.argmax(np.where(kp["y"] >= 1e-5, kp["P"], -np.inf)))
            rows.append(dict(
                x=float(x), H=st["H"], d99=st["d99"], edge_ok=st["edge_ok"],
                Rt=float(st["u_e"] * st["theta"] / nu),
                chimax=float(np.nanmax(np.where(
                    (st["y"] <= 0.04) & st["valid"], st["chi"], np.nan))),
                P_solver=float(st["P"][j]),
                P_planar=float(kp["P"][jp])))
            print(f"  ref {name} x={x:g}: H={rows[-1]['H']:.3f} "
                  f"Rt={rows[-1]['Rt']:.0f} P_solver={rows[-1]['P_solver']:.4f}"
                  f" P_planar={rows[-1]['P_planar']:.4f} "
                  f"chimax={rows[-1]['chimax']:.2e}", flush=True)
        refs[name] = rows
        del grid
    try:
        pickle.dump(refs, open(cache, "wb"))
    except OSError:
        pass
    return refs


def main():
    os.makedirs(FIGD, exist_ok=True)
    sw, bal_sts, nu_ref = get_data()
    x = np.array([s["x"] for s in sw])

    # fronts: measured / Stock from the committed digitized JSON; model chi=1
    # re-extracted from THIS sweep (near-wall max chi crossing 1)
    stock = json.load(open(STOCK))
    meas = float(np.mean([m["xL"] for m in stock["measured_squares"]]))
    stock_ts = float(stock["computed_ts_front"]["mean_xL"])
    chimax = np.array([float(np.nanmax(np.where(
        (s["y"] <= 0.04) & s["valid"], s["chi"], np.nan))) for s in sw])
    chi_front = front_crossing(x, chimax, 1.0)
    fronts = dict(meas=meas, stock=stock_ts, chi1=chi_front)
    print(f"fronts: measured {meas:.4f}, Stock e^N(8) {stock_ts:.4f}, "
          f"model chi=1 (this sweep) {chi_front:.4f}", flush=True)

    # station report incl. the measured-front station
    nu0 = float(np.median(sw[0]["nu"]))
    Rt = np.array([s["u_e"] * s["theta"] / np.median(s["nu"]) for s in sw])
    H = np.array([s["H"] for s in sw])
    for xq in (0.42, meas):
        print(f"  x/L={xq:.3f}: Re_theta={np.interp(xq, x, Rt):.0f}, "
              f"H={np.interp(xq, x, H):.3f}", flush=True)

    # e^N on the field's edge conditions (edge-ok stations only: the nose
    # stagnation rays have no local speed max and their u_e is unreliable)
    eok = np.array([s["edge_ok"] for s in sw])
    ue = np.array([s["u_e"] for s in sw])
    th = thwaites_eN(x[eok], ue[eok], nu0, (1.0, 6.0, 8.0, 11.65))
    print(f"e^N (Thwaites-Mangler shapes): onset x/L={th['onset_x']:.3f}; "
          f"crossings {th['crossings']} "
          f"(FS-conversion variant {th['crossings2']})", flush=True)
    # e^N on the field's OWN extracted profile shape (H anomalously full)
    fh = field_H_envelope(x[eok], Rt[eok], H[eok], (1.0, 6.0, 8.0, 11.65))
    print(f"e^N (field's own H): onset x/L={fh['onset_x']:.3f}; "
          f"crossings {fh['crossings']}", flush=True)

    # kernel bound + realized growth budgets along the meridian
    dNdx = np.array([float(np.nanmax(np.where(
        (s["y"] >= 1e-5) & (s["y"] <= YBAND) & (s["U"] > 0.3 * s["u_e"]),
        s["rate"] * s["om"] / s["U"], np.nan))) for s in sw])
    kernN = np.concatenate([[0.0], np.cumsum(
        0.5 * (dNdx[1:] + dNdx[:-1]) * np.diff(x))])
    chi_amb = np.array([float(s["chi"][-1]) for s in sw])
    actN = np.log(np.maximum(chimax, 1e-12) / chi_amb)
    print(f"  kernel-bound N at meas front: "
          f"{np.interp(meas, x, kernN):.2f}; realized "
          f"{np.interp(meas, x, actN):.2f}", flush=True)

    # eigenvalue ladder along the sweep (coarser x for cost)
    x_eig = x[(x >= 0.14) & (x <= 0.90)][::3]
    eig_sw = {f"{c:g}": [] for c in CNU_LADDER}
    for xq in x_eig:
        st = sw[int(np.argmin(np.abs(x - xq)))]
        for c in CNU_LADDER:
            eig_sw[f"{c:g}"].append(eig_station(st, c)[0])
    sreal = np.gradient(savgol_filter(actN, 21, 3), x)
    eig_sw["realized"] = list(np.interp(x_eig, x, sreal))

    # station band table + FS calibration reference
    fs = fs_reference((400.0, 900.0, 1600.0), CNU_LADDER)
    tab = []
    for xq in STATIONS:
        st = sw[int(np.argmin(np.abs(x - xq)))]
        w, ylo, j = band_width(st["y"], st["P"])
        om_pk = st["om"][j]
        pen = CNU * float(np.median(st["nu"])) / SIGMA * (np.pi / w)**2
        gr = A_MAX * max(st["P"][j], 0.0) * st["gate"][j] * om_pk
        # planar-convention max P in the layer (FS Tables 2/3 convention,
        # smoothed estimator -- see PLANAR_W calibration note)
        kp = planar_kernel_smooth(st)
        jp = int(np.argmax(np.where(kp["y"] >= 1e-5, kp["P"], -np.inf)))
        row = dict(x=float(st["x"]),
                   Rt=float(st["u_e"] * st["theta"] / np.median(st["nu"])),
                   H=float(st["H"]), theta=st["theta"], d99=st["d99"],
                   Pmax=float(st["P"][j]), gate=float(st["gate"][j]),
                   Pmax_planar=float(kp["P"][jp]),
                   gate_planar=float(kp["gate"][jp]),
                   yP_over_d99=float(st["y"][j] / st["d99"]),
                   w=float(w), w_over_theta=float(w / st["theta"]),
                   w_over_d99=float(w / st["d99"]),
                   growth_peak=float(gr), penalty_pi_w=float(pen),
                   net_heuristic=float(gr - pen),
                   sup_dNdx=float(dNdx[int(np.argmin(np.abs(x - xq)))]),
                   s_realized=float(np.interp(xq, x, sreal)),
                   fs_fav_w_over_theta=fs["FS favorable b=+0.10"][
                       "w_over_theta"],
                   fs_blasius_w_over_theta=fs["Blasius"]["w_over_theta"])
        for c in CNU_LADDER:
            row[f"s_eig_{c:g}"] = eig_station(st, c)[0]
        row["s_eig_planar_1/6"] = eig_station(st, CNU, planar=True)[0]
        tab.append(row)
        print(f"  band x/L={xq:.2f}: Rt={row['Rt']:.0f} H={row['H']:.2f} "
              f"Pmax={row['Pmax']:.3f} Ppl={row['Pmax_planar']:.3f} "
              f"w/theta={row['w_over_theta']:.2f} "
              f"w/d99={row['w_over_d99']:.3f} s_eig(1/6)="
              f"{row['s_eig_0.166667']:.2f}/L (planar "
              f"{row['s_eig_planar_1/6']:.2f}) sup={row['s_eig_0']:.2f}/L "
              f"realized={row['s_realized']:.2f}/L", flush=True)
    print("FS reference (verify vs committed Table 2: b=+0.10 H=2.48 "
          "maxP=0.037; Blasius H=2.59 maxP=0.078):",
          json.dumps(fs, indent=1), flush=True)

    # transport balance at the stalled station
    bal = transport_balance(bal_sts, X_BAL, DX_BAL)
    st = bal["st"]
    w, ylo, j = band_width(st["y"], st["P"])
    bsel = (st["y"] >= ylo) & (st["y"] <= ylo + w)
    integ = {k: float(np.trapz(bal[k][bsel], st["y"][bsel]))
             for k in ("prod", "diff_n", "cb2_n", "diff_s", "destr",
                       "conv_s", "conv_n", "resid")}
    jpk = int(np.argmax(np.where((st["y"] >= 1e-5) & (st["y"] <= 0.04),
                                 st["chi"], -np.inf)))
    peak = {k: float(bal[k][jpk]) for k in ("prod", "diff_n", "cb2_n",
                                            "diff_s", "destr", "conv_s",
                                            "conv_n", "resid")}
    print("balance (band-integrated, /prod):",
          {k: round(v / max(integ["prod"], 1e-30), 3)
           for k, v in integ.items()}, flush=True)
    print("balance (at chi-peak height, /prod):",
          {k: round(v / max(abs(peak["prod"]), 1e-30), 3)
           for k, v in peak.items()}, flush=True)
    s_real_peak = peak["conv_s"] / (st["us"][jpk] * st["nuhat"][jpk])
    print(f"  realized d(ln nuHat)/ds at chi peak = {s_real_peak:.2f}/L; "
          f"chimax={st['chi'][jpk]:.3e} at y/d99="
          f"{st['y'][jpk]/st['d99']:.2f}", flush=True)

    # operator/grid cross-checks (plate + L0/L1)
    refs = reference_checks()

    # figures
    fig_blchar(sw, th, fronts, os.path.join(
        FIGD, "spheroid_a0_physics_blchar.png"))
    fig_eN(sw, th, fh, fronts, (x, kernN), (x, actN), os.path.join(
        FIGD, "spheroid_a0_physics_eN.png"))
    fig_chifield(sw, fronts, os.path.join(
        FIGD, "spheroid_a0_physics_chifield.png"))
    fig_band(x_eig, tab, eig_sw, fs, os.path.join(
        FIGD, "spheroid_a0_physics_band.png"))
    fig_balance(bal, os.path.join(FIGD, "spheroid_a0_physics_balance.png"))
    print("figures written to", FIGD, flush=True)

    out = dict(fronts=fronts,
               station_Rt={f"{xq:g}": float(np.interp(xq, x, Rt))
                           for xq in (0.42, meas, 0.55, 0.70)},
               station_H={f"{xq:g}": float(np.interp(xq, x, H))
                          for xq in (0.42, meas, 0.55, 0.70)},
               eN=dict(onset_x=th["onset_x"], crossings=th["crossings"],
                       crossings_fs_conversion=th["crossings2"],
                       fieldH_onset_x=fh["onset_x"],
                       fieldH_crossings=fh["crossings"]),
               reference_checks=refs,
               planar_estimator=dict(
                   window=PLANAR_W,
                   fs_recovery="Blasius 0.0782->0.0780, b+0.10 0.0370->0.0368"
                               " at W=61 (exact profiles on same ray)",
                   field_W_convergence_x042="0.28/0.13/0.056/0.050/0.048 at "
                                            "W=15/31/61/91/121"),
               kernel_N_at_meas=float(np.interp(meas, x, kernN)),
               realized_N_at_meas=float(np.interp(meas, x, actN)),
               band_table=tab, fs_reference=fs,
               balance=dict(x=X_BAL, band_integrated_over_prod={
                   k: v / max(integ["prod"], 1e-30) for k, v in integ.items()},
                   at_chi_peak_over_prod={
                   k: v / max(abs(peak["prod"]), 1e-30)
                   for k, v in peak.items()},
                   s_realized_peak=float(s_real_peak)),
               conventions=dict(
                   phi=PHI, ray=[float(RAY[0]), float(RAY[-1]), len(RAY)],
                   theta="trapz us/ue(1-us/ue) to speed-max height",
                   d99="first crossing us=0.99ue",
                   eig="gated [rate*om + cnu*nu/sigma d2/dy2]v = s u v, "
                       "u floored 0.02ue, v=0 at wall/4*d99",
                   balance="frozen-field terms, sigma_D=1-0.2489 laminar tie",
                   seed=SEED))
    fj = os.path.join(FIGD, "spheroid_a0_physics.json")
    json.dump(out, open(fj, "w"), indent=1)
    print("tables:", fj, flush=True)


if __name__ == "__main__":
    main()
