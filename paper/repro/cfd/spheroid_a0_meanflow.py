"""WHY does the RANS laminar BL on the alpha=0 spheroid (re72a0, Re_L=7.2e6)
run anomalously FULL (H = 2.44-2.51 mid-body) where laminar-BL theory on the
same u_e says Blasius-class 2.59-2.61?  Follow-on to the physics deep-dive
(agent-paper-review/2026-07-28-0040-spheroid-a0-physics.md, Sec 1 + closing
paragraph).  Candidates:
  (a) solver numerics (unpreconditioned low-Mach Roe dissipation, O-grid),
  (b) axisymmetric/transverse-curvature second-order BL physics that
      Thwaites-Mangler misses,
  (c) genuine non-equilibrium/history effects (nose-acceleration H-lag that
      the equilibrium H(lambda) closure cannot represent),
  (d) extraction geometry (edge/u_e convention on a convex body).

Discriminating instruments, in order:

1. LAMINAR BL PROFILE MARCH (not just Thwaites integrals): implicit
   finite-difference space-march of the axisymmetric laminar BL equations in
   physical (s, y) coordinates, driven by the FIELD's own extracted u_e(x)
   and the analytic body radius r0(x):
       u u_s + v u_y = u_e du_e/ds + (nu/r) (r u_y)_y ,
       (r u)_s + (r v)_y = 0,
   with r = 1 (planar), r = r0(s) (Mangler first-order axisymmetric), or
   r = r0(s) + y n_r(s) (first-order TRANSVERSE CURVATURE kept).  The march
   carries full profile history, so it answers (c) exactly at first order;
   the r-ladder isolates (b)'s first-order part; running the SAME
   edge/integral operator on the marched profiles removes (d)'s
   operator part.  Validation: constant-u_e run must reproduce Blasius
   (H = 2.5905, cf theta u_e/nu = 0.2205); IC-shape and grid halving checks.

2. MOMENTUM-BALANCE PROBE of the frozen RANS field at x/L = 0.42 (and the
   verified flat plate at x = 2.0, matched Re_theta ~ 909, plus NLF0416
   x/c = 0.25 as the airfoil control; spheroid L1 for the grid trend):
   evaluate rho(u.grad u).t_s + dp/ds - mu[d2u_s/dy2 + (n_r/r) du_s/dy
   + d2u_s/ds2] through the layer from 5-ray FD stencils.  The imbalance is
   the numerical-dissipation footprint; nu_eff/nu = (conv + dp/ds)/(rho
   d2u_s/dy2-op) reads the effective viscosity the field actually obeys.
   All cases share the SAME numerics (Roe, lowMachPreconditioner=false,
   M=0.1, 2nd order, kappaMUSCL=-1), so a spheroid-only excess indicts the
   grid/geometry interaction, not the scheme per se.

3. AIRFOIL CONTROL at matched Re_theta: NLF0416 Re=4e6 alpha=0 upper
   (front 0.391): stations x/c = 0.12/0.20/0.30 (Re_theta ~ 450-750), and
   Eppler387 Re=2e5 alpha=2 upper (front 0.60): x/c = 0.25/0.40.  H via the
   IDENTICAL operator; local Thwaites lambda from the same slice's u_e(x).

4. STREAMWISE-RESOLUTION measure: meridional spacing ds(x) per level from
   the committed generator (spheroid/ogrid_spheroid.py meridian_points),
   airfoil chordwise spacing from the case's own surface-contour walk,
   plate spacing from the mesh points; expressed as cells per delta99 and
   cell aspect ratio.

Inputs: the a0-physics extraction caches (SAAI_A0PHYS_CACHE_DIR, rebuilt
automatically if absent via spheroid_a0_physics.get_data/reference_checks)
plus plain (no-gradient-filter) probes of the committed L2/L1 spheroid
volumes, the Sec-III flat plate, and the NLF/Eppler L2 slices.

5. STAGGERING PROBE: u_s along x at fixed wall heights, dx = 2e-4 (12
   samples per meridional cell) -- measures the one-cell tangential
   staggering wiggle of the discrete solution directly (cf. the high-lift
   Cf-wiggle finding); plus the hx-robustness ladder of the 0.42 balance,
   and a march on the NLF's OWN u_e (is the airfoil's favorable-region H
   LE-history-consistent?  Answer: yes, to 0.02).

Outputs (exploratory, NOT paper figures; no in-figure titles):
  paper/repro/cfd/figs_explore/spheroid_a0_meanflow_march.png
  paper/repro/cfd/figs_explore/spheroid_a0_meanflow_profiles.png
  paper/repro/cfd/figs_explore/spheroid_a0_meanflow_balance.png
  paper/repro/cfd/figs_explore/spheroid_a0_meanflow_resolution.png
  paper/repro/cfd/figs_explore/spheroid_a0_meanflow_wiggle.png
  paper/repro/cfd/figs_explore/spheroid_a0_meanflow.json

Run:  python3 -u paper/repro/cfd/spheroid_a0_meanflow.py
"""
import json
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_banded
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figs_explore")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "spheroid"))

from spheroid_a0_physics import (                     # noqa: E402
    edge_and_integrals, get_data, reference_checks, spheroid_arc, CASE, RAY,
    YBAND, CACHED)
from surface_map import surface_frame                 # noqa: E402

SPH_ROOT = os.environ.get("SAAI_SPH_ROOT", os.path.join(REPO, "spheroid_fv1"))
FP_ROOT = os.environ.get("SAAI_CFD_ROOT", os.path.join(REPO, "flow360_fv1"))

A_ELL, B_ELL = 0.5, 1.0 / 12.0        # spheroid semi-axes (x centered at 0.5L)
STATIONS = (0.20, 0.42, 0.70)         # profile-comparison stations

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 15, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "lines.linewidth": 2.0, "figure.dpi": 110, "savefig.dpi": 150})
C = dict(field="#3b6bb5", lam="#b0483a", tc="#7a5aa8", planar="#3e8f5c",
         gray="0.45", plate="#666666", nlf="#3e8f5c", epp="#e0821f")


# ============================================================ Part 1: marcher
def ellipse_geo(x):
    """Arc length s(x), body radius r0(x), and outward-normal radial
    component n_r(x) of the 6:1 spheroid meridian (x = x/L from nose)."""
    s, r0 = spheroid_arc(np.asarray(x, float))
    xc = np.asarray(x, float) - A_ELL                 # centered coordinate
    nx, nr = xc / A_ELL**2, r0 / B_ELL**2
    nn = np.hypot(nx, nr) + 1e-30
    return s, r0, nr / nn


def ygrid(h1=4.0e-7, ytop=0.03, n=420):
    """Geometric wall-normal grid: y_0 = 0, first spacing h1, top ytop."""
    lo, hi = 1.0001, 1.2
    for _ in range(80):
        g = 0.5 * (lo + hi)
        if h1 * (g**n - 1) / (g - 1) < ytop:
            lo = g
        else:
            hi = g
    dy = h1 * g**np.arange(n)
    return np.concatenate([[0.0], np.cumsum(dy)]) * (
        ytop / (h1 * (g**n - 1) / (g - 1)))


def _deriv_weights(y):
    """First-derivative central weights on a nonuniform grid, rows j=1..N-1:
    returns (wm, w0, wp) with du/dy_j = wm u_{j-1} + w0 u_j + wp u_{j+1}."""
    hm = y[1:-1] - y[:-2]
    hp = y[2:] - y[1:-1]
    wm = -hp / (hm * (hm + hp))
    wp = hm / (hp * (hm + hp))
    return wm, -(wm + wp), wp


def bl_march(x_march, ue_march, nu, mode, x_profiles=(), ic="quartic",
             theta0=None, y=None, verbose=False, geom="spheroid"):
    """Implicit space-march of the laminar BL equations (module docstring
    eq.) along the spheroid meridian.  mode: 'planar' | 'axi' | 'tc'.
    x_march: monotone x/L grid (first entry = IC station); ue_march: edge
    velocity on it.  theta0: IC momentum thickness (default: axisymmetric
    Thwaites at x_march[0] from the same u_e).  Returns dict with x, theta,
    dstar, H, cf_theta (= cf theta u_e / nu shear-shape parameter) and
    profiles {x: (y, u)} at the requested stations."""
    x_march = np.asarray(x_march, float)
    ue = np.asarray(ue_march, float)
    if geom == "flat":
        s, r0, n_r = (x_march.copy(), np.ones_like(x_march),
                      np.zeros_like(x_march))
    else:
        s, r0, n_r = ellipse_geo(x_march)
    # forcing DISCRETELY CONSISTENT with the backward-Euler march: with
    # due_ds[i] = (ue[i]-ue[i-1])/ds, u = u_e is an exact outer solution of
    # the discrete momentum equation, so no spurious inviscid "wake" deficit
    # accumulates (np.gradient's central stencil at the nose-stub kink left
    # a permanent 2.5e-3 outer deficit that doubled theta -- found the hard
    # way; the O(ds) bias vs central is ~1e-7 in u_e here)
    due_ds = np.concatenate([[0.0], np.diff(ue) / np.diff(s)])
    if y is None:
        y = ygrid()
    ny = len(y)
    wm, w0, wp = _deriv_weights(y)
    hm = y[1:-1] - y[:-2]
    hp = y[2:] - y[1:-1]
    hc = 0.5 * (hm + hp)

    if theta0 is None:                      # axisymmetric Thwaites IC theta
        w = ue**5 * r0**2
        integ = np.concatenate([[0.0], np.cumsum(
            0.5 * (w[1:] + w[:-1]) * np.diff(s))])
        # nose stub assumed linear-u_e (tiny weight); start-integral only
        th2 = 0.45 * nu * integ[0:1] / max(ue[0]**6 * r0[0]**2, 1e-30)
        theta0 = float(np.sqrt(0.45 * nu * (integ[0] + w[0] * s[0] / 6.0)
                               / max(ue[0]**6 * r0[0]**2, 1e-30)))
    if ic == "quartic":                     # Pohlhausen lambda=0
        dP = theta0 * 315.0 / 37.0
        eta = np.clip(y / dP, 0.0, 1.0)
        u = ue[0] * (2 * eta - 2 * eta**3 + eta**4)
    else:                                   # crude tanh (IC-sensitivity)
        u = ue[0] * np.tanh(y / (1.35 * theta0))

    def rad(i):
        if mode == "planar":
            return np.ones_like(y)
        if mode == "axi":
            return np.full_like(y, r0[i])
        return r0[i] + y * n_r[i]           # tc: first-order transverse curv

    def integrals(u_i, ue_i):
        """delta*/theta cut at 2 x the 0.99-u_e height: guards the integrals
        against any residual outer-region deficit (inviscid, undamped)."""
        f = np.clip(u_i / ue_i, 0.0, 1.2)
        j99 = int(np.argmax(u_i >= 0.99 * ue_i))
        cut = y <= min(2.0 * y[max(j99, 1)], y[-1])
        dstar = np.trapz((1 - f)[cut], y[cut])
        theta = np.trapz((f * (1 - f))[cut], y[cut])
        return dstar, theta

    out = dict(x=[], theta=[], dstar=[], H=[], cf_theta=[])
    profiles = {}
    want = sorted(x_profiles)
    r_prev = rad(0)
    for i in range(1, len(x_march)):
        ds = s[i] - s[i - 1]
        r_i = rad(i)
        u_prev = u.copy()
        ug = u.copy()                        # Picard iterate
        rhs_pg = ue[i] * due_ds[i]
        for it in range(30):
            # continuity -> v (trapezoid of d(ru)/ds from the wall up)
            dru = (r_i * ug - r_prev * u_prev) / ds
            rv = -np.concatenate([[0.0], np.cumsum(
                0.5 * (dru[1:] + dru[:-1]) * np.diff(y))])
            v = rv / r_i
            # tridiagonal assembly (rows j = 1..ny-2)
            rp = 0.5 * (r_i[1:-1] + r_i[2:])
            rm = 0.5 * (r_i[1:-1] + r_i[:-2])
            dfp = nu * rp / (r_i[1:-1] * hp * hc)
            dfm = nu * rm / (r_i[1:-1] * hm * hc)
            vj = v[1:-1]
            lo = -dfm + vj * wm
            di = ug[1:-1] / ds + dfm + dfp + vj * w0
            up = -dfp + vj * wp
            rhs = ug[1:-1] * u_prev[1:-1] / ds + rhs_pg
            rhs[-1] -= up[-1] * ue[i]        # top Dirichlet u = u_e
            ab = np.zeros((3, ny - 2))
            ab[0, 1:] = up[:-1]
            ab[1] = di
            ab[2, :-1] = lo[1:]
            sol = solve_banded((1, 1), ab, rhs)
            u_new = np.concatenate([[0.0], sol, [ue[i]]])
            err = np.max(np.abs(u_new - ug)) / ue[i]
            ug = u_new
            if err < 1e-10:
                break
        u, r_prev = ug, r_i
        dstar, theta = integrals(u, ue[i])
        dudy_w = u[1] / y[1]
        out["x"].append(x_march[i])
        out["theta"].append(theta)
        out["dstar"].append(dstar)
        out["H"].append(dstar / max(theta, 1e-30))
        out["cf_theta"].append(nu * dudy_w / ue[i]**2 * theta * ue[i] / nu)
        while want and x_march[i] >= want[0] - 1e-12:
            profiles[want.pop(0)] = (y.copy(), u.copy(), float(ue[i]))
        if verbose and i % 400 == 0:
            print(f"    march {mode} x={x_march[i]:.3f} H={out['H'][-1]:.4f}"
                  f" it={it}", flush=True)
    out = {k: np.array(v) for k, v in out.items()}
    out["profiles"] = profiles
    out["mode"] = mode
    return out


def marcher_validation(nu=1.388888888888889e-08):
    """Constant-u_e planar march must reproduce Blasius: H = 2.5905,
    cf theta u_e/nu = 0.2205, theta = 0.664 sqrt(nu x / u_e)."""
    ue0 = 0.1044
    x = np.concatenate([np.linspace(0.02, 0.06, 200, endpoint=False),
                        np.arange(0.06, 1.2001, 5e-4)])
    res = {}
    for tag, kw in (("base", {}),
                    ("ic_tanh", dict(ic="tanh")),
                    ("coarse", dict(y=ygrid(8e-7, 0.03, 210))),):
        m = bl_march(x if tag != "coarse" else x[::2],
                     np.full_like(x if tag != "coarse" else x[::2], ue0),
                     nu, "planar", geom="flat", **kw)
        j = int(np.argmin(np.abs(m["x"] - 1.0)))
        th_bl = 0.664 * np.sqrt(nu * m["x"][j] / ue0)
        res[tag] = dict(H=float(m["H"][j]),
                        cf_theta=float(m["cf_theta"][j]),
                        theta_over_blasius=float(m["theta"][j] / th_bl))
        print(f"  validation {tag}: H(x=1) = {res[tag]['H']:.4f} "
              f"(Blasius 2.5905), cf*th*ue/nu = {res[tag]['cf_theta']:.4f} "
              f"(0.2205), theta/Blasius = "
              f"{res[tag]['theta_over_blasius']:.4f}", flush=True)
    return res


def spheroid_ue_march_grid(sw):
    """March grid + field u_e on it (linear-in-s nose stub below the first
    edge-ok station, savgol-smoothed Pchip above -- thwaites_eN convention)."""
    x = np.array([s["x"] for s in sw])
    eok = np.array([s["edge_ok"] for s in sw])
    ue_f = savgol_filter(np.array([s["u_e"] for s in sw])[eok], 15, 3)
    x_f = x[eok]
    ue_i = PchipInterpolator(x_f, ue_f)
    x0 = float(x_f[0])
    s_all, _, _ = ellipse_geo(np.concatenate([[x0], x_f]))
    xm = np.concatenate([np.linspace(0.02, 0.08, 300, endpoint=False),
                         np.arange(0.08, 0.9301, 4e-4)])
    s_m, _, _ = ellipse_geo(xm)
    ue_m = np.where(xm >= x0, ue_i(np.clip(xm, x0, x_f[-1])),
                    ue_f[0] * s_m / s_all[0])
    return xm, ue_m


def spheroid_march(sw, nu):
    """The three-mode march on the field's own u_e."""
    xm, ue_m = spheroid_ue_march_grid(sw)
    runs = {}
    for mode in ("planar", "axi", "tc"):
        runs[mode] = bl_march(xm, ue_m, nu, mode, x_profiles=STATIONS,
                              verbose=True)
        H42 = float(np.interp(0.42, runs[mode]["x"], runs[mode]["H"]))
        print(f"  spheroid march [{mode}]  H(0.42) = {H42:.4f}", flush=True)
    # IC sensitivity on the axi mode
    for tag, kw in (("ic_tanh", dict(ic="tanh")),
                    ("th0_x1.3", dict(theta0=None)),):
        m = bl_march(xm, ue_m, nu, "axi", ic=kw.get("ic", "quartic"),
                     theta0=(None if tag != "th0_x1.3" else 1.3 * float(
                         runs["axi"]["theta"][0])))
        runs[f"axi_{tag}"] = dict(x=m["x"], H=m["H"], theta=m["theta"],
                                  mode=f"axi_{tag}")
        print(f"  spheroid march [axi,{tag}] H(0.42) = "
              f"{float(np.interp(0.42, m['x'], m['H'])):.4f}", flush=True)
    return runs


def operator_matched_H(march, sw):
    """Run the FIELD's edge/integral operator (edge_and_integrals on the
    probe RAY grid) on the marched profiles at the comparison stations --
    removes any operator-convention component from the H comparison."""
    rows = {}
    for xq, (y, u, ue) in march["profiles"].items():
        st = dict(y=RAY, U=np.interp(RAY, y, u, right=ue),
                  us=np.interp(RAY, y, u, right=ue), nu=np.array([1.0]))
        edge_and_integrals(st)
        rows[f"{xq:g}"] = dict(H_op=float(st["H"]),
                               theta_op=float(st["theta"]))
    return rows


# ============================================= Part 2: momentum balance probe
def load_plain(case_dir, fname="volume.pvtu"):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(os.path.join(case_dir, fname))
    r.Update()
    g = r.GetOutput()
    print(f"  {os.path.basename(case_dir)}: {g.GetNumberOfPoints():,} pts "
          f"(plain)", flush=True)
    mu = json.load(open(os.path.join(case_dir, "Flow360.json"))
                   )["freestream"]["muRef"]
    return g, mu


def probe_pts(grid, pts):
    pd = vtk.vtkPolyData()
    vp = vtk.vtkPoints()
    vp.SetData(numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
    pd.SetPoints(vp)
    pr = vtk.vtkProbeFilter()
    pr.SetInputData(pd)
    pr.SetSourceData(grid)
    pr.Update()
    out = pr.GetOutput().GetPointData()
    return {out.GetArray(i).GetName(): vtk_to_numpy(out.GetArray(i))
            for i in range(out.GetNumberOfArrays())}


SAVGOL_W = 61          # calibrated in spheroid_a0_physics (PLANAR_W note)


def balance_station(grid, mu_ref, hx, make_stencil, d99,
                    axis_pts=None, ray=None, label=""):
    """5-ray streamwise stencil momentum balance, projected on the center
    station's t_s.  make_stencil(k) -> (surface point, n, t_s) of stencil
    station k in -2..2 (surface-arc spacing hx).  axis_pts: callable r(pts)
    giving the distance-from-axis of ray points (None = planar: no metric
    term).  Returns uniform-grid profiles of the terms + nu_eff."""
    ray = RAY if ray is None else ray
    stations = [make_stencil(k) for k in (-2, -1, 0, 1, 2)]
    _, n0, t0 = stations[2]
    pts = np.concatenate([P + ray[:, None] * n for (P, n, t) in stations])
    res = probe_pts(grid, pts)
    ny = len(ray)
    u5 = res["velocity"].reshape(5, ny, 3).astype(float)
    p5 = res["p"].reshape(5, ny).astype(float)
    rho5 = res.get("rho", np.ones((5 * ny,))).reshape(5, ny).astype(float)

    yg = np.linspace(0.0, 3.0 * d99, 600)
    hg = yg[1] - yg[0]

    def resm(f):                       # resample to uniform + savgol smooth
        return savgol_filter(np.interp(yg, ray, f), SAVGOL_W, 4)

    def dy(f, d=1):
        return savgol_filter(np.interp(yg, ray, f), SAVGOL_W, 4, deriv=d,
                             delta=hg)

    # center-ray fields
    u0 = np.stack([resm(u5[2, :, i]) for i in range(3)], axis=1)
    du0_dy = np.stack([dy(u5[2, :, i]) for i in range(3)], axis=1)
    rho0 = resm(rho5[2])
    us = u0 @ t0
    un = u0 @ n0
    dus_dy = du0_dy @ t0
    d2us_dy2 = np.stack([dy(u5[2, :, i], 2) for i in range(3)], axis=1) @ t0
    # streamwise 5-point FD at fixed y (rays treated as parallel; the
    # neglected metric factor is 1 + kappa*y ~ 1 + 3e-4 here)
    c1 = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * hx)
    c2 = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * hx * hx)
    u_i = np.stack([np.stack([resm(u5[k, :, i]) for i in range(3)], axis=1)
                    for k in range(5)])
    p_i = np.stack([resm(p5[k]) for k in range(5)])
    du_ds = np.einsum("k,kij->ij", c1, u_i)
    dp_ds = c1 @ p_i
    d2us_ds2 = np.einsum("k,kij->ij", c2, u_i) @ t0

    conv = rho0 * (us * (du_ds @ t0) + un * dus_dy)
    metric = np.zeros_like(yg)
    if axis_pts is not None:
        P0, n_0, _ = stations[2]
        rr = axis_pts(P0[None] + yg[:, None] * n_0)
        drdy = np.gradient(rr, yg)
        metric = drdy / np.maximum(rr, 1e-9)
    visc_op = d2us_dy2 + metric * dus_dy
    visc = mu_ref * (visc_op + d2us_ds2)
    imbal = conv + dp_ds - visc
    nu = mu_ref / rho0
    with np.errstate(all="ignore"):
        nu_eff = (conv + dp_ds) / (rho0 * visc_op)
    # summary in the shear region (0.1-0.75 d99: away from the wall probe
    # floor and from the u'' ~ 0 outer zero-crossing)
    sel = (yg > 0.1 * d99) & (yg < 0.75 * d99) & (np.abs(visc_op)
                                                  > 0.1 * np.nanmax(
                                                      np.abs(visc_op)))
    scale = float(np.nanmax(np.abs(visc[sel])))
    summ = dict(
        label=label, d99=float(d99), hx=float(hx),
        nu_eff_over_nu_median=float(np.nanmedian((nu_eff / nu)[sel])),
        nu_eff_over_nu_iqr=[float(np.nanpercentile((nu_eff / nu)[sel], q))
                            for q in (25, 75)],
        imbalance_over_maxvisc_median=float(
            np.nanmedian(np.abs(imbal[sel])) / scale),
        conv_max_over_visc_max=float(np.nanmax(np.abs(conv[sel])) / scale))
    prof = dict(y=yg, conv=conv, dpds=dp_ds, visc=visc, imbal=imbal,
                nu_eff_over_nu=nu_eff / nu, sel=sel, us=us, d99=d99,
                label=label)
    print(f"  balance [{label}]: nu_eff/nu median = "
          f"{summ['nu_eff_over_nu_median']:.3f} "
          f"IQR {summ['nu_eff_over_nu_iqr'][0]:.3f}-"
          f"{summ['nu_eff_over_nu_iqr'][1]:.3f}; |imbal|/max|visc| = "
          f"{summ['imbalance_over_maxvisc_median']:.3f}", flush=True)
    return prof, summ


def spheroid_stencil(x0, hx):
    """Surface stencil generator along the phi=90 meridian."""
    s0, _, _ = ellipse_geo(np.array([x0]))

    def make(k):
        # invert arc length for the stencil station
        xg = np.linspace(max(x0 - 3 * hx, 1e-3), min(x0 + 3 * hx, 0.999), 400)
        sg, _, _ = ellipse_geo(xg)
        xk = float(np.interp(s0[0] + k * hx, sg, xg))
        P, n3, t_s, _ = surface_frame(np.array([xk]),
                                      np.array([np.radians(90.0)]))
        return P[0], n3[0], t_s[0]
    return make


def axis_r(pts):
    return np.hypot(pts[:, 1], pts[:, 2])


# ================================================= Part 3: airfoil H control
def airfoil_stations(case_dir, af, z_te_half, xqs, nu_ref, ray):
    """Wall-normal profiles at x/c stations on the UPPER surface of an
    airfoil slice; H etc. via the identical operator.  Also returns the
    local chordwise mesh spacing and Thwaites lambda from the slice's own
    u_e(x)."""
    import regen_nlf_v2 as m
    Xm, Zm, up_idx, lo_idx = m.walk_contour_xz(case_dir, af=af,
                                               z_te_half=z_te_half)
    xs, zs = Xm[up_idx], Zm[up_idx]
    o = np.argsort(xs)
    xs, zs = xs[o], zs[o]
    tx, tz = np.gradient(xs), np.gradient(zs)
    tn = np.hypot(tx, tz) + 1e-30
    tx, tz = tx / tn, tz / tn
    nx, nz = -tz, tx
    if np.mean(nz) < 0:
        nx, nz = -nx, -nz
    y0 = m.slice_y_plane(case_dir)
    g, _, _ = m.load_slice(case_dir)
    # u_e sweep along the chord for lambda (probe at all surface nodes)
    rows = []
    stash = {}
    for xq in xqs:
        j = int(np.argmin(np.abs(xs - xq)))
        P0 = np.array([xs[j], y0, zs[j]])
        n0 = np.array([nx[j], 0.0, nz[j]])
        t0 = np.array([tx[j], 0.0, tz[j]])
        pts = P0[None] + ray[:, None] * n0
        res = probe_pts(g, pts)
        u = res["velocity"].astype(float)
        rho = res.get("rho", np.ones(len(ray))).astype(float)
        st = dict(y=ray, U=np.linalg.norm(u, axis=1), us=u @ t0,
                  nu=nu_ref / np.maximum(rho, 1e-6))
        edge_and_integrals(st)
        nu = float(np.median(st["nu"]))
        chi = rho * res["nuHat"].astype(float) / nu_ref
        ds_local = float(np.hypot(xs[j + 1] - xs[j], zs[j + 1] - zs[j]))
        rows.append(dict(x=float(xs[j]), H=float(st["H"]),
                         Rt=float(st["u_e"] * st["theta"] / nu),
                         theta=float(st["theta"]), d99=float(st["d99"]),
                         u_e=float(st["u_e"]), edge_ok=bool(st["edge_ok"]),
                         chimax=float(np.nanmax(chi[ray <= 0.02])),
                         ds_mesh=ds_local,
                         cells_per_d99=float(st["d99"] / ds_local)))
        stash[xq] = (st, P0, n0, t0, j)
    # Thwaites lambda at each station from u_e(x) of the extracted stations
    if len(rows) >= 2:
        xr = np.array([r["x"] for r in rows])
        uer = np.array([r["u_e"] for r in rows])
        for k, r in enumerate(rows):
            due = np.gradient(uer, xr)[k]
            nu = nu_ref
            r["lambda_thwaites"] = float(r["theta"]**2 * due / nu)
            r["H_thwaites_eq"] = float(2.61 - 3.75 * r["lambda_thwaites"]
                                       + 5.24 * r["lambda_thwaites"]**2)
    return rows, stash, (xs, zs, tx, tz, nx, nz, y0, g)


def airfoil_ue_march(case_dir, af, z_te_half, nu_ref, ray, x_lo=0.025,
                     x_hi=0.42, n_st=64, x_check=(0.12, 0.20, 0.30)):
    """2D laminar BL march on the AIRFOIL's own u_e (upper surface):
    extract u_e at n_st chordwise stations via the same edge operator, then
    march in surface arc length with a linear-in-s stagnation stub.  Answers
    whether the airfoil's favorable-region H is LE-acceleration history
    (genuine BL physics) or the same anomaly as the spheroid."""
    import regen_nlf_v2 as m
    Xm, Zm, up_idx, lo_idx = m.walk_contour_xz(case_dir, af=af,
                                               z_te_half=z_te_half)
    xs, zs = Xm[up_idx], Zm[up_idx]
    o = np.argsort(xs)
    xs, zs = xs[o], zs[o]
    tx, tz = np.gradient(xs), np.gradient(zs)
    tn = np.hypot(tx, tz) + 1e-30
    tx, tz = tx / tn, tz / tn
    nx, nz = -tz, tx
    if np.mean(nz) < 0:
        nx, nz = -nx, -nz
    y0 = m.slice_y_plane(case_dir)
    g, _, _ = m.load_slice(case_dir)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs),
                                                    np.diff(zs)))])
    xq = np.linspace(x_lo, x_hi, n_st)
    ue, sq, thq = [], [], []
    for xv in xq:
        j = int(np.argmin(np.abs(xs - xv)))
        P0 = np.array([xs[j], y0, zs[j]])
        n0 = np.array([nx[j], 0.0, nz[j]])
        t0 = np.array([tx[j], 0.0, tz[j]])
        res = probe_pts(g, P0[None] + ray[:, None] * n0)
        u = res["velocity"].astype(float)
        st = dict(y=ray, U=np.linalg.norm(u, axis=1), us=u @ t0)
        edge_and_integrals(st)
        if st["edge_ok"]:
            ue.append(st["u_e"])
            sq.append(arc[j])
            thq.append(st["theta"])
    ue = savgol_filter(np.array(ue), 11, 3)
    sq = np.array(sq)
    # march grid in arc length from just aft of the LE
    s0 = sq[0]
    sm = np.concatenate([np.linspace(0.55 * s0, s0, 120, endpoint=False),
                         np.arange(s0, sq[-1], 3e-4)])
    ue_i = PchipInterpolator(sq, ue)
    ue_m = np.where(sm >= s0, ue_i(np.clip(sm, s0, sq[-1])),
                    ue[0] * sm / s0)
    mres = bl_march(sm, ue_m, nu_ref, "planar", geom="flat",
                    x_profiles=(), theta0=None,
                    ic="quartic")
    # theta0 default uses the spheroid weights on geom flat: r0 = 1 -> fine
    x_of_s = PchipInterpolator(arc, xs)
    x_m = x_of_s(np.clip(mres["x"], arc[0], arc[-1]))
    out = dict(x=x_m, H=mres["H"], theta=mres["theta"],
               s=mres["x"], ue_stations=list(map(float, ue)),
               s_stations=list(map(float, sq)))
    out["H_at"] = {f"{xv:g}": float(np.interp(xv, x_m, mres["H"]))
                   for xv in x_check}
    out["theta_field_at_stations"] = list(map(float, thq))
    print(f"  NLF u_e march: H at {x_check} = "
          + " ".join(f"{v:.3f}" for v in out["H_at"].values()), flush=True)
    return out


def wiggle_and_hx(d99_42=8.95e-4):
    """(cached) Direct staggering probe of the L2 field: u_s along x at
    fixed wall heights (dx = 2e-4 << the 2.6e-3 meridional cell), plus the
    hx-robustness ladder of the 0.42 balance.  The near-wall one-cell
    wiggle is the discrete solution's structural signature (cf. the
    high-lift Cf-wiggle finding: tangential BL staggering)."""
    cache = os.path.join(CACHED, "a0meanflow_wiggle.pkl")
    if os.path.exists(cache):
        print(f"  using cached wiggle probe {cache}", flush=True)
        return pickle.load(open(cache, "rb"))
    g, mu = load_plain(CASE)
    xg = np.arange(0.38, 0.4601, 2e-4)
    fr = (0.15, 0.30, 0.60, 1.2)
    P, n3, t_s, _ = surface_frame(xg, np.full_like(xg, np.radians(90.0)))
    pts = np.concatenate([P + (f * d99_42) * n3 for f in fr])
    res = probe_pts(g, pts)
    u = res["velocity"].astype(float).reshape(len(fr), len(xg), 3)
    us = np.einsum("fij,ij->fi", u, t_s)
    out = dict(x=xg, fr=fr, us=us)
    bals = {}
    for hx in (0.003, 0.006, 0.012):
        bals[hx] = balance_station(
            g, mu, hx=hx, make_stencil=spheroid_stencil(0.42, hx),
            d99=d99_42, axis_pts=axis_r, label=f"L2 0.42 hx={hx}")
    try:
        pickle.dump((out, bals), open(cache, "wb"))
    except OSError:
        pass
    return out, bals


def wiggle_stats(out):
    rows = {}
    x, us = out["x"], out["us"]
    for i, f in enumerate(out["fr"]):
        v = us[i]
        sm = np.convolve(v, np.ones(25) / 25, mode="same")
        r = (v - sm)[12:-12]
        F = np.fft.rfft(r * np.hanning(len(r)))
        freq = np.fft.rfftfreq(len(r), d=float(x[1] - x[0]))
        j = int(np.argmax(np.abs(F[3:]))) + 3
        rows[f"{f:g}"] = dict(rms_du_over_u=float(np.std(r) / np.mean(v)),
                              p2p_du_over_u=float((r.max() - r.min())
                                                  / np.mean(v)),
                              dominant_wavelength=float(1.0 / freq[j]))
    return rows


def fig_wiggle(out, fp, d99=8.95e-4, ds_cell=2.58e-3):
    x, us = out["x"], out["us"]
    fig, axs = plt.subplots(len(out["fr"]), 1, figsize=(9.5, 9.6),
                            sharex=True)
    for ax, i in zip(axs, range(len(out["fr"]))):
        v = us[i]
        sm = np.convolve(v, np.ones(25) / 25, mode="same")
        ax.plot(x[12:-12], 1e3 * (v - sm)[12:-12] / np.mean(v),
                color=C["field"], lw=1.2)
        ax.set_ylabel(r"$10^3\,\delta u_s/\bar u_s$")
        ax.text(0.015, 0.83, f"$y/\\delta_{{99}} = {out['fr'][i]:g}$",
                transform=ax.transAxes, fontsize=12)
    axs[0].set_title("")
    for k in range(int(0.38 / ds_cell), int(0.47 / ds_cell) + 1):
        for ax in axs:
            ax.axvline(k * ds_cell, color=C["gray"], lw=0.4, alpha=0.5)
    axs[-1].set_xlabel(r"$x/L$ (thin gray: meridional cell boundaries, "
                       r"$\Delta s = 2.6\times10^{-3}$)")
    axs[-1].set_xlim(0.383, 0.457)
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def challenge_cp_pg(sw, runs, nu):
    """User-challenge probes (2026-07-28): (1) Cp(x) and u_e(x) from the L2
    field with the suction peak and PG-sign regions marked; (2) equilibrium
    H_eq(lambda(x)) + the marched (non-equilibrium) H(x) against the field;
    (3) aft-station (0.55-0.85) H re-verification with denser rays and an
    edge-convention ladder, and the planar-kernel maxP re-extracted there,
    against the SAME kernel run on the marched profiles (the physical
    expectation, which must RISE into the adverse region)."""
    from spheroid_a0_physics import planar_kernel_smooth
    cache = os.path.join(CACHED, "a0meanflow_challenge.pkl")
    x = np.array([s["x"] for s in sw])
    eok = np.array([s["edge_ok"] for s in sw])
    ue = np.array([s["u_e"] for s in sw])
    th = np.array([s["theta"] for s in sw])
    d99 = np.array([s["d99"] for s in sw])

    # (1)+(2) analytic parts from the cache: peak, lambda, H_eq
    s_arc, _, _ = ellipse_geo(x[eok])
    ue_s = savgol_filter(ue[eok], 15, 3)
    due = np.gradient(ue_s, s_arc)
    xf = x[eok]
    j0 = np.where(due <= 0)[0][0]
    x_peak = float(np.interp(0.0, [due[j0], due[j0 - 1]],
                             [xf[j0], xf[j0 - 1]]))
    lam = np.clip(th[eok]**2 * due / nu, -0.12, 0.25)
    H_eq = np.where(lam >= 0, 2.61 - 3.75 * lam + 5.24 * lam**2,
                    2.088 + 0.0731 / (lam + 0.14))     # Cebeci-Bradshaw
    print(f"  u_e suction peak at x/L = {x_peak:.3f} (user expectation: "
          f"at/slightly before 0.5); lambda range on 0.06-0.85: "
          f"[{lam[(xf > 0.06) & (xf < 0.85)].min():+.4f}, "
          f"{lam[(xf > 0.06) & (xf < 0.85)].max():+.4f}]", flush=True)

    if os.path.exists(cache):
        print(f"  using cached challenge probes {cache}", flush=True)
        cp_x, cp, aft = pickle.load(open(cache, "rb"))
    else:
        g, mu_ref = load_plain(CASE)
        # Cp along the meridian: p at 0.3*d99 (p constant through the BL),
        # referenced to a far probe; q_inf = 0.5 rho_inf Mach^2
        m_ok = eok & np.isfinite(d99)
        cp_x = x[m_ok]
        P, n3, _, _ = surface_frame(cp_x, np.full_like(cp_x,
                                                       np.radians(90.0)))
        pts = np.concatenate([P + (0.3 * d99[m_ok])[:, None] * n3,
                              np.array([[0.5, 20.0, 0.0]])])
        res = probe_pts(g, pts)
        p = res["p"].astype(float)
        cp = (p[:-1] - p[-1]) / (0.5 * 1.0 * 0.1**2)
        # aft re-verification: dense rays + edge ladder + planar maxP
        ray_d = np.geomspace(2.5e-6, 0.12, 400)
        aft = []
        for xq in (0.55, 0.60, 0.70, 0.80, 0.85):
            Pq, nq, tq, _ = surface_frame(np.array([xq]),
                                          np.array([np.radians(90.0)]))
            r2 = probe_pts(g, Pq[0][None] + ray_d[:, None] * nq[0])
            u = r2["velocity"].astype(float)
            rho = r2.get("rho", np.ones(len(ray_d))).astype(float)
            st = dict(y=ray_d, U=np.linalg.norm(u, axis=1), us=u @ tq[0],
                      nu=mu_ref / np.maximum(rho, 1e-6))
            edge_and_integrals(st)
            row = dict(x=xq, H_localmax=float(st["H"]),
                       d99=float(st["d99"]), u_e=float(st["u_e"]),
                       Rt=float(st["u_e"] * st["theta"]
                                / np.median(st["nu"])))
            for fac in (1.5, 2.0, 3.0):     # fixed-edge-height ladder
                i_e = int(np.argmin(np.abs(ray_d - fac * st["d99"])))
                ue_f2 = st["us"][i_e]
                f = np.clip(st["us"] / ue_f2, 0.0, 1.2)
                ds_ = float(np.trapz((1 - f)[:i_e + 1], ray_d[:i_e + 1]))
                th_ = float(np.trapz((f * (1 - f))[:i_e + 1],
                                     ray_d[:i_e + 1]))
                row[f"H_edge{fac:g}d99"] = ds_ / max(th_, 1e-30)
            kp = planar_kernel_smooth(st)
            jp = int(np.argmax(np.where(kp["y"] >= 1e-5, kp["P"], -np.inf)))
            row["maxP_planar_field"] = float(kp["P"][jp])
            aft.append(row)
            print(f"  aft recheck x={xq}: H(local-max edge)="
                  f"{row['H_localmax']:.3f} "
                  f"H(1.5/2/3 d99)={row['H_edge1.5d99']:.3f}/"
                  f"{row['H_edge2d99']:.3f}/{row['H_edge3d99']:.3f} "
                  f"maxP={row['maxP_planar_field']:.4f}", flush=True)
        del g
        try:
            pickle.dump((cp_x, cp, aft), open(cache, "wb"))
        except OSError:
            pass

    # planar maxP on the MARCHED profiles (the physical expectation),
    # including the aft stations: re-march with the extended profile list
    xm, ue_mm = spheroid_ue_march_grid(sw)
    m2 = bl_march(xm, ue_mm, nu, "axi",
                  x_profiles=(0.20, 0.30, 0.42, 0.55, 0.60, 0.70, 0.80,
                              0.85))
    march_P = {}
    for xq, (y_m, u_m, ue_m) in sorted(m2["profiles"].items()):
        st = dict(y=RAY, us=np.interp(RAY, y_m, u_m, right=ue_m),
                  U=np.interp(RAY, y_m, u_m, right=ue_m),
                  nu=np.full(1, nu))
        edge_and_integrals(st)
        kp = planar_kernel_smooth(st)
        jp = int(np.argmax(np.where(kp["y"] >= 1e-5, kp["P"], -np.inf)))
        march_P[f"{xq:g}"] = float(kp["P"][jp])
    print(f"  planar maxP on marched profiles: {march_P}", flush=True)
    return dict(x_peak=x_peak, lam_x=list(map(float, xf)),
                lam=list(map(float, lam)), H_eq=list(map(float, H_eq)),
                cp_x=list(map(float, cp_x)), cp=list(map(float, cp)),
                aft=aft, march_maxP_planar=march_P)


def fig_cp_pg(sw, runs, ch, fp):
    x = np.array([s["x"] for s in sw])
    H = np.array([s["H"] for s in sw])
    ue = np.array([s["u_e"] for s in sw])
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9.5, 12.2), sharex=True)
    # Cp + u_e
    a1.plot(ch["cp_x"], ch["cp"], color=C["field"], label=r"$C_p$ (field)")
    a1.plot(x, 1.0 - (ue / 0.1)**2, "--", color=C["gray"], lw=1.4,
            label=r"$1-(u_e/U_\infty)^2$ (Bernoulli check)")
    a1.axvline(ch["x_peak"], color=C["lam"], lw=1.2, ls="-.")
    a1.text(ch["x_peak"] + 0.008, 0.35,
            f"$u_e$ peak x/L = {ch['x_peak']:.3f}", color=C["lam"],
            fontsize=11, rotation=90)
    a1.invert_yaxis()
    a1.set_ylabel(r"$C_p$")
    a1.legend(loc="upper right")
    # H: field, march, equilibrium
    a2.plot(x[(x > 0.05) & (x < 0.93)], H[(x > 0.05) & (x < 0.93)],
            color=C["field"], lw=2.4, label="RANS L2 field")
    r = runs["axi"]
    mk = r["x"] <= 0.86
    a2.plot(r["x"][mk], r["H"][mk], "--", color=C["lam"], lw=2.0,
            label="laminar BL march (non-equilibrium)")
    a2.plot(ch["lam_x"], ch["H_eq"], ":", color=C["tc"], lw=1.8,
            label=r"equilibrium $H_{eq}(\lambda)$ (Cebeci-Bradshaw)")
    a2.axhline(2.5905, color=C["gray"], lw=0.9, ls=":")
    a2.axvline(ch["x_peak"], color=C["lam"], lw=1.2, ls="-.")
    a2.set_ylim(2.35, 2.75)
    a2.set_ylabel(r"$H$")
    a2.legend(loc="lower left", fontsize=10)
    # maxP: field re-check vs marched expectation; fore-body field values
    # from the committed physics-pass table (same planar W=61 estimator)
    fj = os.path.join(FIGD, "spheroid_a0_physics.json")
    if os.path.exists(fj):
        bt = json.load(open(fj)).get("band_table", [])
        xs_b = [r_["x"] for r_ in bt if r_["x"] < 0.5]
        Ps_b = [r_["Pmax_planar"] for r_ in bt if r_["x"] < 0.5]
        a3.plot(xs_b, Ps_b, "o", mfc="none", color=C["field"],
                label="field maxP (physics-pass table, fore-body)")
    a3.plot([r_["x"] for r_ in ch["aft"]],
            [r_["maxP_planar_field"] for r_ in ch["aft"]], "o-",
            color=C["field"], label="field maxP (planar kernel, aft "
                                    "re-check, dense rays)")
    xs_m = sorted(float(k) for k in ch["march_maxP_planar"])
    a3.plot(xs_m, [ch["march_maxP_planar"][f"{v:g}"] for v in xs_m], "s--",
            color=C["lam"], label="marched laminar profiles (physical "
                                  "expectation)")
    a3.axvline(ch["x_peak"], color=C["lam"], lw=1.2, ls="-.")
    a3.set_ylabel(r"$\max_y \hat\Omega\hat I$")
    a3.set_xlabel(r"$x/L$")
    a3.legend(loc="upper left", fontsize=10)
    for ax in (a1, a2, a3):
        ax.axvspan(0.0, ch["x_peak"], color="#3e8f5c", alpha=0.05)
        ax.axvspan(ch["x_peak"], 0.965, color="#b0483a", alpha=0.05)
        ax.set_xlim(0.02, 0.965)
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


# ============================================== Part 4: streamwise resolution
def spheroid_ds_per_level():
    """Meridional spacing at the comparison stations per level, from the
    committed generator."""
    from ogrid_spheroid import meridian_points, D
    LEV = {0: (150, 4.5e-3), 1: (300, 2.25e-3), 2: (600, 1.125e-3)}
    out = {}
    for lev, (nm, dsp) in LEV.items():
        P, n, s, beta = meridian_points(nm, dsp * D)
        xg = P[:, 0] + A_ELL                      # nose at x=0
        ds = np.diff(s)
        xc = 0.5 * (xg[1:] + xg[:-1])
        out[f"L{lev}"] = {f"{xq:g}": float(np.interp(xq, xc, ds))
                          for xq in (0.20, 0.42, 0.70)}
    return out


def plate_dx(grid, x0=2.0):
    """Streamwise mesh spacing of the flat plate at x0 from the wall-row
    mesh points (wall plane z = 0)."""
    P = vtk_to_numpy(grid.GetPoints().GetData())
    wall = (np.abs(P[:, 2]) < 1e-9) & (np.abs(P[:, 0] - x0) < 1.0)
    xw = np.unique(np.round(P[wall, 0], 10))
    if len(xw) < 3:
        return np.nan
    dx = np.diff(xw)
    xc = 0.5 * (xw[1:] + xw[:-1])
    return float(np.interp(x0, xc, dx))


# ===================================================================== figures
def fig_march(sw, runs, refs, fp):
    x = np.array([s["x"] for s in sw])
    H = np.array([s["H"] for s in sw])
    Rt = np.array([s["u_e"] * s["theta"] / np.median(s["nu"]) for s in sw])
    m = (x >= 0.06) & (x <= 0.93)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 8.6), sharex=True)
    th_axi = runs["axi"]
    mm = th_axi["x"] <= 0.86              # marcher separates ~0.87: mask
    a1.plot(x[m], Rt[m], color=C["field"], label="RANS L2 field")
    nu = 1.388888888888889e-08
    ue_f = np.array([s["u_e"] for s in sw])
    ue_m = np.interp(th_axi["x"], x, ue_f)
    a1.plot(th_axi["x"][mm], (ue_m * th_axi["theta"] / nu)[mm], "--",
            color=C["lam"], label="laminar BL march (axisym) on field $u_e$")
    a1.set_ylabel(r"$Re_\theta$")
    a1.legend(loc="upper left")
    a2.plot(x[m], H[m], color=C["field"], lw=2.4, label="RANS L2 field")
    for mode, col, ls, lab in (
            ("planar", C["planar"], ":", "march, planar (no Mangler)"),
            ("axi", C["lam"], "--", "march, axisymmetric (Mangler)"),
            ("tc", C["tc"], "-.", "march + transverse curvature")):
        r = runs[mode]
        mk = r["x"] <= 0.86
        a2.plot(r["x"][mk], r["H"][mk], ls, color=col, lw=1.9, label=lab)
    for tag, col in (("axi_ic_tanh", "0.6"), ("axi_th0_x1.3", "0.75")):
        r = runs[tag]
        mk = np.asarray(r["x"]) <= 0.86
        a2.plot(np.asarray(r["x"])[mk], np.asarray(r["H"])[mk], lw=0.9,
                color=col, alpha=0.8,
                label=("IC-shape / IC-$\\theta_0$ sensitivity"
                       if tag == "axi_ic_tanh" else None))
    a2.axhline(2.5905, color=C["gray"], lw=1.0, ls=":")
    a2.text(0.62, 2.60, "Blasius H = 2.5905", fontsize=11, color=C["gray"])
    # grid-ladder points at the stations
    for lev, mk in (("L0", "^"), ("L1", "s")):
        rows = refs[lev]
        a2.plot([r["x"] for r in rows], [r["H"] for r in rows], mk,
                color=C["field"], ms=7, mfc="none",
                label=f"RANS {lev} (stations)")
    a2.set_xlabel(r"$x/L$")
    a2.set_ylabel(r"$H = \delta^*/\theta$")
    a2.set_ylim(2.30, 2.75)
    a2.legend(loc="lower left", ncol=2, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def fig_profiles(sw, runs, fp):
    x = np.array([s["x"] for s in sw])
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.4), sharey=True)
    for ax, xq in zip(axs, STATIONS):
        st = sw[int(np.argmin(np.abs(x - xq)))]
        th = st["theta"]
        ax.plot(st["us"] / st["u_e"], st["y"] / th, color=C["field"],
                lw=2.4, label="RANS L2 field")
        for mode, col, ls in (("axi", C["lam"], "--"), ("tc", C["tc"], "-.")):
            y, u, ue = runs[mode]["profiles"][xq]
            thm = float(np.interp(xq, runs[mode]["x"], runs[mode]["theta"]))
            ax.plot(u / ue, y / thm, ls, color=col, lw=1.9,
                    label=("march, axisymmetric" if mode == "axi"
                           else "march + transv. curv."))
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 12)
        ax.set_xlabel(r"$u_s/u_e$")
        ax.text(0.06, 11.2, f"$x/L = {xq:g}$", fontsize=13)
    axs[0].set_ylabel(r"$y/\theta$ (each its own $\theta$)")
    axs[0].legend(loc="center right", fontsize=10)
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def fig_balance(profs, fp):
    fig, axs = plt.subplots(1, len(profs), figsize=(4.4 * len(profs), 5.6),
                            sharey=True)
    if len(profs) == 1:
        axs = [axs]
    for ax, pr in zip(axs, profs):
        sc = np.nanmax(np.abs(pr["visc"]))
        yy = pr["y"] / pr["d99"]
        ax.plot(pr["conv"] / sc, yy, color=C["field"], label="convection")
        ax.plot(pr["dpds"] / sc, yy, color=C["epp"], lw=1.6,
                label=r"$\partial p/\partial s$")
        ax.plot(pr["visc"] / sc, yy, color=C["lam"], label="viscous (laminar)")
        ax.plot(pr["imbal"] / sc, yy, "k:", lw=1.8, label="imbalance")
        ax.axvline(0, color="k", lw=0.7)
        ax.set_ylim(0, 1.6)
        ax.set_xlim(-1.3, 1.3)
        ax.set_xlabel("terms / max|viscous|")
        ax.text(0.03, 0.97, pr["label"], transform=ax.transAxes,
                va="top", fontsize=11.5)
        ax2 = ax.twiny()
        ax2.plot(pr["nu_eff_over_nu"], yy, color=C["tc"], lw=1.4, alpha=0.85)
        ax2.set_xlim(0, 2.5)
        ax2.axvline(1.0, color=C["tc"], lw=0.7, ls=":")
        ax2.set_xlabel(r"$\nu_{\rm eff}/\nu$ (purple)", fontsize=11,
                       color=C["tc"])
        ax2.grid(False)
    axs[0].set_ylabel(r"$y/\delta_{99}$")
    axs[0].legend(loc="center right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def fig_resolution(res_tab, fp):
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    rows = res_tab["rows"]
    names = [r["case"] for r in rows]
    vals = [r["cells_per_d99"] for r in rows]
    cols = [C["field"] if r["biased"] else C["planar"] for r in rows]
    yp = np.arange(len(rows))
    ax.barh(yp, vals, color=cols, alpha=0.85)
    ax.axvline(1.0, color="k", lw=0.9, ls=":")
    ax.set_yticks(yp)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel(r"streamwise cells per $\delta_{99}$ at the station")
    ax.set_xscale("log")
    for y_, r in zip(yp, rows):
        ax.text(max(r["cells_per_d99"] * 1.06, 0.05), y_,
                f"H = {r['H']:.3f}", va="center", fontsize=10)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


# ======================================================================== main
def main():
    os.makedirs(FIGD, exist_ok=True)
    nu_sph = 1.388888888888889e-08
    out = {}

    print("== marcher validation (Blasius) ==", flush=True)
    out["validation"] = marcher_validation(nu_sph)

    print("== field caches ==", flush=True)
    sw, bal_sts, nu_ref = get_data()
    refs = reference_checks()
    x = np.array([s["x"] for s in sw])
    H_f = np.array([s["H"] for s in sw])

    print("== spheroid laminar BL march (3 modes + IC sensitivity) ==",
          flush=True)
    cache = os.path.join(CACHED, "a0meanflow_march.pkl")
    if os.path.exists(cache):
        runs = pickle.load(open(cache, "rb"))
        print(f"  using cached march {cache}", flush=True)
    else:
        runs = spheroid_march(sw, nu_sph)
        try:
            pickle.dump(runs, open(cache, "wb"))
        except OSError:
            pass
    tab_H = {}
    for xq in STATIONS:
        row = dict(RANS_L2=float(np.interp(xq, x, H_f)))
        for lev in ("L0", "L1"):
            for r in refs[lev]:
                if abs(r["x"] - xq) < 1e-9:
                    row[f"RANS_{lev}"] = r["H"]
        for mode in ("planar", "axi", "tc", "axi_ic_tanh", "axi_th0_x1.3"):
            row[f"march_{mode}"] = float(np.interp(
                xq, runs[mode]["x"], runs[mode]["H"]))
        tab_H[f"{xq:g}"] = row
        print(f"  H @ {xq:g}: " + "  ".join(f"{k}={v:.3f}"
                                            for k, v in row.items()),
              flush=True)
    out["H_table"] = tab_H
    out["operator_matched_H_march_axi"] = operator_matched_H(runs["axi"], sw)
    print("  operator-matched march H:",
          out["operator_matched_H_march_axi"], flush=True)

    # u_e-scale sensitivity of the field H at 0.42 (candidate-d bound)
    st42 = sw[int(np.argmin(np.abs(x - 0.42)))]
    sens = {}
    for eps in (-0.01, -0.005, 0.005, 0.01):
        s2 = dict(st42)
        s2 = dict(y=st42["y"], U=st42["U"], us=st42["us"], nu=st42["nu"])
        edge_and_integrals(s2)
        ue = s2["u_e"] * (1 + eps)
        f = np.clip(st42["us"] / ue, 0.0, 1.2)
        i_e = s2["i_e"]
        dstar = float(np.trapz((1 - f)[:i_e + 1], st42["y"][:i_e + 1]))
        theta = float(np.trapz((f * (1 - f))[:i_e + 1], st42["y"][:i_e + 1]))
        sens[f"{eps:+g}"] = dstar / theta
    out["ue_scale_sensitivity_H_042"] = sens
    print(f"  u_e-scale sensitivity of H(0.42): {sens}", flush=True)

    print("== momentum balances ==", flush=True)
    cache = os.path.join(CACHED, "a0meanflow_balance.pkl")
    if os.path.exists(cache):
        profs, summs, res_extra = pickle.load(open(cache, "rb"))
        print(f"  using cached balances {cache}", flush=True)
        for sm in summs:
            print(f"  (cached) balance [{sm['label']}]: nu_eff/nu = "
                  f"{sm['nu_eff_over_nu_median']:.3f}", flush=True)
    else:
        profs, summs, res_extra = [], [], {}
        # --- spheroid L2 at 0.42 and 0.70
        g2, mu2 = load_plain(CASE)
        for x0, d99 in ((0.42, st42["d99"]),
                        (0.70, float(np.interp(0.70, x, [s["d99"]
                                                         for s in sw])))):
            pr, sm = balance_station(
                g2, mu2, hx=0.006, make_stencil=spheroid_stencil(x0, 0.006),
                d99=d99, axis_pts=axis_r, label=f"spheroid L2 x/L={x0:g}")
            profs.append(pr)
            summs.append(sm)
        del g2
        # --- spheroid L1 at 0.42 (grid trend of the footprint)
        g1, mu1 = load_plain(os.path.join(os.path.dirname(CASE),
                                          "case_ogrid_L1_saai_re72a0"))
        d99_l1 = [r["d99"] for r in refs["L1"] if abs(r["x"] - 0.42) < 1e-9][0]
        pr, sm = balance_station(
            g1, mu1, hx=0.012, make_stencil=spheroid_stencil(0.42, 0.012),
            d99=d99_l1, axis_pts=axis_r, label="spheroid L1 x/L=0.42")
        profs.append(pr)
        summs.append(sm)
        del g1
        # --- flat plate at x = 2.0 (matched Re_theta ~ 909)
        gp, mup = load_plain(os.path.join(FP_ROOT, "flatplate_sphere_Tu0040"))
        dxp = plate_dx(gp)
        plate_hx = 2.0 * dxp if np.isfinite(dxp) else 0.04
        d99_p = [r["d99"] for r in refs["plate"] if r["x"] == 2.0][0]

        def plate_stencil(k):
            return (np.array([2.0 + k * plate_hx, -0.05, 0.0]),
                    np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
        pr, sm = balance_station(
            gp, mup, hx=plate_hx, make_stencil=plate_stencil, d99=d99_p,
            axis_pts=None, label="plate x=2 (Re_th 909)")
        profs.append(pr)
        summs.append(sm)
        res_extra["plate_dx_at_x2"] = dxp
        del gp
        try:
            pickle.dump((profs, summs, res_extra), open(cache, "wb"))
        except OSError:
            pass
    out["balance"] = summs

    print("== airfoil controls ==", flush=True)
    cache = os.path.join(CACHED, "a0meanflow_airfoil.pkl")
    ray_nlf = np.geomspace(1e-6, 0.02, 240)
    ray_epp = np.geomspace(2e-6, 0.05, 240)
    if os.path.exists(cache):
        nlf_rows, nlf_bal_prof, nlf_bal_summ, epp_rows = pickle.load(
            open(cache, "rb"))
        print(f"  using cached airfoil control {cache}", flush=True)
    else:
        nlf_case = os.path.join(FP_ROOT, "cavL2prop_nlf0416_Re4M_a0")
        nlf_rows, stash, geo = airfoil_stations(
            nlf_case, "nlf0416", 0.00125, (0.12, 0.20, 0.30), 2.5e-8,
            ray_nlf)
        for r in nlf_rows:
            print(f"  NLF a0 x/c={r['x']:.3f}: H={r['H']:.3f} "
                  f"Rt={r['Rt']:.0f} lam={r.get('lambda_thwaites', 0):+.4f} "
                  f"H_eq={r.get('H_thwaites_eq', 0):.3f} "
                  f"chimax={r['chimax']:.2e} cells/d99="
                  f"{r['cells_per_d99']:.1f}", flush=True)
        # NLF momentum balance at 0.25 on the slice
        (xs, zs, tx, tz, nx, nz, y0, g_nlf) = geo
        j0 = int(np.argmin(np.abs(xs - 0.25)))
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs),
                                                        np.diff(zs)))])
        hx_n = 3.0 * float(np.hypot(xs[j0 + 1] - xs[j0], zs[j0 + 1] - zs[j0]))

        def nlf_stencil(k):
            aq = arc[j0] + k * hx_n
            xq = np.interp(aq, arc, xs)
            zq = np.interp(aq, arc, zs)
            jq = int(np.argmin(np.abs(arc - aq)))
            n0 = np.array([nx[jq], 0.0, nz[jq]])
            t0 = np.array([tx[jq], 0.0, tz[jq]])
            return np.array([xq, y0, zq]), n0, t0
        d99_n = float(np.interp(0.25, [r["x"] for r in nlf_rows],
                                [r["d99"] for r in nlf_rows]))
        nlf_bal_prof, nlf_bal_summ = balance_station(
            g_nlf, 2.5e-8, hx=hx_n, make_stencil=nlf_stencil, d99=d99_n,
            axis_pts=None, label="NLF0416 a0 x/c=0.25", ray=ray_nlf)
        # Eppler control
        epp_case = os.path.join(FP_ROOT, "cavL2prop_eppler387_Re200k_a2")
        epp_rows, _, _ = airfoil_stations(
            epp_case, "eppler387", 0.000833, (0.25, 0.40), 5e-7, ray_epp)
        for r in epp_rows:
            print(f"  Eppler a2 x/c={r['x']:.3f}: H={r['H']:.3f} "
                  f"Rt={r['Rt']:.0f} lam={r.get('lambda_thwaites', 0):+.4f} "
                  f"H_eq={r.get('H_thwaites_eq', 0):.3f} "
                  f"chimax={r['chimax']:.2e} cells/d99="
                  f"{r['cells_per_d99']:.1f}", flush=True)
        try:
            pickle.dump((nlf_rows, nlf_bal_prof, nlf_bal_summ, epp_rows),
                        open(cache, "wb"))
        except OSError:
            pass
    profs.append(nlf_bal_prof)
    summs.append(nlf_bal_summ)
    out["nlf_stations"] = nlf_rows
    out["eppler_stations"] = epp_rows

    print("== NLF u_e march control (is the airfoil H history-consistent?) "
          "==", flush=True)
    cache = os.path.join(CACHED, "a0meanflow_nlfmarch.pkl")
    if os.path.exists(cache):
        nlfm = pickle.load(open(cache, "rb"))
        print(f"  using cached NLF march {cache}: H_at = {nlfm['H_at']}",
              flush=True)
    else:
        nlfm = airfoil_ue_march(
            os.path.join(FP_ROOT, "cavL2prop_nlf0416_Re4M_a0"), "nlf0416",
            0.00125, 2.5e-8, ray_nlf)
        try:
            pickle.dump(nlfm, open(cache, "wb"))
        except OSError:
            pass
    out["nlf_ue_march_H"] = nlfm["H_at"]

    print("== staggering wiggle + hx robustness (L2) ==", flush=True)
    wout, wbals = wiggle_and_hx(d99_42=st42["d99"])
    out["wiggle"] = wiggle_stats(wout)
    out["balance_hx_ladder"] = {f"{hx:g}": sm for hx, (pr, sm)
                                in wbals.items()}
    print("  wiggle stats:", json.dumps(out["wiggle"], indent=1), flush=True)

    print("== user-challenge: Cp / PG-sign / equilibrium-H / aft re-check "
          "==", flush=True)
    ch = challenge_cp_pg(sw, runs, nu_sph)
    out["challenge"] = {k: ch[k] for k in ("x_peak", "aft",
                                           "march_maxP_planar")}

    print("== streamwise resolution ==", flush=True)
    ds_sph = spheroid_ds_per_level()
    out["spheroid_ds_per_level"] = ds_sph
    rows = []
    d99_by_lev = {"L0": {f"{r['x']:g}": r["d99"] for r in refs["L0"]},
                  "L1": {f"{r['x']:g}": r["d99"] for r in refs["L1"]},
                  "L2": {"0.2": float(np.interp(0.2, x, [s["d99"] for s in
                                                         sw])),
                         "0.42": st42["d99"],
                         "0.7": float(np.interp(0.7, x, [s["d99"] for s in
                                                         sw]))}}
    H_by_lev = {"L0": {f"{r['x']:g}": r["H"] for r in refs["L0"]},
                "L1": {f"{r['x']:g}": r["H"] for r in refs["L1"]},
                "L2": {"0.2": float(np.interp(0.2, x, H_f)),
                       "0.42": float(np.interp(0.42, x, H_f)),
                       "0.7": float(np.interp(0.7, x, H_f))}}
    for lev in ("L0", "L1", "L2"):
        for xq in ("0.2", "0.42", "0.7"):
            key = "0.2" if xq == "0.2" else xq
            rows.append(dict(
                case=f"spheroid {lev} x/L={xq}",
                ds=ds_sph[lev]["0.2" if xq == "0.2" else xq],
                d99=d99_by_lev[lev][key], H=H_by_lev[lev][key],
                cells_per_d99=d99_by_lev[lev][key]
                / ds_sph[lev]["0.2" if xq == "0.2" else xq], biased=True))
    dxp = None
    cacheb = os.path.join(CACHED, "a0meanflow_balance.pkl")
    if os.path.exists(cacheb):
        _, _, res_extra = pickle.load(open(cacheb, "rb"))
        dxp = res_extra.get("plate_dx_at_x2")
    if dxp:
        d99_p = [r["d99"] for r in refs["plate"] if r["x"] == 2.0][0]
        H_p = [r["H"] for r in refs["plate"] if r["x"] == 2.0][0]
        rows.append(dict(case="plate x=2", ds=dxp, d99=d99_p, H=H_p,
                         cells_per_d99=d99_p / dxp, biased=False))
    for r in nlf_rows:
        rows.append(dict(case=f"NLF0416 L2 x/c={r['x']:.2f}",
                         ds=r["ds_mesh"], d99=r["d99"], H=r["H"],
                         cells_per_d99=r["cells_per_d99"], biased=False))
    for r in epp_rows:
        rows.append(dict(case=f"Eppler387 L2 x/c={r['x']:.2f}",
                         ds=r["ds_mesh"], d99=r["d99"], H=r["H"],
                         cells_per_d99=r["cells_per_d99"], biased=False))
    res_tab = dict(rows=rows)
    out["resolution"] = rows
    for r in rows:
        print(f"  {r['case']:>26}: ds={r['ds']:.2e} d99={r['d99']:.2e} "
              f"cells/d99={r['cells_per_d99']:.2f} H={r['H']:.3f}",
              flush=True)

    print("== figures ==", flush=True)
    fig_march(sw, runs, refs, os.path.join(
        FIGD, "spheroid_a0_meanflow_march.png"))
    fig_profiles(sw, runs, os.path.join(
        FIGD, "spheroid_a0_meanflow_profiles.png"))
    fig_balance(profs, os.path.join(
        FIGD, "spheroid_a0_meanflow_balance.png"))
    fig_resolution(res_tab, os.path.join(
        FIGD, "spheroid_a0_meanflow_resolution.png"))
    fig_wiggle(wout, os.path.join(FIGD, "spheroid_a0_meanflow_wiggle.png"),
               d99=st42["d99"],
               ds_cell=ds_sph["L2"]["0.42"])
    fig_cp_pg(sw, runs, ch, os.path.join(
        FIGD, "spheroid_a0_meanflow_cp_pg.png"))

    fj = os.path.join(FIGD, "spheroid_a0_meanflow.json")
    json.dump(out, open(fj, "w"), indent=1, default=float)
    print("tables:", fj, flush=True)


if __name__ == "__main__":
    main()
