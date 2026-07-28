"""Converged-state spheroid post-processing (alpha = 0, Re_L = 7.2e6):
circumferential uniformity, Cp(x), and the laminar BL profile series.

USER DIRECTIVE (2026-07-28): for the spheroid, (1) first make sure the flow
is circumferentially uniform; (2) plot pressure versus x; (3) a series of
boundary-layer profile plots: at each x, draw a line from the surface in the
surface-normal direction till it reaches outside the BL, probe/interpolate
the velocity along that line, and plot it at each x station.

Primary case: the CONVERGED (43k-step) full-body structured O-grid arm A,
  /local_data/qiqi/sa-ai/spheroid_fv1/case_ogridfull_L1_saai_re72a0
(agent-paper-review/2026-07-28-0413-spheroid-fullbody.md).  Cross-checks
(same instruments, summary numbers only): arm B (lowMach preconditioner,
25k) and the converged unstructured case_unstr_L1_saai_re72a0 (43k,
2026-07-28-0350 record).

Conventions (all inherited, not reinvented):
  - rays/rings are RE-BASED on the actual discrete wall facets via per-ray
    Moller-Trumbore intersection (spheroid_unstruct_a0_verdict.wall_offsets):
    the raw analytic-origin convention manufactures a fake one-azimuthal-cell
    mode (sag ~1.6e-5 L = up to 12% of a 0.15 d99 probe height) that
    collapses ~500x under re-basing (0413 record Sec 5).  The raw convention
    is shown ONCE (x/L = 0.42, 0.15 d99) to display the contrast.
  - edge/integral operator: spheroid_a0_physics.edge_and_integrals on the
    RAY grid (edge = first wall-normal speed max); front = near-wall
    (y <= 0.02) max-chi crossing of chi = 1 (spheroid_fullbody_check
    convention; the 0413 verdict table numbers are the validation gate).
  - ring residual: signal minus 25-sample running mean, 12 samples per
    azimuthal cell, trim 12 (the 0105/0413 wiggle_stats convention).
  - Cp = (p - p_inf) / (0.5 rho_inf M^2), rho_inf = 1, M = 0.1, p_inf =
    1/gamma (surface_map.py convention); the solver's own surface Cp output
    is used for the wall curves.
  - laminar reference: the validated implicit axisymmetric laminar-BL march
    of spheroid_a0_meanflow.bl_march (Blasius-validated to H 2.5905), driven
    by the CONVERGED field's own u_e, profiles compared through the SAME
    edge/integral operator (operator-matched).

Outputs (exploratory, NOT paper figures; house style: no in-figure titles,
captions in the companion md):
  paper/repro/cfd/figs_explore/spheroid_uniformity_rings.png
  paper/repro/cfd/figs_explore/spheroid_uniformity_contrast.png
  paper/repro/cfd/figs_explore/spheroid_cp_x.png
  paper/repro/cfd/figs_explore/spheroid_bl_profiles.png
  paper/repro/cfd/figs_explore/spheroid_uniformity_profiles_captions.md
  paper/repro/cfd/figs_explore/spheroid_uniformity_profiles.json

Run:  python3 -u paper/repro/cfd/spheroid_uniformity_profiles.py
      [--arms A B unstr]   (A alone regenerates all figures)
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figs_explore")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "spheroid"))

from spheroid_a0_physics import (                     # noqa: E402
    edge_and_integrals, front_crossing, planar_kernel_smooth, RAY)
from spheroid_unstruct_a0_verdict import (            # noqa: E402
    load_wall_facets, wall_offsets, forces_tail)
from spheroid_fullbody_check import (                 # noqa: E402
    load_grid, probe_pts, ring_points, XS, CHI_BAND, MESH_FULL)
from spheroid_a0_meanflow import (                    # noqa: E402
    bl_march, spheroid_ue_march_grid, marcher_validation)
from surface_map import surface_frame                 # noqa: E402

SPH_ROOT = os.environ.get("SAAI_SPH_ROOT", "/local_data/qiqi/sa-ai/spheroid_fv1")
CASES = {
    "A": dict(dir=os.path.join(SPH_ROOT, "case_ogridfull_L1_saai_re72a0"),
              src="slice", n_az=160, mesh=MESH_FULL,
              surf="surface_fluid_wall.pvtu",
              label="full body L1, 43k (arm A)"),
    "B": dict(dir=os.path.join(SPH_ROOT,
                               "case_ogridfull_L1_saai_re72a0_lowmach"),
              src="slice", n_az=160, mesh=MESH_FULL,
              surf="surface_fluid_wall.pvtu",
              label="full body L1, precond, 25k (arm B)"),
    "unstr": dict(dir=os.path.join(SPH_ROOT, "case_unstr_L1_saai_re72a0"),
                  src="volume", n_az=160, mesh=None,   # mesh.cgns in-case
                  surf="surface_farfield_body.pvtu",
                  label="unstructured L1, 43k"),
}
# 0413-record verdict numbers = the instrument-validation gate for arm A
GATE_A = dict(H={"0.2": 2.5248, "0.42": 2.5549, "0.7": 2.6065},
              front_chi1=0.8581)

RING_STATIONS = (0.20, 0.42, 0.70)      # the cases' constant-x slice planes
RING_FR = (0.15, 0.50, 1.00)            # heights / delta99 (edge = 1.0 d99)
SAMPLES_PER_CELL = 12
WALL_RING_X = (0.20, 0.42, 0.70, 0.90)  # solver-native surface-Cp/Cf rings
PROF_STATIONS = (0.10, 0.20, 0.30, 0.42, 0.50, 0.60, 0.70, 0.80, 0.84, 0.88)
# 0.84/0.88 straddle the converged chi=1 front (0.855-0.858 on every family)
GAMMA, MACH = 1.4, 0.1
QINF = 0.5 * MACH**2                    # rho_inf = 1

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 14, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.fontsize": 10.5, "figure.dpi": 110, "savefig.dpi": 150})
C = dict(field="#3b6bb5", march="#b0483a", raw="#e0821f", p="#3e8f5c",
         gray="0.45")


# ------------------------------------------------------------- meridian sweep
def sweep_case(grid, mu_ref, facets, phi_deg=90.0, xs=XS):
    """spheroid_fullbody_check.meridian_sweep generalized to any meridian
    (needed for the phi = 270 symmetry cross-check); identical RAY grid,
    facet re-basing, edge/integral operator and chi-band convention --
    validated by reproducing the 0413 verdict table on arm A."""
    xs = np.asarray(xs, float)
    P, n3, t_s, _ = surface_frame(xs, np.full_like(xs, np.radians(phi_deg)))
    t0 = np.zeros(len(xs))
    if facets is not None:
        t0 = wall_offsets(P, n3, *facets)
    P = P + t0[:, None] * n3
    pts = np.concatenate([P[k] + RAY[:, None] * n3[k] for k in range(len(xs))])
    res, valid = probe_pts(grid, pts)
    ny = len(RAY)
    u_all = res["velocity"].astype(float)
    rho_all = res.get("rho", np.ones(len(pts))).astype(float)
    nuhat = (res["solutionTurbulence"] if "solutionTurbulence" in res
             else res["nuHat"]).astype(float)
    sw = []
    for k, xq in enumerate(xs):
        sl = slice(k * ny, (k + 1) * ny)
        u = u_all[sl]
        st = dict(y=RAY, U=np.linalg.norm(u, axis=1), us=u @ t_s[k],
                  nu=mu_ref / np.maximum(rho_all[sl], 1e-6),
                  chi=rho_all[sl] * nuhat[sl] / mu_ref,
                  valid=valid[sl], x=float(xq), t0=float(t0[k]))
        edge_and_integrals(st)
        st["chimax_nearwall"] = float(np.nanmax(
            np.where(RAY <= CHI_BAND, st["chi"], -np.inf)))
        sw.append(st)
    return sw


def sweep_summary(sw, stations=RING_STATIONS):
    x = np.array([s["x"] for s in sw])
    chim = np.array([s["chimax_nearwall"] for s in sw])
    rows = {}
    for xq in stations:
        s = sw[int(np.argmin(np.abs(x - xq)))]
        rows[f"{xq:g}"] = dict(
            H=float(s["H"]), d99=float(s["d99"]), u_e=float(s["u_e"]),
            Rt=float(s["u_e"] * s["theta"] / np.median(s["nu"])))
    return dict(stations=rows,
                front_chi1=front_crossing(x, chim, 1.0),
                front_cv1=front_crossing(x, chim, 7.1))


def ue_peak(sw):
    """Suction-peak (u_e maximum) location from the sweep's edge velocity
    (savgol + first due/ds = 0 crossing -- the challenge_cp_pg convention)."""
    from scipy.signal import savgol_filter
    from spheroid_a0_meanflow import ellipse_geo
    x = np.array([s["x"] for s in sw])
    eok = np.array([s["edge_ok"] for s in sw])
    xf = x[eok]
    ue_s = savgol_filter(np.array([s["u_e"] for s in sw])[eok], 15, 3)
    s_arc, _, _ = ellipse_geo(xf)
    due = np.gradient(ue_s, s_arc)
    j0 = np.where((xf > 0.2) & (due <= 0))[0][0]
    xpk = float(np.interp(0.0, [due[j0], due[j0 - 1]], [xf[j0], xf[j0 - 1]]))
    return xpk, float(np.interp(xpk, xf, ue_s))


# ---------------------------------------------------------- azimuthal rings
def ring_probe(grid, xq, d99, n_az, facets, heights=RING_FR, corrected=True):
    """u_s and p around the full azimuth at heights x d99; ray origins
    re-based on the wall facets when corrected (t0 per phi, reused across
    heights).  Full-circle convention of spheroid_fullbody_check."""
    cell = 2.0 * np.pi / n_az
    phi = np.arange(0.0, 2.0 * np.pi, cell / SAMPLES_PER_CELL)
    t0 = None
    if corrected and facets is not None:
        P0, n3, _, _ = surface_frame(np.full_like(phi, xq), phi)
        t0 = wall_offsets(P0, n3, *facets)
    out = dict(phi=phi, cell_deg=float(np.degrees(cell)), heights=heights,
               us={}, p={}, n_invalid={},
               t0_absmax=float(np.abs(t0).max()) if t0 is not None else 0.0)
    for f in heights:
        pts, t_s = ring_points(xq, phi, f * d99, t0)
        res, valid = probe_pts(grid, pts)
        u = res["velocity"].astype(float)
        us = np.einsum("ij,ij->i", u, t_s)
        p = res["p"].astype(float)
        n_bad = int((~valid).sum())
        if n_bad and n_bad <= max(8, len(us) // 100):
            ph = np.degrees(phi)
            us = np.interp(ph, ph[valid], us[valid], period=360.0)
            p = np.interp(ph, ph[valid], p[valid], period=360.0)
        elif n_bad:
            us, p = np.where(valid, us, np.nan), np.where(valid, p, np.nan)
        out["us"][f], out["p"][f], out["n_invalid"][f] = us, p, n_bad
    return out


def residual(v):
    """The 0105/0413 wiggle convention: 25-sample running-mean detrend."""
    sm = np.convolve(v, np.ones(25) / 25, mode="same")
    return (v - sm)[12:-12]


def ring_stats(rg, n_az):
    rows = {}
    for f in rg["heights"]:
        v, p = rg["us"][f], rg["p"][f]
        if np.isnan(v).any():
            rows[f"{f:g}"] = dict(error="invalid probe points")
            continue
        r, rp = residual(v), residual(p)
        Fp = np.fft.rfft(v - v.mean()) / len(v)
        amp = 2.0 * np.abs(Fp) / np.mean(v)
        jm = int(np.argmax(amp[1:])) + 1
        rows[f"{f:g}"] = dict(
            rms_du_over_u=float(np.std(r) / np.mean(v)),
            p2p_du_over_u=float((r.max() - r.min()) / np.mean(v)),
            rms_dCp=float(np.std(rp) / QINF),
            p2p_dCp=float((rp.max() - rp.min()) / QINF),
            mode_ncell_amp=float(amp[n_az]),
            strongest_mode=int(jm), strongest_mode_amp=float(amp[jm]),
            low_mode_max_amp=float(amp[1:9].max()))
    return rows


# ------------------------------------------------------- wall (surface) data
def load_surface(case):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(os.path.join(case["dir"], case["surf"]))
    r.Update()
    g = r.GetOutput()
    P = vtk_to_numpy(g.GetPoints().GetData()).astype(float)
    pd = g.GetPointData()
    arr = {pd.GetArray(i).GetName(): vtk_to_numpy(pd.GetArray(i))
           for i in range(pd.GetNumberOfArrays())}
    return P, arr


def surface_cp_curves(P, arr, structured=True, n_az=160):
    """Meridian (z = 0, y > 0 node line -- O-grid only) and azimuthally
    averaged wall-Cp curves + the global azimuthal-scatter measure."""
    xL = P[:, 0] + 0.5
    cp, cf = arr["Cp"].astype(float), arr["Cf"].astype(float)
    out = {}
    if structured:                    # group by exact meridian-node x
        xr = np.round(xL, 9)
        ux = np.unique(xr)
        med = (np.abs(P[:, 2]) < 1e-7) & (P[:, 1] > 0)
        o = np.argsort(xL[med])
        out["meridian"] = (xL[med][o], cp[med][o])
        avg, std, cfr = [], [], []
        for xv in ux:
            m = xr == xv
            avg.append(cp[m].mean())
            std.append(cp[m].std())
            cfr.append(cf[m])
        out["az_x"], out["az_mean"] = ux, np.array(avg)
        out["az_std"] = np.array(std)
        # per-azimuth Cf-rise front (paper's third front convention, k=1.5
        # of the pre-rise laminar minimum): the uniformity measure of the
        # FRONT itself -- raw Cf rms just aft of the front only reads the
        # steep dCf/dx.  Grouping: exact azimuth node lines (round to the
        # 2.25-deg cell; a 1e-6-deg float grouping SPLITS lines and
        # decimates the x sampling).  Search window (0.5, 0.95): the
        # near-tail Cf dip at ~0.977 undercuts the pre-transition minimum
        # and, if included, hijacks the k=1.5 crossing.
        phi_r = np.degrees(np.arctan2(P[:, 1], P[:, 2])) % 360.0
        kaz = np.round(phi_r / (360.0 / n_az)).astype(int) % n_az
        fronts = []
        for kk in range(n_az):
            m = kaz == kk
            o2 = np.argsort(xL[m])
            xm, cm = xL[m][o2], cf[m][o2]
            sel = (xm > 0.5) & (xm < 0.95)
            xm, cm = xm[sel], cm[sel]
            jm = int(np.argmin(cm))
            hit = np.where(cm[jm:] >= 1.5 * cm[jm])[0]
            if len(hit):
                j = jm + hit[0]
                f = ((1.5 * cm[jm] - cm[j - 1]) / (cm[j] - cm[j - 1]))
                fronts.append(xm[j - 1] + f * (xm[j] - xm[j - 1]))
        fronts = np.array(fronts)
        out["cf_front"] = dict(n=len(fronts), median=float(np.median(fronts)),
                               rms=float(fronts.std()),
                               p2p=float(np.ptp(fronts)))
        out["rings"] = {}
        for x0 in WALL_RING_X:
            j = int(np.argmin(np.abs(ux - x0)))
            m = xr == ux[j]
            phi = np.degrees(np.arctan2(P[m, 1], P[m, 2]))  # mesh azimuth
            o2 = np.argsort(phi)
            out["rings"][f"{x0:g}"] = dict(
                x=float(ux[j]), phi=phi[o2], cp=cp[m][o2], cf=cf[m][o2],
                cp_rms=float(cp[m].std()), cp_p2p=float(np.ptp(cp[m])),
                cf_rms_over_mean=float(cf[m].std() / cf[m].mean()),
                cf_p2p_over_mean=float(np.ptp(cf[m]) / cf[m].mean()))
    else:                             # unstructured: bin in x
        bins = np.linspace(0.0, 1.0, 241)
        jb = np.clip(np.digitize(xL, bins) - 1, 0, 239)
        xc, avg = [], []
        for b in range(240):
            m = jb == b
            if m.sum() >= 8:
                xc.append(0.5 * (bins[b] + bins[b + 1]))
                avg.append(cp[m].mean())
        out["az_x"], out["az_mean"] = np.array(xc), np.array(avg)
    return out


# ------------------------------------------------------------ profile series
def march_reference(sw, nu):
    """Axisymmetric implicit laminar-BL march on the converged field's own
    u_e; profiles at PROF_STATIONS, run through the SAME edge/integral
    operator (operator-matched H, d99).  Stations past the marcher's
    separation point are dropped."""
    xm, ue_m = spheroid_ue_march_grid(sw)
    m = bl_march(xm, ue_m, nu, "axi", x_profiles=PROF_STATIONS)
    Hm = np.asarray(m["H"])
    bad = np.where(~np.isfinite(Hm) | (Hm > 3.2))[0]
    x_sep = float(m["x"][bad[0]]) if len(bad) else float(m["x"][-1])
    prof = {}
    for xq, (y, u, ue) in m["profiles"].items():
        if xq >= x_sep - 0.01:
            continue
        st = dict(y=RAY, U=np.interp(RAY, y, u, right=ue),
                  us=np.interp(RAY, y, u, right=ue), nu=np.array([1.0]))
        edge_and_integrals(st)
        prof[f"{xq:g}"] = dict(y=RAY, us=st["us"], u_e=float(st["u_e"]),
                               H_op=float(st["H"]), d99=float(st["d99"]))
    return prof, x_sep, dict(x=np.asarray(m["x"]), H=Hm)


def profile_rows(sw, march_prof):
    x = np.array([s["x"] for s in sw])
    rows = []
    for xq in PROF_STATIONS:
        st = sw[int(np.argmin(np.abs(x - xq)))]
        kp = planar_kernel_smooth(st)
        jp = int(np.argmax(np.where(kp["y"] >= 1e-5, kp["P"], -np.inf)))
        row = dict(x=float(st["x"]), H=float(st["H"]), d99=float(st["d99"]),
                   Rt=float(st["u_e"] * st["theta"] / np.median(st["nu"])),
                   u_e=float(st["u_e"]), edge_ok=bool(st["edge_ok"]),
                   maxP_planar=float(kp["P"][jp]),
                   chimax_nearwall=float(st["chimax_nearwall"]))
        mp = march_prof.get(f"{xq:g}")
        if mp:
            row["H_march_op"] = mp["H_op"]
            row["dH_vs_march"] = row["H"] - mp["H_op"]
        rows.append(row)
    return rows


# ------------------------------------------------------------------- figures
def fig_rings(rings, fp):
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 9.6), sharex=True)
    for i, xq in enumerate(RING_STATIONS):
        rg = rings[f"{xq:g}"]
        ph = np.degrees(rg["phi"][12:-12])
        n_c = len(rg["phi"]) // SAMPLES_PER_CELL
        ph_c = np.degrees(rg["phi"]).reshape(n_c, SAMPLES_PER_CELL).mean(1)
        for j, f in enumerate(RING_FR):
            ax = axs[i, j]
            v, p = rg["us"][f], rg["p"][f]
            ax.plot(ph, 1e3 * residual(v) / np.mean(v), color=C["field"],
                    lw=0.5, alpha=0.45,
                    label=r"$10^3\,\delta u_s/\bar u_s$ (probe residual)")
            vm = v.reshape(n_c, SAMPLES_PER_CELL).mean(1)
            ax.plot(ph_c, 1e3 * (vm - vm.mean()) / vm.mean(), color="k",
                    lw=1.2, label=r"$10^3\,\delta u_s/\bar u_s$ (cell mean)")
            pm = p.reshape(n_c, SAMPLES_PER_CELL).mean(1)
            ax.plot(ph_c, 1e3 * (pm - pm.mean()) / QINF, color=C["p"],
                    lw=1.2, ls="--", label=r"$10^3\,\delta C_p$ (cell mean)")
            ym = 1.15 * max(1e3 * np.abs(residual(v)).max() / np.mean(v),
                            1e-3)
            ax.set_ylim(-ym, ym)
            ax.text(0.02, 0.87, f"$x/L={xq:g}$,  $n={f:g}\\,\\delta_{{99}}$",
                    transform=ax.transAxes, fontsize=10.5)
            if i == 0 and j == 0:
                ax.legend(loc="lower right", ncol=1, fontsize=8.5)
    for ax in axs[-1]:
        ax.set_xlabel(r"azimuth $\varphi$ [deg]")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"residual $\times 10^3$")
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def fig_contrast(rg_raw, rg_cor, fp):
    f = RING_FR[0]
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    for rg, col, lab in (
            (rg_raw, C["raw"], "raw analytic-surface ray origin "
                               "(0105 convention)"),
            (rg_cor, C["field"], "origin re-based on the wall facets "
                                 "(Moller-Trumbore)")):
        ph = np.degrees(rg["phi"][12:-12])
        v = rg["us"][f]
        ax.plot(ph, 1e3 * residual(v) / np.mean(v), color=col, lw=0.9,
                label=lab)
    ax.set_xlabel(r"azimuth $\varphi$ [deg] at $x/L=0.42$, "
                  r"$n = 0.15\,\delta_{99}$")
    ax.set_ylabel(r"$10^3\,\delta u_s/\bar u_s$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def fig_cp(sc, sw, xpk, front, fp):
    x = np.array([s["x"] for s in sw])
    eok = np.array([s["edge_ok"] for s in sw])
    ue = np.array([s["u_e"] for s in sw])
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.plot(*sc["meridian"], color=C["field"], lw=1.8,
            label=r"wall $C_p$, $\varphi=90^\circ$ meridian (solver output)")
    ax.plot(sc["az_x"], sc["az_mean"], "--", color=C["raw"], lw=1.4,
            label=r"wall $C_p$, azimuthal average")
    ax.plot(x[eok], 1.0 - (ue[eok] / MACH)**2, ":", color=C["gray"], lw=1.4,
            label=r"$1-(u_e/U_\infty)^2$ from the BL-edge sweep")
    ax.axvline(xpk, color=C["march"], lw=1.2, ls="-.")
    ax.text(xpk + 0.008, 0.55, f"$u_e$ peak $x/L = {xpk:.3f}$",
            color=C["march"], fontsize=11, rotation=90, va="bottom")
    ax.axvline(front, color=C["p"], lw=1.0, ls=":")
    ax.text(front + 0.008, 0.55, f"$\\chi=1$ front {front:.3f}",
            color=C["p"], fontsize=11, rotation=90, va="bottom")
    ax.invert_yaxis()
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel(r"$C_p$")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def fig_profiles(sw, march_prof, rows, fp):
    x = np.array([s["x"] for s in sw])
    fig = plt.figure(figsize=(15.0, 10.6))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.9], hspace=0.28,
                          wspace=0.08)
    axs = [fig.add_subplot(gs[k // 5, k % 5]) for k in range(10)]
    for ax, xq, row in zip(axs, PROF_STATIONS, rows):
        st = sw[int(np.argmin(np.abs(x - xq)))]
        ax.plot(st["us"] / st["u_e"], st["y"] / st["d99"], color=C["field"],
                lw=2.0, label="RANS (converged)")
        mp = march_prof.get(f"{xq:g}")
        if mp:
            ax.plot(mp["us"] / mp["u_e"], mp["y"] / mp["d99"], "--",
                    color=C["march"], lw=1.1,
                    label="laminar BL march on field $u_e$")
        ax.set_xlim(0, 1.12)
        ax.set_ylim(0, 1.6)
        txt = (f"$x/L={xq:g}$\n$H={row['H']:.3f}$"
               + (f" (march {row['H_march_op']:.3f})" if mp else "")
               + f"\n$\\delta_{{99}}={row['d99']:.2e}$"
               f"\n$Re_\\theta={row['Rt']:.0f}$"
               f"\nmax$P={row['maxP_planar']:.3f}$")
        ax.text(0.05, 0.97, txt, transform=ax.transAxes, fontsize=8.6,
                va="top")
        if PROF_STATIONS.index(xq) % 5:
            ax.set_yticklabels([])
        if xq == PROF_STATIONS[0]:
            ax.legend(loc="lower right", fontsize=8.2)
    for k in (5, 6, 7, 8, 9):
        axs[k].set_xlabel(r"$u_t/u_e$")
    axs[0].set_ylabel(r"$n/\delta_{99}$")
    axs[5].set_ylabel(r"$n/\delta_{99}$")
    # bottom: physical wall distance, log scale, all stations
    axb = fig.add_subplot(gs[2, :])
    cm = plt.get_cmap("viridis")
    for k, xq in enumerate(PROF_STATIONS):
        st = sw[int(np.argmin(np.abs(x - xq)))]
        axb.plot(st["us"] / st["u_e"], st["y"], color=cm(k / 9.0), lw=1.5,
                 label=f"{xq:g}")
    axb.set_yscale("log")
    axb.set_ylim(3e-6, 0.03)
    axb.set_xlim(0, 1.12)
    axb.set_xlabel(r"$u_t/u_e$")
    axb.set_ylabel(r"$n/L$ (log)")
    axb.legend(loc="lower right", ncol=5, fontsize=9, title=r"$x/L$",
               title_fontsize=9)
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- main
def run_arm(tag, out, full_figs):
    c = CASES[tag]
    print(f"== {tag}: {c['dir']}", flush=True)
    mu_ref = json.load(open(os.path.join(c["dir"], "Flow360.json"))
                       )["freestream"]["muRef"]
    facets = load_wall_facets(c["mesh"] or os.path.join(c["dir"],
                                                        "mesh.cgns"))
    res = dict(label=c["label"], forces_tail=forces_tail(c["dir"]))
    ft = res["forces_tail"]
    print(f"  forces: step {ft['steps']}  CD {ft['CD']:.5f}  "
          f"CL {ft['CL']:.2e}  drift {ft['CD_drift_per_1k']:+.2e}/1k",
          flush=True)

    gm = load_grid(c["dir"], c["src"], slice_name="meridian")
    sw = sweep_case(gm, mu_ref, facets, phi_deg=90.0)
    summ = sweep_summary(sw)
    res["meridian"] = summ
    print("  H: " + "  ".join(f"{k}={v['H']:.4f}"
                              for k, v in summ["stations"].items())
          + f"  front chi1 {summ['front_chi1']:.4f} "
          f"cv1 {summ['front_cv1']:.4f}", flush=True)
    # symmetry cross-check on the opposite meridian
    sw270 = sweep_case(gm, mu_ref, facets, phi_deg=270.0)
    s270 = sweep_summary(sw270)
    res["meridian_phi270"] = dict(front_chi1=s270["front_chi1"],
                                  H=s270["stations"])
    print(f"  phi=270: front {s270['front_chi1']:.4f}  H(0.42) "
          f"{s270['stations']['0.42']['H']:.4f}", flush=True)
    xpk, uepk = ue_peak(sw)
    res["ue_peak"] = dict(x=xpk, u_e=uepk)
    print(f"  u_e peak x/L = {xpk:.4f} (u_e = {uepk:.5f})", flush=True)

    # rings (corrected; raw once at 0.42/0.15 for the convention contrast)
    stations = RING_STATIONS if full_figs else (0.42,)
    rings, rg_raw = {}, None
    for xq in stations:
        d99 = summ["stations"][f"{xq:g}"]["d99"]
        gs = (load_grid(c["dir"], "slice",
                        slice_name=f"x{int(round(100 * xq)):03d}")
              if c["src"] == "slice" else gm)
        rg = ring_probe(gs, xq, d99, c["n_az"], facets, corrected=True)
        rings[f"{xq:g}"] = rg
        if abs(xq - 0.42) < 1e-9:
            rg_raw = ring_probe(gs, 0.42, d99, c["n_az"], facets,
                                corrected=False)
        if c["src"] == "slice":
            del gs
    res["rings"] = {k: ring_stats(rg, c["n_az"]) for k, rg in rings.items()}
    res["rings_raw_042"] = ring_stats(rg_raw, c["n_az"])
    for k, st_ in res["rings"].items():
        for f, r in st_.items():
            print(f"  ring x={k} n={f}d99: rms du/u {r['rms_du_over_u']:.2e}"
                  f" p2p {r['p2p_du_over_u']:.2e}  rms dCp "
                  f"{r['rms_dCp']:.2e}", flush=True)
    r0 = res["rings_raw_042"][f"{RING_FR[0]:g}"]
    c0 = res["rings"]["0.42"][f"{RING_FR[0]:g}"]
    print(f"  RAW 0.42/0.15d99: rms du/u {r0['rms_du_over_u']:.2e} "
          f"(corrected {c0['rms_du_over_u']:.2e})", flush=True)

    # wall Cp/Cf (solver output on the actual wall facets)
    P, arr = load_surface(c)
    sc = surface_cp_curves(P, arr, structured=(tag != "unstr"),
                           n_az=c["n_az"])
    jmin = int(np.argmin(sc["az_mean"]))
    res["cp"] = dict(cp_min=float(sc["az_mean"][jmin]),
                     x_cp_min=float(sc["az_x"][jmin]),
                     az_std_max=float(sc["az_std"].max())
                     if "az_std" in sc else None)
    if "rings" in sc:
        res["cp"]["wall_rings"] = {
            k: {q: v[q] for q in ("x", "cp_rms", "cp_p2p",
                                  "cf_rms_over_mean", "cf_p2p_over_mean")}
            for k, v in sc["rings"].items()}
        print("  wall rings: " + "  ".join(
            f"x={k}: Cp rms {v['cp_rms']:.1e} Cf rms/mean "
            f"{v['cf_rms_over_mean']:.1e}"
            for k, v in res["cp"]["wall_rings"].items()), flush=True)
    if "cf_front" in sc:
        res["cp"]["cf_front_azimuth"] = sc["cf_front"]
        cff = sc["cf_front"]
        print(f"  Cf-rise front (k=1.5) over azimuth: median "
              f"{cff['median']:.4f}  rms {cff['rms']:.2e}  p2p "
              f"{cff['p2p']:.2e}  (n={cff['n']})", flush=True)
    if "meridian" in sc:
        cp_mer = np.interp(sc["az_x"], *sc["meridian"])
        res["cp"]["max_meridian_minus_azavg"] = float(
            np.max(np.abs(cp_mer - sc["az_mean"])))
    print(f"  Cp min {res['cp']['cp_min']:.4f} at x/L = "
          f"{res['cp']['x_cp_min']:.4f}", flush=True)

    # profile series + march reference
    nu = mu_ref                       # rho_inf = 1
    march_prof, x_sep, march_H = march_reference(sw, nu)
    res["march_separation_x"] = x_sep
    rows = profile_rows(sw, march_prof)
    res["profiles"] = rows
    for r in rows:
        m = (f" march {r['H_march_op']:.3f} (dH {r['dH_vs_march']:+.3f})"
             if "H_march_op" in r else " (no march: past separation)")
        print(f"  prof x={r['x']:g}: H={r['H']:.3f}{m}  Rt={r['Rt']:.0f}"
              f"  maxP={r['maxP_planar']:.3f}", flush=True)

    out[tag] = res
    if full_figs:
        fig_rings(rings, os.path.join(FIGD, "spheroid_uniformity_rings.png"))
        fig_contrast(rg_raw, rings["0.42"],
                     os.path.join(FIGD, "spheroid_uniformity_contrast.png"))
        fig_cp(sc, sw, xpk, summ["front_chi1"],
               os.path.join(FIGD, "spheroid_cp_x.png"))
        fig_profiles(sw, march_prof, rows,
                     os.path.join(FIGD, "spheroid_bl_profiles.png"))
        print("  figures written", flush=True)
    del gm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A", "B", "unstr"])
    args = ap.parse_args()
    os.makedirs(FIGD, exist_ok=True)
    out = dict(conventions=dict(
        ring="12 samples/azimuthal cell, 25-sample running-mean residual, "
             "ray origins Moller-Trumbore re-based on the wall facets "
             "(raw shown once at 0.42/0.15d99)",
        cp="solver surface Cp; qinf = 0.5*M^2, rho_inf = 1",
        profiles="RAY grid, edge = first wall-normal speed max, "
                 "front = near-wall (y<=0.02) chi max crossing 1",
        march="axisymmetric implicit laminar-BL march on the converged "
              "field's own u_e, operator-matched (spheroid_a0_meanflow)",
        gate_0413=GATE_A))
    out["march_validation"] = marcher_validation()
    for tag in args.arms:
        run_arm(tag, out, full_figs=(tag == "A"))
    fj = os.path.join(FIGD, "spheroid_uniformity_profiles.json")
    json.dump(out, open(fj, "w"), indent=1, default=float)
    print("wrote", fj, flush=True)


if __name__ == "__main__":
    main()
