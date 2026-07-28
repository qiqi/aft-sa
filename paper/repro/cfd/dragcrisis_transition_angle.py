"""Drag-crisis transition angle theta_tr(Re, seed, branch) by RADIAL-RAY chi max,
separation angles, and the sphere-kernel FPG diagnosis.

Executes the 2026-07-28 user directive: "a transition angle as a function of
Reynolds number across the range ... the angle can be obtained by shooting
lines from the surface radially to a large distance (or to infinity) and
search maximum xi along that line" (xi = chi = nuHat/muRef, the solver's
transition variable), extended per coordinator brief with (1) separation
angles, (2) kernel coordinates max Re_Omega and max P = max(Shat*Ihat) per
ray, (3) the high-Re favorable-gradient gate-vs-rate readout.

For every case of the 129-case drag-crisis steady campaign
(run_dragcrisis_matrix.py + run_dragcrisis_extension.py, records
2026-07-28-0010/-0412) that has slice_centerSpan.pvtu:
  - polar angle theta in [0, 180] deg from the FORWARD stagnation point
    (theta = 0 upstream; cylinder center (0.5, 0), R = 0.5 D -- verified
    rmin = 0.5 on all three mesh families), step 0.5 deg;
  - each ray runs from the wall to the slice edge (pilot/highre R_out =
    100 D, lowre 1000 D, lowre300 300 D -- all beyond the 20 D target;
    actual per-case reach recorded), N_S = 480 samples log-spaced in wall
    distance from 2e-7 D (below the smallest first-cell height, 1e-6 D);
  - maxchi(theta) = max chi over the ray (chi = nuHat/muRef, the
    fig:dragcrisisfields convention); BOTH sides are probed, symmetry is
    verified (the steady solver holds CL ~ 1e-8), and every reported
    profile is the two-side MEAN;
  - theta_tr = smallest grid theta with maxchi >= 1 (solver front
    convention) and, as robustness companion, >= c_v1 = 7.1 (chi where
    f_v1 = 1/2, the fully-active level); also theta_tr restricted to the
    attached layer (s <= y_e, y_e = ray speed-max height within 0.05 D,
    the flank-audit edge convention);
  - kernel coordinates along each ray, REUSED from
    spheroid_flank_kernel_audit (canon __aiRateFromXYZ + saai_env
    constants): X = |u|, Y = |omega| d (solver vorticityMagnitude),
    Re_Omega = d^2 |omega| / nu with nu = muRef/rho, P = Shat*Ihat,
    onset Re_Omega_c = softmin_2(1851.2, 124.6 + 1.424/P^2), rate =
    0.19 clip(P, 0, 1) * gate.  Z two ways, both recorded: the
    solver-as-implemented Z_i = +1/2 d^2 (lap u).u_hat via the audit's
    chained-VTK-gradient convention ON THE SLICE (in-plane = full
    Laplacian here: the flow is quasi-2D, spanwise derivatives vanish),
    and the ray operator Z_ray = 1/2 d^2 (d2u/ds2).u_hat (radial rays ARE
    wall-normal on a cylinder; the audit's kzray variant).  Per theta:
    max P (attached layer + unrestricted), max Re_Omega (both), max
    pointwise gate ratio Re_Omega/Re_Omega_c(P_i) and max rate in the
    attached layer;
  - separation angles are CROSS-REFERENCED from matrix_summary.jsonl
    (tangential-Cf sign crossings, dragcrisis_pilot_forces.crossings; the
    spurious endpoint crossing at theta = 180 is dropped): first
    (laminar) separation, first reattachment, final separation --
    two-side mean; near-wall chi=1 front (band d < 2e-3 D) likewise
    cross-referenced, min of the two sides, NOT recomputed.

Outputs (regenerable from /local_data + this script):
  figs_explore/data/dragcrisis_theta_tr.json   per-case record: theta_tr at
      both thresholds (+ attached-layer variant), per-side values +
      symmetry metric, ray reach, separation angles, compressed theta
      profiles (log10 maxchi, log10 maxReOm, maxP both operators,
      gate ratio, rate, y_e), near-wall front cross-reference.
  figs_explore/dragcrisis_theta_tr.png/.pdf    theta_tr AND separation
      angles vs Re_D, log-x: thick = radial-ray transition front, thin
      dash-dot = first separation ('v') and final separation ('D'),
      loose small triangles = the campaign's near-wall chi=1 front,
      dotted = chi >= c_v1 companion; Tu 0.05/0.2/0.7 = the paper's
      CVD-validated trio (fig:dragcrisiscd colors), up solid/filled,
      dn dashed/open; gray band = the crisis band (steepest Cd drop,
      Re 3e5-7e5 across the seeds).  Cases whose maxchi never reaches 1
      are omitted from curves ("no transition") but keep their largest
      maxchi in the JSON.
  figs_explore/dragcrisis_thetatr_fpg.png      second figure: theta
      profiles of max P and the gate ratio for the Tu 0.2% up ladder
      2e6 / 7e6 / 2e7 (single-hue Re ramp) -- does the ONSET GATE or the
      RATE keep the front out of the nose region at transcritical Re?

Pure CPU post-processing; slices are read one at a time and released
(129 slices, 11-39 MB each).

Run from anywhere:
  python3 repro/cfd/dragcrisis_transition_angle.py
    [--root /local_data/qiqi/sa-ai/dragcrisis_matrix] [--plot-only]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# canon kernel tail + onset gate (verbatim solver conventions; do not fork)
from spheroid_flank_kernel_audit import kernel_from_xyz, onset_gate  # noqa: E402,F401

PREV = os.path.join(HERE, "figs_explore")
DATA = os.path.join(PREV, "data")
JOUT = os.path.join(DATA, "dragcrisis_theta_tr.json")

CENTER = (0.5, 0.0)
R_WALL = 0.5
DTHETA = 0.5                      # deg
THETA = np.arange(0.0, 180.0 + 1e-9, DTHETA)
N_S = 480
S_MIN = 2e-7                      # below the smallest first-cell height
YBAND = 0.05                      # edge-search band (flank-audit convention)
C_V1 = 7.1
# paper trio (fig:dragcrisiscd, CVD-validated; fixed identity order)
TU_COLOR = {"0.05": "#2f6fa8", "0.2": "#c95f2b", "0.7": "#6d55a3"}
TU_ORDER = ("0.05", "0.2", "0.7")
CRISIS_BAND = (3e5, 7e5)          # steepest Cd drop across the three seeds
FPG_LADDER = ["cyl_Re2000000_Tu0.2_up", "cyl_Re7000000_Tu0.2_up_highre",
              "cyl_Re20000000_Tu0.2_up_highre"]


def slice_with_lap(case_dir):
    """Slice grid + chained-VTK-gradient Laplacian of velocity
    (spheroid_flank_kernel_audit.load_case_with_derived convention; the
    slice is quasi-2D so in-plane == full Laplacian)."""
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    rd = vtk.vtkXMLPUnstructuredGridReader()
    rd.SetFileName(os.path.join(case_dir, "slice_centerSpan.pvtu"))
    rd.Update()
    gf1 = vtk.vtkGradientFilter()
    gf1.SetInputData(rd.GetOutput())
    gf1.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "velocity")
    gf1.SetResultArrayName("grad_u")
    gf1.Update()
    gf2 = vtk.vtkGradientFilter()
    gf2.SetInputData(gf1.GetOutput())
    gf2.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "grad_u")
    gf2.SetResultArrayName("grad2_u")
    gf2.Update()
    g = gf2.GetOutput()
    pd = g.GetPointData()
    g2u = vtk_to_numpy(pd.GetArray("grad2_u"))  # [(j*3+i)*3+k]=d2u_j/dxi dxk
    lap = np.stack([sum(g2u[:, (j * 3 + i) * 3 + i] for i in range(3))
                    for j in range(3)], axis=1)
    va = numpy_to_vtk(np.ascontiguousarray(lap.astype(np.float32)), deep=True)
    va.SetName("lap_u")
    pd.AddArray(va)
    pd.RemoveArray("grad2_u")
    pd.RemoveArray("grad_u")
    return g


def ray_profiles(case_dir):
    """Two-side-mean theta profiles of ray maxima (chi, kernel coords)."""
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    g = slice_with_lap(case_dir)
    P = vtk_to_numpy(g.GetPoints().GetData())
    y0 = P[0, 1]
    rmax = float(np.hypot(P[:, 0] - CENTER[0], P[:, 2]).max())
    reach = rmax - R_WALL - 1e-4 * rmax        # stay inside the outer ring
    s = np.geomspace(S_MIN, reach, N_S)
    th = np.radians(THETA)
    # theta = 0 at the forward stagnation point (upstream, x = 0)
    dx, dz = -np.cos(th), np.sin(th)
    r = R_WALL + s
    pts = np.empty((2, THETA.size, N_S, 3))
    for k, sgn in enumerate((+1.0, -1.0)):     # upper (z>0), lower
        pts[k, :, :, 0] = CENTER[0] + np.outer(dx, r)
        pts[k, :, :, 1] = y0
        pts[k, :, :, 2] = sgn * np.outer(dz, r)
    vp = vtk.vtkPoints()
    vp.SetData(numpy_to_vtk(pts.reshape(-1, 3), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vp)
    pr = vtk.vtkProbeFilter()
    pr.SetInputData(poly)
    pr.SetSourceData(g)
    pr.Update()
    od = pr.GetOutput().GetPointData()
    sh = pts.shape[:3]                         # (2, ntheta, ns)

    def arr(name):
        return vtk_to_numpy(od.GetArray(name)).reshape(sh + (-1,)).squeeze()

    valid = np.zeros(np.prod(sh), bool)
    valid[vtk_to_numpy(pr.GetValidPoints())] = True
    valid = valid.reshape(sh)
    mu = json.load(open(os.path.join(case_dir, "Flow360.json"))
                   )["freestream"]["muRef"]
    u = arr("velocity")
    om = arr("vorticityMagnitude")
    rho = arr("rho")
    lapu = arr("lap_u")
    chi = np.where(valid, arr("nuHat") / mu, -np.inf)
    U = np.linalg.norm(u, axis=-1)
    uh = u / (U[..., None] + 1e-30)
    nu = mu / np.maximum(rho, 1e-6)
    d = s[None, None, :]
    reom = np.where(valid, d * d * om / nu, -np.inf)
    Zi = 0.5 * d * d * np.einsum("ktsj,ktsj->kts", lapu, uh)
    d2u = np.gradient(np.gradient(u, s, axis=2), s, axis=2)
    Zr = 0.5 * d * d * np.einsum("ktsj,ktsj->kts", d2u, uh)
    ki = kernel_from_xyz(U, om * d, Zi, np.maximum(reom, 0.0))
    kr = kernel_from_xyz(U, om * d, Zr, np.maximum(reom, 0.0))
    Pi = np.where(valid, ki["P"], -np.inf)
    Pr = np.where(valid, kr["P"], -np.inf)
    gate_ratio = np.where(valid, np.maximum(reom, 0.0) / ki["re_c"], -np.inf)
    rate = np.where(valid, ki["rate"], -np.inf)
    # attached-layer edge: ray speed-max height within YBAND (audit conv.)
    band = s <= YBAND
    i_e = np.argmax(np.where(valid & band[None, None, :], U, -np.inf), axis=2)
    y_e = s[i_e]                               # (2, ntheta)
    bl = s[None, None, :] <= y_e[:, :, None]

    def mx(f, m=None):
        v = f if m is None else np.where(m, f, -np.inf)
        return 0.5 * (v[0].max(axis=1) + v[1].max(axis=1))   # two-side mean

    imax = chi.argmax(axis=2)
    prof = {
        "maxchi": mx(chi), "maxchi_bl": mx(chi, bl),
        "maxP_i_bl": mx(Pi, bl), "maxP_i": mx(Pi),
        "maxP_ray_bl": mx(Pr, bl),
        "maxReOm_bl": mx(reom, bl), "maxReOm": mx(reom),
        "gate_ratio_bl": mx(gate_ratio, bl), "rate_bl": mx(rate, bl),
        "y_e": 0.5 * (y_e[0] + y_e[1]),
        "s_at_maxchi": 0.5 * (s[imax[0]] + s[imax[1]]),
    }
    sides_maxchi = np.stack([chi[0].max(axis=1), chi[1].max(axis=1)])
    return prof, sides_maxchi, reach, float(valid.mean())


def first_cross(prof, level):
    """Smallest grid theta with prof >= level (None if never)."""
    m = prof >= level
    return float(THETA[np.argmax(m)]) if m.any() else None


def separations(row):
    """(first separation, first reattachment, final separation), two-side
    mean, from the campaign's tangential-Cf crossings; the spurious
    endpoint crossing at theta=180 is dropped."""
    out = {"sep_first": [], "sep_reattach": [], "sep_final": []}
    for side in ("upper", "lower"):
        cr = [(a, k) for a, k in row.get(f"crossings_{side}", [])
              if a < 179.5]
        seps = [a for a, k in cr if k == "separation"]
        if seps:
            out["sep_first"].append(seps[0])
            out["sep_final"].append(seps[-1])
            rea = [a for a, k in cr if k == "reattachment" and a > seps[0]]
            if rea:
                out["sep_reattach"].append(rea[0])
    return {k: (round(float(np.mean(v)), 2) if v else None)
            for k, v in out.items()}


def compress(v, dec=3, log=False):
    a = np.asarray(v, float)
    if log:
        a = np.log10(np.clip(a, 1e-30, None))
    return [float(round(float(x), dec)) for x in a]


def compute(root):
    meta_rows = {}
    with open(os.path.join(root, "matrix_summary.jsonl")) as f:
        for ln in f:
            meta_rows[json.loads(ln)["case"]] = json.loads(ln)

    cases = sorted(d for d in os.listdir(root) if d.startswith("cyl_") and
                   os.path.exists(os.path.join(root, d,
                                               "slice_centerSpan.pvtu")))
    out = {}
    for i, case in enumerate(cases):
        prof, sides, reach, fv = ray_profiles(os.path.join(root, case))
        row = meta_rows.get(case, {})
        nw = [row.get(k) for k in ("chi1_front_upper", "chi1_front_lower")]
        nw = [v for v in nw if v is not None]
        t1 = first_cross(prof["maxchi"], 1.0)
        imax = int(np.argmax(prof["maxchi"]))
        ent = {"re": row.get("re"), "Tu": row.get("Tu"),
               "dir": row.get("dir"), "mesh": row.get("mesh", "pilot"),
               "nearwall_chi1_front": min(nw) if nw else None,
               "phi_Cp_min": row.get("phi_Cp_shoulder_upper"),
               "ray_reach_D": round(reach, 3), "n_s": N_S,
               "frac_valid_samples": round(fv, 4),
               "theta_tr_chi1": t1,
               "theta_tr_cv1": first_cross(prof["maxchi"], C_V1),
               "theta_tr_chi1_bl": first_cross(prof["maxchi_bl"], 1.0),
               "theta_tr_chi1_upper": first_cross(sides[0], 1.0),
               "theta_tr_chi1_lower": first_cross(sides[1], 1.0),
               "side_asym_max_dlog10chi": round(float(np.max(np.abs(
                   np.log10(np.clip(sides[0], 1e-30, None)) -
                   np.log10(np.clip(sides[1], 1e-30, None))))), 4),
               "maxchi_global": float(f"{prof['maxchi'][imax]:.4g}"),
               "theta_at_maxchi": float(THETA[imax]),
               "s_at_maxchi_D": float(f"{prof['s_at_maxchi'][imax]:.3g}"),
               "s_at_theta_tr_D": (
                   float(f"{prof['s_at_maxchi'][int(round(t1/DTHETA))]:.3g}")
                   if t1 is not None else None),
               "profiles": {
                   "log10_maxchi": compress(prof["maxchi"], log=True),
                   "log10_maxchi_bl": compress(prof["maxchi_bl"], log=True),
                   "log10_maxReOm_bl": compress(prof["maxReOm_bl"], log=True),
                   "log10_maxReOm": compress(prof["maxReOm"], log=True),
                   "maxP_i_bl": compress(prof["maxP_i_bl"], 4),
                   "maxP_i": compress(prof["maxP_i"], 4),
                   "maxP_ray_bl": compress(prof["maxP_ray_bl"], 4),
                   "log10_gate_ratio_bl": compress(prof["gate_ratio_bl"],
                                                   log=True),
                   "rate_bl": compress(prof["rate_bl"], 5),
                   "log10_y_e": compress(prof["y_e"], log=True)},
               }
        ent.update(separations(row))
        out[case] = ent
        print(f"[{i + 1:3d}/{len(cases)}] {case:40s} reach={reach:7.1f}D "
              f"maxchi={ent['maxchi_global']:9.3g} th1={t1} "
              f"thv1={ent['theta_tr_cv1']} nw={ent['nearwall_chi1_front']} "
              f"sep={ent['sep_first']}/{ent['sep_final']}", flush=True)
    return out


def plot_main(rec):
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.axvspan(*CRISIS_BAND, color="0.92", zorder=0)
    for tu in TU_ORDER:
        c = TU_COLOR[tu]
        for d, ls, mk, mfc in (("up", "-", "o", c), ("dn", "--", "s", "none")):
            pts = sorted(((v["re"], v) for v in rec.values()
                          if v.get("Tu") == tu and v.get("dir") == d and
                          v["theta_tr_chi1"] is not None),
                         key=lambda p: p[0])
            if not pts:
                continue
            re_ = np.array([p[0] for p in pts])
            g = lambda k: np.array([p[1][k] if p[1][k] is not None  # noqa: E731
                                    else np.nan for p in pts])
            # seam overlaps (same Re+Tu+dir on two meshes) plot as
            # overlapping markers, as in fig:dragcrisiscd full-span
            ax.plot(re_, g("theta_tr_chi1"), ls=ls, marker=mk, color=c,
                    mfc=mfc, mew=1.3, ms=5.5, lw=1.6, zorder=5)
            ax.plot(re_, g("theta_tr_cv1"), ls=":", color=c, lw=1.0,
                    alpha=0.8, zorder=3)
            ax.plot(re_, g("nearwall_chi1_front"), ls="none", marker="^",
                    color=c, mfc="none", mew=0.9, ms=3.8, alpha=0.55,
                    zorder=2)
            mfc2 = c if d == "up" else "none"
            ax.plot(re_, g("sep_first"), ls="-.", marker="v", color=c,
                    mfc=mfc2, mew=0.8, ms=3.2, lw=0.9, alpha=0.75, zorder=4)
            ax.plot(re_, g("sep_final"), ls="-.", marker="D", color=c,
                    mfc=mfc2, mew=0.8, ms=3.0, lw=0.9, alpha=0.75, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(8e3, 3e7)
    ax.set_ylim(40, 180)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.xaxis.set_major_locator(mticker.LogLocator(numticks=12))
    ax.xaxis.set_minor_locator(mticker.LogLocator(subs=(2, 3, 5),
                                                  numticks=12))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(r"$Re_D$")
    ax.set_ylabel(r"angle from forward stagnation [deg]")
    ax.grid(alpha=0.3, which="major")
    for tu, y in (("0.05", 88.5), ("0.2", 81.5), ("0.7", 74.5)):
        ax.text(2.15e7, y, rf"$Tu\,{tu}\%$", color=TU_COLOR[tu],
                fontsize=10, ha="left", va="center")
    hs = [Line2D([], [], color="0.25", ls="-", marker="o", ms=5.5,
                 label=r"$\theta_{tr}$: radial-ray $\max\chi\geq 1$, up"),
          Line2D([], [], color="0.25", ls="--", marker="s", mfc="none",
                 mew=1.3, ms=5.5,
                 label=r"$\theta_{tr}$: radial-ray $\max\chi\geq 1$, dn"),
          Line2D([], [], color="0.25", ls=":", lw=1.0,
                 label=rf"radial-ray $\max\chi\geq c_{{v1}}={C_V1}$"),
          Line2D([], [], color="0.25", ls="none", marker="^", mfc="none",
                 mew=0.9, ms=3.8, alpha=0.7,
                 label=r"near-wall $\chi=1$ front"),
          Line2D([], [], color="0.25", ls="-.", marker="v", ms=3.2, lw=0.9,
                 label="first (laminar) separation"),
          Line2D([], [], color="0.25", ls="-.", marker="D", ms=3.0, lw=0.9,
                 label="final separation"),
          ]
    ax.legend(handles=hs, fontsize=8.5, frameon=False, loc="upper right",
              handlelength=2.6, labelspacing=0.3, borderaxespad=0.3)
    fig.tight_layout()
    os.makedirs(PREV, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(PREV, f"dragcrisis_theta_tr.{ext}"),
                    dpi=150 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote {PREV}/dragcrisis_theta_tr.png/.pdf")


def plot_fpg(rec):
    """Tu 0.2% up ladder: is the nose blocked by the RATE (P) or the
    ONSET GATE (Re_Omega vs Re_Omega_c)?  Single-hue Re ramp."""
    shades = ["#e8a87b", "#c95f2b", "#7a3413"]        # light -> dark
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True,
                            gridspec_kw=dict(hspace=0.08, top=0.97,
                                             bottom=0.09, left=0.15,
                                             right=0.97))
    for case, col in zip(FPG_LADDER, shades):
        v = rec.get(case)
        if v is None:
            continue
        lab = rf"$Re_D={v['re']:.0e}$".replace("e+0", r"\times 10^")
        p = v["profiles"]
        # ray operator solid (smooth; radial rays are exactly wall-normal);
        # the chained-VTK-Laplacian operator thin (noise-dominated in the
        # thin FPG boundary layer at these Re -- see the record)
        axs[0].semilogy(THETA, np.clip(p["maxP_ray_bl"], 1e-6, None), "-",
                        color=col, lw=1.5, label=lab)
        axs[0].semilogy(THETA, np.clip(p["maxP_i_bl"], 1e-6, None), "--",
                        color=col, lw=0.7, alpha=0.55)
        axs[1].semilogy(THETA, 10.0 ** np.asarray(p["log10_gate_ratio_bl"]),
                        "-", color=col, lw=1.5)
        for a in axs:
            if v.get("sep_first"):
                a.axvline(v["sep_first"], color=col, lw=0.8, ls=":",
                          alpha=0.8)
            if v.get("theta_tr_chi1"):
                a.plot([v["theta_tr_chi1"]], [0.03], marker="^", color=col,
                       ms=6, transform=a.get_xaxis_transform(), zorder=6)
    if rec.get(FPG_LADDER[0]):
        for a in axs:
            a.axvspan(0, rec[FPG_LADDER[0]]["phi_Cp_min"] or 65.0,
                      color="0.94", zorder=0)
    axs[0].axhline(1.0, color="0.4", lw=0.8)
    axs[0].set_ylim(1e-4, 2.0)
    axs[0].set_ylabel(r"$\max_{s\leq y_e} P=\hat\Omega\hat I$")
    axs[0].legend(fontsize=9, frameon=False, loc="upper left")
    axs[1].axhline(1.0, color="0.4", lw=0.8)
    axs[1].set_ylim(1e-3, 30)
    axs[1].set_ylabel(r"$\max_{s\leq y_e}\;Re_\Omega/Re_\Omega^c(P)$")
    axs[1].set_xlabel(r"$\theta$ from forward stagnation [deg]")
    axs[1].set_xlim(0, 130)
    for a in axs:
        a.grid(alpha=0.3)
    fig.savefig(os.path.join(PREV, "dragcrisis_thetatr_fpg.png"), dpi=150)
    plt.close(fig)
    print(f"wrote {PREV}/dragcrisis_thetatr_fpg.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/local_data/qiqi/sa-ai/dragcrisis_matrix")
    ap.add_argument("--plot-only", action="store_true",
                    help="re-plot from the existing JSON")
    args = ap.parse_args()
    if args.plot_only:
        rec = json.load(open(JOUT))["cases"]
    else:
        rec = compute(args.root)
        os.makedirs(DATA, exist_ok=True)
        meta = {"theta_deg": "0..180 step 0.5 from forward stagnation",
                "ray": f"{N_S} samples log-spaced in wall distance, "
                       f"{S_MIN} D to the slice edge (per-case reach_D)",
                "profiles": "two-side MEAN of per-ray maxima (symmetry "
                            "verified per case); _bl = restricted to the "
                            "attached layer s <= y_e (ray speed max within "
                            f"{YBAND} D)",
                "chi": "nuHat/muRef (fig:dragcrisisfields convention)",
                "kernel": "canon __aiRateFromXYZ via "
                          "spheroid_flank_kernel_audit.kernel_from_xyz; "
                          "Re_Omega = d^2 |omega|/nu, nu = muRef/rho; "
                          "P_i: Z from chained-VTK-gradient Laplacian on "
                          "the slice; P_ray: Z from radial d2u/ds2; "
                          "Re_Omega_c = softmin2(1851.2, 124.6+1.424/P^2), "
                          "rate = 0.19 clip(P) gate(ramp 0.35)",
                "thresholds": {"chi1": 1.0, "cv1": C_V1},
                "separations": "cross-referenced matrix_summary.jsonl Cf "
                               "crossings (endpoint theta=180 dropped), "
                               "two-side mean",
                "nearwall_chi1_front": "cross-reference from "
                                       "matrix_summary.jsonl (band d<2e-3 D,"
                                       " min of upper/lower)",
                "script": "repro/cfd/dragcrisis_transition_angle.py"}
        json.dump({"meta": meta, "cases": rec}, open(JOUT, "w"),
                  default=float)
        print(f"wrote {JOUT}")
    plot_main(rec)
    plot_fpg(rec)


if __name__ == "__main__":
    main()
