"""FULL-BODY spheroid discriminator harvest (alpha = 0, Re_L = 7.2e6, L1).

The half-model alpha=0 anomaly (agent-paper-review/2026-07-28-0105): RANS
laminar BL anomalously full (H 2.49-2.44 vs marched 2.56-2.63), carried by a
structured near-wall numerical momentum source, with a cell-locked
ONE-AZIMUTHAL-CELL staggering mode (du/u p2p 2.6% at 0.15 delta99 at L2, rms
2.8e-2 at L1) on an azimuthally uniform flow.  The half-model O-grid has
symmetry sheets at y = 0; this harvest reads the FULL-CIRCUMFERENCE rerun
(spheroid/ogrid_spheroid_full.py + build_fullbody_case.py, arms A/B =
lowMachPreconditioner false/true) with the SAME instruments as the 0105
record and asks:
  1. H at x/L = 0.20/0.42/0.70 (identical edge/integral operator,
     spheroid_a0_physics.edge_and_integrals on the identical RAY grid) --
     does the fullness survive without symmetry sheets?  Does the low-Mach
     preconditioner move it?
  2. near-wall chi = 1 (and chi = c_v1) front along the meridian;
  3. the azimuthal staggering spectrum at fixed heights (0.15/0.3/0.6/1.2
     delta99) -- does the one-cell azimuthal mode survive?

Same-instrument control: every probe is ALSO run on the committed half-model
case_ogrid_L1_saai_re72a0 volume, so half-vs-full numbers share one
extraction (the full-body cases write SLICES, not volumes -- disk
discipline; the meridian probe rays and azimuthal rings are constructed
in-plane so slice probing is exact).

Ring convention: points in the constant-x slice plane at radial offset
h / n_r above the analytic surface (wall-normal height h to 0.04% at 0.42;
the analytic-surface convention of the 0105 record's probes).  Residual =
signal minus 25-sample running mean (samples at cell/12, window ~ 2 cells --
the record's wiggle_stats convention); spectra additionally from the
periodic full-circle FFT (full body) and the open-arc Hann FFT (both).

Outputs (exploratory, NOT paper figures):
  paper/repro/cfd/figs_explore/spheroid_fullbody_rings.png
  paper/repro/cfd/figs_explore/spheroid_fullbody_H.png
  paper/repro/cfd/figs_explore/spheroid_fullbody.json

Run:  python3 -u paper/repro/cfd/spheroid_fullbody_check.py [--arms A B half]
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
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figs_explore")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "spheroid"))

from spheroid_a0_physics import (                     # noqa: E402
    edge_and_integrals, front_crossing, RAY)
from spheroid_unstruct_a0_verdict import (            # noqa: E402
    load_wall_facets, wall_offsets, forces_tail)
from surface_map import surface_frame, A, B           # noqa: E402

SPH_ROOT = os.environ.get("SAAI_SPH_ROOT", "/local_data/qiqi/sa-ai/spheroid_fv1")
MESH_HALF = "/local_data/qiqi/sa-ai/spheroid_meshes/mesh_re65_L1.cgns"
MESH_FULL = "/local_data/qiqi/sa-ai/spheroid_meshes/mesh_re65full_L1.cgns"
CASES = {
    "half": dict(dir=os.path.join(SPH_ROOT, "case_ogrid_L1_saai_re72a0"),
                 src="volume", n_az=80, full=False, mesh=MESH_HALF,
                 label="half-model L1 20k (symmetry sheets)"),
    "ext": dict(dir=os.path.join(SPH_ROOT, "case_ogrid_L1_saai_re72a0_ext"),
                src="volume", n_az=80, full=False, mesh=MESH_HALF,
                label="half-model L1 extended (og-ext control)"),
    "A": dict(dir=os.path.join(SPH_ROOT, "case_ogridfull_L1_saai_re72a0"),
              src="slice", n_az=160, full=True, mesh=MESH_FULL,
              label="full body L1, lowMach OFF (arm A)"),
    "B": dict(dir=os.path.join(SPH_ROOT,
                               "case_ogridfull_L1_saai_re72a0_lowmach"),
              src="slice", n_az=160, full=True, mesh=MESH_FULL,
              label="full body L1, lowMach ON (arm B)"),
}
# converged references (agent-paper-review/2026-07-28-0350 Sec 6 + 0105):
REF_UNSTR43K = dict(H={"0.2": 2.515, "0.42": 2.556, "0.7": 2.611},
                    front_chi1=0.8584)
REF_MARCH = dict(H={"0.2": 2.533, "0.42": 2.561, "0.7": 2.613})
XS = np.arange(0.02, 0.9651, 0.004)     # meridian sweep (physics-pass grid)
STATIONS = (0.20, 0.42, 0.70)
FR = (0.15, 0.30, 0.60, 1.2)            # ring heights / delta99
SAMPLES_PER_CELL = 12                   # ring azimuthal sampling density
CHI_BAND = 0.02                         # near-wall band for the chi front

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 14, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.fontsize": 10.5, "figure.dpi": 110, "savefig.dpi": 150})
COL = {"half": "#3b6bb5", "ext": "#7a5aa8", "A": "#b0483a", "B": "#3e8f5c"}


def load_grid(case_dir, src, slice_name=None):
    fn = ("volume.pvtu" if src == "volume"
          else f"slice_{slice_name}.pvtu")
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(os.path.join(case_dir, fn))
    r.Update()
    g = r.GetOutput()
    if g.GetNumberOfPoints() == 0:
        raise RuntimeError(f"empty grid {case_dir}/{fn}")
    return g


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
    res = {out.GetArray(i).GetName(): vtk_to_numpy(out.GetArray(i))
           for i in range(out.GetNumberOfArrays())}
    valid = vtk_to_numpy(out.GetArray(pr.GetValidPointMaskArrayName())
                         ).astype(bool)
    return res, valid


def nuhat_of(res):
    return res["solutionTurbulence" if "solutionTurbulence" in res
               else "nuHat"].astype(float)


# ------------------------------------------------------- meridian sweep
def meridian_sweep(grid, mu_ref, xs=XS, facets=None):
    """Rays at phi_lit = 90 deg (mesh z = 0, y > 0 -- in the meridian slice
    plane), identical RAY grid and edge/integral operator as the record.
    facets: (wall_pts, wall_tris) -> ray origins re-based on the discrete
    wall (Moller-Trumbore, the 0350 record's sag correction; on the O-grid
    node meridian the offsets are meridional-chord only, <= ~1.2e-6 L)."""
    P, n3, t_s, _ = surface_frame(xs, np.full_like(xs, np.radians(90.0)))
    t0 = np.zeros(len(xs))
    if facets is not None:
        t0 = wall_offsets(P, n3, *facets)
    P = P + t0[:, None] * n3
    pts = np.concatenate([P[k] + RAY[:, None] * n3[k] for k in range(len(xs))])
    res, valid = probe_pts(grid, pts)
    ny = len(RAY)
    u_all = res["velocity"].astype(float)
    rho_all = res.get("rho", np.ones(len(pts))).astype(float)
    nuhat_all = nuhat_of(res)
    sw = []
    for k, xq in enumerate(xs):
        sl = slice(k * ny, (k + 1) * ny)
        u = u_all[sl]
        st = dict(y=RAY, U=np.linalg.norm(u, axis=1), us=u @ t_s[k],
                  nu=mu_ref / np.maximum(rho_all[sl], 1e-6),
                  chi=rho_all[sl] * nuhat_all[sl] / mu_ref,
                  valid=valid[sl], x=float(xq), t0=float(t0[k]))
        edge_and_integrals(st)
        st["chimax_nearwall"] = float(np.nanmax(
            np.where(RAY <= CHI_BAND, st["chi"], -np.inf)))
        sw.append(st)
    return sw


def sweep_summary(sw):
    x = np.array([s["x"] for s in sw])
    chim = np.array([s["chimax_nearwall"] for s in sw])
    rows = {}
    for xq in STATIONS:
        s = sw[int(np.argmin(np.abs(x - xq)))]
        rows[f"{xq:g}"] = dict(
            H=float(s["H"]), theta=float(s["theta"]), d99=float(s["d99"]),
            u_e=float(s["u_e"]),
            Rt=float(s["u_e"] * s["theta"] / np.median(s["nu"])),
            edge_ok=bool(s["edge_ok"]))
    t0s = np.array([s.get("t0", 0.0) for s in sw])
    return dict(
        stations=rows,
        front_chi1=front_crossing(x, chim, 1.0),
        front_cv1=front_crossing(x, chim, 7.1),
        valid_frac=float(np.mean([s["valid"].mean() for s in sw])),
        t0_absmax=float(np.abs(t0s).max()))


# ------------------------------------------------------- azimuthal rings
def ring_points(xq, phi_lit, h, t0=None):
    """Points in the constant-x plane at wall-normal height h above the
    wall: radial offset (t0 + h) / n_r from the analytic surface (n_r =
    radial component of the outward normal), so slice probing is exact.
    t0 (per-phi, <= 0): facet-sag offset -- heights are then measured from
    the DISCRETE wall, removing the one-cell geometric height modulation
    (sag ~1.6e-5 L = up to 12%% of 0.15 d99 at L1).  Returns pts and t_s."""
    x = -A + xq                              # mesh x (body centered at 0)
    r0 = B * np.sqrt(max(1.0 - (x / A)**2, 0.0))
    nvec = np.array([x / A**2, r0 / B**2])
    n_r = nvec[1] / np.hypot(*nvec)
    if t0 is None:
        t0 = np.zeros_like(phi_lit)
    R = r0 + (t0 + h) / n_r
    _, _, t_s, _ = surface_frame(np.full_like(phi_lit, xq), phi_lit)
    phi_mesh = np.pi - phi_lit
    pts = np.stack([np.full_like(phi_lit, x), R * np.sin(phi_mesh),
                    R * np.cos(phi_mesh)], axis=1)
    return pts, t_s


def ring_probe(grid, xq, d99, n_az, full, facets=None):
    """u_s around the azimuth at FR x d99; full: phi_lit in [0, 2pi) closed;
    half: open arc, 5..175 deg (away from the symmetry sheets).  facets:
    per-phi Moller-Trumbore wall re-basing (see ring_points)."""
    cell = (2.0 * np.pi if full else np.pi) / n_az
    dphi = cell / SAMPLES_PER_CELL
    if full:
        phi = np.arange(0.0, 2.0 * np.pi, dphi)
    else:
        phi = np.arange(np.radians(5.0), np.radians(175.0), dphi)
    t0 = None
    if facets is not None:
        P0, n3, _, _ = surface_frame(np.full_like(phi, xq), phi)
        t0 = wall_offsets(P0, n3, *facets)
    out = dict(phi=phi, cell_deg=float(np.degrees(cell)), fr=FR, us={},
               t0_absmax=float(np.abs(t0).max()) if t0 is not None else 0.0)
    for f in FR:
        pts, t_s = ring_points(xq, phi, f * d99, t0)
        res, valid = probe_pts(grid, pts)
        u = res["velocity"].astype(float)
        us = np.einsum("ij,ij->i", u, t_s)
        n_bad = int((~valid).sum())
        if n_bad:
            # a handful of points can land on degenerate slice-triangulation
            # edges; fill by interpolation over phi (periodic for full rings)
            if n_bad <= max(8, len(us) // 100):
                ph = np.degrees(phi)
                if full:
                    us = np.interp(ph, ph[valid], us[valid],
                                   period=360.0)
                else:
                    us = np.interp(ph, ph[valid], us[valid])
            else:
                us = np.where(valid, us, np.nan)
        out["us"][f] = us
        out.setdefault("n_invalid", {})[f] = n_bad
    return out


def ring_stats(out, full):
    """The record's wiggle_stats convention (running-mean detrend, Hann FFT
    on the residual) + periodic full-circle mode spectrum for full rings."""
    rows = {}
    phi = out["phi"]
    dph = float(phi[1] - phi[0])
    for f in FR:
        v = out["us"][f]
        if np.isnan(v).any():
            rows[f"{f:g}"] = dict(error="invalid probe points",
                                  invalid=int(np.isnan(v).sum()))
            continue
        sm = np.convolve(v, np.ones(25) / 25, mode="same")
        r = (v - sm)[12:-12]
        F = np.fft.rfft(r * np.hanning(len(r)))
        freq = np.fft.rfftfreq(len(r), d=dph)      # cycles per radian
        j = int(np.argmax(np.abs(F[3:]))) + 3
        row = dict(rms_du_over_u=float(np.std(r) / np.mean(v)),
                   p2p_du_over_u=float((r.max() - r.min()) / np.mean(v)),
                   dominant_wavelength_deg=float(np.degrees(1.0 / freq[j])))
        if full:                                   # clean periodic spectrum
            Fp = np.fft.rfft(v - v.mean()) / len(v)
            m = np.arange(len(Fp))                 # azimuthal mode number
            amp = 2.0 * np.abs(Fp) / np.mean(v)
            n_cell = int(round(2.0 * np.pi / (dph * SAMPLES_PER_CELL)))
            jm = int(np.argmax(amp[1:])) + 1
            row.update(
                mode_ncell_amp=float(amp[n_cell]),
                mode_ncell=n_cell,
                strongest_mode=int(m[jm]),
                strongest_mode_amp=float(amp[jm]),
                low_mode_max_amp=float(amp[1:9].max()))
        rows[f"{f:g}"] = row
    return rows


# ------------------------------------------------------------------ figures
def fig_rings(rings, fp):
    labels = list(rings)
    fig, axs = plt.subplots(len(FR), 1, figsize=(11.0, 10.0), sharex=True)
    for ax, f in zip(axs, FR):
        for tag in labels:
            out = rings[tag]
            v = out["us"][f]
            if np.isnan(v).any():
                continue
            sm = np.convolve(v, np.ones(25) / 25, mode="same")
            r = (v - sm)[12:-12]
            ph = np.degrees(out["phi"][12:-12])
            ax.plot(ph, 1e3 * r / np.mean(v), color=COL[tag], lw=0.9,
                    alpha=0.9, label=CASES[tag]["label"] if f == FR[0]
                    else None)
        ax.set_ylabel(r"$10^3\,\delta u_s/\bar u_s$")
        ax.text(0.008, 0.82, f"$y/\\delta_{{99}} = {f:g}$",
                transform=ax.transAxes, fontsize=11)
    axs[0].legend(loc="upper right", ncol=3, fontsize=9)
    axs[-1].set_xlabel(r"azimuth $\varphi$ [deg] at $x/L = 0.42$")
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def fig_H(sweeps, fp):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for tag, sw in sweeps.items():
        x = np.array([s["x"] for s in sw])
        H = np.array([s["H"] for s in sw])
        m = (x >= 0.06) & (x <= 0.93)
        ax.plot(x[m], H[m], color=COL[tag], lw=1.9,
                label=CASES[tag]["label"])
    ax.axhline(2.5905, color="0.45", lw=0.9, ls=":")
    ax.text(0.62, 2.595, "Blasius 2.5905", fontsize=10, color="0.45")
    # marched laminar reference at the stations (0105 record, march axi)
    ax.plot(STATIONS, (2.564, 2.581, 2.626), "k*", ms=11,
            label="laminar BL march on field $u_e$ (0105 record)")
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel(r"$H$")
    ax.set_ylim(2.30, 2.75)
    ax.legend(loc="lower left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["half", "A", "B"])
    ap.add_argument("--ring-x", type=float, default=0.42)
    args = ap.parse_args()
    os.makedirs(FIGD, exist_ok=True)

    out = {"ring_x": args.ring_x,
           "refs": dict(unstr43k=REF_UNSTR43K, march_op_matched=REF_MARCH)}
    sweeps, rings = {}, {}
    for tag in args.arms:
        c = CASES[tag]
        mu_ref = json.load(open(os.path.join(c["dir"], "Flow360.json"))
                           )["freestream"]["muRef"]
        print(f"== {tag}: {c['dir']}", flush=True)
        facets = load_wall_facets(c["mesh"])
        gm = load_grid(c["dir"], c["src"], slice_name="meridian")
        sw = meridian_sweep(gm, mu_ref, facets=facets)
        sweeps[tag] = sw
        summ = sweep_summary(sw)
        out[tag] = dict(label=c["label"], meridian=summ,
                        forces_tail=forces_tail(c["dir"]))
        hrow = "  ".join("H(%s)=%.4f" % (k, v["H"])
                         for k, v in summ["stations"].items())
        print(f"  {hrow}  front chi=1: {summ['front_chi1']:.4f}"
              f"  cv1: {summ['front_cv1']:.4f}"
              f"  valid {summ['valid_frac']:.3f}"
              f"  |t0|max {summ['t0_absmax']:.2e}", flush=True)
        ft = out[tag]["forces_tail"]
        print(f"  forces: step {ft['steps']}  CD {ft['CD']:.5f}  CL "
              f"{ft['CL']:.2e}  end-drift {ft['CD_drift_per_1k']:+.2e}/1k",
              flush=True)
        del gm
        # rings at ring-x on the constant-x source
        d99 = summ["stations"][f"{args.ring_x:g}"]["d99"]
        if c["src"] == "slice":
            gs = load_grid(c["dir"], "slice",
                           slice_name=f"x{int(round(100 * args.ring_x)):03d}")
        else:
            gs = load_grid(c["dir"], "volume")
        rg = ring_probe(gs, args.ring_x, d99, c["n_az"], c["full"],
                        facets=facets)
        rings[tag] = rg
        st = ring_stats(rg, c["full"])
        out[tag]["ring"] = dict(cell_deg=rg["cell_deg"], stats=st,
                                t0_absmax=rg["t0_absmax"],
                                n_invalid_filled=rg.get("n_invalid", {}))
        for f, row in st.items():
            print(f"  ring y/d99={f}: " + " ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()), flush=True)
        del gs

    fig_rings(rings, os.path.join(FIGD, "spheroid_fullbody_rings.png"))
    fig_H(sweeps, os.path.join(FIGD, "spheroid_fullbody_H.png"))
    fj = os.path.join(FIGD, "spheroid_fullbody.json")
    json.dump(out, open(fj, "w"), indent=1, default=float)
    print("wrote", fj, flush=True)


if __name__ == "__main__":
    main()
