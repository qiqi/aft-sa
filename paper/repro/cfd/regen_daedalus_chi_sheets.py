"""Daedalus wall-normal chi sheets -> figs/chi_sheet_daedalus_a{a}_eta{NN}_upper.pdf.

Same full-page layout as the Eppler chi sheets (regen_chi_sheets.py):
  left  = u_x / U_inf  (negative levels mark reverse flow)
  right = log10(chi)
Rows = meshes paired by level (cavity L0, O-grid L0, ...). L2 volume
dumps were purged from the fv1 tree; those rows are omitted when the
volume is absent (L0/L1 are present for a4/a5/a6).

One sheet per (alpha, eta) on the upper surface. Wall-normal range and
chi scaling use the local chord Reynolds number Re_c(eta).

  SAAI_DAEDALUS_ROOT=/local_data/qiqi/sa-ai/daedalus_fv1 \\
    python3 regen_daedalus_chi_sheets.py [alpha ...]
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PD, "repro"))
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/daedalus")
import regen_chi_sheets as CS  # noqa: E402
from wing_geometry import chord, HALF_SPAN, XQC, C_ROOT  # noqa: E402
import sectional_compare as SC  # noqa: E402

ROOT = os.environ.get("SAAI_DAEDALUS_ROOT", "/local_data/qiqi/sa-ai/daedalus_fv1")
FIGS = os.path.join(PD, "figs")
MACH = 0.1
RE_ROOT = 5.0e5
ALPHAS = (4, 5, 6)
ETAS = (0.10, 0.75, 0.92)
# probe depth in local chords -- match Eppler upper L_up * sqrt(Re_e / Re_loc)
# with Eppler L_up=0.0335 at Re=2e5 as the visual reference
L_REF, RE_REF = 0.0335, 2.0e5
N_ANCH, N_PROBE = 160, 280
ROWS = [
    ("cavity", "L0", "unstructured L0", "surface_farfield_body.pvtu"),
    ("ogrid", "L0", "O-grid L0", "surface_fluid_wing.pvtu"),
    ("cavity", "L1", "unstructured L1", "surface_farfield_body.pvtu"),
    ("ogrid", "L1", "O-grid L1", "surface_fluid_wing.pvtu"),
    ("cavity", "L2", "unstructured L2", "surface_farfield_body.pvtu"),
    ("ogrid", "L2", "O-grid L2", "surface_fluid_wing.pvtu"),
]


def load_pvtu(path):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def case_dir(fam, lvl, alpha):
    return f"{ROOT}/case_{fam}_{lvl}_saai_a{alpha}"


def section_contour(surf_pts, eta_q):
    """Ordered upper/lower (x,y,z) at the nearest span ring (signed y)."""
    y0 = eta_q * HALF_SPAN
    yy = surf_pts[:, 1]
    # prefer the positive-y half (one-sided ogrid; cavity is two-sided)
    if np.any(yy > 0.1 * HALF_SPAN):
        pool = surf_pts[yy > 0.0]
    else:
        pool = surf_pts
    y_star = pool[:, 1][np.argmin(np.abs(pool[:, 1] - y0))]
    m = np.abs(surf_pts[:, 1] - y_star) <= 1e-4
    p = surf_pts[m]
    c_loc = float(chord(eta_q))
    x_le = XQC - 0.25 * c_loc
    xc = (p[:, 0] - x_le) / c_loc
    zcam = np.interp(np.clip(xc, 0, 1), SC._CAM_X, SC._CAM_Z) * c_loc
    up = p[:, 2] >= zcam
    out = {}
    for side, sel in (("upper", up), ("lower", ~up)):
        q = p[sel]
        out[side] = q[np.argsort(q[:, 0])]
    return out, c_loc, x_le


def scan_section(vol, surf_pts, eta, L_probe_c, mu_ref, side="upper"):
    """Probe wall-normal sheet: returns xc[M], d[n], chi[n,M], ux[n,M]."""
    contours, c_loc, x_le = section_contour(surf_pts, eta)
    q = contours[side]
    xc_all = (q[:, 0] - x_le) / c_loc
    ok = np.where((xc_all > 0.01) & (xc_all < 0.985))[0]
    if len(ok) < 8:
        raise RuntimeError(f"too few section anchors at eta={eta}")
    idx = ok[np.linspace(0, len(ok) - 1, min(N_ANCH, len(ok))).astype(int)]
    anch = q[idx]
    # in-plane tangent / outward normal (y frozen)
    n = len(q)
    tan = q[np.minimum(idx + 2, n - 1)] - q[np.maximum(idx - 2, 0)]
    tan[:, 1] = 0.0
    tan /= (np.linalg.norm(tan, axis=1)[:, None] + 1e-30)
    if side == "upper":
        nrm = np.stack([-tan[:, 2], np.zeros(len(idx)), tan[:, 0]], axis=1)
        want = 1.0
    else:
        nrm = np.stack([tan[:, 2], np.zeros(len(idx)), -tan[:, 0]], axis=1)
        want = -1.0
    nrm[np.sign(nrm[:, 2]) != want] *= -1.0
    d = np.linspace(1e-6, L_probe_c, N_PROBE) * c_loc
    pts = (anch[:, None, :] + nrm[:, None, :] * d[None, :, None]).reshape(-1, 3)
    pd = vtk.vtkPolyData()
    vp = vtk.vtkPoints()
    vp.SetData(numpy_to_vtk(pts, deep=1))
    pd.SetPoints(vp)
    pf = vtk.vtkProbeFilter()
    pf.SetInputData(pd)
    pf.SetSourceData(vol)
    pf.Update()
    out = pf.GetOutput().GetPointData()
    vel = vtk_to_numpy(out.GetArray("velocity")).reshape(len(idx), N_PROBE, 3)
    # Daedalus fv1 volumes export SA working variable as solutionTurbulence;
    # Eppler airfoil dumps use nuHat. Accept either.
    turb = out.GetArray("nuHat") or out.GetArray("solutionTurbulence")
    if turb is None:
        names = [out.GetArrayName(i) for i in range(out.GetNumberOfArrays())]
        raise RuntimeError(f"no nuHat/solutionTurbulence in probe; have {names}")
    nh = vtk_to_numpy(turb).reshape(len(idx), N_PROBE)
    vmask = out.GetArray(pf.GetValidPointMaskArrayName())
    if vmask is not None:
        mask = vtk_to_numpy(vmask).astype(bool).reshape(len(idx), N_PROBE)
    else:
        valid = vtk_to_numpy(pf.GetValidPoints())
        mask = np.zeros(len(idx) * N_PROBE, bool)
        mask[valid] = True
        mask = mask.reshape(len(idx), N_PROBE)
    ux = np.where(mask, vel[:, :, 0] / MACH, np.nan)          # [M, n]
    chi = np.where(mask, nh / mu_ref, np.nan)
    xc = (anch[:, 0] - x_le) / c_loc
    o = np.argsort(xc)
    # return as [n, M] to match regen_chi_sheets.contour convention
    return xc[o], d, chi[o].T, ux[o].T, c_loc


def sheet(alpha, eta):
    Re_loc = RE_ROOT * float(chord(eta)) / C_ROOT
    L_c = L_REF * np.sqrt(RE_REF / Re_loc)   # local-chord probe depth
    rows = []
    for fam, lvl, glabel, sname in ROWS:
        cd = case_dir(fam, lvl, alpha)
        vpath = f"{cd}/volume.pvtu"
        spath = f"{cd}/{sname}"
        if os.path.exists(vpath) and os.path.exists(spath):
            rows.append((glabel, cd, vpath, spath))
    if not rows:
        raise FileNotFoundError(f"no volumes for a{alpha} eta={eta}")
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 2, figsize=(11.5, 2.1 * nrows + 0.6),
                             sharex=True, sharey=True)
    if nrows == 1:
        axes = np.asarray([axes])
    for r, (glabel, cd, vpath, spath) in enumerate(rows):
        mu = float(json.load(open(f"{cd}/Flow360.json"))["freestream"]["muRef"])
        vol = load_pvtu(vpath)
        surf = load_pvtu(spath)
        spts = vtk_to_numpy(surf.GetPoints().GetData())
        xc, d, chi, ux, c_loc = scan_section(vol, spts, eta, L_c, mu)
        # wall units: d U_inf / nu with U_inf=Mach, nu=muRef (solver nondim)
        d_wall = d * MACH / mu
        axU, axC = axes[r]
        cs = axU.contour(xc, d_wall, ux, levels=CS.UX_LEV, colors="k",
                         linewidths=0.6)
        axU.clabel(cs, levels=CS.UX_LABEL, fmt="%g", fontsize=6.5,
                   inline_spacing=2)
        logchi = np.log10(np.clip(chi, 1e-8, None))
        axC.contour(xc, d_wall, logchi, levels=CS.CHI_MINOR, colors="k",
                    linewidths=0.4)
        cs = axC.contour(xc, d_wall, logchi, levels=CS.CHI_MAJOR, colors="k",
                         linewidths=0.8)
        axC.clabel(cs, fmt=CS.CHI_FMT, fontsize=6.5, inline_spacing=2)
        axU.set_xlim(0, 1)
        axU.set_ylabel(f"{glabel}\n" + r"$d\,U_\infty/\nu$", fontsize=9)
        print(f"  a{alpha} eta={eta:.2f} {glabel}: scanned", flush=True)
    # Fixed wall-unit window so all (alpha, eta, mesh) sheets are comparable.
    for ax in axes.ravel():
        ax.set_ylim(0, 6000)
    axes[0, 0].set_title(r"$u_x/U_\infty$", fontsize=10)
    axes[0, 1].set_title(r"$\log_{10}\chi$", fontsize=10)
    for c in range(2):
        axes[-1, c].set_xlabel("wall-anchor x/c")
    eta_tag = f"eta{int(round(100 * eta)):02d}"
    out = os.path.join(FIGS, f"chi_sheet_daedalus_a{alpha}_{eta_tag}_upper.pdf")
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main(alphas=None):
    alphas = alphas or ALPHAS
    for a in alphas:
        for eta in ETAS:
            try:
                print(f"daedalus a{a} eta={eta:.2f}:", flush=True)
                sheet(a, eta)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"  SKIP: {e}", flush=True)


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else None
    main(args)
