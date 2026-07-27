"""Frozen-field sphere-kernel check on the drag-crisis PILOT cylinder wake.

Risk 1 of the campaign design (2026-07-27-1540 note): the kernel indicators
use WALL distance, and at d ~ O(D) in the separated near wake the triple's
geometric meaning weakens. This script evaluates the kernel quantities on a
URANS slice (instantaneous and/or time-averaged) and asks:

  does the kernel see the wake shear-layer instability (P = Omega_hat*I_hat
  > 0, onset gate open) for the right geometric reason -- an inflectional
  layer, Y > X + Z -- or do the wall-distance semantics degrade?

Method
  1. Derived fields via the PAPER'S slice convention: reuse
     add_derived_to_slice.augment() verbatim (X = |u|, Y = omega d,
     Z = 1/2 d^2 n.grad(omega) with n = grad(wallDistance)); the output
     slice_with_derived.pvtu is renamed per input slice.
  2. Onset gate from the canon constants (saai_env / ModelConstants.h):
     Re_Omega_crit = softmin_2(1851.2, 124.6 + 1.424/P^2),
     onset = (1 + tanh((Re_Omega/Re_Omega_crit - 1)/0.35))/2.
  3. Wake shear-layer RIDGES: per x-station aft of the cylinder, the
     |omega|-maximum point in each half-plane (outside the immediate wall
     layer). Along each ridge, record the triple decomposition (X, Y, Z),
     P, Re_Omega margin, chi -- plus two SEMANTICS diagnostics:
       misalign_deg : angle between the kernel's normal n = grad(d) and the
                      local layer normal estimated as grad(|u|)/|grad(|u|)|;
       P_layer      : P recomputed with Z evaluated along that layer normal
                      (convention sensitivity of the curvature indicator).
  4. Diagnostic multi-panel figure to paper/repro/cfd/figs_explore/
     (exploratory only, NOT a paper figure) + JSON of the ridge tables.

Usage: python dragcrisis_kernel_check.py [CASE_DIR] [--slices avg,inst]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figs_explore")
sys.path.insert(0, HERE)
import add_derived_to_slice as _derived   # noqa: E402  (paper convention, reused)

# canon sphere-kernel onset constants (saai_env.py / ModelConstants.h)
REOM_CEIL, REOM_A, REOM_B, RAMP_W = 1851.2, 124.6, 1.424, 0.35
CENTER = (0.5, 0.0)


def onset_gate(re_omega, P):
    Pf = np.maximum(P, 1e-6)
    pw = REOM_A + REOM_B / (Pf * Pf)
    re_c = REOM_CEIL * pw / np.sqrt(REOM_CEIL ** 2 + pw ** 2)
    return 0.5 * (1.0 + np.tanh((re_omega / re_c - 1.0) / RAMP_W)), re_c


def load_derived(case_dir, slice_name):
    """augment() the given slice (paper convention) and load the arrays."""
    src = os.path.join(case_dir, slice_name)
    ok, msg = _derived.augment(src)
    if not ok:
        raise RuntimeError(f"augment failed on {src}: {msg}")
    gen = os.path.join(case_dir, "slice_with_derived.pvtu")
    dst = os.path.join(case_dir, slice_name.replace(".pvtu", "_derived.pvtu"))
    os.replace(gen, dst)
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(dst)
    r.Update()
    g = r.GetOutput()
    pd = g.GetPointData()
    names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    A = {n: vtk_to_numpy(pd.GetArray(n)) for n in names}
    P3 = vtk_to_numpy(g.GetPoints().GetData())
    if "velocity" not in A and "primitiveVars" in A:
        A["velocity"] = A["primitiveVars"][:, 1:4]
    return P3, A, g


def in_plane_gradient(g, field_name):
    gf = vtk.vtkGradientFilter()
    gf.SetInputData(g)
    gf.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                              field_name)
    gf.SetResultArrayName("grad_tmp")
    gf.Update()
    return vtk_to_numpy(gf.GetOutput().GetPointData().GetArray("grad_tmp"))


def ridge_table(P3, A, g, nu, side, x_stations, halfwidth=0.03):
    """Shear-layer ridge (max |omega|) per x-station in one half-plane."""
    x, z = P3[:, 0], P3[:, 2]
    om = A["vorticityMagnitude"]
    d = A["wallDistance"]
    v = A["velocity"]
    umag = np.sqrt((v ** 2).sum(1))
    if "umag_tmp" not in [g.GetPointData().GetArrayName(i)
                          for i in range(g.GetPointData().GetNumberOfArrays())]:
        from vtkmodules.util.numpy_support import numpy_to_vtk
        va = numpy_to_vtk(umag.astype(np.float64), deep=True)
        va.SetName("umag_tmp")
        g.GetPointData().AddArray(va)
    grad_u = in_plane_gradient(g, "umag_tmp")
    grad_d = in_plane_gradient(g, "wallDistance")
    grad_w = in_plane_gradient(g, "vorticityMagnitude")

    half = (z >= 0) if side == "upper" else (z < 0)
    rows = []
    for xs in x_stations:
        m = half & (np.abs(x - xs) < halfwidth) & (d > 0.02) & (np.abs(z) < 1.5)
        if not m.any():
            continue
        i = np.flatnonzero(m)[np.argmax(om[m])]
        X0, Y0, Z0 = A["sph_X"][i], A["sph_Y"][i], A["sph_Z"][i]
        Pv = A["OmegaI"][i]
        reo = A["Re_Omega"][i]
        gate, re_c = onset_gate(np.array([reo]), np.array([max(Pv, 0.0)]))
        # layer normal from grad|u|; kernel normal from grad d
        en = grad_u[i][[0, 2]]
        en = en / (np.linalg.norm(en) + 1e-30)
        nd = grad_d[i][[0, 2]]
        nd = nd / (np.linalg.norm(nd) + 1e-30)
        mis = np.degrees(np.arccos(np.clip(np.abs(en @ nd), 0, 1)))
        # Z with the LAYER normal (convention sensitivity)
        dwdn_layer = grad_w[i][[0, 2]] @ en
        Zl = 0.5 * d[i] ** 2 * dwdn_layer
        R = np.sqrt(X0 ** 2 + Y0 ** 2 + Zl ** 2) + 1e-30
        P_layer = (Y0 / (np.hypot(X0, Y0) + 1e-30)) * ((Y0 - X0 - Zl) / R)
        rows.append({
            "x": float(xs), "z": float(z[i]), "d": float(d[i]),
            "X": float(X0), "Y": float(Y0), "Z": float(Z0),
            "P": float(Pv), "P_layer": float(P_layer),
            "Re_Omega": float(reo), "Re_Omega_crit": float(re_c[0]),
            "onset": float(gate[0]),
            "rate": float(0.19 * min(max(Pv, 0.0), 1.0) * gate[0]),
            "chi": float(A["nuHat"][i] / nu),
            "misalign_deg": float(mis),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir", nargs="?",
                    default="/local_data/qiqi/sa-ai/dragcrisis_pilot/cylL1_Re100k")
    ap.add_argument("--slices", default="avg,inst")
    args = ap.parse_args()
    case_dir = args.case_dir
    nu = json.load(open(os.path.join(case_dir, "Flow360.json")))["freestream"]["muRef"]
    slice_map = {"avg": "slice_centerSpan_time_avg.pvtu",
                 "inst": "slice_centerSpan.pvtu"}
    os.makedirs(FIGD, exist_ok=True)
    results = {}

    for tag in args.slices.split(","):
        sname = slice_map[tag]
        if not os.path.exists(os.path.join(case_dir, sname)):
            print(f"skip {tag}: {sname} not found")
            continue
        P3, A, g = load_derived(case_dir, sname)
        x, z = P3[:, 0], P3[:, 2]
        chi = A["nuHat"] / nu
        gate, _ = onset_gate(A["Re_Omega"], np.maximum(A["OmegaI"], 0.0))
        rate = 0.19 * np.clip(A["OmegaI"], 0.0, 1.0) * gate

        xs = np.arange(1.1, 4.01, 0.1)     # 0.6..3.5 D aft of center
        tabs = {s: ridge_table(P3, A, g, nu, s, xs) for s in ("upper", "lower")}
        results[tag] = tabs

        # ---- figure ----
        win = (x > -0.8) & (x < 4.2) & (np.abs(z) < 1.8)
        tri = mtri.Triangulation(x[win], z[win])
        # mask skinny triangles (concave hull artifacts across the cylinder)
        xt = x[win][tri.triangles]
        zt = z[win][tri.triangles]
        rc = np.hypot(xt.mean(1) - CENTER[0], zt.mean(1))
        big = np.hypot(xt.max(1) - xt.min(1), zt.max(1) - zt.min(1))
        tri.set_mask((rc < 0.5) | (big > 0.3))
        fig, axs = plt.subplots(3, 2, figsize=(13, 12))

        def field_map(ax, vals, title, cmap, vmin=None, vmax=None):
            tp = ax.tripcolor(tri, vals[win], shading="gouraud", cmap=cmap,
                              vmin=vmin, vmax=vmax, rasterized=True)
            th = np.linspace(0, 2 * np.pi, 200)
            ax.fill(CENTER[0] + 0.5 * np.cos(th), 0.5 * np.sin(th), color="0.85",
                    ec="0.4", lw=0.8, zorder=5)
            for s, c in (("upper", "#1f77b4"), ("lower", "#d62728")):
                if tabs[s]:
                    ax.plot([r["x"] for r in tabs[s]], [r["z"] for r in tabs[s]],
                            ".", ms=2.5, color=c, zorder=6)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=10)
            fig.colorbar(tp, ax=ax, shrink=0.85)

        field_map(axs[0, 0], np.log10(np.maximum(chi, 1e-6)),
                  "log10 chi (seed -3.06, turb > 0)", "viridis", -4, 3)
        field_map(axs[0, 1], np.clip(A["OmegaI"], -0.5, 0.5),
                  "P = Omega_hat * I_hat (amplifying > 0)", "RdBu_r", -0.5, 0.5)
        field_map(axs[1, 0], gate, "onset gate S(Re_Omega/Re_Omega_crit)",
                  "viridis", 0, 1)
        field_map(axs[1, 1], rate / 0.19, "rate / a_max = clip(P) * onset",
                  "viridis", 0, 1)

        ax = axs[2, 0]
        for s, c in (("upper", "#1f77b4"), ("lower", "#d62728")):
            t = tabs[s]
            if not t:
                continue
            xr = [r["x"] - CENTER[0] for r in t]
            ax.plot(xr, [r["P"] for r in t], "-", color=c, lw=1.4,
                    label=f"P ({s})")
            ax.plot(xr, [r["P_layer"] for r in t], "--", color=c, lw=1.0,
                    label=f"P layer-normal ({s})")
            ax.plot(xr, [r["onset"] for r in t], ":", color=c, lw=1.0,
                    label=f"onset ({s})")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xlabel("(x - x_c) / D along ridge")
        ax.set_ylabel("P, onset")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        ax.set_title("kernel along wake shear-layer ridges", fontsize=10)

        ax = axs[2, 1]
        for s, c in (("upper", "#1f77b4"), ("lower", "#d62728")):
            t = tabs[s]
            if not t:
                continue
            xr = [r["x"] - CENTER[0] for r in t]
            ax.plot(xr, [r["misalign_deg"] for r in t], "-", color=c, lw=1.4,
                    label=f"misalign {s}")
        ax.set_xlabel("(x - x_c) / D along ridge")
        ax.set_ylabel("angle(grad d, layer normal) [deg]")
        ax.set_ylim(0, 90)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("wall-distance vs layer-normal misalignment", fontsize=10)

        fig.suptitle(f"{os.path.basename(case_dir)} - sphere-kernel frozen-field "
                     f"check ({tag} slice)")
        fig.tight_layout()
        fp = os.path.join(FIGD, f"dragcrisis_kernel_check_{tag}.png")
        fig.savefig(fp, dpi=130)
        plt.close(fig)
        print(f"[{tag}] figure: {fp}")

    out = os.path.join(FIGD, "dragcrisis_kernel_check.json")
    json.dump(results, open(out, "w"), indent=1)
    print(f"ridge tables: {out}")


if __name__ == "__main__":
    main()
