"""Whitepaper figure: the two-branch low-H "vg" kernel is benign on the airfoil
cases we ran it on (NLF(1)-0416 a0/a4 Re=4e6, Eppler 387 a2 Re=2e5, structured L2).

Three panels, upper-surface skin friction Cf(x/c): canon (black solid) vs vg
(gray dashed), with the near-wall chi=1 transition front marked for each. The
point: the Cf curves and fronts essentially overlap -- the low-H FPG change does
not trip natural laminar flow.

Extraction reuses the paper's own conventions:
  Cf(x/c)   : surface_fluid_<wall>.pvtu 'Cf' field, upper surface (z>0),
              median per x-bin (run.py _extract_xtr convention).
  chi=1 front: near-wall (wallDistance<3e-4) max chi per x-bin crossing 1.0,
              chi = nuHat/muRef (flatplate/regen_flatplate_flow360 convention;
              muRef is the case's own laminar viscosity). 300 bins over [0,1]
              (~0.003c), matching the 2026-07-29-1401 NLF-vg memo, so the canon
              fronts here reproduce that memo (a0 0.396, a4 0.266).

Canon vs vg is a kernel-only delta (same mesh/config/seed, vg warm-restarted
from the canon converged state).

Run from paper/:  python repro/cfd/regen_vg_airfoils.py
Outputs:
  figs/newkernel_vg_airfoils.png
  repro/cfd/figs_explore/data/vg_airfoils.json   (cached Cf + fronts + forces)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/local_data/qiqi/sa-ai/flow360_fv1")
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent                        # .../sa-ai/paper
FIG = PAPER / "figs" / "newkernel_vg_airfoils.png"
DATA = HERE / "figs_explore" / "data" / "vg_airfoils.json"
BAND = 3e-4                                        # near-wall wallDistance band
NBIN_CHI = 300                                     # ~0.003c, matches the NLF memo

# case -> (wall patch, muRef, canon dir, vg dir, front method, panel title)
# NLF: near-wall chi=1 crossing (2026-07-29-1401 memo convention). Eppler: the
# paper's Eppler convention (run.py _extract_xtr, upper Cf-jump) -- its laminar
# separation bubble keeps the wall-adjacent band laminar (chi<1) while the
# turbulent rise sits in the lifted shear layer, so a near-wall chi=1 probe
# does not resolve the reattachment front.
CASES = [
    ("nlf_a0", "nlf0416", 2.5e-8,
     "strL2prop_nlf0416_Re4M_a0", "strL2prop_nlf0416_Re4M_a0_vg", "chi1",
     r"NLF(1)-0416  $Re=4\times10^6$, $\alpha=0^\circ$"),
    ("nlf_a4", "nlf0416", 2.5e-8,
     "strL2prop_nlf0416_Re4M_a4", "strL2prop_nlf0416_Re4M_a4_vg", "chi1",
     r"NLF(1)-0416  $Re=4\times10^6$, $\alpha=4^\circ$"),
    ("eppler_a2", "eppler387", 5.0e-7,
     "strL2prop_eppler387_Re200k_a2", "strL2prop_eppler387_Re200k_a2_vg", "cfjump",
     r"Eppler 387  $Re=2\times10^5$, $\alpha=2^\circ$"),
]


def _read_pvtu(path: Path):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    pd = g.GetPointData()
    arr = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i))
           for i in range(pd.GetNumberOfArrays())}
    return pts, arr


def cf_upper(case: str, wall: str):
    """Upper-surface Cf(x/c): median per x-bin (run.py convention)."""
    pts, arr = _read_pvtu(ROOT / case / f"surface_fluid_{wall}.pvtu")
    a = arr["Cf"]
    cf = np.linalg.norm(a, axis=1) if a.ndim > 1 else a
    x, z = pts[:, 0], pts[:, 2]
    up = z > 0.0
    xs, cfs = x[up], cf[up]
    bins = np.linspace(0, 1, 51)               # run.py _extract_xtr: 0.02c bins
    xc = 0.5 * (bins[1:] + bins[:-1])
    cb = np.array([np.median(cfs[(xs >= bins[i]) & (xs < bins[i + 1])])
                   if ((xs >= bins[i]) & (xs < bins[i + 1])).any() else np.nan
                   for i in range(len(bins) - 1)])
    ok = np.isfinite(cb)
    return xc[ok], cb[ok]


def chi1_front_upper(case: str, muRef: float):
    """Near-wall chi=1 crossing on the upper surface (paper convention)."""
    pts, arr = _read_pvtu(ROOT / case / "volume.pvtu")
    x, z = pts[:, 0], pts[:, 2]
    chi = arr["nuHat"] / muRef
    m = (arr["wallDistance"] < BAND) & (z > 0)
    xb, cb = x[m], chi[m]
    bins = np.linspace(0, 1, NBIN_CHI + 1)
    xc = 0.5 * (bins[1:] + bins[:-1])
    cmax = np.array([cb[(xb >= bins[i]) & (xb < bins[i + 1])].max()
                     if ((xb >= bins[i]) & (xb < bins[i + 1])).any() else np.nan
                     for i in range(NBIN_CHI)])
    ok = np.isfinite(cmax)
    xc, cmax = xc[ok], cmax[ok]
    cr = np.where(cmax >= 1.0)[0]
    return float(xc[cr[0]]) if len(cr) else None


def cfjump_front_upper(x, cf):
    """Upper-surface transition x/c from the max Cf jump (run.py convention)."""
    x = np.asarray(x)
    cf = np.asarray(cf)
    d = np.diff(cf)
    dm = 0.5 * (x[1:] + x[:-1])
    w = (dm > 0.04) & (dm < 0.95)
    if not w.any():
        return None
    return float(dm[w][np.argmax(d[w])])


def total_forces(case: str):
    import csv
    f = ROOT / case / "total_forces_v2.csv"
    if not f.exists():
        return None, None
    rows = list(csv.DictReader(open(f)))
    last = rows[-1]
    return float(last[" CL"]), float(last[" CD"])


def build_data():
    out = {}
    for key, wall, muRef, cdir, vdir, method, title in CASES:
        rec = {"title": title, "wall": wall, "muRef": muRef,
               "canon_dir": cdir, "vg_dir": vdir, "front_method": method}
        for tag, d in (("canon", cdir), ("vg", vdir)):
            xc, cf = cf_upper(d, wall)
            cl, cd = total_forces(d)
            xtr = (chi1_front_upper(d, muRef) if method == "chi1"
                   else cfjump_front_upper(xc, cf))
            rec[tag] = {"x": xc.tolist(), "cf": cf.tolist(),
                        "xtr": xtr, "CL": cl, "CD": cd}
        out[key] = rec
        print(f"{key}: canon xtr={rec['canon']['xtr']:.3f} "
              f"vg xtr={rec['vg']['xtr']:.3f} | "
              f"CL {rec['canon']['CL']:.4f}->{rec['vg']['CL']:.4f} "
              f"CD {rec['canon']['CD']:.5f}->{rec['vg']['CD']:.5f}")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(out, indent=1))
    return out


def make_figure(data):
    CANON = "#000000"
    VG = "#8a8a8a"
    keys = ["nlf_a0", "nlf_a4", "eppler_a2"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    for ax, key in zip(axes, keys):
        rec = data[key]
        c, v = rec["canon"], rec["vg"]
        ax.plot(c["x"], c["cf"], color=CANON, lw=1.6, label="canon", zorder=3)
        ax.plot(v["x"], v["cf"], color=VG, lw=1.6, ls=(0, (5, 2)),
                label="vg (low-H)", zorder=4)
        ymax = max(max(c["cf"]), max(v["cf"]))
        ymin = min(min(c["cf"]), min(v["cf"]))
        pad = 0.08 * (ymax - ymin)
        ax.set_ylim(min(0, ymin - pad), ymax + pad)
        # transition-front ticks (chi=1)
        for xtr, col, dash in ((c["xtr"], CANON, False), (v["xtr"], VG, True)):
            if xtr is not None:
                ax.axvline(xtr, color=col, lw=1.0, ls=":" if dash else "-",
                           alpha=0.75, zorder=2)
        # annotate the two fronts
        txt = (r"$x_{tr}$(canon)=%.3f" % c["xtr"] + "\n" +
               r"$x_{tr}$(vg)=%.3f" % v["xtr"])
        ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#222222",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                          alpha=0.85))
        ax.set_title(rec["title"], fontsize=9.5)
        ax.set_xlabel("x/c")
        ax.set_xlim(0, 1)
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel(r"upper-surface $C_f$")
    axes[0].legend(loc="upper left", fontsize=8, frameon=True,
                   facecolor="white", edgecolor="#cccccc")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"wrote {FIG}")


if __name__ == "__main__":
    import sys
    # Plot from the cached JSON when present (reproducible without the CFD tree);
    # pass --rebuild to re-read the volume/surface .pvtu on /local_data.
    if DATA.exists() and "--rebuild" not in sys.argv:
        make_figure(json.loads(DATA.read_text()))
    else:
        make_figure(build_data())
