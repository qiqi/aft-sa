"""Collect the drag-crisis steady matrix + extension (matrix_summary.jsonl)
into the campaign figures + table: Cd(Re; Tu) families with up/dn hysteresis
overlays, separation-knee and chi-front angle families, base/shoulder Cp
families, and the FULL-SPAN log-log composite Cd(Re) (2026-07-28 extension:
mesh families lowre/pilot/highre, Re 1 .. 1-2e7, low-Re steady benchmarks +
literature guide levels). Exploratory output (figs_explore/), NOT a paper
figure.

Usage: python regen_dragcrisis_matrix_figs.py [--root .../dragcrisis_matrix]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figs_explore")

# fixed identity colors per Tu level (do not re-map when a series is absent)
TU_COLOR = {"0.05": "#1f77b4", "0.2": "#ff7f0e", "0.7": "#2ca02c"}
DIR_STYLE = {"up": "-o", "dn": "--s", "cold": ":^"}
# mesh-family membership markers for the full-span composite
MESH_MARK = {"pilot": "o", "lowre": "v", "lowre300": "P", "highre": "D",
             "ultra": "^"}
MESHES = ("lowre", "lowre300", "pilot", "highre", "ultra")

# Low-Re steady-branch benchmarks (litrange record 2026-07-28-0117 Sec. 2):
# Dennis & Chang 1970 JFM 42 (Cd(20), Cd(40) verified; 10/100 widely
# reproduced [mem]); Fornberg 1980 JFM 98 (Cd(20)=2.00, Cd(40)=1.498
# verified; 1985 JCP 61 steady symmetric branch to Re=600, Cd(600)~0.54
# [mem]). Below Re~46-47 (Henderson onset) steady IS the physical flow.
DENNIS_CHANG = {10: 2.846, 20: 2.045, 40: 1.522, 100: 1.056}
FORNBERG = {20: 2.00, 40: 1.498, 600: 0.54}


def load(root):
    rows = {}
    fp = os.path.join(root, "matrix_summary.jsonl")
    with open(fp) as f:
        for ln in f:
            r = json.loads(ln)
            r.setdefault("mesh", "pilot")      # pre-extension rows = pilot
            rows[(r["Tu"], r["dir"], r["re"], r["mesh"])] = r  # later rows win
    return sorted(rows.values(),
                  key=lambda r: (r["Tu"], r["dir"], r["mesh"], r["re"]))


def series(rows, tu, d, key, mesh=None):
    pts = [(r["re"], r.get(key)) for r in rows
           if r["Tu"] == tu and r["dir"] == d and r.get(key) is not None
           and (mesh is None or r["mesh"] == mesh)]
    if not pts:
        return np.array([]), np.array([])
    re, v = zip(*sorted(pts))
    return np.array(re), np.array(v)


def fullspan_figure(rows, figd):
    """Log-log composite Cd(Re): every mesh family, Tu colors, up/dn/cold
    line styles, mesh-membership markers, limit-cycle bands (approximated as
    Cd +/- Cd_tail_p2p/2 from the jsonl; the paper-figure script recomputes
    exact tail min/max from the force histories), low-Re steady benchmarks
    and literature guide levels."""
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for tu, c in TU_COLOR.items():
        for d, st in DIR_STYLE.items():
            ls = {"up": "-", "dn": "--", "cold": ":"}[d]
            for mesh in MESHES:
                re, v = series(rows, tu, d, "Cd", mesh=mesh)
                if len(re) == 0:
                    continue
                ax.plot(re, v, ls=ls, marker=MESH_MARK[mesh], color=c,
                        lw=1.2, ms=5, mfc=(c if d == "up" else "none"),
                        label=f"Tu {tu}% {d} [{mesh}]")
    # limit-cycle bands
    for r in rows:
        if any("converged" not in v for v in r["verdicts"]) \
                and r.get("Cd") and r.get("Cd_tail_p2p"):
            h = r["Cd_tail_p2p"] / 2.0
            ax.plot([r["re"]] * 2, [r["Cd"] - h, r["Cd"] + h],
                    color="0.3", lw=2.2, alpha=0.55,
                    solid_capstyle="butt", zorder=1)
    # steady-branch benchmarks (physical below Re~47; unstable branch above)
    ax.plot(list(DENNIS_CHANG), list(DENNIS_CHANG.values()), "k*", ms=11,
            mfc="none", label="Dennis-Chang 1970 (steady)")
    ax.plot(list(FORNBERG), list(FORNBERG.values()), "kh", ms=8, mfc="none",
            label="Fornberg 1980/85 (steady symm.)")
    # literature guide levels (qualitative): subcritical plateau ~1.2
    # (Wieselsberger), transcritical recovery Cd->~0.7 by 3.5e6 (Roshko 1961)
    ax.plot([1e3, 2e5], [1.2, 1.2], color="0.5", ls=":", lw=1)
    ax.text(4e3, 1.26, "shedding-mean plateau ~1.2 (exp.)", color="0.4",
            fontsize=7)
    ax.plot([3.5e6, 1e7], [0.7, 0.7], color="0.5", ls=":", lw=1)
    ax.text(3.6e6, 0.74, "Roshko transcritical ~0.7", color="0.4", fontsize=7)
    # 47 < Re < ~1e3: steady symmetric branch is unphysical (continuity band)
    ax.axvspan(47, 1e3, color="0.85", alpha=0.5, zorder=0)
    ax.text(2.2e2, 6.5, "steady branch\nunstable (47<Re<1e3):\ncontinuity, "
            "not validation", fontsize=7, ha="center", color="0.35")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Re_D")
    ax.set_ylabel("steady-branch Cd")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower left")
    ax.set_title("single-model steady Cd(Re) traverse "
                 "(bands = limit-cycle tail p2p; markers = mesh family)")
    fig.tight_layout()
    fp = os.path.join(figd, "dragcrisis_fullspan.png")
    fig.savefig(fp, dpi=140)
    print(f"figure: {fp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/local_data/qiqi/sa-ai/dragcrisis_matrix")
    args = ap.parse_args()
    rows = load(args.root)
    os.makedirs(FIGD, exist_ok=True)

    panels = [
        ("Cd", "steady-branch Cd", None),
        ("knee_upper", "separation knee phi [deg]", (60, 150)),
        ("chi1_front_upper", "chi=1 near-wall front phi [deg]", (60, 185)),
        ("Cp_base_upper", "base Cp", None),
        ("Cp_shoulder_upper", "shoulder Cp (min, 40-110 deg)", None),
        ("CL", "CL (asymmetry detector)", None),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (key, label, ylim) in zip(axs.ravel(), panels):
        for tu, c in TU_COLOR.items():
            for d in DIR_STYLE:
                ls = {"up": "-", "dn": "--", "cold": ":"}[d]
                for mesh in MESHES:
                    re, v = series(rows, tu, d, key, mesh=mesh)
                    if len(re) == 0:
                        continue
                    ax.plot(re, v, ls=ls, marker=MESH_MARK[mesh], color=c,
                            lw=1.3, ms=4,
                            label=f"Tu {tu}% {d} [{mesh}]")
        ax.set_xscale("log")
        ax.set_xlabel("Re_D")
        ax.set_ylabel(label)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, lw=0.5)
    axs[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    # non-convergence markers
    for r in rows:
        if any("converged" not in v for v in r["verdicts"]):
            for ax, (key, _, _) in zip(axs.ravel(), panels):
                val = r.get(key)
                if val is not None:
                    ax.plot(r["re"], val, "kx", ms=10, mew=2, zorder=9)
    fig.suptitle("drag-crisis STEADY matrix (x = not force-converged)")
    fig.tight_layout()
    fp = os.path.join(FIGD, "dragcrisis_matrix.png")
    fig.savefig(fp, dpi=140)
    print(f"figure: {fp}")

    fullspan_figure(rows, FIGD)

    # compact table to stdout
    print(f"{'case':34s} {'verdicts':22s} {'Cd':>7s} {'CL':>8s} "
          f"{'knee':>6s} {'front':>6s} {'Cpb':>7s}")
    for r in rows:
        knee = r.get("knee_upper")
        fr = r.get("chi1_front_upper")
        print(f"{r['case']:34s} {'|'.join(v.split(':')[1] for v in r['verdicts']):22s} "
              f"{r.get('Cd', float('nan')):7.3f} {r.get('CL', 0):+8.4f} "
              f"{knee if knee else float('nan'):6.1f} "
              f"{fr if fr else float('nan'):6.1f} "
              f"{r.get('Cp_base_upper', float('nan')):7.3f}")


if __name__ == "__main__":
    main()
