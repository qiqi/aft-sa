"""Collect the drag-crisis steady matrix (matrix_summary.jsonl) into the
campaign figure + table: Cd(Re; Tu) families with up/dn hysteresis overlays,
separation-knee and chi-front angle families, base/shoulder Cp families.
Exploratory output (figs_explore/), NOT a paper figure.

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


def load(root):
    rows = {}
    fp = os.path.join(root, "matrix_summary.jsonl")
    with open(fp) as f:
        for ln in f:
            r = json.loads(ln)
            rows[(r["Tu"], r["dir"], r["re"])] = r   # later rows win
    return sorted(rows.values(), key=lambda r: (r["Tu"], r["dir"], r["re"]))


def series(rows, tu, d, key):
    pts = [(r["re"], r.get(key)) for r in rows
           if r["Tu"] == tu and r["dir"] == d and r.get(key) is not None]
    if not pts:
        return np.array([]), np.array([])
    re, v = zip(*sorted(pts))
    return np.array(re), np.array(v)


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
            for d, st in DIR_STYLE.items():
                re, v = series(rows, tu, d, key)
                if len(re) == 0:
                    continue
                ax.plot(re, v, st, color=c, lw=1.3, ms=4,
                        label=f"Tu {tu}% {d}")
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
