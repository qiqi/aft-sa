"""fig:dragcrisiscd -> paper/figs/dragcrisis_cd_re.pdf (systematic source).

Rebuild the drag-crisis Cd(Re_D) figure with the SA-AI curve taken from the
clean systematic Tu=0.2% up/down ladder campaign (2026-07-29, 50 rows,
matched Re grid 1..1e10) instead of the old 84-case patchwork matrix,
reusing the exact literature overlay from regen_dragcrisis_cd_re. Only the
UP-LADDER is drawn (user directive 2026-07-29; the down ladder is dropped
from the drag-crisis plots pending the ultra-Re branch-split narrative -- see
draw_saai). The literature overlay, log axes and full 1..1e10 span match the
committed figure.

NOTE (whitepaper-first, 2026-07-29): this writes the SHARED figs PDF used by
BOTH the whitepaper and the main paper. The main paper's Table t:dragcrisis
and several tied text numbers still describe the OLD 84-case 3-seed matrix
(regen_dragcrisis_cd_re.py) and are a deferred migration the user signs off.

Run from anywhere:
  python3 repro/cfd/regen_dragcrisis_cd_systematic.py [--xmax 1e10]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

import regen_dragcrisis_cd_re as base

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "figs"))
PREV = os.path.join(HERE, "figs_explore")
SYS = ("/local_data/qiqi/sa-ai/dragcrisis_matrix/"
       "systematic_Tu0.2_summary.jsonl")
TU_C = base.TU_COLOR["0.2"]      # the single systematic seed's identity color

# mesh-family draw order (segments drawn per family so seam overlaps do not
# make a spurious vertical kink at a shared Re)
FAM_ORDER = ("lowre", "pilot", "highre", "ultra")


def load_sys():
    rows = {"up": [], "dn": []}
    with open(SYS) as f:
        for ln in f:
            d = json.loads(ln)
            rows[d["dir"]].append(d)
    for k in rows:
        rows[k].sort(key=lambda r: r["re"])
    return rows


def draw_saai(ax, rows):
    # UP-LADDER ONLY for now (user directive 2026-07-29): the down ladder is
    # dropped from the drag-crisis plots pending the ultra-Re branch-split
    # narrative. Re-enable by adding ("dn", "--") to the loop.
    for d, ls in (("up", "-"),):
        rr = rows[d]
        for fam in FAM_ORDER:
            fp = [r for r in rr if r.get("mesh") == fam]
            if not fp:
                continue
            re = np.array([r["re"] for r in fp])
            cd = np.array([r["Cd"] for r in fp])
            ax.plot(re, cd, ls=ls, color=TU_C, lw=1.7,
                    alpha=base.ALPHA_SAAI, zorder=4)
        # open ring on every case the steady monitor did not accept
        # (verdict != converged); this is the authoritative flag that also
        # brackets the peak-to-peak in Table~\ref{tab:data_dragcrisis}, so
        # rings and bracketed rows coincide exactly.
        lc = [r for r in rr if r.get("verdict") != "converged"]
        if lc:
            ax.plot([r["re"] for r in lc], [r["Cd"] for r in lc],
                    ls="none", marker="o", ms=4.5, mfc="none",
                    mec=TU_C, mew=1.0, alpha=base.ALPHA_SAAI, zorder=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmin", type=float, default=0.7)
    ap.add_argument("--xmax", type=float, default=1.3e10)
    ap.add_argument("--no-lit", action="store_true")
    args = ap.parse_args()

    rows = load_sys()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
    fig, ax = plt.subplots(figsize=(7.8, 5.4))

    draw_saai(ax, rows)

    lit_handles = [] if args.no_lit else base.overlay_literature(ax)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(args.xmin, args.xmax)
    allcd = [r["Cd"] for d in rows for r in rows[d]]
    ax.set_ylim(0.7 * min(allcd), 1.35 * max(allcd))
    ax.xaxis.set_major_locator(mticker.LogLocator(numticks=14))
    ax.set_xlabel(r"$Re_D$")
    ax.set_ylabel(r"$C_d$")
    ax.grid(alpha=0.3, which="major")

    ax.text(3.5e5, 0.60, r"$Tu\,0.2\%$", color=TU_C, fontsize=11, ha="right")
    ours = [
        Line2D([], [], color=TU_C, ls="-", lw=1.7, label="SA-AI (up-ladder)"),
        Line2D([], [], color=TU_C, ls="none", marker="o", ms=4.5,
               mfc="none", mew=1.0, label="limit cycle"),
    ]
    leg1 = ax.legend(handles=ours, fontsize=9, frameon=False,
                     loc="upper right", handlelength=2.4)
    ax.add_artist(leg1)
    if lit_handles:
        ax.legend(handles=lit_handles, fontsize=8, frameon=False,
                  loc="lower left", handlelength=2.2, handletextpad=0.5,
                  borderaxespad=0.3, labelspacing=0.32)

    fig.tight_layout()
    os.makedirs(PREV, exist_ok=True)
    pdf = os.path.join(OUT, "dragcrisis_cd_re.pdf")
    png = os.path.join(PREV, "dragcrisis_cd_re_systematic.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=140)
    print(f"wrote {pdf}\nwrote {png}")
    print(f"  up rows {len(rows['up'])} (dn dropped), "
          f"Re {min(r['re'] for r in rows['up']):g}"
          f"..{max(r['re'] for r in rows['up']):g}")


if __name__ == "__main__":
    main()
