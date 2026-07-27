"""fig:dragcrisiscd -> paper/figs/dragcrisis_cd_re.pdf.

The drag-crisis steady matrix, macroscopic view: Cd(Re_D) for the three
freestream seeds chi_inf(Tu) (color per Tu, CVD-validated trio), up-ladder
solid/filled vs dn-ladder dashed/open, cold two-stage references (middle
seed) as open diamonds. Cases whose convergence monitor flagged a steady
limit cycle carry a capped vertical band spanning the tail-window Cd
min--max (the band, not the marker style, is the limit-cycle marking).

Data: matrix_summary.jsonl of the 84-case campaign
(repro/cfd/run_dragcrisis_matrix.py); the band extents are recomputed here
from each flagged case's final-stage force history over the same tail
window (min(3000, len) rows) the campaign medians use, and dumped with the
plotted medians into data/dragcrisis_cd_re_computed.json (the appendix
table reads that dump + the jsonl copy in data/).

Run from anywhere: python3 repro/cfd/regen_dragcrisis_cd_re.py
  [--root /local_data/qiqi/sa-ai/dragcrisis_matrix]
"""
import argparse
import csv
import json
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', '..', 'figs'))
DATA = os.path.abspath(os.path.join(HERE, '..', '..', 'data'))
PREV = os.path.join(HERE, 'figs_explore')

# ordered identity colors per Tu level (validated: dataviz six-checks trio)
TU_COLOR = {"0.05": "#2f6fa8", "0.2": "#c95f2b", "0.7": "#6d55a3"}
TU_ORDER = ("0.05", "0.2", "0.7")


def load_rows(root):
    rows = {}
    with open(os.path.join(root, "matrix_summary.jsonl")) as f:
        for ln in f:
            r = json.loads(ln)
            rows[(r["Tu"], r["dir"], r["re"])] = r     # later rows win
    return rows


def tail_minmax(case_dir):
    """Final-stage Cd tail min/max, same windowing as the campaign
    extractor (run_dragcrisis_matrix.extract)."""
    ps, cd = [], []
    with open(os.path.join(case_dir, "total_forces_v2.csv")) as f:
        for row in csv.reader(f):
            try:
                ps.append(int(float(row[1])))
                cd.append(float(row[3]))
            except (ValueError, IndexError):
                continue
    k = len(ps) - 1
    while k > 0 and ps[k - 1] <= ps[k]:
        k -= 1
    w = np.array(cd[k:])
    w = w[-min(len(w), 3000):]
    return float(w.min()), float(w.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/local_data/qiqi/sa-ai/dragcrisis_matrix")
    args = ap.parse_args()
    rows = load_rows(args.root)
    # keep the committed copy of the campaign summary in sync (small file;
    # the case trees themselves stay on /local_data)
    shutil.copy2(os.path.join(args.root, "matrix_summary.jsonl"),
                 os.path.join(DATA, "dragcrisis_matrix_summary.jsonl"))

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    dump = {}
    for tu in TU_ORDER:
        c = TU_COLOR[tu]
        for d, ls, mk, mfc in (("up", "-", "o", c), ("dn", "--", "s", "none"),
                               ("cold", "none", "D", "none")):
            pts = sorted((re, r) for (t, dd, re), r in rows.items()
                         if t == tu and dd == d)
            if not pts:
                continue
            re = np.array([p[0] for p in pts])
            cd = np.array([p[1]["Cd"] for p in pts])
            ax.plot(re, cd, ls=ls, marker=mk, color=c, mfc=mfc, mew=1.3,
                    ms=5.5 if mk != 'D' else 6.5, lw=1.6, zorder=4)
            for reval, r in pts:
                lc = any("limit_cycle" in v for v in r["verdicts"])
                ent = {"Cd": r["Cd"], "Cd_tail_p2p": r["Cd_tail_p2p"],
                       "limit_cycle": lc, "verdicts": r["verdicts"]}
                if lc:
                    lo, hi = tail_minmax(os.path.join(args.root, r["case"]))
                    ent["Cd_tail_min"], ent["Cd_tail_max"] = lo, hi
                    ax.errorbar([reval], [r["Cd"]],
                                yerr=[[r["Cd"] - lo], [hi - r["Cd"]]],
                                fmt='none', ecolor=c, elinewidth=1.4,
                                capsize=3.5, capthick=1.4, zorder=3)
                dump[r["case"]] = ent

    ax.set_xscale("log")
    ax.set_xlim(5.2e4, 2.4e6)
    ax.set_ylim(0.15, 0.92)
    ax.xaxis.set_major_locator(
        mticker.FixedLocator([6e4, 1e5, 2e5, 3e5, 5e5, 1e6, 2e6]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(
        mticker.FixedFormatter(['$0.6$', '$1$', '$2$', '$3$', '$5$',
                                '$10$', '$20$']))
    ax.set_xlabel(r'$Re_D\ (\times 10^5)$')
    ax.set_ylabel(r'$C_d$')
    ax.grid(alpha=0.3, which='both')

    # direct family labels where the three seeds separate
    ax.text(2.35e5, 0.545, r'$Tu\,0.7\%$', color=TU_COLOR["0.7"],
            fontsize=12, ha='right')
    ax.text(6.3e5, 0.575, r'$Tu\,0.2\%$', color=TU_COLOR["0.2"],
            fontsize=12, ha='left')
    ax.text(4.6e5, 0.755, r'$Tu\,0.05\%$', color=TU_COLOR["0.05"],
            fontsize=12, ha='left')

    handles = [
        Line2D([], [], color='0.25', ls='-', marker='o', ms=5.5,
               label='up-ladder (warm from lower $Re$)'),
        Line2D([], [], color='0.25', ls='--', marker='s', mfc='none',
               mew=1.3, ms=5.5, label='dn-ladder (warm from higher $Re$)'),
        Line2D([], [], color='0.25', ls='none', marker='D', mfc='none',
               mew=1.3, ms=6.5, label='cold two-stage ($Tu\\,0.2\\%$)'),
        Line2D([], [], color='0.25', ls='none', marker='|', ms=13,
               mew=1.6, label='steady limit cycle: tail $C_d$ min--max'),
    ]
    ax.legend(handles=handles, fontsize=10.5, frameon=False,
              loc='lower left', handlelength=2.6)
    fig.tight_layout()
    os.makedirs(PREV, exist_ok=True)
    fig.savefig(os.path.join(OUT, 'dragcrisis_cd_re.pdf'))
    fig.savefig(os.path.join(PREV, 'dragcrisis_cd_re.png'), dpi=140)
    json.dump(dump, open(os.path.join(DATA, 'dragcrisis_cd_re_computed.json'),
                         'w'), indent=1)
    n_lc = sum(1 for e in dump.values() if e["limit_cycle"])
    print(f"wrote {OUT}/dragcrisis_cd_re.pdf ({len(dump)} cases, "
          f"{n_lc} limit-cycle bands)")


if __name__ == "__main__":
    main()
