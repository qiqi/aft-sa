"""fig:dragcrisisangles -> paper/figs/dragcrisis_angles.pdf.

Single seed Tu=0.2%, the systematic clean up/down ladders (2026-07-29
campaign, /local_data .../systematic_Tu0.2_summary.jsonl), across the FULL
Re_D = 1..1e10 range.
  (a) transition angle theta_tr (radial-ray max-chi >= 1) and wall separation
      angles (first / final tangential-Cf zero crossings) vs Re_D, up (solid)
      and down (dashed) ladders.
  (b) max-chi(theta) radial-ray profiles, one Re per decade (10^2..10^10),
      up-ladder, log scale, chi = 1 marked.
House style: line plots, no in-figure titles, caption carries the story.
Run from anywhere: python3 repro/cfd/regen_dragcrisis_angles.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', '..', 'figs'))
PREV = os.path.join(HERE, 'figs_explore')
SUMMARY = ('/local_data/qiqi/sa-ai/dragcrisis_matrix/'
           'systematic_Tu0.2_summary.jsonl')
CRISIS_BAND = (3e5, 7e5)
PROFILE_DECADES = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
# UP-LADDER ONLY for now (user directive 2026-07-29): the down ladder is
# dropped from the drag-crisis plots pending the ultra-Re branch-split
# narrative. Set True to restore the up/dn (solid/dashed) pair.
PLOT_DN = False


def sep_first_final(crossings):
    seps = [a for a, t in (crossings or []) if t == 'separation']
    if not seps:
        return None, None
    return min(seps), max(seps)


def load():
    rows = {}
    with open(SUMMARY) as f:
        for ln in f:
            d = json.loads(ln)
            rows.setdefault(d['dir'], []).append(d)
    for d in rows:
        rows[d].sort(key=lambda r: r['re'])
    return rows


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is not None:
            xs.append(r['re']); ys.append(v)
    return np.array(xs), np.array(ys)


def sep_series(rows, which):
    xs, ys = [], []
    for r in rows:
        f, l = sep_first_final(r.get('crossings_upper'))
        v = f if which == 'first' else l
        if v is not None and 1.0 < v < 179.9:
            xs.append(r['re']); ys.append(v)
    return np.array(xs), np.array(ys)


def main():
    rows = load()
    plt.rcParams.update({"font.size": 11})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.4),
                                 gridspec_kw=dict(wspace=0.22))

    # ---- (a) angles vs Re, up-ladder (dn dropped for now, PLOT_DN) ----
    ax.axvspan(*CRISIS_BAND, color="0.92", zorder=0)
    branches = [("up", "-", "k")]
    if PLOT_DN:
        branches.append(("dn", "--", "none"))
    for br, ls, mfc in branches:
        rr = rows.get(br, [])
        re_, th = series(rr, "theta_tr_chi1")
        ax.plot(re_, th, ls, color="k", marker="o", ms=4.5, mfc=mfc,
                lw=1.6, zorder=4)
        # "final separation" dropped (user directive 2026-07-29); keep first.
        for which, mk in (("first", "v"),):
            re_, sp = sep_series(rr, which)
            ax.plot(re_, sp, ls, color="0.5", marker=mk, ms=3.5, mfc=mfc,
                    lw=0.7, alpha=0.8, zorder=3)
    leg = [plt.Line2D([], [], color="k", ls="-", marker="o", ms=4.5, label=r"$\theta_{tr}$")]
    if PLOT_DN:
        leg.append(plt.Line2D([], [], color="k", ls="--", marker="o", ms=4.5, mfc="none", label=r"$\theta_{tr}$ dn"))
    leg += [plt.Line2D([], [], color="0.5", ls="-", marker="v", ms=3.5, lw=0.7, label="first separation")]
    ax.legend(handles=leg, loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 1.2e10)
    ax.set_ylim(0, 185)
    ax.set_xlabel("$Re_D$")
    ax.set_ylabel(r"angle from forward stagnation [deg]")
    ax.grid(alpha=0.25, which="both")

    # ---- (b) max-chi(theta) at one Re per decade (up-ladder) ----
    up = {r['re']: r for r in rows.get("up", [])}
    cmap = plt.cm.viridis(np.linspace(0.0, 0.92, len(PROFILE_DECADES)))
    for re, col in zip(PROFILE_DECADES, cmap):
        r = up.get(re)
        if r is None or r.get('log10_maxchi') is None:
            continue
        th = np.asarray(r['theta_deg'], float)
        prof = 10.0 ** np.asarray(r['log10_maxchi'], float)
        e = int(round(np.log10(re)))
        bx.plot(th, prof, '-', color=col, lw=1.3, label=rf"$10^{{{e}}}$")
    bx.axhline(1.0, color="k", lw=0.9, ls=":")
    bx.set_yscale("log")
    bx.set_xlim(0, 180)
    bx.set_ylim(1e-3, 3e8)
    bx.set_xticks([0, 30, 60, 90, 120, 150, 180])
    bx.set_xlabel(r"angle from forward stagnation [deg]")
    bx.set_ylabel(r"$\max_s \chi$ along radial ray")
    bx.legend(loc="lower right", fontsize=8, title="$Re_D$",
              title_fontsize=9, framealpha=0.9, ncol=2)
    bx.grid(alpha=0.25, which="both")

    fig.savefig(os.path.join(OUT, "dragcrisis_angles.pdf"), bbox_inches="tight")
    os.makedirs(PREV, exist_ok=True)
    fig.savefig(os.path.join(PREV, "dragcrisis_angles.png"), dpi=130,
                bbox_inches="tight")
    print(f"wrote {OUT}/dragcrisis_angles.pdf")


if __name__ == "__main__":
    main()
