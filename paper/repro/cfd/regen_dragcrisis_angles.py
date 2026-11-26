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
# Panel (b) used to show one Reynolds number per decade, nine of the
# twenty-five the ladder holds. It now shows ALL of them, split into panes over
# contiguous ranges of roughly equal COUNT (6-7 curves each) rather than by
# grid family -- the families are very unevenly populated (7/9/3/6), which left
# one pane crowded and another nearly empty.
N_PANES = 4
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


def _relab(re):
    e = int(np.floor(np.log10(re) + 1e-9))
    m = re / 10.0 ** e
    if abs(re - 1.0) < 1e-9:
        return "1"
    return (r"10^{%d}" % e) if abs(m - 1) < 1e-6 else (r"%g\times10^{%d}" % (round(m, 2), e))


def main():
    rows = load()
    plt.rcParams.update({"font.size": 11})
    # (a) spans the top; the four profile panes sit in a 2x2 block below it.
    import matplotlib.gridspec as _gs
    fig = plt.figure(figsize=(11.2, 10.4))
    G = _gs.GridSpec(3, 2, height_ratios=[1.15, 1.0, 1.0], hspace=0.30,
                     wspace=0.20, figure=fig)
    ax = fig.add_subplot(G[0, :])
    bxs = [fig.add_subplot(G[1, 0]), fig.add_subplot(G[1, 1]),
           fig.add_subplot(G[2, 0]), fig.add_subplot(G[2, 1])]

    # ---- (a) angles vs Re, up-ladder (dn dropped for now, PLOT_DN) ----
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

    # ---- (b) max-chi(theta) for EVERY up-ladder Reynolds number ----
    up = {r['re']: r for r in rows.get("up", [])}
    all_re = sorted(r for r in up if up[r].get('log10_maxchi') is not None)
    # contiguous, near-equal chunks: 25 points over 4 panes -> 6,6,7,6
    k, m = divmod(len(all_re), N_PANES)
    panes, i0 = [], 0
    for j in range(N_PANES):
        n = k + (1 if j >= N_PANES - m else 0)
        panes.append(all_re[i0:i0 + n]); i0 += n
    for bx, res in zip(bxs, panes):
        cmap = plt.cm.viridis(np.linspace(0.0, 0.92, max(len(res), 1)))
        for re, col in zip(res, cmap):
            r = up.get(re)
            if r is None or r.get('log10_maxchi') is None:
                continue
            th = np.asarray(r['theta_deg'], float)
            prof = 10.0 ** np.asarray(r['log10_maxchi'], float)
            e = int(np.floor(np.log10(re) + 1e-9))
            m = re / 10.0 ** e
            lbl = (rf"$10^{{{e}}}$" if abs(m - 1) < 1e-6 and e else
                   ("$1$" if abs(re - 1) < 1e-9 else
                    rf"${m:g}\times10^{{{e}}}$"))
            bx.plot(th, prof, '-', color=col, lw=1.3, label=lbl)
        bx.axhline(1.0, color="k", lw=0.9, ls=":")
        bx.set_yscale("log")
        bx.set_xlim(0, 180)
        bx.set_ylim(1e-3, 3e8)
        bx.set_xticks([0, 30, 60, 90, 120, 150, 180])
        bx.set_xlabel(r"angle from forward stagnation [deg]")
        bx.set_ylabel(r"$\max_s \chi$ along radial ray")
        bx.set_title(rf"$Re_D={_relab(res[0])}$–${_relab(res[-1])}$"
                     rf"  ({len(res)} points)", fontsize=9)
        # upper left: the curves climb from bottom-left to top-right, so that
        # corner is the empty one in every pane (chi sits at the seed until
        # the front arrives).
        bx.legend(loc="upper left", fontsize=7, title="$Re_D$",
                  title_fontsize=8, framealpha=0.9, ncol=2)
        bx.grid(alpha=0.25, which="both")

    fig.savefig(os.path.join(OUT, "dragcrisis_angles.pdf"), bbox_inches="tight")
    os.makedirs(PREV, exist_ok=True)
    fig.savefig(os.path.join(PREV, "dragcrisis_angles.png"), dpi=130,
                bbox_inches="tight")
    print(f"wrote {OUT}/dragcrisis_angles.pdf")


if __name__ == "__main__":
    main()
