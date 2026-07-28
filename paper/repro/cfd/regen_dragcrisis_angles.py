"""fig:dragcrisisangles -> paper/figs/dragcrisis_angles.pdf.

Two panels from the committed radial-ray extraction JSON
(figs_explore/data/dragcrisis_theta_tr.json, built by
dragcrisis_transition_angle.py; no field data re-read here):
  (a) transition angle theta_tr (radial-ray max-chi >= 1, the user's
      convention) AND wall separation angles (first / final tangential-Cf
      crossings) vs Re_D, three seeds, up/dn ladders, authoritative mesh
      family per Re (pilot <= 2e6, highre >= 4e6; low arm = the pilot dn
      continuation). The lowre creeping-arm angles are far-wake features
      (record 2026-07-28-1033) and are excluded.
  (b) max-chi(theta) ray profiles for a representative Re ladder at the
      middle seed, log scale, chi = 1 marked.
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
JSON = os.path.join(PREV, 'data', 'dragcrisis_theta_tr.json')

SEED_COLOR = {"0.05": "tab:blue", "0.2": "tab:orange", "0.7": "tab:purple"}
CRISIS_BAND = (3e5, 7e5)                      # shading as in fig:dragcrisiscd
REP = [("cyl_Re10000_Tu0.2_dn", "$10^4$"),
       ("cyl_Re60000_Tu0.2_up", r"$6\times10^4$"),
       ("cyl_Re200000_Tu0.2_up", r"$2\times10^5$"),
       ("cyl_Re500000_Tu0.2_up", r"$5\times10^5$"),
       ("cyl_Re2000000_Tu0.2_up", r"$2\times10^6$"),
       ("cyl_Re20000000_Tu0.2_up_highre", r"$2\times10^7$")]


def authoritative(c):
    """Keep the per-Re authoritative mesh family (0412 statement)."""
    if c["re"] < 1e4:
        return False                          # creeping/low: far-wake angles
    if c["mesh"] == "pilot":
        return c["re"] <= 2e6
    if c["mesh"] == "highre":
        return c["re"] >= 4e6
    return False


def series(cases, tu, branch, key):
    pts = [(c["re"], c[key]) for c in cases.values()
           if c["Tu"] == tu and c["dir"] == branch and authoritative(c)
           and c.get(key) is not None]
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def main():
    cases = json.load(open(JSON))["cases"]
    plt.rcParams.update({"font.size": 11})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.4),
                                 gridspec_kw=dict(wspace=0.22))

    # ---- (a) angles vs Re ----
    ax.axvspan(*CRISIS_BAND, color="0.92", zorder=0)
    for tu, col in SEED_COLOR.items():
        for branch, ls, mfc in (("up", "-", col), ("dn", "--", "none")):
            re_, th = series(cases, tu, branch, "theta_tr_chi1")
            ax.plot(re_, th, ls, color=col, marker="o", ms=5, mfc=mfc,
                    lw=1.6, zorder=4)
            for key, mk in (("sep_first", "v"), ("sep_final", "d")):
                re_, sp = series(cases, tu, branch, key)
                ax.plot(re_, sp, ls, color=col, marker=mk, ms=3.5, mfc=mfc,
                        lw=0.7, alpha=0.65, zorder=3)
    # line-class legend (gray), seed identity via text labels
    leg = [plt.Line2D([], [], color="0.3", ls="-", marker="o", ms=5,
                      label=r"$\theta_{tr}$ up"),
           plt.Line2D([], [], color="0.3", ls="--", marker="o", ms=5,
                      mfc="none", label=r"$\theta_{tr}$ dn"),
           plt.Line2D([], [], color="0.3", ls="-", marker="v", ms=3.5,
                      lw=0.7, label="first separation"),
           plt.Line2D([], [], color="0.3", ls="-", marker="d", ms=3.5,
                      lw=0.7, label="final separation")]
    ax.legend(handles=leg, loc="upper left", fontsize=9, framealpha=0.9)
    ax.text(1.1e7, 91, "$Tu\\,0.05\\%$", color=SEED_COLOR["0.05"], fontsize=10)
    ax.text(1.1e7, 84.5, "$Tu\\,0.2\\%$", color=SEED_COLOR["0.2"], fontsize=10)
    ax.text(1.1e7, 71, "$Tu\\,0.7\\%$", color=SEED_COLOR["0.7"], fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(8e3, 4e7)
    ax.set_ylim(60, 145)
    ax.set_xlabel("$Re_D$")
    ax.set_ylabel(r"angle from forward stagnation [deg]")
    ax.grid(alpha=0.25)

    # ---- (b) max-chi(theta) at representative Re ----
    cmap = plt.cm.viridis(np.linspace(0.0, 0.92, len(REP)))
    meta_theta = None
    for (name, lab), col in zip(REP, cmap):
        c = cases[name]
        prof = np.asarray(c["profiles"]["log10_maxchi"], float)
        th = np.linspace(0.0, 180.0, len(prof))
        meta_theta = th
        bx.plot(th, 10.0 ** prof, "-", color=col, lw=1.4, label=lab)
    bx.axhline(1.0, color="k", lw=0.9, ls=":")
    bx.set_yscale("log")
    bx.set_xlim(0, 180)
    bx.set_ylim(1e-3, 3e4)
    bx.set_xticks([0, 30, 60, 90, 120, 150, 180])
    bx.set_xlabel(r"angle from forward stagnation [deg]")
    bx.set_ylabel(r"$\max_s \chi$ along radial ray")
    bx.legend(loc="lower right", fontsize=9, title="$Re_D$",
              title_fontsize=9, framealpha=0.9)
    bx.grid(alpha=0.25)

    fig.savefig(os.path.join(OUT, "dragcrisis_angles.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(PREV, "dragcrisis_angles.png"), dpi=130,
                bbox_inches="tight")
    print(f"wrote {OUT}/dragcrisis_angles.pdf")


if __name__ == "__main__":
    main()
