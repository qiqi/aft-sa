"""fig:dragcrisisfpg -> paper/figs/dragcrisis_fpg.pdf.

Two panels from the committed FPG-audit JSON
(figs_explore/fpg_rate_audit.json, built by fpg_rate_audit.py; nothing
recomputed here):
  (a) Falkner-Skan favorable-gradient ladder: the model's frozen-profile
      envelope slope dN/dRe_theta (mean-secant and late-window) against
      the Drela-Giles and mfoil envelope fits, log scale in the slope.
      Model points that fall below the axis floor are drawn at the floor
      with a downward marker.
  (b) Cylinder nose amplification-budget N(theta) at Re_D = 2e6/7e6/2e7
      (middle seed): Drela envelope integrated on the marched laminar
      nose flow (solid) vs the model's frozen-profile instrument on the
      same march (dashed); the chi=1 budget N = ln(1/chi_inf) = 4.53 and
      the solver front angles are marked.
House style: line plots, no in-figure titles, caption carries the story.
Run from anywhere: python3 repro/cfd/regen_dragcrisis_fpg.py
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
JSON = os.path.join(PREV, 'fpg_rate_audit.json')

N_BUDGET = 4.53                     # ln(1/chi_inf), Tu = 0.2% seed
FLOOR = 1e-6                        # axis floor for dead model rates
RE_COLOR = {2e6: "tab:blue", 7e6: "tab:orange", 2e7: "tab:purple"}
RE_LAB = {2e6: r"$2\times10^6$", 7e6: r"$7\times10^6$",
          2e7: r"$2\times10^7$"}


def main():
    d = json.load(open(JSON))
    plt.rcParams.update({"font.size": 11})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.2),
                                 gridspec_kw=dict(wspace=0.25))

    # ---- (a) FS ladder: envelope slope vs beta ----
    lad = d["ladder"]
    beta = np.array([e["beta"] for e in lad])
    drela = np.array([e["drela_dNdRt"] for e in lad])
    mfoil = np.array([e["mfoil_dNdRt"] for e in lad])
    s_mean = np.array([e["s_mean"] if e["s_mean"] == e["s_mean"]
                       else 0.0 for e in lad])
    s_late = np.array([e["s_late"] if e["s_late"] == e["s_late"]
                       else 0.0 for e in lad])
    ax.plot(beta, drela, "-o", color="0.25", lw=1.6, ms=5,
            label="Drela--Giles envelope")
    ax.plot(beta, mfoil, "--s", color="0.55", lw=1.2, ms=4.5,
            label="mfoil fit")
    dead = s_mean < FLOOR
    ax.plot(beta[~dead], s_mean[~dead], "-o", color="tab:red", lw=1.6,
            ms=5, label="model, mean secant")
    ax.plot(beta[~dead], np.where(s_late[~dead] > FLOOR,
                                  s_late[~dead], np.nan),
            "--o", color="tab:red", lw=1.1, ms=4, mfc="none",
            label="model, late window")
    ax.plot(beta[dead], np.full(dead.sum(), FLOOR), "v",
            color="tab:red", ms=7, mfc="none")
    ax.set_yscale("log")
    ax.set_ylim(5e-7, 3e-2)
    ax.set_xlabel(r"Falkner--Skan $\beta$")
    ax.set_ylabel(r"$dN/dRe_\theta$")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    keep = [0, 3, 4, 5]                # beta = 0, 0.2, 0.5, 1: uncrowded
    top = ax.secondary_xaxis("top")
    top.set_xticks(beta[keep])
    top.set_xticklabels([f"{lad[i]['H']:.2f}" for i in keep], fontsize=8)
    top.set_xlabel("$H$", fontsize=9)

    # ---- (b) cylinder nose N(theta) ----
    for c in d["cylinder"]:
        col = RE_COLOR[c["re"]]
        th = np.asarray(c["st_theta"], float)
        bx.plot(c["theta_deg"], c["N_drela"], "-", color=col, lw=1.6)
        bx.plot(th, c["N_model_sup"], "--", color=col, lw=1.2)
        bx.axvline(c["chi1_front_solver"], color=col, lw=0.9, ls=":",
                   ymin=0, ymax=1)
        bx.plot([], [], "-", color=col, label=RE_LAB[c["re"]])
    bx.axhline(N_BUDGET, color="k", lw=0.9, ls="-.")
    bx.text(4, N_BUDGET + 0.12, r"$N=\ln(1/\chi_\infty)=4.53$",
            fontsize=9)
    bx.plot([], [], "-", color="0.3", label="Drela envelope")
    bx.plot([], [], "--", color="0.3", label="model instrument")
    bx.plot([], [], ":", color="0.3", label="solver front")
    bx.set_xlim(0, 95)
    bx.set_ylim(0, 6.2)
    bx.set_xlabel(r"angle from forward stagnation [deg]")
    bx.set_ylabel(r"$N$ on the marched nose flow")
    bx.legend(loc="upper left", fontsize=9, framealpha=0.9, ncol=2)
    bx.grid(alpha=0.25)

    fig.savefig(os.path.join(OUT, "dragcrisis_fpg.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(PREV, "dragcrisis_fpg.png"), dpi=130,
                bbox_inches="tight")
    print(f"wrote {OUT}/dragcrisis_fpg.pdf")


if __name__ == "__main__":
    main()
