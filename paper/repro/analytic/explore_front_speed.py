"""Front-speed sweeps and figures -> paper/figs/front_*.pdf.

Drives front_travelling_wave.py. Three figures, all from the same cached
fields:

    transition_front_contours.pdf
                           log10(chi) line contours of the travelling wave on
                           frozen Blasius at Re_theta = 150..4800, one panel
                           each, annotated with the front speed over the edge
                           velocity and over u_tau. Drawn at page width so the
                           in-plot type prints at caption size. The paper's
                           Figure 5; the line styles are named in its caption.
    front_trueaspect.pdf   the Re_theta = 600 field with the axes to scale --
                           a ~4 deg ramp about 20 delta_99 long
    front_locus_speed.pdf  the locus nuHat = nuHat_eq/2 in outer and in wall
                           units, and c against Re_x

and, behind flags, the sweeps that back the numbers quoted in
notes_front_propagation.md:

    --family     Falkner-Skan beta = +0.10 .. -0.1988 at Re_theta = 600
    --backflow   the reverse-flow knob that locates the c = 0 crossing
    --grid       nx / ny / y_max / domain-length / scheme-order convergence
    --gate       amplification kernel on vs off

Fields are cached in sa-ai/repro/front_cache/ -- OUTSIDE paper/repro/, which
holds no large binaries (see paper/repro/README.md, "No large binaries here").
Override with $SAAI_FRONT_CACHE. ~0.9 MB each, gitignored; delete to force a
recompute. A cold run of the six figure cases is ~15 min on 6 cores; replotting
from cache is seconds.

EXPLORATORY, not part of regenerate_all.py: nothing in the paper cites these
figures yet. Findings live in notes_front_propagation.md.

Run from paper/: python3 repro/analytic/explore_front_speed.py [--flags]
"""
import _saai  # noqa: F401
import argparse
import os
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import front_travelling_wave as ftw
from front_travelling_wave import front_position

HERE = os.path.dirname(os.path.abspath(__file__))
# Cached fields live OUTSIDE paper/repro/ -- that folder carries code and text
# only (paper/repro/README.md, "No large binaries here").
CACHE = os.environ.get(
    "SAAI_FRONT_CACHE",
    os.path.join(os.path.dirname(_saai.PAPER), "repro", "front_cache"))
FIGS = "figs"                      # _saai has chdir'd to paper/
RETS = [150.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0]
CV1 = 7.1
YHI = 1.6
# The WHOLE domain is plotted, xi measured from the inflow, so both
# boundaries are in frame and their flatness is visible rather than
# asserted. Cropping to a window around the front hides exactly the
# thing that needs checking.
MINOR = np.arange(-2.5, 3.01, 1.0)
MAJOR = np.arange(-2.0, 3.01, 1.0)


# --------------------------------------------------------------- computation
def _cache_path(Re_theta):
    return os.path.join(CACHE, f"blasius_Re{int(Re_theta)}.npz")


def compute(Re_theta):
    """Solve one Blasius case and cache it. Returns nothing: this runs in a
    Pool worker and an open NpzFile cannot be pickled back to the parent."""
    os.makedirs(CACHE, exist_ok=True)
    f = _cache_path(Re_theta)
    if os.path.exists(f):
        return
    p, nh_eq, r = ftw.solve(0.0, Re_theta, gate=False, ny=220, nx=1000,
                            Lfac=90.0, nsteps=40000)
    if not r["ok"]:
        raise RuntimeError(f"Re_theta={Re_theta}: {r.get('reason', 'failed')}")
    np.savez_compressed(f, xi=r["xi"], y=r["y"], nh=r["nh"], nh_eq=nh_eq,
                        u=p["u"], c=r["c"], yc=r["yc"], d99=p["d99"],
                        utau=p["utau"], Ret=Re_theta, Rex=p["Rex"],
                        drift=r["drift"], xfy=r["xfy"])
    print(f"  Re_theta={Re_theta:6.0f}  c/Ue={r['c']:.5f}  "
          f"c/utau={r['c']/p['utau']:.2f}  yc+={r['yc']*p['utau']:.2f}",
          flush=True)


def field(Re_theta):
    """The cached field for one case, computing it first if absent."""
    compute(Re_theta)
    return np.load(_cache_path(Re_theta))


def report(tag, beta, Re_theta, **kw):
    """One line of a sweep table."""
    gate = kw.pop("gate", False)
    bf = kw.pop("backflow", 0.0)
    p, nh_eq, r = ftw.solve(beta, Re_theta, gate=gate, backflow=bf, **kw)
    if not r["ok"]:
        print(f"{tag:>22}  {r.get('reason', 'front left the window')}")
        return
    yc = r["yc"]
    print(f"{tag:>22}  chi_eq={nh_eq.max():7.1f}  c/Ue={r['c']:+9.5f}  "
          f"c/utau={r['c']/p['utau']:7.2f}  yc+={yc*p['utau']:6.2f}  "
          f"yc/d99={yc/p['d99']:6.3f}  umin={p['u'].min():+7.4f}  "
          f"drift={r['drift']:+.1e}", flush=True)


# ------------------------------------------------------------------ plotting
def draw(ax, d, window=None, critical=True):
    xi = (d["xi"] - d["xi"][0]) / d["d99"]      # 0 at the inflow boundary
    y = d["y"] / d["d99"]
    X, Y = np.meshgrid(xi, y, indexing="ij")
    L = np.log10(np.maximum(d["nh"], 1e-12))
    ax.contour(X, Y, L, levels=MINOR, colors="0.72", linewidths=0.5)
    ax.contour(X, Y, L, levels=MAJOR, colors="0.35", linewidths=0.8)
    ax.contour(X, Y, L, levels=[np.log10(CV1)], colors="tab:red",
               linewidths=1.4, linestyles="--")
    ax.contour(X, Y, L, levels=[0.0], colors="k", linewidths=2.2)
    yc = float(d["yc"]) / float(d["d99"])
    if critical:                 # the height where u(y) equals the front speed
        ax.axhline(yc, color="tab:blue", lw=1.0, ls=":")
    ax.axhline(1.0, color="0.35", lw=0.8, ls="-.")
    ax.set_xlim(*(window if window else (xi[0], xi[-1])))
    ax.set_ylim(0, YHI)
    return yc


def save(fig, stem, png):
    ftw.save_pdf(fig, os.path.join(FIGS, stem + ".pdf"))
    if png:                      # preview copies only; the PDF is the artifact
        fig.savefig(os.path.join(FIGS, stem + ".png"), dpi=165)


def figures(cases, png=False):
    n = len(cases)
    # 6.5 in = the paper's text width, so the PDF is placed at scale 1 and the
    # type below prints at the size declared here.
    fig, axes = plt.subplots(n, 1, figsize=(6.5, 1.0 * n + 0.45), sharex=True)
    for i, (ax, d) in enumerate(zip(axes, cases)):
        draw(ax, d, critical=False)
        c, ut = float(d["c"]), float(d["utau"])
        ax.text(0.012, 0.93,
                rf"$Re_\theta={int(d['Ret'])}$     front speed "
                rf"${c:.4f}\,U_e={c/ut:.2f}\,u_\tau$",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(fc="white", ec="0.75", pad=2.0))
        ax.set_ylabel(r"$y/\delta_{99}$", fontsize=9.5)
        ax.set_yticks([0, 0.5, 1.0, 1.5])
        ax.tick_params(labelsize=9)
    axes[-1].set_xlabel(r"streamwise distance from the inflow, in $\delta_{99}$"
                        r" (frame moving with the front; flow "
                        r"left$\rightarrow$right)", fontsize=9.5)
    fig.subplots_adjust(top=0.995, bottom=0.42 / fig.get_figheight(),
                        left=0.085, right=0.99, hspace=0.13)
    save(fig, "transition_front_contours", png)

    # ---- boundary quietness: is the solution flat at BOTH ends? ----------
    figq, (q1, q2) = plt.subplots(2, 1, figsize=(9.6, 5.4), sharex=False)
    cmq = plt.cm.viridis(np.linspace(0.05, 0.88, n))
    for col, d in zip(cmq, cases):
        xi = (d["xi"] - d["xi"][0]) / d["d99"]
        nh, eq, d99 = d["nh"], d["nh_eq"], float(d["d99"])
        cmax = nh.max()
        slope = np.abs(np.gradient(nh, d["xi"], axis=0)).max(axis=1) * d99 / cmax
        dep = np.abs(nh - eq[None, :]).max(axis=1) / cmax
        lab = rf"$Re_\theta={int(d['Ret'])}$"
        q1.semilogy(xi, np.maximum(slope, 1e-16), color=col, lw=1.3, label=lab)
        q2.semilogy(xi, np.maximum(dep, 1e-16), color=col, lw=1.3, label=lab)
    for q, ttl, yl in (
            (q1, r"(a) streamwise slope   "
                 r"$\max_y|\partial_\xi\tilde\nu|\,\delta_{99}/\chi_{\max}$",
             r"slope"),
            (q2, r"(b) departure from the downstream state   "
                 r"$\max_y|\tilde\nu-\tilde\nu_{\rm eq}|/\chi_{\max}$",
             r"departure")):
        q.axhline(1e-3, color="0.4", lw=0.9, ls="--")
        q.text(0.3, 1.3e-3, r"$10^{-3}$", fontsize=8, color="0.4")
        q.set_ylim(1e-9, 3)
        q.set_ylabel(yl, fontsize=9)
        q.set_title(ttl, fontsize=9.5)
        q.grid(alpha=0.25, lw=0.5, which="both")
        q.tick_params(labelsize=8)
    q1.legend(fontsize=7.5, frameon=False, ncol=3, loc="lower right")
    q2.set_xlabel(r"$\xi/\delta_{99}$ from the inflow "
                  r"(left edge = inflow BC, right edge = outflow BC)",
                  fontsize=9.5)
    figq.suptitle("Are the streamwise boundaries quiet? Both curves must reach "
                  "the floor AT the two edges.", fontsize=10.5)
    figq.subplots_adjust(top=0.89, bottom=0.11, left=0.085, right=0.985,
                         hspace=0.30)
    save(figq, "front_boundaries", png)

    figt, axt = plt.subplots(figsize=(9.6, 1.75))
    dm = cases[min(2, n - 1)]
    xf = (front_position(dm["nh"], dm["xi"]) - dm["xi"][0]) / dm["d99"]
    draw(axt, dm, window=(xf - 4.0, xf + 24.0))
    axt.set_aspect("equal", adjustable="box")
    axt.set_xlabel(r"$\xi/\delta_{99}$ from the inflow", fontsize=9.5)
    axt.set_ylabel(r"$y/\delta_{99}$", fontsize=9)
    axt.set_yticks([0, 1])
    axt.tick_params(labelsize=8)
    axt.set_title(rf"$Re_\theta={int(dm['Ret'])}$, ZOOM on the front at TRUE "
                  r"aspect ratio: a $\sim4^\circ$ ramp, "
                  r"$\sim20\,\delta_{99}$ long", fontsize=9.5, pad=5)
    figt.subplots_adjust(top=0.80, bottom=0.30, left=0.072, right=0.988)
    save(figt, "front_trueaspect", png)

    fig2, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11.0, 3.8))
    cm = plt.cm.viridis(np.linspace(0.05, 0.88, n))
    for col, d in zip(cm, cases):
        y, xf, ut = d["y"], d["xfy"], float(d["utau"])
        m = np.isfinite(xf)
        lab = rf"$Re_\theta={int(d['Ret'])}$"
        a1.plot(xf[m] / d["d99"], y[m] / d["d99"], color=col, lw=1.7, label=lab)
        a2.plot(xf[m] * ut, y[m] * ut, color=col, lw=1.7, label=lab)
        a2.plot([np.interp(float(d["yc"]), y, xf) * ut],
                [float(d["yc"]) * ut], "o", color=col, ms=5, zorder=5)
    a1.set_xlabel(r"$\xi_{50\%}/\delta_{99}$", fontsize=10)
    a1.set_ylabel(r"$y/\delta_{99}$", fontsize=10)
    a1.set_ylim(0, 1.6)
    a1.set_xlim(-1, 22)
    a1.set_title("front locus, outer units", fontsize=10)
    a1.legend(fontsize=8, frameon=False, loc="upper left")
    a1.grid(alpha=0.25, lw=0.5)
    a2.set_xlabel(r"$\xi_{50\%}^{+}$", fontsize=10)
    a2.set_ylabel(r"$y^{+}$", fontsize=10)
    a2.set_yscale("log")
    a2.set_ylim(1, 5e3)
    a2.set_xlim(-200, 6000)
    a2.axhline(14.8, color="k", lw=0.9, ls=":")
    a2.text(-150, 16.5, r"$y^+=14.8$   ($\bullet$ = critical layer)",
            fontsize=8, va="bottom")
    a2.set_title("front locus, wall units", fontsize=10)
    a2.grid(alpha=0.25, lw=0.5, which="both")

    Rex = np.array([float(d["Rex"]) for d in cases])
    c = np.array([float(d["c"]) for d in cases])
    ut = np.array([float(d["utau"]) for d in cases])
    Ret = np.array([float(d["Ret"]) for d in cases])
    a3.loglog(Rex, c, "o", color="tab:blue", ms=6, label="computed")
    rr = np.logspace(np.log10(Rex.min() / 3), np.log10(Rex.max() * 3), 50)
    a3.loglog(rr, 8.53 * rr**-0.25, "k--", lw=1.1, label=r"$8.53\,Re_x^{-1/4}$")
    a3.set_xlabel(r"$Re_x$", fontsize=10)
    a3.set_ylabel(r"$c/U_e$", fontsize=10)
    a3.legend(fontsize=9, frameon=False, loc="lower left")
    a3.grid(alpha=0.25, which="both", lw=0.5)
    ai = a3.inset_axes([0.52, 0.60, 0.45, 0.36])
    ai.semilogx(Ret, c / ut, "o-", color="tab:red", ms=3.5, lw=1.0)
    ai.axhline(14.8, color="k", ls=":", lw=0.9)
    ai.set_ylim(14.3, 15.3)
    ai.set_ylabel(r"$c/u_\tau$", fontsize=7.5)
    ai.set_xlabel(r"$Re_\theta$", fontsize=7.5)
    ai.tick_params(labelsize=6.5)
    ai.grid(alpha=0.25, lw=0.4, which="both")
    fig2.suptitle(
        r"Left/centre: locus of $\tilde\nu=\frac{1}{2}\tilde\nu_{\rm eq}(y)$ — "
        r"the front leans downstream with height (bottom-led).     "
        r"Right: speed.", fontsize=10)
    fig2.subplots_adjust(top=0.83, bottom=0.14, left=0.06, right=0.985,
                         wspace=0.30)
    save(fig2, "front_locus_speed", png)


# ---------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="store_true")
    ap.add_argument("--backflow", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--png", action="store_true",
                    help="also emit PNG copies (for HTML previews only)")
    a = ap.parse_args(argv)

    if not a.no_figs:
        print("Blasius figure cases (cached in .front_cache/):")
        todo = [R for R in RETS
                if not os.path.exists(
                    os.path.join(CACHE, f"blasius_Re{int(R)}.npz"))]
        if todo:
            os.makedirs(CACHE, exist_ok=True)
            with Pool(min(len(todo), 6)) as pool:
                pool.map(compute, todo)
        cases = [field(R) for R in RETS]
        figures(cases, png=a.png)
        print()
        for d in cases:
            ut = float(d["utau"])
            print(f"  Re_theta={int(d['Ret']):5d}  c/Ue={float(d['c']):.5f}  "
                  f"c/utau={float(d['c'])/ut:6.2f}  "
                  f"yc+={float(d['yc'])*ut:5.2f}  "
                  f"yc/d99={float(d['yc'])/float(d['d99']):.3f}  "
                  f"chi_max={d['nh'].max():7.1f}  "
                  f"drift={float(d['drift']):+.1e}")

    if a.family:
        print("\nFalkner-Skan family, Re_theta = 600, gate off")
        for beta in (0.10, 0.0, -0.10, -0.1988):
            report(f"beta={beta:+.4f}", beta, 600.0)

    if a.backflow:
        print("\nback-flow knob on beta = -0.1988, Re_theta = 600")
        for bf in (0.0, 0.02, 0.05, 0.10, 0.20):
            report(f"backflow={bf:.2f}", -0.1988, 600.0, backflow=bf, c0=0.2)

    if a.gate:
        print("\namplification gate off vs on (Blasius, Re_theta = 600)")
        report("gate off", 0.0, 600.0, gate=False)
        report("gate on", 0.0, 600.0, gate=True)

    if a.grid:
        print("\ngrid / domain / scheme convergence (Blasius, Re_theta = 600)")
        report("nx=200", 0.0, 600.0, nx=200)
        report("nx=400 (base)", 0.0, 600.0, nx=400)
        report("nx=800", 0.0, 600.0, nx=800)
        report("ny=140", 0.0, 600.0, ny=140)
        report("ny=280", 0.0, 600.0, ny=280)
        report("Lfac=60", 0.0, 600.0, Lfac=60.0, nx=600)
        report("order=1", 0.0, 600.0, order=1)


if __name__ == "__main__":
    main()
