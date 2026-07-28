"""Three-way N(x) decomposition of the spheroid alpha=0 front miss on the
CONVERGED field: why doesn't the model amplify chi to transition at the
measured 0.438 when everything environmental is clean?

Context (agent-paper-review records): the converged full-body O-grid arm A
(43k steps) is circumferentially uniform, its laminar BL IS the laminar
march at profile level (2026-07-28-1030), topology/numerics/convergence are
exonerated (0350/0413), and the model's chi = 1 front sits at 0.858 vs the
measured 0.438 (Stock 2006 Fig. 14a digitization, committed).  This script
transplants the cylinder three-way method of fpg_rate_audit.py (2026-07-28
-1041) onto the spheroid to decompose the remaining 0.42 L:

  1. March the validated axisymmetric implicit laminar BL
     (spheroid_a0_meanflow.bl_march, mode='axi', Blasius-validated) on the
     CONVERGED field's own u_e(x) from the nose to laminar separation.
  2. Along the march, three N(x) curves against the seed budget
     N = ln(1/chi_inf), chi_inf read from the case:
       a. N_drela(x): Drela-Giles 1987 Eq. 29/30 envelope
          (lib/correlations.py -- the fit the model is calibrated against),
          chain rule dN/ds = dN/dRe_theta * dRe_theta/ds on the march;
          mfoil/XFOIL envelope as the independent cross-check
          (fpg_rate_audit.mfoil_envelope, cross-checked against a direct
          get_damp call at import of that machinery);
       b. N_model(x): the model's frozen-profile instrument on the SAME
          marched profiles (fpg_rate_audit.profile_kernel: savgol W=61
          kernel estimator, calibrated <1% on exact FS profiles) -- both
          the pointwise kernel sup-bound max_y(a S omega / u) and the gated
          c_nu_ai = 1/6 eigenvalue of eq:frozeneig, integrated over arc
          length;
       c. N_transported(x): the ACTUAL solver field's near-wall
          ln(chi_max/chi_inf) along the meridian (facet-re-based ray
          extraction, spheroid_uniformity_profiles.sweep_case -- validated
          against the 0413 verdict table).
  3. Readouts: the three x(N = budget) stations; the model/Drela rate ratio
     along the body (FS-ladder context: 0.74-0.84 at the H = 2.48-2.59
     class); the transport-realization gap N_model - N_transported; the
     seed lever x(N) table (where the front would sit at N = 9, 7, ...)
     and the N Drela accumulates by the measured 0.438 (the disturbance
     level the measurement implies within envelope physics).

Primary case (converged, arm A):
  /local_data/qiqi/sa-ai/spheroid_fv1/case_ogridfull_L1_saai_re72a0
Cross-check (transported curve + front only):
  /local_data/qiqi/sa-ai/spheroid_fv1/case_ogrid_L1_saai_re72a0_ext

Figure convention (USER DIRECTIVE 2026-07-28): the paper's AIRFOIL dual-axis
style (regen_nlf_v2.make_cf_figure row 3 / regen_eppler_v2): transported chi
on a LOG left axis, envelope N on a LINEAR right axis, axes MATCHED so
N = ln(chi/chi_inf) aligns exactly (left limits chi_inf*exp(N_LO..N_HI)),
chi = 1 and the budget line coincide by construction and are marked once.
House style: no in-figure titles; caption in the companion md.

Outputs (caption in the companion record
agent-paper-review/2026-07-28-*-spheroid-threeway.md):
  paper/figs/spheroid_threeway_N.pdf            (paper-quality; NO tex edits)
  paper/repro/cfd/figs_explore/spheroid_threeway_N.png
  paper/repro/cfd/figs_explore/spheroid_threeway.json      (all numbers)

Run:  python3 -u paper/repro/cfd/spheroid_threeway.py   (CPU only, ~10 min)
"""
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figs_explore")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (HERE, os.path.join(REPO, "paper", "repro"),
          os.path.join(REPO, "paper", "repro", "analytic"),
          os.path.join(REPO, "spheroid")):
    sys.path.insert(0, p)

# the cylinder three-way instruments, reused verbatim (canon-constant
# asserts + mfoil formula cross-check live in that module)
from fpg_rate_audit import (                          # noqa: E402
    profile_kernel, mfoil_envelope, mfoil_crosscheck, A_MAX, C_NU_AI, C_V1)
from spheroid_uniformity_profiles import (            # noqa: E402
    sweep_case, ue_peak, GATE_A)
from spheroid_fullbody_check import (                 # noqa: E402
    load_grid, MESH_FULL)
from spheroid_unstruct_a0_verdict import (            # noqa: E402
    load_wall_facets, forces_tail)
from spheroid_a0_physics import front_crossing        # noqa: E402
from spheroid_a0_meanflow import (                    # noqa: E402
    bl_march, spheroid_ue_march_grid, ellipse_geo, marcher_validation)
from lib.correlations import dN_dRe_theta, Re_theta0  # noqa: E402

SPH_ROOT = os.environ.get("SAAI_SPH_ROOT",
                          "/local_data/qiqi/sa-ai/spheroid_fv1")
CASE_A = os.path.join(SPH_ROOT, "case_ogridfull_L1_saai_re72a0")
CASE_X = os.path.join(SPH_ROOT, "case_ogrid_L1_saai_re72a0_ext")
STOCK14A = os.path.join(REPO, "paper", "data",
                        "stock2006_fig14a_digitized.json")
FPG_JSON = os.path.join(FIGD, "fpg_rate_audit.json")   # FS-ladder context
FIG_PDF = os.path.join(REPO, "paper", "figs", "spheroid_threeway_N.pdf")

ST_X = np.round(np.arange(0.04, 0.876, 0.01), 3)   # instrument stations
N_LEVELS = (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0)
KEY_X = (0.10, 0.20, 0.30, 0.42, 0.55, 0.70, 0.80, 0.85)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 110, "savefig.dpi": 150})


def read_seed(case_dir):
    """chi_inf from the case's own Flow360.json + ai_constants echo.
    Physical seed = BC seed / ai_laminarSlowdown (this campaign runs
    slowdown = 1, i.e. NO pre-compensation -- verified, not assumed)."""
    fj = json.load(open(os.path.join(case_dir, "Flow360.json")))
    tq = fj["freestream"]["turbulenceQuantities"]
    assert tq["modelType"] == "ModifiedTurbulentViscosityRatio"
    chi_bc = float(tq["modifiedTurbulentViscosityRatio"])
    for bc in fj["boundaries"].values():
        if bc.get("type") == "Freestream":
            assert float(bc["turbulenceQuantities"][
                "modifiedTurbulentViscosityRatio"]) == chi_bc
    slowdown, echo = None, os.path.join(case_dir, "ai_constants.log")
    consts = {}
    if os.path.exists(echo):
        for ln in open(echo):
            mm = re.search(r"ai_(\w+):\s+([-0-9.]+)", ln)
            if mm:
                consts[mm.group(1)] = float(mm.group(2))
        slowdown = consts.get("laminarSlowdown")
    mu_ref = float(fj["freestream"]["muRef"])
    mach = float(fj["freestream"]["Mach"])
    return dict(chi_bc=chi_bc, slowdown=slowdown, consts=consts,
                mu_ref=mu_ref, mach=mach, re_L=mach / mu_ref)


def crossing(xv, Nv, level):
    xv, Nv = np.asarray(xv, float), np.asarray(Nv, float)
    hit = np.where(Nv >= level)[0]
    if len(hit) and hit[0] > 0:
        j = hit[0]
        return float(np.interp(level, Nv[j - 1:j + 1], xv[j - 1:j + 1]))
    return float("nan")


def transported_curve(case_dir, src, mesh_cgns, mu_ref, chi_inf):
    """Near-wall max-chi along the phi=90 meridian (facet-re-based rays,
    y <= 0.02 band) -> x, chi_max, N = ln(chi/chi_inf), fronts, sweep."""
    grid = load_grid(case_dir, src,
                     slice_name="meridian" if src == "slice" else None)
    facets = load_wall_facets(mesh_cgns)
    sw = sweep_case(grid, mu_ref, facets, phi_deg=90.0)
    x = np.array([s["x"] for s in sw])
    chim = np.array([s["chimax_nearwall"] for s in sw])
    out = dict(x=x, chi=chim, N=np.log(chim / chi_inf),
               front_chi1=front_crossing(x, chim, 1.0),
               front_cv1=front_crossing(x, chim, C_V1),
               nose_chi_ratio=float(np.median(
                   chim[(x >= 0.02) & (x <= 0.10)]) / chi_inf))
    return out, sw


def run_march(sw, nu):
    """Axisymmetric laminar march on the field's own u_e; truncate at
    laminar separation (the fpg_rate_audit cylinder convention)."""
    xm, ue_m = spheroid_ue_march_grid(sw)
    m = bl_march(xm, ue_m, nu, "axi", x_profiles=list(ST_X))
    bad = np.where((m["cf_theta"] <= 1e-4) | (m["H"] >= 4.5))[0]
    x_sep = float(m["x"][bad[0]]) if len(bad) else float(m["x"][-1])
    if len(bad):
        j = int(bad[0])
        for k in ("x", "theta", "dstar", "H", "cf_theta"):
            m[k] = m[k][:j]
        m["profiles"] = {k: v for k, v in m["profiles"].items()
                         if k < x_sep - 1e-9}
    m["ue"] = np.interp(m["x"], xm, ue_m)
    m["x_sep"] = x_sep
    return m


def envelope_N(m, nu):
    """Drela-Giles + mfoil envelope N on the march grid (chain rule)."""
    s, _, _ = ellipse_geo(m["x"])
    Rt = m["ue"] * m["theta"] / nu
    H = m["H"]
    Rt0 = np.asarray(Re_theta0(H))
    dDG = np.asarray(dN_dRe_theta(H))
    dRt_ds = np.gradient(Rt, s)
    dN = np.where(Rt > Rt0, np.maximum(dDG * dRt_ds, 0.0), 0.0)
    N_dg = np.concatenate([[0.0], np.cumsum(
        0.5 * (dN[1:] + dN[:-1]) * np.diff(s))])
    da, _, Rtc = mfoil_envelope(H)
    dNm = np.where(Rt > Rtc, np.maximum(da * dRt_ds, 0.0), 0.0)
    N_mf = np.concatenate([[0.0], np.cumsum(
        0.5 * (dNm[1:] + dNm[:-1]) * np.diff(s))])
    return dict(s=s, Rt=Rt, H=H, Rt0=Rt0, dN_ds_dg=dN, N_dg=N_dg, N_mf=N_mf,
                onset_x_dg=front_crossing(m["x"], Rt - Rt0, 0.0),
                onset_x_mf=front_crossing(m["x"], Rt - Rtc, 0.0))


def model_N(m, env, nu):
    """Frozen-profile instrument on the marched profiles at ST_X: kernel
    sup-bound and gated c_nu_ai=1/6 eigenvalue, integrated over arc."""
    rows = []
    for xq in ST_X:
        if xq >= m["x_sep"] - 1e-9 or not m["profiles"]:
            break
        key = min(m["profiles"], key=lambda v: abs(v - xq))
        if abs(key - xq) > 5e-3:
            continue
        y, u, ue = m["profiles"][key]
        kk = profile_kernel(y, u, ue, nu)
        kk["x"] = float(xq)
        kk["H"] = float(np.interp(xq, m["x"], m["H"]))
        kk["Rt"] = float(np.interp(xq, m["x"], env["Rt"]))
        kk["dNds_dg"] = float(np.interp(xq, m["x"], env["dN_ds_dg"]))
        rows.append(kk)
    st_x = np.array([r["x"] for r in rows])
    st_s, _, _ = ellipse_geo(st_x)
    sup = np.array([r["sup"] for r in rows])
    eig = np.clip(np.array([r["s_eig"] for r in rows]), 0.0, None)
    N_sup = np.concatenate([[0.0], np.cumsum(
        0.5 * (sup[1:] + sup[:-1]) * np.diff(st_s))])
    N_eig = np.concatenate([[0.0], np.cumsum(
        0.5 * (eig[1:] + eig[:-1]) * np.diff(st_s))])
    return dict(rows=rows, x=st_x, sup=sup, eig=eig,
                N_sup=N_sup, N_eig=N_eig)


def stock_fronts():
    d = json.load(open(STOCK14A))
    meas = float(np.median([r["xL"] for r in d["measured_squares"]]))
    ts = d["computed_ts_front"]
    xl = ts.get("xL") or ts.get("x_L") or ts.get("xl")
    ts_front = float(np.median(np.asarray(xl, float)))
    return meas, ts_front, d["source"]


def mack_tu_from_budget(N_chi1):
    """Invert eq:tumap: chi_inf = c_v1 exp(-N_crit), N_crit = -8.43
    - 2.4 ln(Tu_frac); budget N(chi=1) = N_crit - ln(c_v1)."""
    N_crit = N_chi1 + np.log(C_V1)
    return float(np.exp(-(N_crit + 8.43) / 2.4))


def figure(tr_A, tr_X, m, env, mod, chi_inf, n_budget, meas, ts_front, fp):
    """Airfoil dual-axis convention (regen_nlf_v2 row 3): chi log-left,
    N linear-right, matched so N = ln(chi/chi_inf) aligns exactly."""
    N_LO, N_HI = -1.0, 14.0
    fig, ax_n = plt.subplots(figsize=(6.4, 4.3))
    ax_N = ax_n.twinx()
    ax_n.semilogy(tr_A["x"], tr_A["chi"], color="C0", lw=1.8,
                  label=r"SA-AI transported $\chi$ (near-wall max,"
                        " converged)")
    ax_n.semilogy(tr_X["x"], tr_X["chi"], color="C0", lw=0.9, ls="--",
                  alpha=0.6, label="cross-check case (ext)")
    ax_N.plot(m["x"], env["N_dg"], ":", color="C3", lw=1.8,
              label="Drela-Giles envelope $N$ (laminar march)")
    ax_N.plot(m["x"], env["N_mf"], ":", color="0.55", lw=1.0,
              label="mfoil envelope")
    ax_N.plot(mod["x"], mod["N_sup"], "--", color="C2", lw=1.6,
              label="model frozen-profile rate (kernel sup)")
    ax_N.plot(mod["x"], mod["N_eig"], "-.", color="C2", lw=1.0, alpha=0.7,
              label=r"model gated eigenvalue ($c_{\nu}{=}1/6$)")
    # chi = 1 and the budget N = ln(1/chi_inf) coincide on matched axes
    ax_n.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax_N.text(0.98, n_budget + 0.22,
              r"$\chi{=}1$:  $N{=}" + f"{n_budget:.2f}$",
              fontsize=8, color="0.35", ha="right")
    for lev, lab in ((9.0, r"$N{=}9$"), (7.0, r"$N{=}7$")):
        ax_N.axhline(lev, color="gray", ls=":", lw=0.5, alpha=0.7)
        ax_N.text(0.985, lev + 0.18, lab, fontsize=8, color="0.35",
                  ha="right")
    # measured onset (filled) + Stock's own e^N(8) front (open): triangles
    # on the x-axis, the airfoil EXP_XTR convention
    ax_N.plot(meas, N_LO, marker="^", ms=9, color="k", clip_on=False,
              zorder=6, ls="none")
    ax_N.plot(ts_front, N_LO, marker="^", ms=9, color="k", mfc="none",
              mew=1.5, clip_on=False, zorder=6, ls="none")
    ax_n.set_xlim(0.0, 1.0)
    ax_n.set_ylim(chi_inf * np.exp(N_LO), chi_inf * np.exp(N_HI))
    ax_N.set_ylim(N_LO, N_HI)
    ax_n.grid(alpha=0.3)
    ax_n.set_xlabel(r"$x/L$")
    ax_n.set_ylabel(r"$\chi$ (log)")
    ax_N.set_ylabel(r"$N$ (linear)")
    h1, l1 = ax_n.get_legend_handles_labels()
    h2, l2 = ax_N.get_legend_handles_labels()
    h3 = [Line2D([], [], color="k", marker="^", ls="none", ms=8,
                 label="measured onset (Kreplin, via Stock Fig. 14a)"),
          Line2D([], [], color="k", marker="^", ls="none", ms=8, mfc="none",
                 mew=1.5, label=r"Stock $e^N$ ($N_{TS}{=}8$) front")]
    ax_N.legend(handles=h1 + h2 + h3, loc="upper left", frameon=False,
                fontsize=7.2)
    fig.tight_layout()
    fig.savefig(FIG_PDF)
    fig.savefig(fp)
    plt.close(fig)


def main():
    os.makedirs(FIGD, exist_ok=True)
    seed_A, seed_X = read_seed(CASE_A), read_seed(CASE_X)
    assert seed_A["chi_bc"] == seed_X["chi_bc"] == 8.76e-6
    assert seed_A["slowdown"] == 1.0, "arm A must echo laminarSlowdown"
    # ext restart leg carries no ai_constants echo; its seed is verified
    # from Flow360.json + the near-nose chi plateau below (field-level)
    chi_inf = seed_A["chi_bc"] / seed_A["slowdown"]
    n_budget = float(np.log(1.0 / chi_inf))
    nu = seed_A["mu_ref"]                      # rho_inf = 1, solver units
    print(f"chi_inf = {chi_inf:.3e} (BC {seed_A['chi_bc']:.3e} / slowdown "
          f"{seed_A['slowdown']}), N budget = ln(1/chi_inf) = "
          f"{n_budget:.3f}, Re_L = {seed_A['re_L']:.3e}", flush=True)
    print(f"canon: a_max={A_MAX}, c_nu_ai={C_NU_AI:.4f}; mfoil cross-check "
          f"{'OK' if mfoil_crosscheck() else 'FAIL'}", flush=True)
    mv = marcher_validation()                  # Blasius gate

    # ---- transported curves (solver truth) + instrument-validation gate
    tr_A, sw = transported_curve(CASE_A, "slice", MESH_FULL,
                                 seed_A["mu_ref"], chi_inf)
    xg = np.array([s["x"] for s in sw])
    H42 = float(sw[int(np.argmin(np.abs(xg - 0.42)))]["H"])
    assert abs(H42 - GATE_A["H"]["0.42"]) < 2e-3, H42
    assert abs(tr_A["front_chi1"] - GATE_A["front_chi1"]) < 1e-3
    xpk, uepk = ue_peak(sw)
    ft_A = forces_tail(CASE_A)
    print(f"arm A: front chi1 {tr_A['front_chi1']:.4f} cv1 "
          f"{tr_A['front_cv1']:.4f}  ue peak {xpk:.4f}  nose chi/chi_inf "
          f"{tr_A['nose_chi_ratio']:.3f}  CD {ft_A['CD']:.5f}", flush=True)
    tr_X, _ = transported_curve(CASE_X, "volume",
                                os.path.join(CASE_X, "mesh.cgns"),
                                seed_X["mu_ref"], chi_inf)
    ft_X = forces_tail(CASE_X)
    print(f"ext:   front chi1 {tr_X['front_chi1']:.4f} cv1 "
          f"{tr_X['front_cv1']:.4f}  nose chi/chi_inf "
          f"{tr_X['nose_chi_ratio']:.3f}  CD {ft_X['CD']:.5f}", flush=True)

    # ---- march + the two instrument curves
    m = run_march(sw, nu)
    print(f"march separation x = {m['x_sep']:.4f}", flush=True)
    env = envelope_N(m, nu)
    mod = model_N(m, env, nu)

    # ---- crossings of the budget + seed-lever table
    cross = dict(
        drela=crossing(m["x"], env["N_dg"], n_budget),
        mfoil=crossing(m["x"], env["N_mf"], n_budget),
        model_sup=crossing(mod["x"], mod["N_sup"], n_budget),
        model_eig=crossing(mod["x"], mod["N_eig"], n_budget),
        transported=tr_A["front_chi1"])
    lever = {f"N{n:g}": crossing(m["x"], env["N_dg"], n) for n in N_LEVELS}
    lever[f"N{n_budget:.3f}"] = cross["drela"]
    # the model's OWN seed lever: where the transported curve crosses a
    # reduced budget (first order in seed linearity; analysis only)
    lever_tr = {f"N{n:g}": crossing(tr_A["x"], tr_A["N"], n)
                for n in N_LEVELS}
    lever_tr[f"N{n_budget:.3f}"] = tr_A["front_chi1"]
    meas, ts_front, stock_src = stock_fronts()
    N_at_meas = float(np.interp(meas, m["x"], env["N_dg"]))
    N_mf_at_meas = float(np.interp(meas, m["x"], env["N_mf"]))
    tu_impl = mack_tu_from_budget(N_at_meas)
    chi_impl = float(np.exp(-N_at_meas))
    print(f"budget crossings: Drela {cross['drela']:.3f}  mfoil "
          f"{cross['mfoil']:.3f}  model sup {cross['model_sup']:.3f}  eig "
          f"{cross['model_eig']:.3f}  transported {cross['transported']:.4f}",
          flush=True)
    print("seed lever x(N), Drela:       " + "  ".join(
        f"{k}={v:.3f}" for k, v in lever.items()), flush=True)
    print("seed lever x(N), transported: " + "  ".join(
        f"{k}={v:.3f}" for k, v in lever_tr.items()), flush=True)
    print(f"Drela N at measured {meas:.4f}: {N_at_meas:.2f} (mfoil "
          f"{N_mf_at_meas:.2f}) -> implied chi_inf {chi_impl:.2e}, Mack Tu "
          f"{100 * tu_impl:.3f}%", flush=True)

    # ---- rate ratio + transport gap at key stations
    key_rows = []
    for xq in KEY_X:
        j = int(np.argmin(np.abs(mod["x"] - xq)))
        r = mod["rows"][j]
        dg = r["dNds_dg"]
        key_rows.append(dict(
            x=float(mod["x"][j]), H=r["H"], Rt=r["Rt"], maxP=r["maxP"],
            gate=r["gate_max"], sup_perL=r["sup"], eig_perL=float(
                max(r["s_eig"], 0.0)), dNds_drela=dg,
            ratio_sup=(r["sup"] / dg if dg > 0 else float("nan")),
            ratio_eig=(max(r["s_eig"], 0.0) / dg if dg > 0
                       else float("nan")),
            N_dg=float(np.interp(mod["x"][j], m["x"], env["N_dg"])),
            N_sup=float(mod["N_sup"][j]), N_eig=float(mod["N_eig"][j]),
            N_tr=float(np.interp(mod["x"][j], tr_A["x"], tr_A["N"]))))
        k = key_rows[-1]
        print(f"  x={k['x']:.2f} H={k['H']:.3f} Rt={k['Rt']:.0f} "
              f"maxP={k['maxP']:.3f} sup={k['sup_perL']:.1f}/L "
              f"DG={k['dNds_drela']:.1f}/L ratio={k['ratio_sup']:.2f} "
              f"(eig {k['ratio_eig']:.2f}) | N: DG {k['N_dg']:.2f} "
              f"sup {k['N_sup']:.2f} eig {k['N_eig']:.2f} "
              f"tr {k['N_tr']:.2f}", flush=True)
    xf = tr_A["front_chi1"]
    at_front = dict(
        x=xf,
        N_drela=float(np.interp(xf, m["x"], env["N_dg"])),
        N_model_sup=float(np.interp(xf, mod["x"], mod["N_sup"])),
        N_model_eig=float(np.interp(xf, mod["x"], mod["N_eig"])),
        N_transported=n_budget)
    at_meas = dict(
        x=meas, N_drela=N_at_meas, N_mfoil=N_mf_at_meas,
        N_model_sup=float(np.interp(meas, mod["x"], mod["N_sup"])),
        N_model_eig=float(np.interp(meas, mod["x"], mod["N_eig"])),
        N_transported=float(np.interp(meas, tr_A["x"], tr_A["N"])))
    # segment bookkeeping: forward (nose-0.70, on-calibration class) vs the
    # aft steepening-adverse run (0.70-front) where the envelope blows up
    seg = {}
    for xq in (0.70, xf):
        seg[f"{xq:.3f}"] = dict(
            N_drela=float(np.interp(xq, m["x"], env["N_dg"])),
            N_eig=float(np.interp(xq, mod["x"], mod["N_eig"])),
            N_sup=float(np.interp(xq, mod["x"], mod["N_sup"])),
            N_tr=float(np.interp(xq, tr_A["x"], tr_A["N"])))
    print("segments: " + json.dumps(seg), flush=True)
    print(f"at solver front {xf:.4f}: DG {at_front['N_drela']:.2f}, "
          f"sup {at_front['N_model_sup']:.2f}, eig "
          f"{at_front['N_model_eig']:.2f}, transported = budget "
          f"{n_budget:.2f}", flush=True)
    print(f"at measured {meas:.4f}: DG {at_meas['N_drela']:.2f}, sup "
          f"{at_meas['N_model_sup']:.2f}, eig {at_meas['N_model_eig']:.2f},"
          f" transported {at_meas['N_transported']:.2f}", flush=True)

    # FS-ladder context (committed fpg_rate_audit.json, 1041 record)
    ladder_ctx = None
    if os.path.exists(FPG_JSON):
        lad = json.load(open(FPG_JSON)).get("ladder", [])
        ladder_ctx = {f"beta{r['beta']:g}": dict(
            H=r["H"], ratio_mean=r["ratio_mean"], ratio_late=r["ratio_late"])
            for r in lad if r["beta"] <= 0.10}

    figure(tr_A, tr_X, m, env, mod, chi_inf, n_budget, meas, ts_front,
           os.path.join(FIGD, "spheroid_threeway_N.png"))
    print(f"wrote {FIG_PDF}", flush=True)

    out = dict(
        cases=dict(A=CASE_A, ext=CASE_X),
        seed=dict(chi_bc=seed_A["chi_bc"], slowdown_A=seed_A["slowdown"],
                  slowdown_ext_echo=seed_X["slowdown"], chi_inf=chi_inf,
                  N_budget=n_budget,
                  nose_chi_ratio=dict(A=tr_A["nose_chi_ratio"],
                                      ext=tr_X["nose_chi_ratio"]),
                  mack_tu_equiv=mack_tu_from_budget(n_budget)),
        forces=dict(A=ft_A, ext=ft_X),
        marcher_validation=mv,
        gate=dict(H42=H42, front_A=tr_A["front_chi1"], ref=GATE_A),
        fronts=dict(A_chi1=tr_A["front_chi1"], A_cv1=tr_A["front_cv1"],
                    ext_chi1=tr_X["front_chi1"], ext_cv1=tr_X["front_cv1"],
                    measured=meas, stock_ts8=ts_front, stock_src=stock_src),
        ue_peak=dict(x=xpk, ue=uepk), march_separation=m["x_sep"],
        onset=dict(drela=env["onset_x_dg"], mfoil=env["onset_x_mf"]),
        budget_crossings=cross, seed_lever_xN=lever,
        seed_lever_xN_transported=lever_tr, segments=seg,
        measured_implies=dict(N_drela=N_at_meas, N_mfoil=N_mf_at_meas,
                              chi_inf=chi_impl, mack_tu=tu_impl),
        at_solver_front=at_front, at_measured_front=at_meas,
        key_stations=key_rows, ladder_context=ladder_ctx,
        curves=dict(
            x_march=[float(v) for v in m["x"][::20]],
            N_drela=[float(v) for v in env["N_dg"][::20]],
            N_mfoil=[float(v) for v in env["N_mf"][::20]],
            H_march=[float(v) for v in m["H"][::20]],
            Rt_march=[float(v) for v in env["Rt"][::20]],
            x_st=[float(v) for v in mod["x"]],
            N_sup=[float(v) for v in mod["N_sup"]],
            N_eig=[float(v) for v in mod["N_eig"]],
            x_tr=[float(v) for v in tr_A["x"]],
            N_tr=[float(v) for v in tr_A["N"]],
            x_tr_ext=[float(v) for v in tr_X["x"]],
            N_tr_ext=[float(v) for v in tr_X["N"]]))
    fj = os.path.join(FIGD, "spheroid_threeway.json")
    json.dump(out, open(fj, "w"), indent=1, default=float)
    print("wrote", fj, flush=True)


if __name__ == "__main__":
    main()
