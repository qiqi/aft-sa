"""Harvest the spheroid RESEED campaign (2026-07-28): the
freestream-turbulence mis-specification test.

Five converged-protocol L1 half-model O-grid runs (build_reseed_cases.py;
45k cold-start pseudo-steps, run on 017-v100-dev GPUs 2-3, synced back to
/local_data/qiqi/sa-ai/spheroid_fv1/):

  case_ogrid_L1_saai_re72a0_reseed_{n9p66,n8,n6p34}   Re 7.2e6, alpha 0
  case_ogrid_L1_saai_re65a0_reseed_n8                 Re 6.5e6, alpha 0
  case_ogrid_L1_saai_re72a2p5_reseed_n8               Re 7.2e6, alpha 2.5

Per case:
  - seed verification: Flow360.json chi (freestream + farfield BC) +
    ai_constants echo (slowdown MUST be 1.0, campaign convention) + the
    in-field near-nose chi plateau (the 1122 ritual);
  - convergence gate: CD end drift < 3e-5/1k (0350 protocol);
  - meridian sweep (the 0413/1030 instrument, imported: facet-re-based
    analytic-normal rays, near-wall chi band y <= 0.02): chi = 1 front
    + c_v1 companion, H at x/L = 0.20/0.42/0.70, nose chi ratio;
  - per-azimuth Cf-rise front from the wall surface output (k = 1.5 of
    the pre-rise minimum, RUNNING-min variant -- the 1030 fixed window
    (0.5, 0.95) assumes an aft front; re-seeded fronts sit at 0.35-0.65)
    -> front median/rms/p2p over azimuth = the crossflow/uniformity check;
  - alpha = 2.5 only: windward (mesh phi = 180, -z) and leeward (phi = 0,
    +z) meridian sweeps from the y = +1e-5 near-symmetry slice (probe
    rays built at phi = 0/180 and shifted into the slice plane; the
    1e-5 L transverse offset is ~1% of delta99).

Comparisons assembled into the output JSON:
  - measured fronts: paper/data/stock2006_fig14a_digitized.json (a0,
    Re 7.2e6: 0.4381) and ..._fig14b_... (a2.5, Re 7.2e6: squares at
    Stock-phi 0/90/180; NOTE Stock phi=0 = WINDWARD = mesh phi 180);
  - Stock's own e^N (N_TS = 8) fronts from the same digitizations;
  - the 1122 seed-lever prediction (spheroid_threeway.json x(N) tables)
    re-read at each case's budget N = ln(1/chi_inf).

Run (CPU, after sync):  python3 paper/repro/cfd/spheroid_reseed_harvest.py
-> figs_explore/spheroid_reseed_harvest.json + stdout tables
   + figs_explore/spheroid_reseed_front_vs_seed.png
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figs_explore")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (HERE, os.path.join(REPO, "paper", "repro"),
          os.path.join(REPO, "spheroid")):
    sys.path.insert(0, p)

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import vtk                                            # noqa: E402
from vtk.util.numpy_support import vtk_to_numpy      # noqa: E402

from spheroid_threeway import read_seed               # noqa: E402
from spheroid_uniformity_profiles import sweep_case   # noqa: E402
from spheroid_fullbody_check import (                 # noqa: E402
    load_grid, probe_pts, XS, CHI_BAND)
from spheroid_unstruct_a0_verdict import (            # noqa: E402
    load_wall_facets, wall_offsets, forces_tail)
from spheroid_a0_physics import front_crossing        # noqa: E402
from spheroid_uniformity_profiles import RAY          # noqa: E402
from surface_map import surface_frame                 # noqa: E402
from lib.calibrate_kernel import C_V1                 # noqa: E402

SPH_ROOT = os.environ.get("SAAI_SPH_ROOT",
                          "/local_data/qiqi/sa-ai/spheroid_fv1")
MESH_HALF = "/local_data/qiqi/sa-ai/spheroid_meshes/mesh_re65_L1.cgns"
STATIONS = (0.20, 0.42, 0.70)
Y_OFF = 1e-5                       # the near-symmetry slice plane
DRIFT_GATE = 3e-5                  # CD drift/1k convergence gate (0350)

CASES = [
    ("case_ogrid_L1_saai_re72a0_reseed_n9p66", 9.6636, False),
    ("case_ogrid_L1_saai_re72a0_reseed_n8",    8.0,    False),
    ("case_ogrid_L1_saai_re72a0_reseed_n6p34", 6.3364, False),
    ("case_ogrid_L1_saai_re65a0_reseed_n8",    8.0,    False),
    ("case_ogrid_L1_saai_re72a2p5_reseed_n8",  8.0,    True),
]


def offset_meridian_sweep(grid, mu_ref, facets, phi_deg, xs=XS):
    """sweep_case for the symmetry-plane meridians (mesh phi = 0 leeward /
    180 windward): rays built on the analytic meridian, then shifted into
    the y = +1e-5 slice plane (n3 and t_s are in-plane at these phi)."""
    from spheroid_uniformity_profiles import edge_and_integrals
    xs = np.asarray(xs, float)
    P, n3, t_s, _ = surface_frame(xs, np.full_like(xs, np.radians(phi_deg)))
    t0 = wall_offsets(P, n3, *facets) if facets is not None else \
        np.zeros(len(xs))
    P = P + t0[:, None] * n3
    pts = np.concatenate([P[k] + RAY[:, None] * n3[k] for k in range(len(xs))])
    pts[:, 1] = Y_OFF                       # into the slice plane
    res, valid = probe_pts(grid, pts)
    ny = len(RAY)
    u_all = res["velocity"].astype(float)
    rho_all = res.get("rho", np.ones(len(pts))).astype(float)
    nuhat = (res["solutionTurbulence"] if "solutionTurbulence" in res
             else res["nuHat"]).astype(float)
    sw = []
    for k, xq in enumerate(xs):
        sl = slice(k * ny, (k + 1) * ny)
        u = u_all[sl]
        st = dict(y=RAY, U=np.linalg.norm(u, axis=1), us=u @ t_s[k],
                  nu=mu_ref / np.maximum(rho_all[sl], 1e-6),
                  chi=rho_all[sl] * nuhat[sl] / mu_ref,
                  valid=valid[sl], x=float(xq), t0=float(t0[k]))
        edge_and_integrals(st)
        st["chimax_nearwall"] = float(np.nanmax(
            np.where(RAY <= CHI_BAND, st["chi"], -np.inf)))
        sw.append(st)
    return sw


def sweep_report(sw, chi_inf):
    x = np.array([s["x"] for s in sw])
    chim = np.array([s["chimax_nearwall"] for s in sw])
    rows = {}
    for xq in STATIONS:
        s = sw[int(np.argmin(np.abs(x - xq)))]
        rows[f"{xq:g}"] = dict(H=float(s["H"]),
                               Rt=float(s["u_e"] * s["theta"]
                                        / np.median(s["nu"])),
                               d99=float(s["d99"]))
    return dict(
        front_chi1=front_crossing(x, chim, 1.0),
        front_cv1=front_crossing(x, chim, C_V1),
        nose_chi_ratio=float(np.median(
            chim[(x >= 0.02) & (x <= 0.10)]) / chi_inf),
        H=rows)


def cf_fronts_per_azimuth(case_dir, n_az=80, x_lo=0.10, x_hi=0.95, k=1.5):
    """Per-azimuth-line Cf-rise front (k x pre-rise minimum), RUNNING-min
    walk so the window needs no prior front location; structured O-grid
    surface (exact 2.25-deg node lines, the 1030 fixed grouping)."""
    fn = os.path.join(case_dir, "surface_fluid_wall.pvtu")
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(fn)
    r.Update()
    g = r.GetOutput()
    P = vtk_to_numpy(g.GetPoints().GetData())
    pd = g.GetPointData()
    arr = {pd.GetArray(i).GetName(): vtk_to_numpy(pd.GetArray(i))
           for i in range(pd.GetNumberOfArrays())}
    xL = P[:, 0] + 0.5
    cf = arr["Cf"].astype(float)
    phi_r = np.degrees(np.arctan2(P[:, 1], P[:, 2])) % 360.0
    kaz = np.round(phi_r / (180.0 / n_az)).astype(int)   # half model 0..n_az
    fronts, phis = [], []
    for kk in range(n_az + 1):
        m = kaz == kk
        if m.sum() < 50:
            continue
        o = np.argsort(xL[m])
        xm, cm = xL[m][o], cf[m][o]
        sel = (xm > x_lo) & (xm < x_hi)
        xm, cm = xm[sel], cm[sel]
        rmin = np.minimum.accumulate(cm)
        hit = np.where((cm >= k * rmin) & (xm > x_lo + 0.03))[0]
        if len(hit):
            j = hit[0]
            c_t = k * rmin[j]
            f = ((c_t - cm[j - 1]) / (cm[j] - cm[j - 1])
                 if cm[j] != cm[j - 1] else 1.0)
            fronts.append(float(xm[j - 1] + np.clip(f, 0, 1)
                                * (xm[j] - xm[j - 1])))
            phis.append(kk * 180.0 / n_az)
    fronts, phis = np.array(fronts), np.array(phis)
    return dict(n=len(fronts), median=float(np.median(fronts)),
                rms=float(fronts.std()), p2p=float(np.ptp(fronts)),
                phi=phis.tolist(), front=fronts.tolist())


def main():
    facets = load_wall_facets(MESH_HALF)
    out = {"cases": {}}
    for name, n_crit, is_alpha in CASES:
        case = os.path.join(SPH_ROOT, name)
        if not os.path.exists(os.path.join(case, "ai_constants.log")):
            print(f"-- {name}: not run/synced yet (no ai_constants echo), "
                  f"skipped")
            continue
        print(f"== {name}")
        seed = read_seed(case)
        chi_inf = seed["chi_bc"] / (seed["slowdown"] or 1.0)
        n_budget = float(np.log(1.0 / chi_inf))
        n_crit_case = float(np.log(C_V1 / chi_inf))
        assert abs(n_crit_case - n_crit) < 2e-3, (n_crit_case, n_crit)
        assert seed["slowdown"] == 1.0, seed["slowdown"]
        frc = forces_tail(case)
        # forces_tail fits the last 2k steps -- on these reseeded runs a
        # small CD wiggle (~1e-4, several-k period) aliases into that fit;
        # add the 5k-window fit as the primary gate metric
        import csv as _csv
        rows = list(_csv.reader(open(os.path.join(case,
                                                  "total_forces_v2.csv"))))
        hdr = [h.strip() for h in rows[0]]
        dat = np.array([[float(v) for v in r[:len(hdr)] if v.strip()]
                        for r in rows[1:] if len(r) >= 5])
        stp, cd = dat[:, 1], dat[:, hdr.index("CD")]
        m5 = stp >= stp[-1] - 5000
        frc["CD_drift_per_1k_5kwin"] = float(
            np.polyfit(stp[m5], cd[m5], 1)[0] * 1000)
        # de-trended p2p over the last 5k: a natural-transition limit cycle
        # (appears at the earlier tunnel-class fronts) inflates the drift
        # fit; the FRONT is judged stationary separately (cross-leg, in the
        # record).  CD "settled" here = drift below gate after removing the
        # limit-cycle trend.
        cd5 = cd[m5] - np.polyval(np.polyfit(stp[m5], cd[m5], 1), stp[m5])
        frc["CD_p2p_last5k"] = float(np.ptp(cd[m5]))
        frc["CD_detrended_rms_last5k"] = float(cd5.std())
        conv = abs(frc["CD_drift_per_1k_5kwin"]) < DRIFT_GATE
        frc["limit_cycle"] = bool(frc["CD_p2p_last5k"] > 5e-4)
        print(f"   seed chi_inf {chi_inf:.4e} (N_crit {n_crit_case:.3f}, "
              f"budget N {n_budget:.3f}, slowdown {seed['slowdown']}), "
              f"Re {seed['re_L']:.2e}")
        print(f"   forces: CL {frc['CL']:+.4f} CD {frc['CD']:.5f} "
              f"drift/1k {frc['CD_drift_per_1k_5kwin']:+.2e} (5k) "
              f"p2p {frc['CD_p2p_last5k']:.1e} "
              f"{'CD-SETTLED' if conv else ('LIMIT-CYCLE' if frc['limit_cycle'] else 'NOT SETTLED')}")
        gm = load_grid(case, "slice", slice_name="meridian")
        sw90 = sweep_case(gm, seed["mu_ref"], facets, phi_deg=90.0)
        rep = dict(seed=dict(chi_inf=chi_inf, N_crit=n_crit_case,
                             N_budget=n_budget,
                             slowdown=seed["slowdown"],
                             consts=seed["consts"]),
                   re_L=seed["re_L"], forces=frc, converged=bool(conv),
                   phi90=sweep_report(sw90, chi_inf))
        print(f"   phi=90: chi1 front {rep['phi90']['front_chi1']:.4f}  "
              f"cv1 {rep['phi90']['front_cv1']:.4f}  nose chi ratio "
              f"{rep['phi90']['nose_chi_ratio']:.3f}")
        print("   H:", {k: round(v['H'], 4)
                        for k, v in rep["phi90"]["H"].items()})
        rep["cf_front_az"] = cf_fronts_per_azimuth(case)
        c = rep["cf_front_az"]
        print(f"   Cf-rise front over {c['n']} azimuth lines: median "
              f"{c['median']:.4f} rms {c['rms']:.2e} p2p {c['p2p']:.2e}")
        if is_alpha:
            gs = load_grid(case, "slice", slice_name="symm")
            for tag, phi in (("leeward_phi0", 0.0),
                             ("windward_phi180", 180.0)):
                sw = offset_meridian_sweep(gs, seed["mu_ref"], facets, phi)
                rep[tag] = sweep_report(sw, chi_inf)
                print(f"   {tag}: chi1 {rep[tag]['front_chi1']:.4f}  "
                      f"cv1 {rep[tag]['front_cv1']:.4f}  H(0.42) "
                      f"{rep[tag]['H']['0.42']['H']:.4f}")
        out["cases"][name] = rep

    # ---- references ---------------------------------------------------------
    f14a = json.load(open(os.path.join(REPO, "paper", "data",
                                       "stock2006_fig14a_digitized.json")))
    f14b_p = os.path.join(REPO, "paper", "data",
                          "stock2006_fig14b_digitized.json")
    f14b = json.load(open(f14b_p)) if os.path.exists(f14b_p) else None
    out["measured"] = dict(
        a0_re72=float(np.median([q["xL"] for q in f14a["measured_squares"]])),
        a0_re72_stock_eN=f14a["computed_ts_front"]["mean_xL"])
    if f14b:
        out["measured"]["a2p5_re72"] = {
            str(q["phi_deg"]): q["xL"] for q in f14b["measured_squares"]}
        out["measured"]["a2p5_re72_stock_eN"] = \
            f14b["computed_ts_front"]["mean_xL"]
        out["measured"]["a2p5_convention"] = f14b["convention"]

    tw = json.load(open(os.path.join(FIGD, "spheroid_threeway.json")))
    out["threeway_prediction_keys"] = sorted(tw.keys())
    out["notes"] = [
        "chi=1 front = near-wall (y<=0.02) max-chi crossing, facet-re-based "
        "rays (0350/1030 conventions); c_v1 companion; Cf-rise k=1.5 "
        "running-min variant",
        "mesh azimuth: phi=0=+z=LEEWARD at alpha>0; Stock figs use "
        "phi=0=windward -> mirror phi_stock=180-phi_mesh",
        "budget N = ln(1/chi_inf); N_crit = ln(c_v1/chi_inf) = Mack map",
    ]
    fp = os.path.join(FIGD, "spheroid_reseed_harvest.json")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    json.dump(clean(out), open(fp, "w"), indent=1)
    print("wrote", fp)

    # ---- front-vs-seed figure (re72a0 bracket vs 1122 prediction) ----------
    def lever(key):
        d = tw[key]
        N = sorted(float(k[1:]) for k in d)
        return np.array(N), np.array([d[f"N{g:g}"] for g in N])

    meas = out["measured"]["a0_re72"]
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    N, xd = lever("seed_lever_xN")
    ax.plot(N, xd, "-", color="#b0483a", lw=1.8,
            label="Drela-Giles envelope $x(N)$ (1122)")
    N, xt = lever("seed_lever_xN_transported")
    ax.plot(N, xt, "-", color="#3b6bb5", lw=1.8,
            label="model transported remap $x(N)$ (1122)")
    nb, f1, fc = [], [], []
    for name, n_crit, _ in CASES[:3]:
        rep = out["cases"].get(name)
        if rep:
            nb.append(rep["seed"]["N_budget"])
            f1.append(rep["phi90"]["front_chi1"])
            fc.append(rep["cf_front_az"]["median"])
    ax.plot(nb, f1, "o", ms=10, color="#1d2e4f", zorder=5,
            label=r"reseed run, $\chi\!=\!1$ front")
    ax.plot(nb, fc, "s", ms=8, mfc="none", mec="#1d2e4f", mew=1.6,
            zorder=5, label="reseed run, Cf-rise front")
    # measurement-implied budget from the actual bracket (chi=1 interp)
    n_impl = float(np.interp(meas, f1[::-1], nb[::-1]))
    ax.axhline(meas, color="0.25", lw=1.3, ls="--",
               label="measured 0.438 (Kreplin/Stock 14a)")
    ax.axhline(out["measured"]["a0_re72_stock_eN"], color="0.55", lw=1.0,
               ls=":", label="Stock $e^N$ ($N_{TS}\\!=\\!8$): 0.425")
    ax.axvline(n_impl, color="0.5", lw=1.0, ls="-.")
    ax.annotate(f"measurement-implied\n$N\\approx{n_impl:.1f}$ "
                f"(Mack $Tu\\approx0.17\\%$)",
                (n_impl, meas), (8.4, 0.44), fontsize=9, color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=1.0))
    out["measured_implied_N_from_bracket"] = n_impl
    ax.set_xlabel("freestream budget $N=\\ln(1/\\chi_\\infty)$")
    ax.set_ylabel("transition front $x/L$")
    ax.set_title("6:1 spheroid, $\\alpha=0$, $Re_L=7.2\\times10^6$: "
                 "front vs freestream seed")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(3.5, 13.2)
    fig.tight_layout()
    png = os.path.join(FIGD, "spheroid_reseed_front_vs_seed.png")
    fig.savefig(png, dpi=150)
    fig.savefig(os.path.join(REPO, "paper", "figs",
                             "spheroid_reseed_front_vs_seed.pdf"))
    json.dump(clean(out), open(fp, "w"), indent=1)   # re-dump w/ implied N
    print("wrote", png, "+ paper/figs PDF")


if __name__ == "__main__":
    main()
