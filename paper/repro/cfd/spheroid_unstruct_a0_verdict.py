"""Discriminating extraction on the UNSTRUCTURED (Flynn360 cavity-family)
spheroid at alpha = 0, Re_L = 7.2e6 -- the mean-flow-fullness topology test.

Question (agent-paper-review/2026-07-28-0040-spheroid-a0-physics.md Sec 1):
the O-grid RANS laminar boundary layer on this body runs H ~= 2.49 where
laminar-BL theory on its own u_e (axisymmetric Thwaites) says
Blasius-class 2.59.  If the unstructured family -- different topology,
different discretization stencils, same solver core -- reproduces
H ~ 2.49, the fullness is solver-core/physics; if it reads ~2.59, the
O-grid discretization is convicted.

Machinery: spheroid_a0_physics.py's EXACT extractor (imported, not
reimplemented): analytic-normal rays along the phi = 90 meridian,
edge = first wall-normal speed maximum, H from the momentum/displacement
integrals, P from kernel_from_xyz on the VTK chained-gradient field
("solver-read" convention), planar savgol variant, chi = 1 front =
near-wall (y <= 0.04) max-chi crossing of 1 (the sweep convention that
re-extracted the O-grid's 0.9244).

O-grid comparison values are read from the committed
figs_explore/spheroid_a0_physics.json (band_table = L2 sweep,
reference_checks = L0/L1 station probes, plate = flat-plate operator
control H = 2.60-2.61).

Outputs:
  figs_explore/spheroid_unstruct_a0_verdict.json   (all numbers)
  stdout: the verdict table (H: unstructured vs O-grid ladder vs
  Thwaites 2.59) + fronts + forces.

Run:  python3 -u paper/repro/cfd/spheroid_unstruct_a0_verdict.py [case_dir]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spheroid_a0_physics import (                    # noqa: E402
    XS, YBAND, extract_meridian, front_crossing, planar_kernel_smooth)
from spheroid_flank_kernel_audit import load_case_with_derived  # noqa: E402

CASE = sys.argv[1] if len(sys.argv) > 1 else \
    "/local_data/qiqi/sa-ai/spheroid_fv1/case_unstr_L1_saai_re72a0"
OGRID_JSON = os.path.join(HERE, "figs_explore", "spheroid_a0_physics.json")
STATIONS = (0.20, 0.42, 0.70)
THWAITES_H = 2.59            # Blasius-class equilibrium on this u_e
                             # (2.59-2.61 across the mid-body, record Sec 1)


def forces_tail(case, n=200):
    import csv
    rows = list(csv.reader(open(os.path.join(case, "total_forces_v2.csv"))))
    hdr = [h.strip() for h in rows[0]]
    dat = np.array([[float(v) for v in r[:len(hdr)] if v.strip() != ""]
                    for r in rows[1:] if len(r) >= 5])
    i_cl, i_cd, i_cy = hdr.index("CL"), hdr.index("CD"), hdr.index("CFy")
    tail = dat[-n:]
    full = dat
    return dict(steps=int(dat[-1, 1]),
                CL=float(np.mean(tail[:, i_cl])),
                CD=float(np.mean(tail[:, i_cd])),
                CFy=float(np.mean(tail[:, i_cy])),
                CD_drift_per_1k=float(
                    np.polyfit(full[-2000 // 10:, 1],
                               full[-2000 // 10:, i_cd], 1)[0] * 1000)
                if len(full) > 210 else np.nan)


def main():
    print(f"case: {CASE}", flush=True)
    frc = forces_tail(CASE)
    print(f"forces (mean last 200 rows @ step {frc['steps']}): "
          f"CL={frc['CL']:+.3e}  CD={frc['CD']:.5f}  CFy={frc['CFy']:+.3e}  "
          f"CD drift/1k={frc['CD_drift_per_1k']:+.2e}", flush=True)

    grid, nu_ref, mach = load_case_with_derived(CASE)
    sw = extract_meridian(grid, nu_ref, XS)
    x = np.array([s["x"] for s in sw])
    H = np.array([s["H"] for s in sw])
    Rt = np.array([s["u_e"] * s["theta"] / np.median(s["nu"]) for s in sw])
    chimax = np.array([float(np.nanmax(np.where(
        (s["y"] <= 0.04) & s["valid"], s["chi"], np.nan))) for s in sw])
    front = front_crossing(x, chimax, 1.0)

    og = json.load(open(OGRID_JSON))
    bt = {round(r["x"], 2): r for r in og["band_table"]}
    rc = og["reference_checks"]
    ogl = {lev: {round(r["x"], 2): r for r in rc[lev]} for lev in ("L0", "L1")}

    rows = []
    print("\n== VERDICT TABLE: laminar mean-flow shape factor H "
          "(phi=90 meridian, alpha=0, Re_L=7.2e6) ==")
    print(f"{'x/L':>5} | {'unstr L1':>9} | {'O-grid L0':>9} | "
          f"{'O-grid L1':>9} | {'O-grid L2':>9} | {'Thwaites':>8} | "
          f"{'P unstr':>8} {'P og-L2':>8}")
    for xq in STATIONS:
        st = sw[int(np.argmin(np.abs(x - xq)))]
        bnd = (st["y"] >= 1e-5) & (st["y"] <= YBAND)
        j = int(np.argmax(np.where(bnd, st["P"], -np.inf)))
        kp = planar_kernel_smooth(st)
        jp = int(np.argmax(np.where(kp["y"] >= 1e-5, kp["P"], -np.inf)))
        r = dict(x=xq, H=float(st["H"]),
                 Rt=float(st["u_e"] * st["theta"] / np.median(st["nu"])),
                 d99=float(st["d99"]), edge_ok=bool(st["edge_ok"]),
                 P_solver=float(st["P"][j]), P_planar=float(kp["P"][jp]),
                 chimax=float(np.interp(xq, x, chimax)))
        rows.append(r)
        print(f"{xq:5.2f} | {r['H']:9.3f} | {ogl['L0'][xq]['H']:9.3f} | "
              f"{ogl['L1'][xq]['H']:9.3f} | {bt[xq]['H']:9.3f} | "
              f"{THWAITES_H:8.2f} | {r['P_solver']:8.4f} "
              f"{bt[xq]['Pmax']:8.4f}")
    print(f"\nfronts: unstructured chi=1 = {front:.4f}  "
          f"(O-grid L2: {og['fronts']['chi1']:.4f}; measured "
          f"{og['fronts']['meas']:.4f}; Stock e^N(8) "
          f"{og['fronts']['stock']:.4f})", flush=True)

    out = dict(case=CASE, forces=frc, front_chi1=float(front),
               stations=rows,
               sweep=dict(x=x.tolist(), H=H.tolist(), Rt=Rt.tolist(),
                          chimax=chimax.tolist()),
               ogrid_fronts=og["fronts"], thwaites_H=THWAITES_H,
               conventions="spheroid_a0_physics.py extractor verbatim: "
                           "edge=first speed max, H from integrals to i_e, "
                           "P=kernel on VTK chained gradients, front="
                           "near-wall (y<=0.04) chi max crossing 1")
    op = os.path.join(HERE, "figs_explore", "spheroid_unstruct_a0_verdict.json")
    json.dump(out, open(op, "w"), indent=1)
    print(f"wrote {op}", flush=True)


if __name__ == "__main__":
    main()
