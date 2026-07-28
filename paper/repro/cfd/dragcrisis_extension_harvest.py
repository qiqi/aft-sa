"""Harvest the drag-crisis matrix EXTENSION (run_dragcrisis_extension.py)
into the record tables: extended Cd(Re) by arm/seed, seam deltas (same Re +
seed on two mesh families), low-Re steady benchmarks (Dennis-Chang 1970 /
Fornberg 1980 -- litrange record 2026-07-28-0117 Sec. 2), MEASURED y+ ladder
(y1 * Re * sqrt(max surface Cf / 2)) for the high arm, creeping-arm
chi-passivity, ai_constants echo diff vs campaign canon, and cost.

Usage: python dragcrisis_extension_harvest.py [--root .../dragcrisis_matrix]
"""
import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys                                                     # noqa: E402
sys.path.insert(0, HERE)
import dragcrisis_pilot_forces as F                            # noqa: E402

MACH = 0.1
Y1 = {"pilot": 8e-6, "lowre": 1e-3, "lowre300": 1e-3, "highre": 1e-6,
      "ultra": 1e-9}
# steady-branch benchmarks; DC 20/40 + Fornberg 20/40 verified (litrange),
# DC 10/100 widely-reproduced [mem]
DENNIS_CHANG = {10: 2.846, 20: 2.045, 40: 1.522, 100: 1.056}
FORNBERG = {20: 2.00, 40: 1.498}
SEAMS = [(300.0, "0.2"), (1e3, "0.2"), (2e6, "0.2"), (4e6, "0.2"),
         # phase 3: highre/ultra overlaps (2e7 all seeds; 5e7 highre is a
         # continuity check only -- highre y+ ~ 3.5 there)
         (2e7, "0.05"), (2e7, "0.2"), (2e7, "0.7"), (5e7, "0.2")]


def load(root):
    rows = {}
    with open(os.path.join(root, "matrix_summary.jsonl")) as f:
        for ln in f:
            r = json.loads(ln)
            r.setdefault("mesh", "pilot")
            rows[(r["Tu"], r["dir"], r["re"], r["mesh"])] = r
    return rows


def vtag(r):
    return "".join("*" if "converged" not in v else "" for v in r["verdicts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/local_data/qiqi/sa-ai/dragcrisis_matrix")
    args = ap.parse_args()
    rows = load(args.root)
    ext = [r for r in rows.values() if "chain" in r]

    print("== extended Cd(Re) (extension cases; * = a stage not converged) ==")
    for r in sorted(ext, key=lambda r: (r["mesh"], r["Tu"], r["dir"], r["re"])):
        print(f"  {r['case']:38s} Cd={r['Cd']:8.4f}{vtag(r):2s} "
              f"CL={r['CL']:+.2e} chi_max={r.get('chi_max_nearwall')} "
              f"front={r.get('chi1_front_upper')} wall={r['wall_s']:.0f}s")

    print("\n== seam deltas (same Re + Tu on two mesh families) ==")
    for re_v, tu in SEAMS:
        pts = [r for r in rows.values() if r["re"] == re_v and r["Tu"] == tu]
        fams = {}
        for r in pts:
            fams.setdefault(r["mesh"], []).append(r)
        if len(fams) < 2:
            print(f"  Re={re_v:g}: only {list(fams)} present")
            continue
        for fam, rs in sorted(fams.items()):
            for r in rs:
                print(f"  Re={re_v:<9g} {fam:7s} {r['dir']:4s} "
                      f"Cd={r['Cd']:.4f}{vtag(r)}")
        cds = {f: np.mean([r["Cd"] for r in rs]) for f, rs in fams.items()}
        vals = sorted(cds.values())
        print(f"    -> seam delta {abs(vals[-1] - vals[0]):.4f} "
              f"({200 * abs(vals[-1] - vals[0]) / (vals[-1] + vals[0]):.2f}%)")

    print("\n== low-Re steady benchmarks (physical below Re~47) ==")
    for re_v, cd_ref in sorted({**DENNIS_CHANG, **{k: None for k in FORNBERG}}.items()):
        ours = [r for r in ext if r["re"] == re_v and r["mesh"] == "lowre"]
        if not ours:
            continue
        cd = ours[0]["Cd"]
        s = f"  Re={re_v:<5g} ours={cd:7.4f}"
        if re_v in DENNIS_CHANG:
            dc = DENNIS_CHANG[re_v]
            s += f"  DC70={dc:6.3f} ({100 * (cd - dc) / dc:+.2f}%)"
        if re_v in FORNBERG:
            fb = FORNBERG[re_v]
            s += f"  Fornberg80={fb:6.3f} ({100 * (cd - fb) / fb:+.2f}%)"
        print(s)

    print("\n== far-field sensitivity (lowre R=1000 vs lowre300 R=300) ==")
    for re_v in (1.0, 30.0):
        a = [r for r in ext if r["re"] == re_v and r["mesh"] == "lowre"]
        b = [r for r in ext if r["re"] == re_v and r["mesh"] == "lowre300"]
        if a and b:
            ca, cb = a[0]["Cd"], b[0]["Cd"]
            print(f"  Re={re_v:<4g} R1000={ca:.4f} R300={cb:.4f} "
                  f"delta={cb - ca:+.4f} ({100 * (cb - ca) / ca:+.2f}%)")

    print("\n== measured y+ ladder (y1 * Re * sqrt(Cf_max/2), final field) ==")
    for r in sorted(ext, key=lambda r: r["re"]):
        if r["mesh"] not in ("highre", "pilot", "ultra") or r["re"] < 1e6 \
                or r["dir"] not in ("up", "cold"):
            continue
        case = os.path.join(args.root, r["case"])
        try:
            sides = F.surface_mean_cfcp(case, avg=False)
            cfmax = max(np.nanmax(np.abs(cf)) for _, cf, _ in sides.values())
            yp = Y1[r["mesh"]] * r["re"] * np.sqrt(cfmax / 2.0)
            print(f"  {r['case']:38s} Cf_max={cfmax:.2e} y+max={yp:.2f}")
        except Exception as e:                                 # noqa: BLE001
            print(f"  {r['case']:38s} ({e!r})")

    print("\n== seed collapse across Tu (ultra arm; phase-3 prediction: "
          "seeds collapse as transition saturates toward the nose) ==")
    for d in ("up", "dn"):
        res = sorted({r["re"] for r in ext
                      if r["mesh"] == "ultra" and r["dir"] == d})
        for re_v in res:
            cds = {r["Tu"]: r["Cd"] for r in ext if r["mesh"] == "ultra"
                   and r["dir"] == d and r["re"] == re_v}
            if len(cds) < 2:
                continue
            v = sorted(cds.values())
            print(f"  {d:2s} Re={re_v:<8g} " +
                  " ".join(f"Tu{t}={cds[t]:.4f}" for t in sorted(cds)) +
                  f"  spread={v[-1] - v[0]:.4f} "
                  f"({200 * (v[-1] - v[0]) / (v[-1] + v[0]):.2f}%)")

    print("\n== ai_constants echo diff vs campaign canon ==")

    def payload(fp):        # strip solver-log timestamps before comparing
        return [ln.split("]: ", 1)[-1] for ln in open(fp).read().splitlines()]

    canon = payload(os.path.join(args.root, "cyl_Re60000_Tu0.2_dn",
                                 "ai_constants_echo.log"))
    ndiff = 0
    for r in sorted(ext, key=lambda r: r["case"]):
        echo = os.path.join(args.root, r["case"], "ai_constants_echo.log")
        if r["Tu"] == "ft":
            # FT-SA control: the AI echo must be ABSENT (AI_SA=0 suppresses
            # the block at the source) -- absence IS the verification
            if os.path.exists(echo) and "SA-AI" in open(echo).read():
                print(f"  {r['case']}: FT case but SA-AI echo PRESENT "
                      f"-- AI was NOT off!")
                ndiff += 1
            continue
        if not os.path.exists(echo):
            print(f"  {r['case']}: NO ECHO")
            ndiff += 1
            continue
        mine = payload(echo)
        if mine != canon:
            print(f"  {r['case']}: DIFFERS")
            for a, b in zip(canon, mine):
                if a != b:
                    print(f"    canon: {a}\n    case : {b}")
            if len(mine) != len(canon):
                print(f"    line count {len(mine)} vs canon {len(canon)}")
            ndiff += 1
    print(f"  {len(ext)} cases checked, {ndiff} deviations" if ndiff else
          f"  all {len(ext)} extension echoes identical to canon "
          f"(timestamps stripped)")

    ftr = [r for r in ext if r["Tu"] == "ft"]
    if ftr:
        print("\n== FT-SA control vs SA-AI at matched Re (fault attribution) ==")
        for r in sorted(ftr, key=lambda r: r["re"]):
            mates = [q for q in ext if q["Tu"] != "ft" and q["re"] == r["re"]
                     and q["mesh"] == r["mesh"] and q["dir"] in ("up", "cold")]
            ai = "; ".join(
                f"Tu{q['Tu']}/{q['dir']}: Cd={q['Cd']:.3f} "
                f"knee={q.get('knee_upper')}" for q in mates)
            print(f"  Re={r['re']:<8g} FT: Cd={r['Cd']:.4f} "
                  f"knee={r.get('knee_upper')} Cpb={r.get('Cp_base_upper')} "
                  f"Cpsh={r.get('Cp_shoulder_upper')}\n"
                  f"    vs SA-AI: {ai}")

    wall = sum(r["wall_s"] for r in ext)
    print(f"\n== cost: {len(ext)} extension cases, {wall / 3600:.2f} GPU-h ==")


if __name__ == "__main__":
    main()
