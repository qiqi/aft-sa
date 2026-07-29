"""vg-kernel drag-crisis validation ladder (2026-07-29).

Runs the high-Re cylinder ladder Re = 2e6..2e7 (Tu = 0.2%) with the two-branch
"vg" SA-AI kernel ON (env AI_A_VISC=0.0276, AI_REOMC_BC=130), reusing the canon
campaign infrastructure (highre mesh template, staged-fSlow steady protocol,
extract()) from run_dragcrisis_matrix.py. Cases are tagged _vg and results are
written to a SEPARATE vg_summary.jsonl -- the canon matrix_summary.jsonl is
NEVER touched. The first point (2e6) is warm-started from the canon converged
2e6 dn state; each higher Re warm-chains from the previous vg case.

The scientific question (agent-paper-review/2026-07-28-1218 Parts V/VIII, and
2026-07-29-0121 Part VIII): does the vg nose-FPG amplification advance the
transition front forward (toward the transcritical ~25-35 deg class) and raise
Cd toward the experimental transcritical 0.5-0.7, i.e. recover the missing rise
of Sec VII, vs the canon kernel at the same Re?

Usage:  python run_dragcrisis_vg.py --gpu 1 [--res 2e6,4e6,7e6,1e7,2e7]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_dragcrisis_matrix import (                                    # noqa: E402
    OUT, SEEDS, WARM_STAGES, COLD_STAGES, ALLOWED_GPUS, make_case, run_case,
    extract, retag, gpu_is_free)
from run_dragcrisis_pilot import canonical_env                        # noqa: E402
from run_dragcrisis_extension import TEMPLATES, free_gb, MIN_FREE_GB  # noqa: E402

TU = "0.2"
CHI = SEEDS[TU]
VG_SUMMARY = OUT / "vg_summary.jsonl"       # separate; canon JSON untouched
A_VISC = "0.0276"
REOMC_BC = "130"


def vg_env(chi_inf: float, fslow: float) -> dict:
    e = canonical_env(chi_inf, fslow)
    e["AI_A_VISC"] = A_VISC
    e["AI_REOMC_BC"] = REOMC_BC
    return e


def vg_name(re: float, mesh: str) -> str:
    return f"cyl_Re{retag(re)}_Tu{TU}_up_{mesh}_vg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, choices=ALLOWED_GPUS)
    ap.add_argument("--res", default="2e6,4e6,7e6,1e7,2e7")
    ap.add_argument("--mesh", default="highre", choices=["highre", "ultra"])
    ap.add_argument("--cold", action="store_true",
                    help="cold two-stage start for the FIRST Re (no warm seed)")
    ap.add_argument("--seed", default="cyl_Re2000000_Tu0.2_dn_highre",
                    help="canon case dir (in OUT) to warm-start the first Re")
    args = ap.parse_args()
    res = [float(x) for x in args.res.split(",")]

    if not gpu_is_free(args.gpu):
        print(f"GPU {args.gpu} is busy -- aborting (occupancy check)",
              flush=True)
        sys.exit(1)
    if free_gb() < MIN_FREE_GB:
        print(f"DISK ABORT: {free_gb():.1f} GiB free", flush=True)
        sys.exit(1)

    if args.cold:
        warm = None
    else:
        warm = OUT / args.seed
        assert (warm / "restartOutput").is_dir(), f"no restart in {warm}"
    for re in res:
        case_dir = OUT / vg_name(re, args.mesh)
        stages = COLD_STAGES if warm is None else WARM_STAGES
        tag = "cold" if warm is None else f"warm<-{warm.name}"
        print(f"=== {vg_name(re, args.mesh)} (gpu {args.gpu}, {tag}) ===",
              flush=True)
        make_case(case_dir, re, CHI, tmpl=TEMPLATES[args.mesh])
        info = run_case(case_dir, CHI, args.gpu, stages, warm, env_fn=vg_env)
        row = {"case": vg_name(re, args.mesh), "re": re, "Tu": TU, "dir": "up",
               "mesh": args.mesh, "kernel": "vg",
               "a_visc": float(A_VISC), "reOmBc": float(REOMC_BC),
               "warm_src": warm.name if warm else None,
               **info, **extract(case_dir, re)}
        json.dump(row, open(case_dir / "summary.json", "w"), indent=1)
        with open(VG_SUMMARY, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  -> {row['verdicts']} Cd={row.get('Cd'):.4f} "
              f"CL={row.get('CL'):+.4f} knee={row.get('knee_upper')} "
              f"chi1_u={row.get('chi1_front_upper')} "
              f"({info['wall_s']:.0f}s)", flush=True)
        warm = case_dir
    print("VG LADDER COMPLETE", flush=True)


if __name__ == "__main__":
    main()
