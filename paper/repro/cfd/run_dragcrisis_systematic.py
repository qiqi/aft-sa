"""Systematic Tu=0.2% drag-crisis Cd(Re) campaign: TWO clean continuation
ladders (one up-Re, one down-Re) on ONE identical Reynolds list, replacing the
six-arm patchwork inventoried in
agent-paper-review/2026-07-29-dragcrisis-Tu0.2-provenance.md.

Reuses the canon infrastructure (run_dragcrisis_matrix.make_case / run_case /
extract, run_dragcrisis_pilot.canonical_env, run_dragcrisis_extension.TEMPLATES)
UNCHANGED: same fSlow staged steady protocol, same Tu=0.2 chi_inf
(SEEDS["0.2"]=1.0835e-2), same COLD/WARM stage budgets. The canon
matrix_summary.jsonl / vg_summary.jsonl are NEVER touched -- output goes to a
SEPARATE systematic_Tu0.2_summary.jsonl.

Re list (identical up & dn): 2/decade over 1..1e10 + 4/decade crisis
densification in [1e5,1e7].  Mesh bands (validated y+<1, R>=100D):
  lowre  (y1=1e-3D, R~1000D): Re 1 .. 1e3
  pilot  (y1=8e-6D, R=100D) : Re 3e3 .. 1.78e6
  highre (y1=1e-6D)         : Re 3e6 .. 1e7
  ultra  (y1=1e-9D)         : Re 3e7 .. 1e10
Restarts do NOT cross meshes, so each band's first Re is cold-started OR
warm-seeded from an existing converged SAME-MESH state (reuse); every other Re
warm-restarts from the previous Re of the same ladder+band.

Per case we record: Cd (tail median), Cd_tail_p2p, theta_tr_chi1 (radial-ray
chi=1 front, dragcrisis_transition_angle.ray_profiles), verdict, warm_src, and
the full 0..180deg log10(max-chi) radial-ray profile (361 pts, step 0.5deg;
compress(..., log=True), same array as dragcrisis_theta_tr.json profiles).

Run under the compute venv python (extract/ray_profiles need vtk):
  python run_dragcrisis_systematic.py --part up   --gpu 1
  python run_dragcrisis_systematic.py --part dn   --gpu 2
  python run_dragcrisis_systematic.py --part seam --gpu 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_dragcrisis_matrix import (                                    # noqa: E402
    OUT, MACH, SEEDS, COLD_STAGES, WARM_STAGES, ALLOWED_GPUS,
    make_case, run_case, extract, retag, gpu_is_free, wait_for_gpu)
from run_dragcrisis_pilot import canonical_env                        # noqa: E402
from run_dragcrisis_extension import TEMPLATES, free_gb, MIN_FREE_GB  # noqa: E402
from dragcrisis_transition_angle import (                             # noqa: E402
    ray_profiles, first_cross, C_V1, THETA, compress, N_S)

# GPU restriction temporarily LIFTED (coordinator 2026-07-29): all 8 GPUs on
# this host are ours for a few hours. Widen the canon driver's allowlist so
# run_case/wait_for_gpu accept GPUs 4-7 too (reverts on next import of canon).
import run_dragcrisis_matrix as _M                                    # noqa: E402
_M.ALLOWED_GPUS = tuple(range(8))

TU = "0.2"
CHI = SEEDS[TU]
SUMMARY = OUT / "systematic_Tu0.2_summary.jsonl"

# 2/decade over 1..1e10 + 4/decade crisis densification in [1e5,1e7]
RE_SYS = sorted([1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6,
                 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10,
                 1.78e5, 5.62e5, 1.78e6, 5.62e6])


def mesh_of(re: float) -> str:
    if re <= 1e3:
        return "lowre"
    if re <= 1.78e6:
        return "pilot"
    if re <= 1e7:
        return "highre"
    return "ultra"


# band-first external seed (existing converged SAME-MESH case with restart);
# None => cold two-stage start.  Keyed (direction, mesh).
SEED_FIRST = {
    ("up", "lowre"):  None,                              # Re 1, cold
    ("up", "pilot"):  None,                              # Re 3e3, cold (subcrit)
    ("up", "highre"): OUT / "cyl_Re2000000_Tu0.2_cold_highre",   # 3e6 <- 2e6
    ("up", "ultra"):  None,                              # Re 3e7, cold
    ("dn", "ultra"):  None,                              # Re 1e10, cold
    ("dn", "highre"): None,                              # Re 1e7, cold (seam)
    ("dn", "pilot"):  OUT / "cyl_Re2000000_Tu0.2_cold",         # 1.78e6 <- 2e6
    ("dn", "lowre"):  OUT / "cyl_Re1000_Tu0.2_up_lowre",       # 1e3 <- 1e3
}

# seam overlap points: boundary-adjacent Re run on the OTHER band's mesh
# (mesh-family agreement check). (re, mesh, seed_case_in_OUT)
SEAMS = [
    (1e3,    "pilot",  "cyl_Re3000_Tu0.2_dn_sys_pilot"),      # lowre|pilot
    (1.78e6, "highre", "cyl_Re3000000_Tu0.2_up_sys_highre"),  # pilot|highre
    (1e7,    "ultra",  "cyl_Re30000000_Tu0.2_up_sys_ultra"),  # highre|ultra
]


def name(re: float, direction: str, mesh: str) -> str:
    return f"cyl_Re{retag(re)}_Tu{TU}_{direction}_sys_{mesh}"


def theta_tr_and_profile(case_dir: Path) -> dict:
    """Radial-ray chi=1 transition angle + the 0..180deg log10(max-chi)
    profile, reusing dragcrisis_transition_angle exactly."""
    out = {"theta_tr_chi1": None, "theta_tr_cv1": None,
           "theta_at_maxchi": None, "maxchi_global": None,
           "log10_maxchi": None, "theta_deg": None}
    if not (case_dir / "slice_centerSpan.pvtu").exists():
        out["theta_note"] = "no slice_centerSpan.pvtu"
        return out
    try:
        prof, sides, reach, fv = ray_profiles(str(case_dir))
        import numpy as np
        imax = int(np.argmax(prof["maxchi"]))
        out.update({
            "theta_tr_chi1": first_cross(prof["maxchi"], 1.0),
            "theta_tr_cv1": first_cross(prof["maxchi"], C_V1),
            "theta_at_maxchi": float(THETA[imax]),
            "maxchi_global": float(f"{prof['maxchi'][imax]:.4g}"),
            "ray_reach_D": round(float(reach), 3),
            "log10_maxchi": compress(prof["maxchi"], log=True),
            "theta_deg": [float(round(float(t), 2)) for t in THETA],
        })
    except Exception as e:                                         # noqa: BLE001
        out["theta_note"] = repr(e)
    return out


def run_one(re: float, direction: str, mesh: str, warm: Path | None,
            gpu: int, role: str = "ladder") -> Path:
    nm = name(re, direction, mesh)
    case_dir = OUT / nm
    if (case_dir / "summary.json").exists():
        print(f"=== {nm} already done, skipping ===", flush=True)
        return case_dir
    if free_gb() < MIN_FREE_GB:
        raise SystemExit(f"DISK ABORT: {free_gb():.1f} GiB free")
    stages = WARM_STAGES if warm is not None else COLD_STAGES
    seedtag = f"warm<-{warm.name}" if warm is not None else "cold"
    print(f"=== {nm} (gpu {gpu}, {seedtag}) ===", flush=True)
    make_case(case_dir, re, CHI, tmpl=TEMPLATES[mesh])
    info = run_case(case_dir, CHI, gpu, stages, warm, env_fn=canonical_env)
    last = info["verdicts"][-1]
    verdict = "converged" if last.endswith(":converged") else (
        "limit_cycle" if last.endswith(":limit_cycle") else "cap")
    row = {"case": nm, "re": float(re), "Tu": TU, "dir": direction,
           "mesh": mesh, "role": role,
           "warm_src": warm.name if warm is not None else None,
           "verdict": verdict, **info, **extract(case_dir, re)}
    row.update(theta_tr_and_profile(case_dir))
    json.dump(row, open(case_dir / "summary.json", "w"), indent=1)
    with open(SUMMARY, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  -> {verdict} Cd={row.get('Cd'):.4f} p2p={row.get('Cd_tail_p2p'):.4f}"
          f" theta_tr={row.get('theta_tr_chi1')} ({info['wall_s']:.0f}s)",
          flush=True)
    return case_dir


def run_ladder(direction: str, gpu: int, band: str | None = None):
    """Run one direction. If `band` is given, only that mesh band's Re are run
    (an independent sub-chain: it re-seeds at the band boundary exactly as the
    full ladder would, since restarts never cross meshes -- so band sub-chains
    are branch-faithful AND parallelizable one-per-GPU)."""
    res = RE_SYS if direction == "up" else RE_SYS[::-1]
    prev_dir = None
    prev_mesh = None
    for re in res:
        mesh = mesh_of(re)
        if band is not None and mesh != band:
            continue
        if mesh != prev_mesh:                     # band-first: cold or seed
            seed = SEED_FIRST[(direction, mesh)]
            warm = seed if (seed and (seed / "restartOutput").is_dir()) else None
            if seed and warm is None:
                print(f"  WARNING seed {seed} has no restart -> cold", flush=True)
        else:
            warm = prev_dir
        cd = run_one(re, direction, mesh, warm, gpu)
        prev_dir, prev_mesh = cd, mesh
    print(f"LADDER {direction}"
          f"{(':' + band) if band else ''} COMPLETE", flush=True)


def run_seams(gpu: int):
    for re, mesh, seed_name in SEAMS:
        seed = OUT / seed_name
        warm = seed if (seed / "restartOutput").is_dir() else None
        if warm is None:
            print(f"  seam {re}@{mesh}: seed {seed_name} missing restart; cold",
                  flush=True)
        run_one(re, "seam", mesh, warm, gpu, role="seam")
    print("SEAMS COMPLETE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["up", "dn", "seam"])
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--band", default=None,
                    choices=["lowre", "pilot", "highre", "ultra"])
    args = ap.parse_args()
    if not gpu_is_free(args.gpu):
        print(f"GPU {args.gpu} busy -- waiting for it to free", flush=True)
        wait_for_gpu(args.gpu)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.part == "seam":
        run_seams(args.gpu)
    else:
        run_ladder(args.part, args.gpu, band=args.band)
    print(f"PART {args.part} done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
