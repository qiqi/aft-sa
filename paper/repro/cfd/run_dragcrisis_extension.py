"""Drag-crisis matrix EXTENSION: single-model Cd(Re) traverse of the composite
curve (user directive 2026-07-28). Extends the completed 84-case steady matrix
(run_dragcrisis_matrix.py, record 2026-07-28-0010) to the smallest and largest
Re the same model + canon env can reach. Same model, same constants, same
staged-fSlow steady protocol; only the MESH changes per arm:

  arm        mesh fam   cases
  ---------  ---------  ---------------------------------------------------
  lowarm*    pilot      dn-continuation below the matrix: 4e4, 2e4, 1e4 (all
                        3 seeds) + 3e3, 1e3, 300 (Tu 0.2 only), warm from the
                        matrix cyl_Re60000_Tu*_dn endpoints.
  creep      lowre      Re 1..1000 (Tu 0.2): physically STEADY below Re~47
                        (validation vs Dennis-Chang/Fornberg); R_OUT=1000 D
                        (low-Re blockage is logarithmic), y1=1e-3 D, 600x121.
  creep300   lowre300   R_OUT=300 twin at Re 1 and 30: far-field sensitivity.
  high*      highre     Re 4e6, 7e6, 1e7 (+2e7 stretch if the 1e7 up-point
                        force-converges), all 3 seeds; cold two-stage at 4e6
                        (restarts do NOT cross meshes), up-ladder, dn-ladder
                        back down (Tu 0.2 dn extended to 2e6 = highre seam
                        point). y1=1e-6 D (y+ est 0.36 @1e7 / 0.69 @2e7),
                        N_SURF=1600 (the shoulder LSB shrinks with Re; 1200
                        gave ~10 cells across the 1.5-deg bubble at 2e6 --
                        1600 keeps ~9 cells/deg as it shrinks further), 170
                        layers, growth<=1.1, R_OUT=100.
  seam4e6    pilot      Re 4e6 Tu 0.2, warm from the matrix 2e6 up endpoint.

SEAM OVERLAPS for the composite curve (same Re + seed on both meshes):
Re=300, 1e3 on lowre+pilot; Re=2e6, 4e6 on pilot+highre. Deltas reported;
a mismatch is a finding, not something to hide.

Case names: pilot-family cases keep the matrix scheme cyl_Re<int>_Tu<tu>_<dir>
(LOSSLESS integer tags -- the retag-collision lesson); new-mesh cases append
_<fam> so seam duplicates cannot collide. Rows appended to the SAME
matrix_summary.jsonl with an extra "mesh" field (absent == pilot).

Discipline: at most 2 GPUs campaign-wide (2 worker processes, one GPU each,
acquired per case via lockfile + fully-idle occupancy check, ALLOWED_GPUS
only); >=6 GiB free on /local_data checked before every launch (abort
otherwise); solver.log truncated + constants echo preserved after each case;
superseded ladder restarts purged as the chain advances (endpoints kept).

Usage (one worker per GPU slot):
  python run_dragcrisis_extension.py --chains high02,high005,high07
  python run_dragcrisis_extension.py --chains creep,creep300,lowarm02,lowarm005,lowarm07,seam4e6
  python run_dragcrisis_extension.py --templates-only   # build family templates
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_dragcrisis_matrix import (                                # noqa: E402
    OUT, MACH, SEEDS, COLD_STAGES, WARM_STAGES, ALLOWED_GPUS, BIGFILES,
    retag, gpu_is_free, make_case, run_case, extract)
from dragcrisis_matrix_janitor import (                            # noqa: E402
    truncate_solver_log, purge_restarts)
import dragcrisis_pilot_forces as F                                # noqa: E402
import numpy as np                                                 # noqa: E402

MESHBUILD = OUT / "meshbuild"
TEMPLATES = {"pilot": OUT / "template_case",
             "lowre": OUT / "template_lowre",
             "lowre300": OUT / "template_lowre300",
             "highre": OUT / "template_highre"}
MIN_FREE_GB = 6.0

# fam, tu, dir, Re list, warm-seed case dir (None = cold first point)
PLAN = {
    "lowarm02":  ("pilot", "0.2", "dn", [4e4, 2e4, 1e4, 3e3, 1e3, 300],
                  OUT / "cyl_Re60000_Tu0.2_dn"),
    "lowarm005": ("pilot", "0.05", "dn", [4e4, 2e4, 1e4],
                  OUT / "cyl_Re60000_Tu0.05_dn"),
    "lowarm07":  ("pilot", "0.7", "dn", [4e4, 2e4, 1e4],
                  OUT / "cyl_Re60000_Tu0.7_dn"),
    "creep":     ("lowre", "0.2", "up", [1, 3, 10, 30, 100, 300, 1000], None),
    "creep300":  ("lowre300", "0.2", "up", [1, 30], None),
    "seam4e6":   ("pilot", "0.2", "up", [4e6], OUT / "cyl_Re2000000_Tu0.2_up"),
    # protocol-neutral highre point at the 2e6 seam: the dn-ladder point
    # alone conflates the mesh-family offset with the up/dn branch spread
    # seen on highre through the supercritical band
    "seam2e6h":  ("highre", "0.2", "cold", [2e6], None),
}
HIGH_TU = {"high005": "0.05", "high02": "0.2", "high07": "0.7"}
HIGH_UP = [4e6, 7e6, 1e7]
HIGH_STRETCH = 2e7


def case_name(fam: str, re: float, tu: str, direction: str) -> str:
    base = f"cyl_Re{retag(re)}_Tu{tu}_{direction}"
    return base if fam == "pilot" else f"{base}_{fam}"


# ---------------------------------------------------------------------------
def build_family_template(fam: str):
    """Template = the family's built steady twin (meshbuild/cyl_<fam>_steady),
    big mesh files hardlinked (one physical copy per family)."""
    tmpl = TEMPLATES[fam]
    if tmpl.exists():
        return
    src = MESHBUILD / f"cyl_{fam}_steady"
    assert src.is_dir(), f"steady twin missing: {src} (run the mesh builds)"
    tmpl.mkdir(parents=True)
    for f in os.listdir(src):
        s = src / f
        if not s.is_file():
            continue
        if f in BIGFILES:
            os.link(s, tmpl / f)
        else:
            shutil.copy2(s, tmpl / f)
    print(f"template ready: {tmpl}", flush=True)


# ---------------------------------------------------------------------------
def free_gb() -> float:
    st = os.statvfs(OUT)
    return st.f_bavail * st.f_frsize / 2 ** 30


def disk_guard():
    if free_gb() >= MIN_FREE_GB:
        return
    # one recovery attempt: truncate logs of all completed extension cases
    for case in OUT.glob("cyl_Re*"):
        if case.is_dir() and (case / "summary.json").exists():
            truncate_solver_log(case)
    if free_gb() < MIN_FREE_GB:
        raise SystemExit(f"DISK ABORT: {free_gb():.1f} GiB free < "
                         f"{MIN_FREE_GB} GiB required -- not launching")


# ---------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_gpu(tag: str) -> int:
    """First ALLOWED gpu that is fully idle AND not locked by another
    extension worker. Lockfile carries pid for staleness detection."""
    announced = False
    while True:
        for g in ALLOWED_GPUS:
            lock = OUT / f"extgpu{g}.lock"
            if lock.exists():
                try:
                    pid = int(lock.read_text().split()[0])
                except (ValueError, IndexError):
                    pid = None
                if pid and _pid_alive(pid):
                    continue
                lock.unlink(missing_ok=True)          # stale lock
            if not gpu_is_free(g):
                continue
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, f"{os.getpid()} {tag}".encode())
            os.close(fd)
            if gpu_is_free(g):                        # re-check post-lock
                return g
            lock.unlink(missing_ok=True)
        if not announced:
            print("  [gpu] no fully idle allowed GPU -- waiting...", flush=True)
            announced = True
        time.sleep(60)


def release_gpu(g: int):
    (OUT / f"extgpu{g}.lock").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
def chi_max_nearwall(case_dir: Path, re: float) -> float | None:
    """Max near-wall chi over both sides (laminar-passivity check for the
    creeping arm: SA-AI must be inert there)."""
    try:
        fronts = F.chi_front(str(case_dir), MACH / re, avg=False)
        return float(max(np.nanmax(p) for _, p in fronts.values()))
    except Exception:                                     # noqa: BLE001
        return None


def run_one(fam: str, re: float, tu: str, direction: str, chain: str,
            warm_dir: Path | None) -> Path:
    name = case_name(fam, re, tu, direction)
    case_dir = OUT / name
    if (case_dir / "summary.json").exists():
        print(f"=== {name} already done, skipping ===", flush=True)
        return case_dir
    disk_guard()
    chi = SEEDS[tu]
    gpu = acquire_gpu(name)
    try:
        make_case(case_dir, re, chi, tmpl=TEMPLATES[fam])
        stages = WARM_STAGES if warm_dir is not None else COLD_STAGES
        print(f"=== {name} (gpu {gpu}, "
              f"{'warm<-' + warm_dir.name if warm_dir else 'cold'}) ===",
              flush=True)
        info = run_case(case_dir, chi, gpu, stages, warm_dir)
    finally:
        release_gpu(gpu)
    row = {"case": name, "re": re, "Tu": tu, "dir": direction, "mesh": fam,
           "chain": chain,
           "warm_src": warm_dir.name if warm_dir else None, **info,
           **extract(case_dir, re)}
    row["chi_max_nearwall"] = chi_max_nearwall(case_dir, re)
    json.dump(row, open(case_dir / "summary.json", "w"), indent=1)
    with open(OUT / "matrix_summary.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"  -> {row['verdicts']} Cd={row.get('Cd'):.4f} "
          f"CL={row.get('CL'):+.4f} chi_max={row.get('chi_max_nearwall')} "
          f"({info['wall_s']:.0f}s)", flush=True)
    truncate_solver_log(case_dir)
    # superseded-restart janitor: the predecessor's restart is no longer
    # needed once this case is summarized (chain endpoints keep theirs)
    if warm_dir is not None and (warm_dir / "summary.json").exists():
        purge_restarts(warm_dir)
    return case_dir


def final_stage_converged(case_dir: Path) -> bool:
    """Gate for the 2e7 stretch: force-converged, or a mild steady breather
    (the supercritical points routinely carry small limit cycles -- the 4e6
    highre cold point flagged at Cd_tail_p2p=0.027; that must not veto the
    stretch)."""
    try:
        s = json.load(open(case_dir / "summary.json"))
        return (s["verdicts"][-1].endswith(":converged")
                or s.get("Cd_tail_p2p", 1.0) < 0.05)
    except Exception:                                     # noqa: BLE001
        return False


def run_chain(chain: str):
    if chain in HIGH_TU:
        tu = HIGH_TU[chain]
        prev = None
        for re in HIGH_UP:
            prev = run_one("highre", re, tu, "up", chain, prev)
        top_re = HIGH_UP[-1]
        if final_stage_converged(prev):
            prev = run_one("highre", HIGH_STRETCH, tu, "up", chain, prev)
            top_re = HIGH_STRETCH
        else:
            print(f"  [chain {chain}] 1e7 up-point not force-converged; "
                  f"skipping the 2e7 stretch", flush=True)
        dn_res = [r for r in [1e7, 7e6, 4e6] if r < top_re]
        if tu == "0.2":
            dn_res.append(2e6)                 # highre seam point
        for re in dn_res:
            prev = run_one("highre", re, tu, "dn", chain, prev)
        return
    fam, tu, direction, res, seed = PLAN[chain]
    prev = seed
    for re in res:
        prev = run_one(fam, re, tu, direction, chain, prev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", default="",
                    help="comma list from: " + ",".join(
                        list(PLAN) + list(HIGH_TU)))
    ap.add_argument("--templates-only", action="store_true")
    args = ap.parse_args()
    for fam in TEMPLATES:
        if fam != "pilot":
            build_family_template(fam)
    if args.templates_only:
        return
    for chain in [c for c in args.chains.split(",") if c]:
        print(f"##### CHAIN {chain} #####", flush=True)
        run_chain(chain)
    print("ALL CHAINS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
