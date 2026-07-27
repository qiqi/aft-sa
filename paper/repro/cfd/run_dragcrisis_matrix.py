"""Drag-crisis STEADY-ONLY campaign matrix driver (user directive 2026-07-27).

Matrix: Re in {6e4, 1e5, 1.5e5, 2e5, 2.5e5, 3e5, 3.5e5, 4e5, 5e5, 7e5, 1e6,
2e6} x chi_inf(Tu ~ {0.05, 0.2, 0.7}%) x {up, dn} continuation ladders
(previous-Re converged state as init; the run_continuation_ladders.py
pattern) + one COLD two-stage start per Re at the middle seed (protocol-
neutral reference). Steady pseudo-transient ONLY (campaign settings; the
pilot's steady twin protocol). L1 pilot mesh family (one mesh, hardlinked
into every case; y+ <= 0.64 at Re 2e6 per mesh_stats).

One driver process = ONE chain (a ladder direction at one seed, or the cold
reference sweep), bound to one GPU. GPU DISCIPLINE (shared team cap): only
GPUs 0-3 are ever considered (4-7 belong to another user); before EVERY case
the driver re-verifies its GPU carries no foreign compute process, else it
WAITS (yields gracefully).

Convergence per case, monitored live from total_forces_v2.csv:
  converged   : |dCD| over the last 1000 pseudo steps < tol -> graceful
                stop.json (solver writes surface/slice/restart outputs).
  limit_cycle : tail CD peak-to-peak > 10*tol at the cap -> FINDING,
                recorded verbatim (residual/force stats), outputs kept.
  cap         : neither -> recorded as unconverged-drift.
Steady non-convergence anywhere (especially through the crisis) is a
FINDING, not a failure; nothing is tuned around it.

Per-case extraction (appended to matrix_summary.jsonl in the out root):
Cd/CL medians + tail stats, Cf-knee separation angle (validated convention),
mean-Cf crossings (for LSB reattachment structure), shoulder/base Cp,
chi=1 near-wall front, verdict + pseudo-step count.

Usage:
  python run_dragcrisis_matrix.py --chain up:0.2 --gpu 0      # ladder
  python run_dragcrisis_matrix.py --chain cold:0.2 --gpu 1    # cold refs
  (--template builds/uses OUT/template_case; default OUT below)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_dragcrisis_pilot import make_env, canonical_env          # noqa: E402
import dragcrisis_pilot_forces as F                               # noqa: E402

OUT = Path("/local_data/qiqi/sa-ai/dragcrisis_matrix")
PILOT_STEADY = Path("/local_data/qiqi/sa-ai/dragcrisis_pilot/cylL1_Re100k_steady")

MACH = 0.1
RE_LIST = [6e4, 1e5, 1.5e5, 2e5, 2.5e5, 3e5, 3.5e5, 4e5, 5e5, 7e5, 1e6, 2e6]
# Mack e^N map (lib/calibrate_kernel.chi_inf_from_Tu_pct), computed once:
SEEDS = {"0.05": 3.8895e-4, "0.2": 1.0835e-2, "0.7": 2.1908e-1}
ALLOWED_GPUS = (0, 1, 2, 3)          # 4-7 are another user's -- never touch

# (min_pseudo, cap, fSlow) per protocol stage
COLD_STAGES = [(6000, 30000, 0.1), (4000, 15000, 0.01)]
WARM_STAGES = [(3000, 15000, 0.1), (3000, 8000, 0.01)]
FLAT_TOL = 1.0e-3                    # |dCD| per 1000 pseudo steps
BIGFILES = ("mesh.cgns", "mesh.cgns_rank_1_of_1.dmp",
            "mesh.cgns.partitionerData.npart.1_rank_1_of_1.dmp",
            "mesh.msh", "cylinder_ogrid.p3d")   # hardlinked (disk: 99% full)


def retag(re):
    # LOSSLESS tag. NOT {:.0e}: that rounds 1.5e5 and 2.5e5 both onto "2e5"
    # and 3.5e5 onto "4e5" -- name collisions that clobber ladder case dirs
    # (hit on the first launch; caught by duplicate ticker rows).
    return str(int(re))


def gpu_is_free(gpu: int) -> bool:
    """True iff GPU carries no compute process (anyone's)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
             "--format=csv,noheader"], capture_output=True, text=True).stdout
        uuid = subprocess.run(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid",
             "--format=csv,noheader"], capture_output=True, text=True
        ).stdout.strip()
        return uuid not in out
    except Exception:
        return False


def wait_for_gpu(gpu: int):
    assert gpu in ALLOWED_GPUS, f"GPU {gpu} is outside the team allotment"
    waited = 0
    while not gpu_is_free(gpu):
        if waited % 600 == 0:
            print(f"  [gpu {gpu}] busy (foreign/team process) -- yielding...",
                  flush=True)
        time.sleep(30)
        waited += 30


# ---------------------------------------------------------------------------
def make_case(case_dir: Path, re: float, chi_inf: float):
    """Clone the steady template: hardlink big mesh files, copy the small
    preprocess products, patch muRef. Seeds/fSlow are patched per stage."""
    tmpl = OUT / "template_case"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    for f in os.listdir(tmpl):
        src = tmpl / f
        if not src.is_file():
            continue
        if f in BIGFILES:
            os.link(src, case_dir / f)
        else:
            shutil.copy2(src, case_dir / f)
    p = case_dir / "Flow360.json"
    d = json.loads(p.read_text())
    d["freestream"]["muRef"] = MACH / re
    # matrix trims: slice+surface only (84 cases); final restart dump only
    d.pop("volumeOutput", None)
    d["runControl"]["dumpRestartFilesFrequency"] = -1
    p.write_text(json.dumps(d, indent=1))
    json.dump({"re": re, "chi_inf": chi_inf},
              open(case_dir / "case_meta.json", "w"), indent=1)


def build_template():
    """Template = the pilot's steady twin case dir (mesh + preprocess
    products), cleaned of outputs."""
    tmpl = OUT / "template_case"
    if tmpl.exists():
        return
    tmpl.mkdir(parents=True)
    for f in os.listdir(PILOT_STEADY):
        src = PILOT_STEADY / f
        if not src.is_file():
            continue
        if (f.endswith((".pvtu", ".vtu", ".csv", ".log", ".gltf", ".json.fork"))
                or f.startswith(("restart", "ipc", "stop", "surface_forces",
                                 "autovis", "progress"))):
            continue
        shutil.copy2(src, tmpl / f)
    print(f"template ready: {tmpl}")


def patch_stage_json(case_dir: Path, chi_inf: float, fslow: float,
                     extra: int, restart: bool):
    p = case_dir / "Flow360.json"
    d = json.loads(p.read_text())
    seed = chi_inf * fslow
    d["freestream"]["turbulenceQuantities"] = {
        "modelType": "ModifiedTurbulentViscosityRatio",
        "modifiedTurbulentViscosityRatio": seed}
    for bc in d.get("boundaries", {}).values():
        if bc.get("type") == "Freestream":
            bc["turbulenceQuantities"] = {
                "modelType": "ModifiedTurbulentViscosityRatio",
                "modifiedTurbulentViscosityRatio": seed}
    d["runControl"]["restart"] = restart
    d["timeStepping"]["maxPseudoSteps"] = extra   # pseudo counter resets (NEW)
    p.write_text(json.dumps(d, indent=1))


def read_cd_trace(case_dir: Path):
    ps, cl, cd = [], [], []
    fp = case_dir / "total_forces_v2.csv"
    if not fp.exists():
        return ps, cl, cd
    import csv as _csv
    with open(fp) as f:
        for row in _csv.reader(f):
            try:
                ps.append(int(float(row[1])))
                cl.append(float(row[2]))
                cd.append(float(row[3]))
            except (ValueError, IndexError):
                continue
    return ps, cl, cd


def solve_stage(case_dir: Path, env: dict, gpu: int, min_pseudo: int,
                cap: int, tol: float) -> str:
    """Popen the solver + postprocessor; poll CD flatness; stop.json when
    flat. Returns 'converged' | 'limit_cycle' | 'cap'."""
    import bisect
    (case_dir / "stop.json").unlink(missing_ok=True)
    senv = dict(env)
    senv.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
                 "OMPI_COMM_WORLD_LOCAL_RANK": "0", "OMPI_COMM_WORLD_RANK": "0",
                 "OMPI_COMM_WORLD_SIZE": "1"})
    find = senv.pop("_find")
    sock = case_dir / "ipc_control.sock"
    sock.unlink(missing_ok=True)
    pp_log = open(case_dir / "postprocessor.log", "a")
    pp = subprocess.Popen([find("columnarDataProcessor.py"), "--asyncMode",
                           "--inputSimulationJson", "simulation.json",
                           "--columnarDataProcessorJson", "columnar.json"],
                          cwd=str(case_dir), env=senv, stdout=pp_log,
                          stderr=subprocess.STDOUT)
    verdict = "cap"
    try:
        for _ in range(40):
            if sock.exists():
                break
            time.sleep(0.5)
        slog = open(case_dir / "solver.log", "a")
        sol = subprocess.Popen([find("Flow360Solver")], cwd=str(case_dir),
                               env=senv, stdout=slog, stderr=subprocess.STDOUT)
        stopped = False
        while sol.poll() is None:
            time.sleep(10)
            ps, cl, cd = read_cd_trace(case_dir)
            # current stage's rows only (pseudo counter resets per stage)
            k = len(ps) - 1
            while k > 0 and ps[k - 1] <= ps[k]:
                k -= 1
            ps, cd = ps[k:], cd[k:]
            if not ps or stopped:
                continue
            cur = ps[-1]
            i = bisect.bisect_left(ps, cur - 1000)
            if cur >= min_pseudo and abs(cd[-1] - cd[i]) < tol:
                (case_dir / "stop.json").write_text("{}")
                verdict = "converged"
                stopped = True
        if sol.returncode not in (0, None):
            raise RuntimeError(f"solver rc={sol.returncode} in {case_dir}")
        slog.close()
    finally:
        pp.terminate()
        try:
            pp.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pp.kill()
        pp_log.close()
    if verdict == "cap":
        ps, cl, cd = read_cd_trace(case_dir)
        k = len(ps) - 1
        while k > 0 and ps[k - 1] <= ps[k]:
            k -= 1
        tail = cd[k:][-2000:]
        if tail and (max(tail) - min(tail)) > 10 * tol:
            verdict = "limit_cycle"
    (case_dir / "stop.json").unlink(missing_ok=True)
    return verdict


def run_case(case_dir: Path, chi_inf: float, gpu: int, stages,
             warm_from: Path | None) -> dict:
    env, find = make_env()
    env["_find"] = find
    restart = False
    if warm_from is not None:
        ro = warm_from / "restartOutput"
        if ro.is_dir():
            for f in os.listdir(ro):
                shutil.copy2(ro / f, case_dir / f)
            restart = True
        else:
            print(f"  WARNING: {ro} missing; falling back to cold stages",
                  flush=True)
            stages = COLD_STAGES
    verdicts = []
    t0 = time.time()
    for si, (min_pseudo, cap, fslow) in enumerate(stages):
        wait_for_gpu(gpu)
        patch_stage_json(case_dir, chi_inf, fslow, cap,
                         restart or si > 0)
        e = dict(env)
        e.update(canonical_env(chi_inf, fslow))
        v = solve_stage(case_dir, e, gpu, min_pseudo, cap, FLAT_TOL)
        verdicts.append(f"fSlow{fslow}:{v}")
        # promote restart for the next stage
        ro = case_dir / "restartOutput"
        if ro.is_dir():
            for f in os.listdir(ro):
                shutil.copy2(ro / f, case_dir / f)
        # archive the stage force history
        tf = case_dir / "total_forces_v2.csv"
        if tf.exists():
            shutil.copy2(tf, case_dir / f"total_forces_v2.stage{si}.csv")
    return {"verdicts": verdicts, "wall_s": round(time.time() - t0, 1)}


def extract(case_dir: Path, re: float) -> dict:
    import numpy as np
    nu = MACH / re
    ps, cl, cd = read_cd_trace(case_dir)
    k = len(ps) - 1
    while k > 0 and ps[k - 1] <= ps[k]:
        k -= 1
    clw, cdw = np.array(cl[k:]), np.array(cd[k:])
    tail = min(len(cdw), 3000)
    s = {"Cd": float(np.median(cdw[-tail:])),
         "CL": float(np.median(clw[-tail:])),
         "Cd_tail_p2p": float(cdw[-tail:].max() - cdw[-tail:].min()),
         "CL_tail_p2p": float(clw[-tail:].max() - clw[-tail:].min()),
         "pseudo_final": int(ps[-1]) if ps else 0}
    try:
        sides = F.surface_mean_cfcp(str(case_dir), avg=False)
        for side in ("upper", "lower"):
            phi, cf, cp = sides[side]
            s[f"knee_{side}"] = F.cf_knee(phi, cf)
            s[f"crossings_{side}"] = F.crossings(phi, cf)
            s[f"Cp_base_{side}"] = float(np.interp(180.0, phi, cp))
            sh = (phi > 40) & (phi < 110)
            i = np.argmin(cp[sh])
            s[f"Cp_shoulder_{side}"] = float(cp[sh][i])
            s[f"phi_Cp_shoulder_{side}"] = float(phi[sh][i])
    except Exception as e:                                    # noqa: BLE001
        s["surface_note"] = repr(e)
    try:
        fronts = F.chi_front(str(case_dir), nu, avg=False)
        for side in ("upper", "lower"):
            b, prof = fronts[side]
            ok = np.isfinite(prof) & (prof >= 1.0)
            s[f"chi1_front_{side}"] = float(b[ok].min()) if ok.any() else None
    except Exception as e:                                    # noqa: BLE001
        s["front_note"] = repr(e)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True,
                    help="up:<Tu> | dn:<Tu> | cold:<Tu>  (Tu in "
                         + "/".join(SEEDS) + ")")
    ap.add_argument("--gpu", type=int, required=True, choices=ALLOWED_GPUS)
    args = ap.parse_args()
    direction, tu = args.chain.split(":")
    chi = SEEDS[tu]
    OUT.mkdir(parents=True, exist_ok=True)
    build_template()

    res = RE_LIST if direction in ("up", "cold") else RE_LIST[::-1]
    prev_dir = None
    summary_fp = OUT / "matrix_summary.jsonl"
    for i, re in enumerate(res):
        name = f"cyl_Re{retag(re)}_Tu{tu}_{direction}"
        case_dir = OUT / name
        warm = prev_dir if (direction in ("up", "dn") and i > 0) else None
        stages = WARM_STAGES if warm is not None else COLD_STAGES
        print(f"=== {name} (gpu {args.gpu}, {'warm' if warm else 'cold'}) ===",
              flush=True)
        make_case(case_dir, re, chi)
        info = run_case(case_dir, chi, args.gpu, stages, warm)
        row = {"case": name, "re": re, "Tu": tu, "dir": direction, **info,
               **extract(case_dir, re)}
        json.dump(row, open(case_dir / "summary.json", "w"), indent=1)
        with open(summary_fp, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  -> {row['verdicts']} Cd={row.get('Cd'):.4f} "
              f"CL={row.get('CL'):+.4f} knee={row.get('knee_upper')} "
              f"({info['wall_s']:.0f}s)", flush=True)
        prev_dir = case_dir
    print(f"CHAIN {args.chain} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
