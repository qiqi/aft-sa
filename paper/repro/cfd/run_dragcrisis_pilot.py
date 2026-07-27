"""Stage driver for the drag-crisis PILOT cylinder case (URANS, Re_D=1e5).

SELF-CONTAINED on purpose: this runs on whatever GPU host holds the case
(pilot: 017-v100-dev), so it embeds its own copies of
paper/repro/driver/env.py::make_env and driver/solve.py::run_solver (adapted),
plus the canonical SA-AI sphere-kernel env of driver/saai_env.py -- the
ai_constants.log echo is diffed against a campaign case to prove identity.

Staged unsteady startup (the paper Sec. VII two-stage protocol adapted to
dual-time; convergence.py::_staged_sweep is the steady original):

  stage    target step  alpha  fSlow  notes
  kick             800    3.0    0.1  break the O-grid's symmetry so shedding
                                      onsets promptly (Travin-style kick)
  develop         4000    0.0    0.1  fast front migration while the wake and
                                      shedding limit-cycle develop
  prod           22000    0.0   0.01  production; time-averaging (surface +
                                      slice) accumulates from step 6000 =
                                      >= 30 shedding periods at St ~ 0.19

Seed convention (HARD RULE): the JSON freestream/farfield seed is ALWAYS
chi_inf * fSlow (chi_BC = chi_inf * f), so the effective chi the model sees is
the physical chi_inf = 8.76e-4 at every stage.

physicalSteps semantics: with runControl.caseType == 0 (NEW) + restart, the
solver runs JSON physicalSteps MORE steps (Flow360Solver.cpp: physicalSteps +=
current physicalStep), so each invocation writes target - current.

Usage:  python run_dragcrisis_pilot.py CASE_DIR --stage kick|develop|prod [--gpu 0]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

CHI_INF = 8.76e-4        # LTPT-class physical seed (c_v1 * e^-9)

STAGES = {
    #  name    : (target_total_physical_steps, alpha_deg, fSlow, avg_start)
    "kick":     (800,    3.0, 0.1,  None),
    "develop":  (4000,   0.0, 0.1,  None),
    "prod":     (22000,  0.0, 0.01, 6000),
}

# STEADY twin (the campaign's staged-fSlow pseudo-transient protocol,
# convergence.py::_staged_sweep semantics: maxPseudoSteps is CUMULATIVE,
# stage 2 restarts from stage 1). Run on the steady case dir built with
# build_dragcrisis_pilot.py --steady.
STEADY_STAGES = {
    #  name     : (extra_pseudo_steps, fSlow)
    "steady1":   (30000, 0.1),
    "steady2":   (10000, 0.01),
}


# ---- driver/saai_env.py::canonical_env, inlined ---------------------------
def canonical_env(chi_inf: float, fslow: float) -> dict[str, str]:
    return {
        "AI_SA": "1",
        "AI_RATESCALE": "0.19",
        "AI_REOMC_CEIL": "1851.2",
        "AI_REOMC_A": "124.6",
        "AI_REOMC_B": "1.424",
        "AI_RAMPWIDTH": "0.35",
        "AI_FV1BYPASS": "1",
        "AI_FV1_SWIDTH": "1",
        "AI_SIGMAD_TIE": "1",
        "AFT_CHI_INF": repr(float(chi_inf)),
        "AI_CHI_INF": repr(float(chi_inf)),
        "AI_LAMINAR_SLOWDOWN": repr(float(fslow)),
    }


# ---- driver/env.py::make_env, inlined -------------------------------------
def make_env(compute_root=None):
    root = Path(compute_root or os.environ.get(
        "FLOW360_COMPUTE_ROOT", "/home/qiqi/flexcompute/compute"))
    release = root / "install" / "release"
    venv_bin = root / ".venv" / "bin"
    bindir = release / "bin"
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = str(root / ".venv")
    env["PATH"] = f"{bindir}:{venv_bin}:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{release}/lib:" + env.get("LD_LIBRARY_PATH", "")
    hits = sorted(release.glob("lib/python3.*/site-packages"))
    env["PYTHONPATH"] = (str(hits[0]) if hits else str(release / "lib")) \
        + ":" + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    env["FLOW360_SUPPRESS_BETA_WARNING"] = "1"

    def find(name: str) -> str:
        for d in (bindir, venv_bin):
            p = d / name
            if p.exists():
                return str(p)
        raise FileNotFoundError(f"tool {name!r} not found in {bindir} or {venv_bin}")
    return env, find


# ---- driver/solve.py::run_solver, inlined (in-session direct launch) ------
def run_solver(workdir: Path, find, env: dict, *, gpu: int = 0,
               timeout: int = 6 * 3600) -> None:
    senv = dict(env)
    senv["CUDA_VISIBLE_DEVICES"] = str(gpu)
    senv["OMP_NUM_THREADS"] = "1"
    senv["OMPI_COMM_WORLD_LOCAL_RANK"] = "0"
    senv["OMPI_COMM_WORLD_RANK"] = "0"
    senv["OMPI_COMM_WORLD_SIZE"] = "1"
    sock = workdir / "ipc_control.sock"
    if sock.exists():
        sock.unlink()
    pp_log = open(workdir / "postprocessor.log", "a")
    pp = subprocess.Popen(
        [find("columnarDataProcessor.py"), "--asyncMode",
         "--inputSimulationJson", "simulation.json",
         "--columnarDataProcessorJson", "columnar.json"],
        cwd=str(workdir), env=senv, stdout=pp_log, stderr=subprocess.STDOUT)
    try:
        for _ in range(40):
            if sock.exists():
                break
            time.sleep(0.5)
        with open(workdir / "solver.log", "a") as slog:
            subprocess.run([find("Flow360Solver")], cwd=str(workdir), env=senv,
                           stdout=slog, stderr=subprocess.STDOUT, check=True,
                           timeout=timeout)
    finally:
        pp.terminate()
        try:
            pp.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pp.kill()
        pp_log.close()


def current_physical_step(case_dir: Path) -> int:
    p = case_dir / "total_forces_v2.csv"
    if not p.exists():
        return 0
    last = 0
    with open(p) as f:
        rows = csv.reader(f)
        hdr = next(rows)
        i = [h.strip() for h in hdr].index("physical_step")
        for r in rows:
            if len(r) > i and r[i].strip():
                try:
                    last = int(float(r[i]))
                except ValueError:
                    pass
    return last + 1          # steps completed = last index + 1 (0-based)


def patch_stage(case_dir: Path, target: int, alpha: float, fslow: float,
                avg_start: int | None, cur: int) -> int:
    p = case_dir / "Flow360.json"
    d = json.loads(p.read_text())
    seed = CHI_INF * fslow   # chi_BC = chi_inf * fSlow, ALWAYS
    d["freestream"]["alphaAngle"] = alpha
    d["freestream"]["turbulenceQuantities"][
        "modifiedTurbulentViscosityRatio"] = seed
    for bc in d.get("boundaries", {}).values():
        if bc.get("type") == "Freestream":
            bc.setdefault("turbulenceQuantities", {
                "modelType": "ModifiedTurbulentViscosityRatio"})[
                "modifiedTurbulentViscosityRatio"] = seed
    add = target - cur
    if add <= 0:
        print(f"  stage already complete (cur={cur} >= target={target})")
        return 0
    d["timeStepping"]["physicalSteps"] = add       # ADDITIVE (caseType NEW)
    d["runControl"]["restart"] = cur > 0
    if avg_start is not None:
        # The solver creates averaged writers ONLY from the top-level
        # timeAverageSurfaceOutput / timeAverageSliceOutput sections
        # (Flow360Solver.cpp; the computeTimeAverages flag is an SDK-
        # preprocessing input we bypass). Mirror the instantaneous sections
        # and add the averaging keys; files land as *_time_avg.pvtu, with a
        # running-average snapshot every 2000 steps + a final write.
        avg_keys = {
            "computeTimeAverages": True,
            "startAverageIntegrationStep": avg_start,
            "animationFrequencyTimeAverage": 2000,
            "animationFrequencyTimeAverageOffset": 0,
        }
        sl = d.get("sliceOutput")
        if sl and "timeAverageSliceOutput" not in d:
            d["timeAverageSliceOutput"] = {**json.loads(json.dumps(sl)), **avg_keys}
        if d.get("surfaceOutput") and "timeAverageSurfaceOutput" not in d:
            d["timeAverageSurfaceOutput"] = [
                {**json.loads(json.dumps(so)), **avg_keys}
                for so in d["surfaceOutput"]]
    p.write_text(json.dumps(d, indent=1))
    # each solver invocation TRUNCATES the force/residual CSVs (observed on the
    # pilot); archive the previous stage's history before it is lost.
    tf = case_dir / "total_forces_v2.csv"
    if tf.exists():
        shutil.copy2(tf, case_dir / f"total_forces_v2.through_step{cur}.csv")
    ro = case_dir / "restartOutput"
    if cur > 0 and ro.is_dir():
        for f in os.listdir(ro):
            shutil.copy2(ro / f, case_dir / f)
    return add


def current_pseudo_step(case_dir: Path) -> int:
    """Last pseudo step of a STEADY run (nonlinear_residual_v2.csv col 2)."""
    p = case_dir / "nonlinear_residual_v2.csv"
    cur = 0
    if p.exists():
        for ln in p.read_text().splitlines():
            try:
                cur = int(float(ln.split(",")[1]))
            except (ValueError, IndexError):
                pass
    return cur


def patch_steady_stage(case_dir: Path, extra: int, fslow: float) -> int:
    """convergence.py::_set_stage semantics for the steady twin."""
    p = case_dir / "Flow360.json"
    d = json.loads(p.read_text())
    seed = CHI_INF * fslow
    d["freestream"]["turbulenceQuantities"][
        "modifiedTurbulentViscosityRatio"] = seed
    for bc in d.get("boundaries", {}).values():
        if bc.get("type") == "Freestream":
            bc.setdefault("turbulenceQuantities", {
                "modelType": "ModifiedTurbulentViscosityRatio"})[
                "modifiedTurbulentViscosityRatio"] = seed
    cur = current_pseudo_step(case_dir)
    d["runControl"]["restart"] = cur > 0
    d["timeStepping"]["maxPseudoSteps"] = cur + extra
    p.write_text(json.dumps(d, indent=1))
    ro = case_dir / "restartOutput"
    if cur > 0 and ro.is_dir():
        for f in os.listdir(ro):
            shutil.copy2(ro / f, case_dir / f)
    return extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--stage", required=True,
                    choices=list(STAGES) + list(STEADY_STAGES))
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    case_dir = Path(args.case_dir).resolve()

    if args.stage in STEADY_STAGES:
        extra, fslow = STEADY_STAGES[args.stage]
        add = patch_steady_stage(case_dir, extra, fslow)
        env, find = make_env()
        env.update(canonical_env(CHI_INF, fslow))
        print(f"[{case_dir.name}] {args.stage}: +{add} pseudo steps "
              f"(fSlow={fslow}, seed={CHI_INF * fslow:.3e})", flush=True)
        t0 = time.time()
        run_solver(case_dir, find, env, gpu=args.gpu)
        print(f"[{case_dir.name}] {args.stage} done in {time.time() - t0:.0f}s",
              flush=True)
        return

    target, alpha, fslow, avg_start = STAGES[args.stage]
    cur = current_physical_step(case_dir)
    add = patch_stage(case_dir, target, alpha, fslow, avg_start, cur)
    if add == 0:
        return
    env, find = make_env()
    env.update(canonical_env(CHI_INF, fslow))
    print(f"[{case_dir.name}] stage {args.stage}: steps {cur} -> {target} "
          f"(alpha={alpha}, fSlow={fslow}, seed={CHI_INF * fslow:.3e})", flush=True)
    t0 = time.time()
    run_solver(case_dir, find, env, gpu=args.gpu)
    dt = time.time() - t0
    print(f"[{case_dir.name}] stage {args.stage} done: {add} steps in "
          f"{dt:.0f}s ({dt / add:.3f} s/step)", flush=True)


if __name__ == "__main__":
    main()
