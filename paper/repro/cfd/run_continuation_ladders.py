"""Slow-continuation ladders for the Eppler a5 bistability study (Sec. eppresweep).

Down-ladder: start from the converged Re=2e5 L2 solution and lower the
Reynolds number ONE STEP AT A TIME (150k -> 100k -> 80k -> 60k), each step
warm-started from the previous step's end-of-run restart dump
(restartOutput/), 20k pseudo-steps each. Tests whether the attached branch
survives a slow descent further than the direct 2e5->1e5 / 2e5->6e4 forks.

Up-ladder: start from the cold-start (burst) 6e4 solution and raise Re
slowly (80k -> 100k). Tests whether the burst branch persists upward into
the model's bistable band, i.e. hysteresis in Re.

One family per invocation (they run in parallel on different GPUs):
  python3 run_continuation_ladders.py str 6
  python3 run_continuation_ladders.py cav 7
Results: FR/ladder_{fam}_results.json (per-step CL/CD medians + drift).
"""
import sys, os, json, csv, shutil, time
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.env import make_env
from rans.solve import run_solver
import saai_env
import numpy as np

FR = "/home/qiqi/flexcompute/sa-ai/flow360_fr"
MACH = 0.1
STEPS = 20000


def canon_env():
    """The campaign env: canonical kernel constants + the laminar pseudo-time
    slowdown (AI_LAMINAR_SLOWDOWN=0.01). The slowdown scales the freestream-BC
    enforcement too, so the campaign case JSONs carry a pre-compensated seed
    (modifiedTurbulentViscosityRatio = 0.01 * chi_inf); every clone made here
    copies that JSON, so the slowdown env var is LOAD-BEARING -- without it
    the effective freestream seed is 100x low."""
    env, find = make_env()
    env.update(saai_env.canonical_ai_env())
    env["AI_LAMINAR_SLOWDOWN"] = "0.01"
    return env, find


def write_ai_constants(wd):
    """Extract the solver's resolved-constants echo into ai_constants.log
    (same per-case provenance record as the campaign runner)."""
    try:
        lines = open(f"{wd}/solver.log", errors='ignore').read().splitlines(keepends=True)
        i = next((k for k, l in enumerate(lines)
                  if 'SA-AI transition constants' in l), None)
        if i is not None:
            open(f"{wd}/ai_constants.log", 'w').writelines(lines[i:i + 16])
    except OSError:
        pass
# outputs never copied into a clone (everything else is a solver input)
OUT_PAT = ('.csv', '.log', '.vtu', '.pvtu', '.gltf', '.sock')
OUT_NAMES = {'restartOutput', 'ipc_data', 'progress.csv', 'timer.json',
             'metaData2DPlots.json', 'Flow360_processed.json',
             'restart.json', 'restart_rank_1_of_1.dmp'}


def clone(src, dst, Rk):
    """New case dir seeded from src's end-of-run restart, at Reynolds Rk*1000."""
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for f in os.listdir(src):
        if f in OUT_NAMES or f.endswith(OUT_PAT):
            continue
        sp = os.path.join(src, f)
        if not os.path.isfile(sp):
            continue
        if os.path.getsize(sp) > 10e6:
            os.link(sp, os.path.join(dst, f))     # mesh + partitioner dumps
        else:
            shutil.copy(sp, os.path.join(dst, f))
    # seed: the PREVIOUS step's end-of-run state
    ro = os.path.join(src, 'restartOutput')
    if not os.path.isdir(ro):                     # cold-start source dirs
        ro = src
    for f in ('restart.json', 'restart_rank_1_of_1.dmp'):
        shutil.copy(os.path.join(ro, f), os.path.join(dst, f))
    j = json.load(open(os.path.join(dst, 'Flow360.json')))
    j['freestream']['muRef'] = MACH/(Rk*1000.0)
    j['runControl']['restart'] = True
    j['timeStepping']['maxPseudoSteps'] = STEPS
    json.dump(j, open(os.path.join(dst, 'Flow360.json'), 'w'), indent=4)


def forces(wd):
    rows = [r for r in list(csv.reader(open(f"{wd}/total_forces_v2.csv")))[1:] if len(r) > 3]
    t = rows[int(0.8*len(rows)):]
    cl = float(np.median([float(r[2]) for r in t]))
    cd = float(np.median([float(r[3]) for r in t]))
    drift = max(np.ptp([float(r[2]) for r in t[-100:]]),
                np.ptp([float(r[3]) for r in t[-100:]]))
    return dict(CL=round(cl, 4), CD=round(cd, 5), drift=round(drift, 6))


def main():
    fam, gpu = sys.argv[1], int(sys.argv[2])
    ladders = [
        ('dn', f"{FR}/{fam}L2prop_eppler387_Re200k_a5", (150, 100, 80, 60)),
        ('up', f"{FR}/sweep_{fam}L2_Re60k_a5", (80, 100)),
    ]
    res_path = f"{FR}/ladder_{fam}_results.json"
    results = {}
    for direction, seed, res in ladders:
        src = seed
        for Rk in res:
            tag = f"ladder_{direction}_{fam}L2_Re{Rk}k_a5"
            wd = f"{FR}/{tag}"
            print(f"START {tag} (seed {os.path.basename(src)})", flush=True)
            t0 = time.time()
            try:
                clone(src, wd, Rk)
                env, find = canon_env()
                run_solver(wd, find, env, gpu=gpu, timeout=14400)
                write_ai_constants(wd)
                dmp = f"{wd}/restartOutput/restart_rank_1_of_1.dmp"
                assert os.path.getmtime(dmp) > t0, "no fresh restart dump"
                results[tag] = forces(wd)
                print(f"DONE {tag}: {results[tag]}", flush=True)
                src = wd                          # chain
            except Exception as e:
                results[tag] = dict(err=str(e)[:160])
                print(f"FAIL {tag}: {e}", flush=True)
                break                             # broken chain: stop ladder
            json.dump(results, open(res_path, "w"), indent=1)
    json.dump(results, open(res_path, "w"), indent=1)
    print(f"LADDER-{fam.upper()}-DONE", flush=True)


if __name__ == '__main__':
    main()
