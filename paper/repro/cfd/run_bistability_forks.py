"""Bistability probe: warm-start Eppler a5 L2 cases at lower Re from the
converged Re=200k (short-bubble) solutions, GPU 7, sequential.

Fork construction is run_continuation_ladders.clone(): copy the solver
inputs from the family's {fam}L2prop Re=200k case, seed with its end-of-run
restart (restartOutput/restart_rank_*.dmp + restart.json), and edit
Flow360.json to muRef=0.1/Re, restart=true, maxPseudoSteps=20000. The env
is the campaign canon incl. AI_LAMINAR_SLOWDOWN=0.01 (load-bearing: the
copied JSON seed is slowdown-compensated -- see canon_env()).

Usage: python3 run_bistability_forks.py [fam gpu]   (default: both, GPU 7)
"""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.solve import run_solver
from run_continuation_ladders import clone, canon_env, write_ai_constants
import numpy as np

FR = "/home/qiqi/flexcompute/sa-ai/flow360_fr"
FAMS = (sys.argv[1],) if len(sys.argv) > 2 else ('str', 'cav')
GPU = int(sys.argv[2]) if len(sys.argv) > 2 else 7
results_path = f"{FR}/fork_bistability_results.json"
results = json.load(open(results_path)) if os.path.exists(results_path) else {}
for tag in [f"fork_{f}L2_Re{Rk}k_a5" for f in FAMS for Rk in (100, 60)]:
    wd = f"{FR}/{tag}"
    fam, Rk = tag.split('_')[1][:3], int(tag.split('_Re')[1].split('k')[0])
    clone(f"{FR}/{fam}L2prop_eppler387_Re200k_a5", wd, Rk)
    env, find = canon_env()
    print(f"START {tag}", flush=True)
    try:
        run_solver(wd, find, env, gpu=GPU, timeout=14400)
        write_ai_constants(wd)
        rows = [r for r in list(csv.reader(open(f"{wd}/total_forces_v2.csv")))[1:] if len(r) > 3]
        t = rows[int(0.8*len(rows)):]
        cl = float(np.median([float(r[2]) for r in t]))
        cd = float(np.median([float(r[3]) for r in t]))
        drift = max(np.ptp([float(r[2]) for r in t[-100:]]),
                    np.ptp([float(r[3]) for r in t[-100:]]))
        results[tag] = dict(CL=round(cl, 4), CD=round(cd, 5), drift=round(drift, 6))
        print(f"DONE {tag}: {results[tag]}", flush=True)
    except Exception as e:
        results[tag] = dict(err=str(e)[:120])
        print(f"FAIL {tag}: {e}", flush=True)
    # reload-merge-write: the two family processes share this file
    merged = json.load(open(results_path)) if os.path.exists(results_path) else {}
    merged.update({k: v for k, v in results.items()})
    json.dump(merged, open(results_path, "w"), indent=1)
print("FORKS-DONE", flush=True)
