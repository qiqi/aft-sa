"""Bistability probe: warm-start Eppler a5 L2 cases at lower Re from the
converged Re=200k (short-bubble) solutions, GPU 7, sequential."""
import sys, os, json, csv
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.env import make_env
from rans.solve import run_solver
import saai_env
import numpy as np

FR = "/home/qiqi/flexcompute/sa-ai/flow360_fr"
results = {}
for tag in ("fork_strL2_Re100k_a5", "fork_cavL2_Re100k_a5",
            "fork_strL2_Re60k_a5", "fork_cavL2_Re60k_a5"):
    wd = f"{FR}/{tag}"
    env, find = make_env()
    env.update(saai_env.canonical_ai_env())
    print(f"START {tag}", flush=True)
    try:
        run_solver(wd, find, env, gpu=7, timeout=14400)
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
json.dump(results, open(f"{FR}/fork_bistability_results.json", "w"), indent=1)
print("FORKS-DONE", flush=True)
