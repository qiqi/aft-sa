"""fv1-bypass inertness check at 2e5 with fSlow=1 (raw pseudo-time):
the earlier 2e5 'not inert' verdict (CL cycle std 0.016) used the canon
slowdown schedule, which the 1e5 fSlow battery showed manufactures a
limit cycle. If this run is steady at the canon 2e5 benchmark values,
the bypass is inert at 2e5 and the buffer-layer guard stands.

  python3 run_fv1bypass_fslow2e5.py [gpu]
"""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.solve import run_solver
from run_continuation_ladders import clone, forces, FR, canon_env, write_ai_constants
import numpy as np

gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 6
JOBS = [("fv1byp_strL2_Re200k_a5_fslow_off", f"{FR}/strL2prop_eppler387_Re200k_a5", 200, "1"),
        ("nobyp_strL2_Re200k_a5_fslow_off",  f"{FR}/strL2prop_eppler387_Re200k_a5", 200, "0")]
for tag, src, Rk, byp in JOBS:
    wd = f"{FR}/{tag}"
    clone(src, wd, Rk)
    j = json.load(open(f"{wd}/Flow360.json"))
    j['runControl']['restart'] = False
    j['timeStepping']['maxPseudoSteps'] = 25000
    def rescale(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == 'modifiedTurbulentViscosityRatio':
                    d[k] = v * 100.0          # seed pre-compensated for 0.01
                else:
                    rescale(v)
    rescale(j)
    json.dump(j, open(f"{wd}/Flow360.json", 'w'), indent=4)
    for f in ("restart.json", "restart_rank_1_of_1.dmp"):
        p = f"{wd}/{f}"
        os.path.exists(p) and os.remove(p)
    env, find = canon_env()
    env["AI_LAMINAR_SLOWDOWN"] = "1.0"
    env["AI_FV1BYPASS"] = byp
    print(f"START {tag}", flush=True)
    run_solver(wd, find, env, gpu=gpu, timeout=14400)
    write_ai_constants(wd)
    r = forces(wd)
    rows = [x for x in list(csv.reader(open(f"{wd}/total_forces_v2.csv")))[1:] if len(x) > 3]
    cl = np.array([float(x[2]) for x in rows])[-500:]
    print(f"DONE {tag}: {r}  cycle {cl.min():.4f}-{cl.max():.4f} std {cl.std():.4f}", flush=True)
print("FSLOW2E5-DONE", flush=True)
