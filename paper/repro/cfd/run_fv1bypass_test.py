"""fv1-bypass test battery (AI_FV1BYPASS=1; see compute ModelConstants.h):
cold Eppler strL2 at 1e5 (does the bubble close, steadily?), cold strL2
at 2e5 (benchmark must stand still), flat plate Tu=0.16% (transition
onset must stand still). nuHat equation untouched; only the nu_t output
map changes in transitional shear layers."""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.solve import run_solver
from run_continuation_ladders import clone, forces, FR, canon_env, write_ai_constants
import numpy as np

fam, gpu = 'str', int(sys.argv[1]) if len(sys.argv) > 1 else 6
JOBS = [("fv1byp_strL2_Re100k_a5", f"{FR}/sweep_strL2_Re100k_a5", 100),
        ("fv1byp_strL2_Re200k_a5", f"{FR}/strL2prop_eppler387_Re200k_a5", 200)]
for tag, src, Rk in JOBS:
    wd = f"{FR}/{tag}"
    clone(src, wd, Rk)
    j = json.load(open(f"{wd}/Flow360.json"))
    j['runControl']['restart'] = False
    j['timeStepping']['maxPseudoSteps'] = 25000
    json.dump(j, open(f"{wd}/Flow360.json", 'w'), indent=4)
    for f in ("restart.json", "restart_rank_1_of_1.dmp"):
        p = f"{wd}/{f}"
        os.path.exists(p) and os.remove(p)
    env, find = canon_env()
    env["AI_FV1BYPASS"] = "1"
    print(f"START {tag}", flush=True)
    run_solver(wd, find, env, gpu=gpu, timeout=14400)
    write_ai_constants(wd)
    r = forces(wd)
    rows = [x for x in list(csv.reader(open(f"{wd}/total_forces_v2.csv")))[1:] if len(x) > 3]
    cl = np.array([float(x[2]) for x in rows])[-500:]
    print(f"DONE {tag}: {r}  cycle {cl.min():.4f}-{cl.max():.4f} std {cl.std():.4f}", flush=True)
print("FV1BYP-DONE", flush=True)
