"""2D verification of the INVARIANT (shear-direction) sphere kernel
(AI_INVARIANT_KERNEL=1) against the standard magnitude-triple form:
per the author's plan, run one attached case (NLF(1)-0416, alpha=4,
Re=4e6, structured L2) and one bubble case (Eppler 387, alpha=5, Re=2e5,
structured L2) with the flag ON, cold start, and compare forces against
the canon record.  Expectations: on parallel layers the two forms are
algebraically identical, so the NLF case should agree to solver noise;
the Eppler case exercises the mixed-sign recirculation layer where the
magnitude triple deviates from the projective class -- differences there
measure exactly the region the invariant form fixes and should be within
line widths.  Only the verified invariant form then goes to the 3D
spheroid (task #23).

  python3 run_invariant_kernel_verify.py [gpu]
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
# (tag, source case to clone, Reynolds in k, canon reference CL/CD)
JOBS = [
    ("invk_nlf0416_Re4M_a4",   f"{FR}/strL2prop_nlf0416_Re4M_a4",      4000,
     "canon: see nlf polar record"),
    ("invk_eppler387_Re200k_a5", f"{FR}/strL2prop_eppler387_Re200k_a5", 200,
     "canon: tab:eppresweep 2e5 row"),
]
for tag, src, Rk, note in JOBS:
    wd = f"{FR}/{tag}"
    clone(src, wd, Rk)
    j = json.load(open(f"{wd}/Flow360.json"))
    j['runControl']['restart'] = False
    j['timeStepping']['maxPseudoSteps'] = 20000
    json.dump(j, open(f"{wd}/Flow360.json", 'w'), indent=4)
    for f in ("restart.json", "restart_rank_1_of_1.dmp"):
        p = f"{wd}/{f}"
        os.path.exists(p) and os.remove(p)
    env, find = canon_env()
    env["AI_INVARIANT_KERNEL"] = "1"
    print(f"START {tag} ({note})", flush=True)
    run_solver(wd, find, env, gpu=gpu, timeout=14400)
    write_ai_constants(wd)
    r = forces(wd)
    rows = [x for x in list(csv.reader(open(f"{wd}/total_forces_v2.csv")))[1:] if len(x) > 3]
    cl = np.array([float(x[2]) for x in rows])[-500:]
    print(f"DONE {tag}: {r}  cycle {cl.min():.4f}-{cl.max():.4f} std {cl.std():.4f}", flush=True)
print("INVK-VERIFY-DONE", flush=True)
