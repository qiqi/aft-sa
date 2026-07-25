"""Negative-incidence NLF(1)-0416 cases (task #22): alpha = -4 (near zero
lift: the experimental lift curve crosses CL=0 at -3.90 deg) and alpha = -8
(the edge of the experimental data, CL ~ -0.47).  Cold start at the canon
whole-equation environment, cloned from the finest-grid a4 cases; finer
levels can follow once these two anchor the polar's negative branch.

  python3 run_negalpha_nlf.py <fam: str|cav> <alpha: -4|-8> <gpu>
"""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.solve import run_solver
from run_continuation_ladders import clone, forces, FR, canon_env, write_ai_constants
import numpy as np

fam, alpha, gpu = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
tag = f"{fam}L2prop_nlf0416_Re4M_am{int(round(-alpha))}"
wd = f"{FR}/{tag}"
clone(f"{FR}/{fam}L2prop_nlf0416_Re4M_a4", wd, 4000)
j = json.load(open(f"{wd}/Flow360.json"))
j['freestream']['alphaAngle'] = alpha
j['runControl']['restart'] = False
j['timeStepping']['maxPseudoSteps'] = 20000
json.dump(j, open(f"{wd}/Flow360.json", 'w'), indent=4)
for f in ("restart.json", "restart_rank_1_of_1.dmp"):
    p = f"{wd}/{f}"
    os.path.exists(p) and os.remove(p)
env, find = canon_env()
print(f"START {tag} alpha={alpha} gpu={gpu}", flush=True)
run_solver(wd, find, env, gpu=gpu, timeout=14400)
write_ai_constants(wd)
r = forces(wd)
rows = [x for x in list(csv.reader(open(f"{wd}/total_forces_v2.csv")))[1:] if len(x) > 3]
cl = np.array([float(x[2]) for x in rows])[-500:]
print(f"DONE {tag}: {r}  cycle {cl.min():.4f}-{cl.max():.4f} std {cl.std():.4f}", flush=True)
print(f"NEGALPHA-DONE {tag}", flush=True)
