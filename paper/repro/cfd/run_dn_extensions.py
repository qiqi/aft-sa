"""Extend the slow-descent (ladder_dn) Re=1e5 states by 40k iterations --
the fourth initialization path of the monostability demonstration."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.solve import run_solver
from run_continuation_ladders import clone, forces, FR, canon_env, write_ai_constants
from run_fork_extensions import history_diag

fam, gpu = sys.argv[1], int(sys.argv[2])
tag = f"ext_dn_{fam}L2_Re100k_a5"
wd = f"{FR}/{tag}"
print(f"START {tag}", flush=True)
clone(f"{FR}/ladder_dn_{fam}L2_Re100k_a5", wd, 100)
j = json.load(open(f"{wd}/Flow360.json"))
j['timeStepping']['maxPseudoSteps'] = 40000
json.dump(j, open(f"{wd}/Flow360.json", 'w'), indent=4)
env, find = canon_env()
run_solver(wd, find, env, gpu=gpu, timeout=28800)
write_ai_constants(wd)
r = forces(wd); r.update(history_diag(wd))
p = f"{FR}/fork_extension_results.json"
m = json.load(open(p)) if os.path.exists(p) else {}
m[tag] = r
json.dump(m, open(p, "w"), indent=1)
print(f"DONE {tag}: {r}", flush=True)
