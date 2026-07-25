"""Extend an invariant-kernel verification case (or its control) by restart to
40k pseudo-steps -- the cold-20k snapshots were still mid-transient (the
transition front converges over ~30k+ steps at Re=4e6), so kernel comparisons
are only valid at matched, converged protocol.
  python3 extend_invk_cases.py <case> <gpu> [invariant:0|1] [steps]
"""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.solve import run_solver
from run_continuation_ladders import forces, FR, canon_env, write_ai_constants
case, gpu = sys.argv[1], int(sys.argv[2])
inv = sys.argv[3] == '1' if len(sys.argv) > 3 else False
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 40000
wd = f"{FR}/{case}"
# the solver reads restart_rank_*.dmp from the case root; the end-of-run
# state is written to restartOutput/
import shutil
for f in ("restart.json", "restart_rank_1_of_1.dmp"):
    src = f"{wd}/restartOutput/{f}"
    if os.path.exists(src):
        shutil.copy(src, f"{wd}/{f}")
j = json.load(open(f"{wd}/Flow360.json"))
j['runControl']['restart'] = True
j['timeStepping']['maxPseudoSteps'] = steps
json.dump(j, open(f"{wd}/Flow360.json", 'w'), indent=4)
env, find = canon_env()
if inv:
    env["AI_INVARIANT_KERNEL"] = "1"
print(f"EXTEND {case} to {steps} (invariant={inv}) gpu={gpu}", flush=True)
run_solver(wd, find, env, gpu=gpu, timeout=14400)
write_ai_constants(wd)
print(f"DONE {case}: {forces(wd)}", flush=True)
print(f"EXTEND-DONE {case}", flush=True)
