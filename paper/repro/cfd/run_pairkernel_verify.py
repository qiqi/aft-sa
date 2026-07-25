"""Verification battery for the PAIR-form invariant kernel with the
shear-significance guard (AI_INVARIANT_KERNEL=2): same three cases as the
original battery, cold 40k, fSlow default (canon env minus bypass to match
the existing controls), plus an Eppler control leg.
Pass criterion: ON == control within line widths.
  python3 run_pairkernel_verify.py <slot 0..3> <gpu>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.solve import run_solver
from run_continuation_ladders import clone, forces, FR, canon_env, write_ai_constants

SLOTS = [
    ("pairk_nlf_str_a4",  "strL2prop_nlf0416_Re4M_a4",      4000, "2"),
    ("pairk_nlf_cav_a4",  "cavL2prop_nlf0416_Re4M_a4",      4000, "2"),
    ("pairk_epp_a5",      "strL2prop_eppler387_Re200k_a5",  200,  "2"),
    ("pairkctl_epp_a5",   "strL2prop_eppler387_Re200k_a5",  200,  "0"),
]
slot, gpu = int(sys.argv[1]), int(sys.argv[2])
tag, src, Rk, inv = SLOTS[slot]
wd = f"{FR}/{tag}"
clone(f"{FR}/{src}", wd, Rk)
j = json.load(open(f"{wd}/Flow360.json"))
j['runControl']['restart'] = False
j['timeStepping']['maxPseudoSteps'] = 40000
json.dump(j, open(f"{wd}/Flow360.json", 'w'), indent=4)
for f in ("restart.json", "restart_rank_1_of_1.dmp"):
    p = f"{wd}/{f}"
    os.path.exists(p) and os.remove(p)
env, find = canon_env()
env["AI_INVARIANT_KERNEL"] = inv
env["AI_FV1BYPASS"] = "0"        # match the existing invkctl controls
print(f"START {tag} gpu={gpu}", flush=True)
run_solver(wd, find, env, gpu=gpu, timeout=14400)
write_ai_constants(wd)
print(f"DONE {tag}: {forces(wd)}", flush=True)
print(f"PAIRK-DONE {tag}", flush=True)
