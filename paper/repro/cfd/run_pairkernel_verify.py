"""Verification battery for the GRAM (pair) invariant-kernel variants:
mode 2 = fully unprojected, mode 3 = <l,l> wall-parallel-projected (the
projected-u form and the s_hat form were falsified earlier the same day).
Same three cases as the original battery, cold 40k, canon env minus bypass
to match the existing controls. VERDICT (2026-07-25): both modes fail
identically (+13.7/+23.6/+23.8 counts) — signedness itself rectifies
omega-direction noise in weak shear; the magnitude kernel (Gram-clothed,
Eq. eq:gram of the paper) is canon.
  python3 run_pairkernel_verify.py <slot 0..5> <gpu>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.solve import run_solver
from run_continuation_ladders import clone, forces, FR, canon_env, write_ai_constants

SLOTS = [
    # mode 2 = fully unprojected Gram; mode 3 = <l,l> projected, <u,u> full
    ("gram2_nlf_str_a4",  "strL2prop_nlf0416_Re4M_a4",      4000, "2"),
    ("gram2_nlf_cav_a4",  "cavL2prop_nlf0416_Re4M_a4",      4000, "2"),
    ("gram2_epp_a5",      "strL2prop_eppler387_Re200k_a5",  200,  "2"),
    ("gram3_nlf_str_a4",  "strL2prop_nlf0416_Re4M_a4",      4000, "3"),
    ("gram3_nlf_cav_a4",  "cavL2prop_nlf0416_Re4M_a4",      4000, "3"),
    ("gram3_epp_a5",      "strL2prop_eppler387_Re200k_a5",  200,  "3"),
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
