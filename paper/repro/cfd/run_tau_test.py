"""Falsification test for the tau-tuning hypothesis (Sec. epphandover
discussion): cold-start Eppler strL2 with AI_SWITCHWIDTH=1 (canon tau=4).
If the handover bottleneck were the sigma_t ramp, a 4x narrower ramp
should visibly shorten the bubble; if it is the c_v1 (f_v1) amplitude
ladder, reattachment should barely move.

RESULT (both Re): tau=1 turns the steady canon solutions into relaxation
oscillators (limit-cycle dCL ~ 0.3 at 1e5, ~ 0.08 at 2e5) without closing
the 1e5 bubble -- the bottleneck is the amplitude ladder, not the ramp.

  python3 run_tau_test.py [gpu]        # runs Re=1e5 then Re=2e5
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.solve import run_solver
from run_continuation_ladders import clone, forces, FR, canon_env, write_ai_constants

gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 6
JOBS = [("tautest_strL2_Re100k_a5_tau1", f"{FR}/sweep_strL2_Re100k_a5", 100),
        ("tautest_strL2_Re200k_a5_tau1", f"{FR}/strL2prop_eppler387_Re200k_a5", 200)]
for tag, src, Rk in JOBS:
    wd = f"{FR}/{tag}"
    clone(src, wd, Rk)
    j = json.load(open(f"{wd}/Flow360.json"))
    j['runControl']['restart'] = False            # COLD start
    j['timeStepping']['maxPseudoSteps'] = 25000
    json.dump(j, open(f"{wd}/Flow360.json", 'w'), indent=4)
    for f in ("restart.json", "restart_rank_1_of_1.dmp"):
        p = f"{wd}/{f}"
        os.path.exists(p) and os.remove(p)
    env, find = canon_env()
    env["AI_SWITCHWIDTH"] = "1.0"
    print(f"START {tag}", flush=True)
    run_solver(wd, find, env, gpu=gpu, timeout=14400)
    write_ai_constants(wd)
    print(f"DONE {tag}: {forces(wd)}", flush=True)
print("TAUTEST-DONE", flush=True)
