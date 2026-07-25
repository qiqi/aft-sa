"""fv1-bypass battery over the Eppler cases (author request 2026-07-25):
bypass ON vs OFF at MATCHED protocol (cold start, fSlow=1 i.e. no laminar
slowdown schedule, 40k steps) -- the earlier 1e5/2e5 pairs showed the canon
slowdown schedule contaminates the comparison (pseudo-time limit cycle) and
the protocol itself moves the 2e5 state, so every ON has its own OFF here.

Cases: structured L2 at the benchmark incidences (a0,2,5,7, Re=2e5), the
cavity L2 at a5 (family check), and the structured L2 Reynolds sweep at a5
(60k, 100k, 300k, 460k) -- the sweep probes whether the bypass moves the
BURSTING BOUNDARY (1e5 closed in the first exploration: CL 0.93/CD 0.0231
vs exp 0.873/0.0237; burst branch was 0.77/0.044).

  python3 run_fv1bypass_battery.py <slot 0..8> <gpu>
Each slot runs one (case, ON) then its (case, OFF) sequentially.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.solve import run_solver
from run_continuation_ladders import clone, forces, FR, canon_env, write_ai_constants

# (tag-stem, source case, Re in k)
SLOTS = [
    ("epp2e5_a0",  "strL2prop_eppler387_Re200k_a0",  200),
    ("epp2e5_a2",  "strL2prop_eppler387_Re200k_a2",  200),
    ("epp2e5_a5",  "strL2prop_eppler387_Re200k_a5",  200),
    ("epp2e5_a7",  "strL2prop_eppler387_Re200k_a7",  200),
    ("epp2e5_a5cav", "cavL2prop_eppler387_Re200k_a5", 200),
    ("epp60k_a5",  "strL2prop_eppler387_Re60k_a5",   60),
    ("epp100k_a5", "strL2prop_eppler387_Re100k_a5",  100),
    ("epp300k_a5", "strL2prop_eppler387_Re300k_a5",  300),
    ("epp460k_a5", "strL2prop_eppler387_Re460k_a5",  460),
]

slot, gpu = int(sys.argv[1]), int(sys.argv[2])
stem, src, Rk = SLOTS[slot]
for onoff, envval in (("on", "1"), ("off", "0")):
    tag = f"fv1b_{stem}_{onoff}"
    wd = f"{FR}/{tag}"
    clone(f"{FR}/{src}", wd, Rk)
    j = json.load(open(f"{wd}/Flow360.json"))
    j['runControl']['restart'] = False
    j['timeStepping']['maxPseudoSteps'] = 40000

    def rescale(d):
        """The campaign seed is pre-compensated for slowdown 0.01; at
        fSlow=1 the freestream chi must carry the physical value."""
        if isinstance(d, dict):
            for k, v in d.items():
                if k == 'modifiedTurbulentViscosityRatio':
                    d[k] = v * 100.0
                else:
                    rescale(v)
    rescale(j)
    json.dump(j, open(f"{wd}/Flow360.json", 'w'), indent=4)
    for f in ("restart.json", "restart_rank_1_of_1.dmp"):
        p = f"{wd}/{f}"
        os.path.exists(p) and os.remove(p)
    env, find = canon_env()
    env["AI_FV1BYPASS"] = envval
    env["AI_LAMINAR_SLOWDOWN"] = "1"      # fSlow=1: no schedule artifact
    print(f"START {tag} gpu={gpu}", flush=True)
    run_solver(wd, find, env, gpu=gpu, timeout=14400)
    write_ai_constants(wd)
    print(f"DONE {tag}: {forces(wd)}", flush=True)
print(f"FV1B-SLOT-DONE {stem}", flush=True)
