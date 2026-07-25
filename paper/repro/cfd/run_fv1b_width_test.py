"""fv1-bypass s(chi)-width probe: does restoring the old window's strength
(AI_FV1_SWIDTH=1 vs the sigma_t-tied default 4) recover the 1e5 bubble
closure while staying inert on NLF / the flat plate?  OFF controls and
width-4 ON runs already exist from the promotion battery.
  python3 run_fv1b_width_test.py <slot 0..7> <gpu>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.solve import run_solver
from run_continuation_ladders import clone, forces, FR, canon_env, write_ai_constants

SLOTS = [
    ("epp100k_a5", "sweep_strL2_Re100k_a5",  100),
    ("epp60k_a5",  "sweep_strL2_Re60k_a5",   60),
    ("epp2e5_a5",  "strL2prop_eppler387_Re200k_a5", 200),
    ("epp2e5_a0",  "strL2prop_eppler387_Re200k_a0", 200),
    ("nlf_a4",  "strL2prop_nlf0416_Re4M_a4",  4000),
    ("nlf_a15", "strL2prop_nlf0416_Re4M_a15", 4000),
    ("fp_Tu0040", "flatplate_sphere_Tu0040", 1000),
    ("fp_Tu0300", "flatplate_sphere_Tu0300", 1000),
]
slot, gpu = int(sys.argv[1]), int(sys.argv[2])
stem, src, Rk = SLOTS[slot]
tag = f"fv1w1_{stem}"
wd = f"{FR}/{tag}"
clone(f"{FR}/{src}", wd, Rk)
j = json.load(open(f"{wd}/Flow360.json"))
j['runControl']['restart'] = False
j['timeStepping']['maxPseudoSteps'] = 40000

def rescale(d):
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
env["AI_FV1BYPASS"] = "1"
env["AI_FV1_SWIDTH"] = "1"
env["AI_LAMINAR_SLOWDOWN"] = "1"
print(f"START {tag} gpu={gpu}", flush=True)
run_solver(wd, find, env, gpu=gpu, timeout=14400)
write_ai_constants(wd)
print(f"DONE {tag}: {forces(wd)}", flush=True)
print(f"FV1W1-DONE {stem}", flush=True)
