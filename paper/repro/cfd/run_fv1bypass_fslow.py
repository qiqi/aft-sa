"""fv1-bypass fSlow-schedule battery: is the 1e5 bypass limit cycle a
pseudo-time artifact of the laminar-slowdown schedule, or a genuine
bubble burst-rebuild oscillation of the modified model?

Baseline (run_fv1bypass_test.py): fSlow=0.01, center 7.1, width 4 ->
CL cycle 0.73-1.08 (std 0.12). Variants, all cold strL2 Re=1e5 a5 with
AI_FV1BYPASS=1:
  wide : fSlow=0.01, center 30, width 10 (slow zone spans the handover)
  mild : fSlow=0.1, canon center/width
  off  : fSlow=1.0 (raw pseudo-time dynamics)
The JSON seed carries 0.01*chi_inf (pre-compensated for the canon
schedule), so each variant rescales it by fSlow/0.01.

  python3 run_fv1bypass_fslow.py [gpu]
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
VARIANTS = [
    ("wide", {"AI_LAMINAR_SLOWDOWN": "0.01", "AI_SLOWCENTER": "30", "AI_SLOWWIDTH": "10"}, 0.01),
    ("mild", {"AI_LAMINAR_SLOWDOWN": "0.1"}, 0.1),
    ("off",  {"AI_LAMINAR_SLOWDOWN": "1.0"}, 1.0),
]
for name, envmod, fslow in VARIANTS:
    tag = f"fv1byp_strL2_Re100k_a5_fslow_{name}"
    wd = f"{FR}/{tag}"
    clone(f"{FR}/sweep_strL2_Re100k_a5", wd, 100)
    j = json.load(open(f"{wd}/Flow360.json"))
    j['runControl']['restart'] = False
    j['timeStepping']['maxPseudoSteps'] = 25000
    # seed is pre-compensated for fSlow=0.01; rescale for this schedule
    # (appears under freestream AND the farfield boundary)
    def rescale(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == 'modifiedTurbulentViscosityRatio':
                    d[k] = v * (fslow / 0.01)
                else:
                    rescale(v)
    rescale(j)
    json.dump(j, open(f"{wd}/Flow360.json", 'w'), indent=4)
    for f in ("restart.json", "restart_rank_1_of_1.dmp"):
        p = f"{wd}/{f}"
        os.path.exists(p) and os.remove(p)
    env, find = canon_env()
    env["AI_FV1BYPASS"] = "1"
    env.update(envmod)
    print(f"START {tag}", flush=True)
    run_solver(wd, find, env, gpu=gpu, timeout=14400)
    write_ai_constants(wd)
    r = forces(wd)
    rows = [x for x in list(csv.reader(open(f"{wd}/total_forces_v2.csv")))[1:] if len(x) > 3]
    cl = np.array([float(x[2]) for x in rows])[-500:]
    print(f"DONE {tag}: {r}  cycle {cl.min():.4f}-{cl.max():.4f} std {cl.std():.4f}", flush=True)
print("FSLOW-DONE", flush=True)
