"""Run the 5 NEW-CANON flat plates in flow360_fv1 immediately on idle GPUs
(the orchestrator has them queued last; the user wants the flat-plate paper
section revised first). Skip-guarded on ai_constants.log, so the
orchestrator's own flat-plate block at campaign end is a harmless no-op
recompute of identical canon (or can be ignored).
"""
import sys, os, threading

os.environ["AI_FV1BYPASS"] = "1"
os.environ["AI_FV1_SWIDTH"] = "1"
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/cfd")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.solve import run_solver
from run_continuation_ladders import canon_env, write_ai_constants, forces

NEW = "/home/qiqi/flexcompute/sa-ai/flow360_fv1"
JOBS = (("Tu0040", 0), ("Tu0080", 1), ("Tu0160", 2),
        ("Tu0300", 4), ("Tu0600", 1))


def one(tu, gpu):
    wd = f"{NEW}/flatplate_sphere_{tu}"
    if os.path.exists(f"{wd}/ai_constants.log"):
        print(f"SKIP flatplate {tu} (completed)", flush=True)
        return
    env, find = canon_env()
    print(f"START flatplate {tu} gpu={gpu}", flush=True)
    run_solver(wd, find, env, gpu=gpu, timeout=14400)
    write_ai_constants(wd)
    print(f"DONE flatplate {tu}: {forces(wd)}", flush=True)


th = [threading.Thread(target=one, args=j) for j in JOBS]
for t in th:
    t.start()
for t in th:
    t.join()
print("FV1-FLATPLATES-DONE")
