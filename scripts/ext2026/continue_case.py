"""Continue an already-cloned extension case that ran out of batches.

converge_by_xtr reloads xtr_history.csv and restarts from the case's current
state, so this simply gives an existing case more batches WITHOUT re-cloning
(which would discard the solution and cold-start again).

Use when a case exits rc=1 having reached --max-batches while its front was
still creeping, e.g. the deep-negative NLF incidences where the pressure-side
front marches most of the chord before settling.

  python3 continue_case.py <case_name> <gpu> [extra_batches]
"""
import sys, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/home/qiqi/flexcompute/flexfoil/rans",
          "/home/qiqi/flexcompute/sa-ai/paper/repro",
          "/home/qiqi/flexcompute/sa-ai/paper/repro/cfd",
          "/home/qiqi/flexcompute/sa-ai/paper/repro/driver",
          HERE):
    sys.path.insert(0, p)

from run_continuation_ladders import forces, FR, canon_env, write_ai_constants

DRIVER = "/home/qiqi/flexcompute/sa-ai/paper/repro/driver/converge_by_xtr.py"

name, gpu = sys.argv[1], sys.argv[2]
extra = sys.argv[3] if len(sys.argv) > 3 else "10"
wd = f"{FR}/{name}"
if not os.path.isdir(wd):
    sys.exit(f"no such case dir: {wd}")

env, _ = canon_env()
penv = dict(os.environ)
penv.update(env)
penv["PYTHONPATH"] = ":".join([
    "/home/qiqi/flexcompute/flexfoil/rans",
    "/home/qiqi/flexcompute/sa-ai/paper/repro",
    "/home/qiqi/flexcompute/sa-ai/paper/repro/cfd",
    "/home/qiqi/flexcompute/sa-ai/paper/repro/driver",
    penv.get("PYTHONPATH", ""),
]).rstrip(":")

t0 = time.time()
print(f"CONTINUE {name} gpu={gpu} extra_batches={extra}", flush=True)
# min-batches 0: this is not a fresh impulsive start, the sentinel plateau is
# long past, so the canon 2-consecutive-batch test applies immediately.
rc = subprocess.run([sys.executable, DRIVER, wd, "--gpu", str(gpu),
                     "--max-batches", str(extra)], env=penv).returncode
write_ai_constants(wd)
try:
    r = forces(wd)
except Exception as e:
    r = f"<forces unavailable: {e}>"
print(f"DONE {name}: rc={rc} {r} wall={time.time()-t0:.0f}s", flush=True)
print(f"EXT2026-CASE-DONE {name} rc={rc}", flush=True)
