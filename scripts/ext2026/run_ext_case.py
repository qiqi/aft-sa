"""Run one case of the 2026-08 extension matrix (see cases_ext.py).

Cloned from the L2 canon base of the same mesh family, cold-started at the
canon environment, front-converged with converge_by_xtr -- byte-for-byte the
recipe of scripts/run_nlfneg_case.py, generalised so that BOTH alphaAngle and
muRef are patched.

  python3 run_ext_case.py <case_name> <gpu>
"""
import sys, os, json, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/home/qiqi/flexcompute/flexfoil/rans",
          "/home/qiqi/flexcompute/sa-ai/paper/repro",
          "/home/qiqi/flexcompute/sa-ai/paper/repro/cfd",
          "/home/qiqi/flexcompute/sa-ai/paper/repro/driver",
          HERE):
    sys.path.insert(0, p)

from run_continuation_ladders import clone, forces, FR, canon_env, write_ai_constants
from cases_ext import BY_NAME

DRIVER = "/home/qiqi/flexcompute/sa-ai/paper/repro/driver/converge_by_xtr.py"

name, gpu = sys.argv[1], sys.argv[2]
c = BY_NAME[name]
wd = f"{FR}/{name}"
src = f"{FR}/{c['base']}"

# clone() patches muRef from its Rk argument (kR in thousands) and flips
# restart on; we re-patch below for the cold start, so Rk here is only a
# first guess. Pass the real Reynolds so the JSON is already correct if the
# re-patch is ever skipped.
clone(src, wd, c["re"] / 1000.0)

j = json.load(open(f"{wd}/Flow360.json"))
j["freestream"]["alphaAngle"] = c["alpha"]
j["freestream"]["muRef"] = c["muRef"]
j["runControl"]["restart"] = False           # cold start at the canon seed
json.dump(j, open(f"{wd}/Flow360.json", "w"), indent=4)
for f in ("restart.json", "restart_rank_1_of_1.dmp"):
    p = f"{wd}/{f}"
    os.path.exists(p) and os.remove(p)

env, _ = canon_env()
penv = dict(os.environ)
penv.update(env)
# converge_by_xtr runs as a subprocess: hand it the import roots explicitly
penv["PYTHONPATH"] = ":".join([
    "/home/qiqi/flexcompute/flexfoil/rans",
    "/home/qiqi/flexcompute/sa-ai/paper/repro",
    "/home/qiqi/flexcompute/sa-ai/paper/repro/cfd",
    "/home/qiqi/flexcompute/sa-ai/paper/repro/driver",
    penv.get("PYTHONPATH", ""),
]).rstrip(":")

t0 = time.time()
print(f"START {name} block={c['block']} alpha={c['alpha']} Re={c['re']:.3g} "
      f"muRef={c['muRef']:.6g} gpu={gpu}", flush=True)
# --min-batches 3: a cold impulsive start sits on a flat all-laminar sentinel
# plateau (~0.99, ~0.99) for 2-4 batches before the true front appears, which
# the 2-consecutive-batch test alone mistakes for convergence. This only DELAYS
# the convergence declaration -- the tolerance test itself is the canon one.
rc = subprocess.run([sys.executable, DRIVER, wd, "--gpu", str(gpu),
                     "--min-batches", "3", "--max-batches", "16"],
                    env=penv).returncode
write_ai_constants(wd)
try:
    r = forces(wd)
except Exception as e:
    r = f"<forces unavailable: {e}>"
print(f"DONE {name}: rc={rc} {r} wall={time.time()-t0:.0f}s", flush=True)
print(f"EXT2026-CASE-DONE {name} rc={rc}", flush=True)
