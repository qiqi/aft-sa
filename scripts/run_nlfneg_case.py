"""One NLF(1)-0416 negative-incidence case on a coarser grid (L0/L1),
cloned from the fv1 a4 case of the same grid and family, cold-started at
the canon environment, and front-converged with the campaign's
converge_by_xtr protocol (the L2 pair's protocol).

  python3 run_nlfneg_case.py <fam: str|cav> <level: L0|L1> <alpha: -4|-8> <gpu>
"""
import sys, os, json, subprocess
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/cfd")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from run_continuation_ladders import clone, forces, FR, canon_env, write_ai_constants

DRIVER = "/home/qiqi/flexcompute/sa-ai/paper/repro/driver/converge_by_xtr.py"

fam, lev, alpha, gpu = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
tag = f"{fam}{lev}prop_nlf0416_Re4M_am{int(round(-alpha))}"
wd = f"{FR}/{tag}"
clone(f"{FR}/{fam}{lev}prop_nlf0416_Re4M_a4", wd, 4000)
j = json.load(open(f"{wd}/Flow360.json"))
j['freestream']['alphaAngle'] = alpha
j['runControl']['restart'] = False
json.dump(j, open(f"{wd}/Flow360.json", 'w'), indent=4)
for f in ("restart.json", "restart_rank_1_of_1.dmp"):
    p = f"{wd}/{f}"
    os.path.exists(p) and os.remove(p)
env, _ = canon_env()
penv = dict(os.environ)
penv.update(env)
# converge_by_xtr runs as a subprocess: hand it the import roots explicitly
penv['PYTHONPATH'] = ':'.join([
    "/home/qiqi/flexcompute/flexfoil/rans",
    "/home/qiqi/flexcompute/sa-ai/paper/repro",
    "/home/qiqi/flexcompute/sa-ai/paper/repro/cfd",
    "/home/qiqi/flexcompute/sa-ai/paper/repro/driver",
    penv.get('PYTHONPATH', ''),
]).rstrip(':')
print(f"START {tag} alpha={alpha} gpu={gpu}", flush=True)
rc = subprocess.run([sys.executable, DRIVER, wd, '--gpu', str(gpu)],
                    env=penv).returncode
write_ai_constants(wd)
print(f"DONE {tag}: rc={rc} {forces(wd)}", flush=True)
print(f"NLFNEG-CASE-DONE {tag}", flush=True)
