"""NLF convergence investigation: longer runs + lower CFL to see if the limit cycle
settles, and cavity at full 10000 steps."""
import os, json, shutil, threading, csv, numpy as np
import sys; sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver, extract_forces
F = "/home/qiqi/flexcompute/aft-sa/flow360"
SKIP = ('.pvtu', '.vtu', '.csv', '.log', '.sock', '.gltf', '.pvd')

def stage(base, wd, alpha, steps, cflmax=None):
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    for f in os.listdir(base):
        if any(f.endswith(x) for x in SKIP) or f in ('ipc_data', 'restartOutput'): continue
        s, d = os.path.join(base, f), os.path.join(wd, f)
        (shutil.copytree if os.path.isdir(s) else shutil.copy)(s, d)
    cf = json.load(open(f"{wd}/Flow360.json")); cf['freestream']['alphaAngle'] = float(alpha)
    cf['timeStepping']['maxPseudoSteps'] = steps
    if cflmax is not None: cf['timeStepping']['CFL']['max'] = float(cflmax)
    json.dump(cf, open(f"{wd}/Flow360.json", "w"), indent=1); return wd

def cdtail(wd):  # mean & std of CD over last 25% of steps (limit-cycle amplitude)
    rows = list(csv.reader(open(f"{wd}/total_forces_v2.csv"))); h = [x.strip() for x in rows[0]]
    ip, icd = h.index('pseudo_step'), h.index('CD'); s, cd = [], []
    for r in rows[1:]:
        r = [c for c in r if c.strip() != '']
        if len(r) > icd:
            try: s.append(float(r[ip])); cd.append(float(r[icd]))
            except: pass
    s, cd = np.array(s), np.array(cd); m = s > 0.75 * s.max()
    return cd[m].mean(), cd[m].std()

res = {}; lock = threading.Lock(); sem = threading.Semaphore(8)
def run(tag, base, alpha, steps, cfl, aft, gpu):
    with sem:
        try:
            wd = stage(base, f"{F}/conv_{tag}", alpha, steps, cfl)
            env, find = make_env()
            if aft: env["AI_SA"] = "1"
            run_solver(wd, find, env, gpu=gpu, timeout=5400)
            f = extract_forces(wd); m, sd = cdtail(wd)
            with lock: res[tag] = dict(CD=f["CD"], CDmean=m, CDstd=sd)
            print("done %-22s CD=%.5f  CDmean=%.5f  CDstd=%.2e (%.1f%%)" % (tag, f["CD"], m, sd, 100*sd/max(m,1e-9)), flush=True)
        except Exception as e:
            with lock: res[tag] = dict(err=str(e)[:80]); print("FAIL %s %s" % (tag, e), flush=True)

OG, CAV = f"{F}/base_nlf_ogrid", f"{F}/base_nlf_cavity"
jobs = [("og_a4_20k", OG, 4, 20000, None, True),
        ("og_a4_cfl30", OG, 4, 20000, 30, True),
        ("og_a0_20k", OG, 0, 20000, None, True),
        ("cav_a0_10k", CAV, 0, 10000, None, True),
        ("cav_a4_10k", CAV, 4, 10000, None, True),
        ("cav_a6_10k", CAV, 6, 10000, None, True)]
ts = [threading.Thread(target=run, args=(t, b, a, s, c, af, i % 8)) for i, (t, b, a, s, c, af) in enumerate(jobs)]
for t in ts: t.start()
for t in ts: t.join()
json.dump(res, open(f"{F}/run_nlfconv_results.json", "w"), indent=1)
print("CONV TEST DONE:", json.dumps({k: (round(v['CDmean'], 5), round(100*v['CDstd']/max(v['CDmean'], 1e-9), 1)) if 'CDmean' in v else 'ERR' for k, v in res.items()}))
