"""10k-step alpha sweeps: SA-AI (chi=0.02) AND fully-turbulent SA (chi=3) for
NACA0012 + NLF0416, parallel across GPUs. Edits Flow360.json in place (no rebuild)."""
import os, sys, json, shutil, threading, csv
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver, extract_forces

B = "/home/qiqi/flexcompute/aft-sa/flow360"
SKIP = ('.pvtu', '.vtu', '.csv', '.log', '.sock', '.gltf', '.pvd')
bases = {'naca0012': f"{B}/out_naca0012_lam", 'nlf0416': f"{B}/out_nlf0416_lam"}
alphas = [-2, 0, 2, 4, 6, 8]
jobs = [(m, af, a) for af in ('naca0012', 'nlf0416') for a in alphas for m in ('aftsa', 'turb')]

def stage(model, af, a):
    wd = f"{B}/run10k_{model}_{af}_a{a}"
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    base = bases[af]
    for f in os.listdir(base):
        if any(f.endswith(x) for x in SKIP):
            continue
        s, d = os.path.join(base, f), os.path.join(wd, f)
        (shutil.copytree if os.path.isdir(s) else shutil.copy)(s, d)
    cf = json.load(open(f"{wd}/Flow360.json"))
    cf['timeStepping']['maxPseudoSteps'] = 10000
    cf['freestream']['alphaAngle'] = float(a)
    if model == 'turb':   # fully turbulent reference
        cf['boundaries']['fluid']['farfield']['turbulenceQuantities']['modifiedTurbulentViscosityRatio'] = 3.0
    json.dump(cf, open(f"{wd}/Flow360.json", "w"), indent=1)
    return wd

results = {}; sem = threading.Semaphore(8)
def worker(model, af, a, gpu):
    with sem:
        try:
            wd = stage(model, af, a)
            env, find = make_env()
            if model == 'aftsa':
                env["AI_SA"] = "1"
            run_solver(wd, find, env, gpu=gpu, timeout=2400)
            f = extract_forces(wd)
            results[(model, af, a)] = (f["CL"], f["CD"])
            print(f"  done {model} {af} a={a}: CL={f['CL']:.4f} CD={f['CD']:.5f}", flush=True)
        except Exception as e:
            results[(model, af, a)] = ("ERR", str(e)[:80])
            print(f"  FAIL {model} {af} a={a}: {e}", flush=True)

threads = []
for i, (m, af, a) in enumerate(jobs):
    t = threading.Thread(target=worker, args=(m, af, a, i % 8)); t.start(); threads.append(t)
for t in threads:
    t.join()

with open(f"{B}/run10k_results.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["model", "airfoil", "alpha", "CL", "CD"])
    for (m, af, a), v in sorted(results.items()):
        w.writerow([m, af, a, v[0], v[1]])
print("ALL DONE")
