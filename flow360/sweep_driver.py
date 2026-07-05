"""Parallel SA-AI alpha-sweep across GPUs for NACA0012 and NLF0416 (chi_inf=0.02)."""
import os, sys, json, shutil, threading, csv
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver, extract_forces

BASE = "/home/qiqi/flexcompute/aft-sa/flow360"
SKIP = ('.pvtu', '.vtu', '.csv', '.log', '.sock', '.gltf', '.pvd')

# (tag, base_case_dir, alpha) — base cases already built at chi_inf=0.02
jobs = [("naca0012", f"{BASE}/out_naca0012_lam", a) for a in (-2, 2, 6, 8)] \
     + [("nlf0416",  f"{BASE}/out_nlf0416_lam",  a) for a in (-2, 0, 2, 4, 6, 8)]

def stage(tag, base, a):
    wd = f"{BASE}/sweep_{tag}_a{a}"
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    for f in os.listdir(base):
        if any(f.endswith(x) for x in SKIP):
            continue
        s, d = os.path.join(base, f), os.path.join(wd, f)
        (shutil.copytree if os.path.isdir(s) else shutil.copy)(s, d)
    cf = json.load(open(f"{wd}/Flow360.json"))
    cf['freestream']['alphaAngle'] = float(a)
    json.dump(cf, open(f"{wd}/Flow360.json", "w"), indent=1)
    return wd

results = {}
sem = threading.Semaphore(6)   # cap concurrency

def worker(tag, base, a, gpu):
    with sem:
        try:
            wd = stage(tag, base, a)
            env, find = make_env(); env["AI_SA"] = "1"
            run_solver(wd, find, env, gpu=gpu, timeout=1200)
            f = extract_forces(wd)
            results[(tag, a)] = (f["CL"], f["CD"])
            print(f"  done {tag} a={a}: CL={f['CL']:.4f} CD={f['CD']:.5f}", flush=True)
        except Exception as e:
            results[(tag, a)] = ("ERR", str(e)[:80])
            print(f"  FAIL {tag} a={a}: {e}", flush=True)

threads = []
for i, (tag, base, a) in enumerate(jobs):
    t = threading.Thread(target=worker, args=(tag, base, a, i % 8))
    t.start(); threads.append(t)
for t in threads:
    t.join()

with open(f"{BASE}/sweep_results.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["tag", "alpha", "CL", "CD"])
    for (tag, a), v in sorted(results.items()):
        w.writerow([tag, a, v[0], v[1]])
print("ALL DONE; wrote sweep_results.csv")
