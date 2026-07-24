"""Extend the Re=1e5 warm-start forks until they plateau or return
(pass-12 major 1: the 20k-iteration fork histories were still decaying).

Each extension is a fresh case dir seeded from the parent's end-of-run
restart (restartOutput/), same Reynolds number, 40,000 further
pseudo-steps. Extended are the two direct forks (2e5 -> 1e5) and the two
up-ladder ends (6e4 burst -> 8e4 -> 1e5), so both branches at 1e5 get the
same 60k-iteration total history.

  python3 run_fork_extensions.py str 6
  python3 run_fork_extensions.py cav 7

Prints quartile CL medians + end slope so the plateau-vs-merge verdict is
read straight off the log; results in FR/fork_extension_results.json.
"""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.solve import run_solver
import numpy as np
from run_continuation_ladders import clone, forces, FR, canon_env, write_ai_constants


def history_diag(wd):
    rows = [r for r in list(csv.reader(open(f"{wd}/total_forces_v2.csv")))[1:] if len(r) > 3]
    cl = np.array([float(r[2]) for r in rows])
    q = [float(np.median(c)) for c in np.array_split(cl, 4)]
    n = len(cl)
    a, b = np.polyfit(np.arange(n - n//10, n), cl[-(n//10):], 1)
    return dict(quartiles=[round(v, 4) for v in q],
                end_slope_per_1000it=round(float(a)*100*10, 6))


def main():
    fam, gpu = sys.argv[1], int(sys.argv[2])
    jobs = [
        (f"ext_fork_{fam}L2_Re100k_a5", f"{FR}/fork_{fam}L2_Re100k_a5"),
        (f"ext_up_{fam}L2_Re100k_a5", f"{FR}/ladder_up_{fam}L2_Re100k_a5"),
    ]
    res_path = f"{FR}/fork_extension_results.json"
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for tag, seed in jobs:
        wd = f"{FR}/{tag}"
        print(f"START {tag} (seed {os.path.basename(seed)})", flush=True)
        try:
            clone(seed, wd, 100)
            j = json.load(open(f"{wd}/Flow360.json"))
            j['timeStepping']['maxPseudoSteps'] = 40000
            json.dump(j, open(f"{wd}/Flow360.json", 'w'), indent=4)
            env, find = canon_env()
            run_solver(wd, find, env, gpu=gpu, timeout=28800)
            write_ai_constants(wd)
            r = forces(wd)
            r.update(history_diag(wd))
            results[tag] = r
            print(f"DONE {tag}: {r}", flush=True)
        except Exception as e:
            results[tag] = dict(err=str(e)[:160])
            print(f"FAIL {tag}: {e}", flush=True)
        json.dump(results, open(res_path, "w"), indent=1)
    print(f"EXT-{fam.upper()}-DONE", flush=True)


if __name__ == '__main__':
    main()
