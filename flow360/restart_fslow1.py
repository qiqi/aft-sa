"""Restart the high-alpha L2 sigma_FPG-off NLF cases with AI_LAMINAR_SLOWDOWN=1
(full-speed laminar dynamics) to converge the transition front ~100x faster and
see where it settles. Continues from the fSlow=0.01 field (restart=True).

CRITICAL: with fSlow=1 the freestream needs NO compensation, so the BC seed is set
to the uncompensated desired value 8.76e-4 (was 8.76e-6 = 8.76e-4 * fSlow). The
current in-domain freestream is already 8.76e-4, so this is a smooth restart.
"""
import os, sys, json, shutil, threading, time
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver

B = "/home/qiqi/flexcompute/aft-sa/flow360"
CASES = [("str", 9), ("cav", 9), ("str", 15), ("cav", 15)]   # L2
DESIRED_CHI = 8.76e-4
MAXSTEPS = 30000

def cdir(m, a): return f"{B}/{m}L2prop_nlf0416_Re4M_a{a}"

def patch(d):
    # preserve the fSlow=0.01 (unconverged) forces for comparison
    src = f"{d}/total_forces_v2.csv"
    if os.path.exists(src) and not os.path.exists(f"{d}/total_forces_v2.csv.fslow01"):
        shutil.copy2(src, f"{d}/total_forces_v2.csv.fslow01")
    # solver reads restart from the case-dir root; the run wrote it to restartOutput/
    ro = f"{d}/restartOutput"
    for fn in ('restart.json', 'restart_rank_1_of_1.dmp'):
        s = f"{ro}/{fn}"
        if os.path.exists(s):
            shutil.copy2(s, f"{d}/{fn}")
    p = f"{d}/Flow360.json"; c = json.load(open(p))
    tq = {'modelType': 'ModifiedTurbulentViscosityRatio', 'modifiedTurbulentViscosityRatio': DESIRED_CHI}
    c['freestream']['turbulenceQuantities'] = tq
    for bn, bc in c.get('boundaries', {}).items():
        if 'farfield' in bn or bc.get('type') == 'Freestream':
            bc['turbulenceQuantities'] = dict(tq)
    c['runControl']['restart'] = True
    c['timeStepping']['maxPseudoSteps'] = MAXSTEPS
    json.dump(c, open(p, 'w'), indent=1)

def main():
    sem = threading.Semaphore(8); lock = threading.Lock(); res = {}
    def worker(m, a, gpu):
        d = cdir(m, a); tag = f"{m}L2_a{a}"
        patch(d)
        env, find = make_env(); env['AI_SA'] = '1'; env['AI_LAMINAR_SLOWDOWN'] = '1'
        t0 = time.time()
        print(f"[{tag}] restart on GPU {gpu} (fSlow=1, BC=8.76e-4)", flush=True)
        try:
            run_solver(d, find, env, gpu=gpu, timeout=10800)
            c = open(f"{d}/total_forces_v2.csv").readlines()[-1].strip().split(',')
            with lock: res[tag] = {'dt': round(time.time()-t0), 'CL': float(c[2]), 'CD': float(c[3])}
            print(f"[{tag}] DONE {res[tag]['dt']}s  CL={res[tag]['CL']:.4f} CD={res[tag]['CD']:.5f}", flush=True)
        except Exception as e:
            with lock: res[tag] = {'error': str(e)}; print(f"[{tag}] FAILED: {e}", flush=True)
    ts = [threading.Thread(target=worker, args=(m, a, i)) for i, (m, a) in enumerate(CASES)]
    for t in ts: t.start()
    for t in ts: t.join()
    json.dump(res, open(f"{B}/restart_fslow1_results.json", 'w'), indent=1)
    print("\n=== fSlow=1 restart results (compare to .sigfpg_on baseline) ===", flush=True)
    for k, v in sorted(res.items()): print(f"  {k}: {v}", flush=True)
    print("DONE_ALL", flush=True)

if __name__ == '__main__':
    main()
