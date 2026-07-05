"""Stage 2 of the staged-convergence protocol for the high-alpha L2 NLF cases.
Stage 1 (restart_fslow1.py) used fSlow=1 to migrate the transition front to its
settled location fast (but with a mild limit cycle). Stage 2 restarts from that
field with fSlow=0.01 to DAMP the cycle and reach a clean steady state -- the
front is already in place, so this only needs to quench the wobble.

BC re-compensation: fSlow=0.01 requires input = desired * fSlow = 8.76e-6
(back to the compensated value); the in-domain freestream stays at 8.76e-4.
"""
import os, sys, json, shutil, threading, time
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver

B = "/home/qiqi/flexcompute/aft-sa/flow360"
CASES = [("str", 9), ("cav", 9), ("str", 15), ("cav", 15)]   # L2, sigma_FPG OFF
COMP_CHI = 8.76e-6      # compensated BC for fSlow=0.01
MAXSTEPS = 40000

def cdir(m, a): return f"{B}/{m}L2prop_nlf0416_Re4M_a{a}"

def patch(d):
    src = f"{d}/total_forces_v2.csv"
    if os.path.exists(src) and not os.path.exists(f"{d}/total_forces_v2.csv.fslow1"):
        shutil.copy2(src, f"{d}/total_forces_v2.csv.fslow1")   # preserve fSlow=1 (Stage 1)
    ro = f"{d}/restartOutput"
    for fn in ('restart.json', 'restart_rank_1_of_1.dmp'):
        s = f"{ro}/{fn}"
        if os.path.exists(s): shutil.copy2(s, f"{d}/{fn}")
    p = f"{d}/Flow360.json"; c = json.load(open(p))
    tq = {'modelType': 'ModifiedTurbulentViscosityRatio', 'modifiedTurbulentViscosityRatio': COMP_CHI}
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
        env, find = make_env(); env['AI_SA'] = '1'; env['AI_LAMINAR_SLOWDOWN'] = '0.01'
        t0 = time.time()
        print(f"[{tag}] stage2 damp on GPU {gpu} (fSlow=0.01, BC=8.76e-6)", flush=True)
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
    json.dump(res, open(f"{B}/restart_stage2_results.json", 'w'), indent=1)
    print("\n=== Stage-2 (damped) results ===", flush=True)
    for k, v in sorted(res.items()): print(f"  {k}: {v}", flush=True)
    print("DONE_ALL", flush=True)

if __name__ == '__main__':
    main()
