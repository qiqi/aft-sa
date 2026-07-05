"""Re-run the 8 low-alpha NLF cases (cav+str × L0+L1 × α∈{0,4}) with the
current SA-AI calibration (Phase 1+1c). All other knobs unchanged; chi_inf
for these cases is still set directly in Flow360.json (not via Tu mapping),
so no JSON edits — just reset outputs and rerun.
"""
import os, sys, json, shutil, threading, time
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver

B = "/home/qiqi/flexcompute/aft-sa/flow360"
MESHES = ['cav', 'str']
LEVELS = ['L0', 'L1']
ALPHAS = [0, 4]
SKIP = ('.pvtu','.vtu','.pvd','.gltf','.log','.sock')

def reset(cd):
    for f in os.listdir(cd):
        if any(f.endswith(s) for s in SKIP) or f.startswith('ipc') or f.endswith('_v2.csv') or f == 'progress.csv':
            sp = os.path.join(cd, f)
            try:
                if os.path.isfile(sp): os.remove(sp)
                elif os.path.isdir(sp): shutil.rmtree(sp)
            except: pass
        elif f in ('restartOutput', 'restart.json', 'restart_rank_1_of_1.dmp'):
            sp = os.path.join(cd, f)
            try:
                if os.path.isfile(sp): os.remove(sp)
                elif os.path.isdir(sp): shutil.rmtree(sp)
            except: pass
    p = f"{cd}/Flow360.json"
    d = json.load(open(p))
    d['runControl']['restart'] = False
    d['timeStepping']['maxPseudoSteps'] = 80000
    json.dump(d, open(p, 'w'), indent=1)

def run_case(mesh, L, a, gpu, sem, lock, results):
    cd = f"{B}/{mesh}{L}prop_nlf0416_Re4M_a{a}"
    with sem:
        env, find = make_env()
        env["AI_SA"] = "1"; env["AI_LAMINAR_SLOWDOWN"] = "0.01"
        t0 = time.time()
        tag = f"{mesh}{L}_a{a}"
        print(f"[{tag}] launching on GPU {gpu}", flush=True)
        try:
            run_solver(cd, find, env, gpu=gpu, timeout=10800)
            lastF = open(f"{cd}/total_forces_v2.csv").readlines()[-1]
            lastR = open(f"{cd}/nonlinear_residual_v2.csv").readlines()[-1]
            with lock:
                results[tag] = {'dt_s': time.time()-t0,
                                'res': lastR.strip().split(',')[2],
                                'CL': lastF.strip().split(',')[2],
                                'CD': lastF.strip().split(',')[3]}
            print(f"[{tag}] DONE in {results[tag]['dt_s']:.0f}s  res={results[tag]['res']}  CL={results[tag]['CL']}  CD={results[tag]['CD']}", flush=True)
        except Exception as e:
            print(f"[{tag}] FAILED: {e}", flush=True)
            with lock: results[tag] = {'error': str(e)}

if __name__ == '__main__':
    cases = [(m, L, a) for m in MESHES for L in LEVELS for a in ALPHAS]
    print(f"Running {len(cases)} cases (cav+str × L0+L1 × α∈{{0,4}})")
    for m, L, a in cases:
        cd = f"{B}/{m}{L}prop_nlf0416_Re4M_a{a}"
        if not os.path.exists(cd):
            print(f"SKIP {m}{L}_a{a}: {cd} does not exist"); continue
        reset(cd)
        print(f"  reset {m}{L}_a{a}")
    sem = threading.Semaphore(8); lock = threading.Lock(); results = {}
    threads = [threading.Thread(target=run_case, args=(m, L, a, i%8, sem, lock, results))
               for i, (m, L, a) in enumerate(cases)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"Done. {len(results)}/{len(cases)}")
