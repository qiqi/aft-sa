"""Rerun the 5 Flow360 flat-plate cases (no mesh rebuild) in parallel."""
import os, sys, shutil, threading, time, json
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver

B = "/home/qiqi/flexcompute/aft-sa/flow360"
TU_LIST = [0.026, 0.06, 0.18, 0.30, 0.85]
def case_dir(Tu): return f"{B}/flatplate_aftsa_Tu{int(round(Tu*1000)):04d}"

SKIP = ('.pvtu','.vtu','.pvd','.gltf','.log','.sock')

def reset(cd):
    for f in os.listdir(cd):
        if any(f.endswith(s) for s in SKIP) or f.startswith('ipc') or f.endswith('_v2.csv') or f == 'progress.csv':
            sp = os.path.join(cd, f)
            try:
                if os.path.isfile(sp): os.remove(sp)
                elif os.path.isdir(sp): shutil.rmtree(sp)
            except: pass
        elif f in ('restartOutput',) or f in ('restart.json',) or f == 'restart_rank_1_of_1.dmp':
            sp = os.path.join(cd, f)
            try:
                if os.path.isfile(sp): os.remove(sp)
                elif os.path.isdir(sp): shutil.rmtree(sp)
            except: pass
    # Make sure restart is False
    p = f"{cd}/Flow360.json"
    d = json.load(open(p))
    d['runControl']['restart'] = False
    json.dump(d, open(p, 'w'), indent=1)

def run_case(Tu, gpu, sem, lock, results):
    cd = case_dir(Tu)
    with sem:
        env, find = make_env()
        env["AI_SA"] = "1"; env["AI_LAMINAR_SLOWDOWN"] = "0.01"
        t0 = time.time()
        tag = f"Tu{Tu}"
        print(f"[{tag}] launching on GPU {gpu}", flush=True)
        try:
            run_solver(cd, find, env, gpu=gpu, timeout=10800)
            lastF = open(f"{cd}/total_forces_v2.csv").readlines()[-1]
            lastR = open(f"{cd}/nonlinear_residual_v2.csv").readlines()[-1]
            with lock:
                results[tag] = {'dt_s': time.time()-t0,
                                'res': lastR.strip().split(',')[2],
                                'CD': lastF.strip().split(',')[3]}
            print(f"[{tag}] DONE in {results[tag]['dt_s']:.0f}s  res={results[tag]['res']}  CD={results[tag]['CD']}", flush=True)
        except Exception as e:
            print(f"[{tag}] FAILED: {e}", flush=True)
            with lock: results[tag] = {'error': str(e)}

if __name__ == '__main__':
    for Tu in TU_LIST:
        reset(case_dir(Tu))
        print(f"reset {case_dir(Tu)}", flush=True)
    sem = threading.Semaphore(8); lock = threading.Lock(); results = {}
    threads = [threading.Thread(target=run_case, args=(Tu, i%8, sem, lock, results))
               for i, Tu in enumerate(TU_LIST)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"Done. {len(results)}/{len(TU_LIST)}")
    for k,v in sorted(results.items()): print(f"  {k}: {v}")
