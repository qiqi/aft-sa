"""Diagnostic: rerun the 5 flat-plate cases with AI_NULAMSCALE=0.25 (1/4 of laminar
viscosity in SA diffusion only) into a parallel set of case dirs *_nulamquarter.
Compare to the existing baseline (nulamscale=1) cases.
"""
import os, sys, json, shutil, threading, time
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver

B = "/home/qiqi/flexcompute/aft-sa/flow360"
TU_LIST = [0.026, 0.06, 0.18, 0.30, 0.85]
SKIP = ('.pvtu','.vtu','.pvd','.gltf','.log','.sock')

def src_cd(Tu): return f"{B}/flatplate_aftsa_Tu{int(round(Tu*1000)):04d}"
def dst_cd(Tu): return f"{B}/flatplate_aftsa_Tu{int(round(Tu*1000)):04d}_nulamquarter"

def clone(src, dst):
    if os.path.exists(dst): shutil.rmtree(dst)
    os.makedirs(dst)
    for f in os.listdir(src):
        if any(f.endswith(s) for s in SKIP) or f.startswith('ipc') or f.endswith('_v2.csv') \
                or f in ('restartOutput','restart.json','restart_rank_1_of_1.dmp','solver.log','postprocessor.log','progress.csv'):
            continue
        sp = os.path.join(src, f)
        if os.path.isfile(sp): shutil.copy2(sp, dst)
    d = json.load(open(f"{dst}/Flow360.json"))
    d['runControl']['restart'] = False
    json.dump(d, open(f"{dst}/Flow360.json", 'w'), indent=1)

def run_case(Tu, gpu, sem, lock, results):
    cd = dst_cd(Tu)
    with sem:
        env, find = make_env()
        env["AI_SA"] = "1"
        env["AI_LAMINAR_SLOWDOWN"] = "0.01"
        env["AI_NULAMSCALE"] = "0.25"   # 1/4 of nu_lam in SA diffusion only
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
        clone(src_cd(Tu), dst_cd(Tu))
        print(f"cloned {src_cd(Tu)} -> {dst_cd(Tu)}")
    sem = threading.Semaphore(8); lock = threading.Lock(); results = {}
    threads = [threading.Thread(target=run_case, args=(Tu, i%8, sem, lock, results))
               for i, Tu in enumerate(TU_LIST)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"Done. {len(results)}/{len(TU_LIST)}")
