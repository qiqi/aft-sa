"""Set up + run the flat-plate natural-transition cases.

Tu->chi_inf map (calibrate_kernel) is Mack's (1977) e^N critical-N-factor
correlation, adopted wholesale:
    N_crit = -8.43 - 2.4 ln(Tu_frac)        [= Mack 1977; borrowed, not fit]
    chi_inf = c_v1 * exp(-N_crit)           [transition at chi = c_v1]
The coupled-RANS onset (chi=1) then lands within ~7% of the Abu-Ghannam & Shaw
(1980) correlation Re_theta_t = 163 + exp(6.91 - Tu%) across Tu=0.04-0.60%, which
is used here only as an independent verification reference (not a fit target).
Flow360 BC seed = chi_inf_from_Tu_pct(Tu) * f_slow,  f_slow = 0.01.

Cases cloned from flatplate_aftsa_Tu0026 (mesh reused), BC seed patched, run with
AI_SA=1, AI_LAMINAR_SLOWDOWN=0.01. New dirs flatplate_ags_Tu#### (old dirs kept).
"""
import os, sys, shutil, threading, time, json
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/scripts")
from rans.env import make_env
from rans.solve import run_solver
from calibrate_kernel import chi_inf_from_Tu_pct  # uses NEW A_TU,B_TU after edit

B = "/home/qiqi/flexcompute/aft-sa/flow360"
SRC = f"{B}/flatplate_aftsa_Tu0026"
FSLOW = 0.01
TU_LIST = [0.04, 0.08, 0.16, 0.30, 0.60]
def case_dir(Tu): return f"{B}/flatplate_ags_Tu{int(round(Tu*1000)):04d}"

# Output/state files NOT to copy from the source case.
SKIP_COPY = ('.pvtu', '.vtu', '.pvd', '.gltf', '.log', '.sock')
def is_output(f):
    return (any(f.endswith(s) for s in SKIP_COPY) or f.startswith('ipc')
            or f.endswith('_v2.csv') or f == 'progress.csv'
            or f in ('restartOutput', 'restart.json', 'restart_rank_1_of_1.dmp'))

def setup(Tu):
    cd = case_dir(Tu)
    shutil.rmtree(cd, ignore_errors=True); os.makedirs(cd)
    for f in os.listdir(SRC):
        if is_output(f): continue
        sp = os.path.join(SRC, f)
        if os.path.isfile(sp): shutil.copy2(sp, cd)
        elif os.path.isdir(sp) and f != 'ipc_data': shutil.copytree(sp, os.path.join(cd, f))
    seed = chi_inf_from_Tu_pct(Tu) * FSLOW
    p = f"{cd}/Flow360.json"; d = json.load(open(p))
    d['freestream']['turbulenceQuantities'] = {
        'modelType': 'ModifiedTurbulentViscosityRatio',
        'modifiedTurbulentViscosityRatio': seed}
    for bn, bc in d.get('boundaries', {}).items():
        if 'farfield' in bn or bc.get('type') == 'Freestream':
            bc['turbulenceQuantities'] = {
                'modelType': 'ModifiedTurbulentViscosityRatio',
                'modifiedTurbulentViscosityRatio': seed}
    d['runControl']['restart'] = False
    json.dump(d, open(p, 'w'), indent=1)
    return seed

def run_case(Tu, gpu, sem, lock, results):
    cd = case_dir(Tu); tag = f"Tu{Tu}"
    with sem:
        env, find = make_env()
        env["AI_SA"] = "1"; env["AI_LAMINAR_SLOWDOWN"] = "0.01"
        t0 = time.time()
        print(f"[{tag}] launching on GPU {gpu}  seed={chi_inf_from_Tu_pct(Tu)*FSLOW:.3e}", flush=True)
        try:
            run_solver(cd, find, env, gpu=gpu, timeout=10800)
            lastF = open(f"{cd}/total_forces_v2.csv").readlines()[-1]
            with lock:
                results[tag] = {'dt_s': time.time()-t0, 'CD': lastF.strip().split(',')[3]}
            print(f"[{tag}] DONE {results[tag]['dt_s']:.0f}s CD={results[tag]['CD']}", flush=True)
        except Exception as e:
            print(f"[{tag}] FAILED: {e}", flush=True)
            with lock: results[tag] = {'error': str(e)}

if __name__ == '__main__':
    for Tu in TU_LIST:
        s = setup(Tu); print(f"setup {case_dir(Tu)}  seed={s:.3e}", flush=True)
    sem = threading.Semaphore(8); lock = threading.Lock(); results = {}
    threads = [threading.Thread(target=run_case, args=(Tu, i % 8, sem, lock, results))
               for i, Tu in enumerate(TU_LIST)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"\nDone. {len(results)}/{len(TU_LIST)}")
    for k, v in sorted(results.items()): print(f"  {k}: {v}")
