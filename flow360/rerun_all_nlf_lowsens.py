"""Re-run all 24 NLF cases (cav+str × L0/L1/L2 × α∈{0,4,9,15}) with the
lowered-sensitivity SA-AI kernel (sigmoidSlope=5, sigmoidCenter=1.277).

The case directories already have the right Flow360.json (Re=4M, M=0.1,
chi_inf=8.76e-6, AI_LAMINAR_SLOWDOWN=0.01). We just wipe the prior
output and restart from scratch with fresh init.
"""
import os, sys, json, shutil, threading, time, csv
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver

B = "/home/qiqi/flexcompute/aft-sa/flow360"
LEVELS = ['L0', 'L1', 'L2']
MESHES = ['cav', 'str']
ALPHAS = [0, 4, 9, 15]
SKIP_FILES = ('.pvtu', '.vtu', '.pvd', '.gltf', '.log', '.sock')

def reset_case(d):
    """Wipe outputs but keep mesh + JSON config; turn off restart so we init fresh."""
    if not os.path.exists(d):
        return False
    for f in os.listdir(d):
        if any(f.endswith(s) for s in SKIP_FILES) or f.startswith('ipc'):
            try: os.remove(os.path.join(d, f))
            except: pass
        elif f in ('restartOutput', 'restart_rank_1_of_1.dmp', 'restart.json',
                   'ipc_data') or f.endswith('_v2.csv'):
            sp = os.path.join(d, f)
            if os.path.isfile(sp): os.remove(sp)
            elif os.path.isdir(sp): shutil.rmtree(sp)
    # Patch config: ensure restart=False, ensure maxPseudoSteps=80000
    p = os.path.join(d, 'Flow360.json')
    cfg = json.load(open(p))
    cfg['runControl']['restart'] = False
    cfg['timeStepping']['maxPseudoSteps'] = 80000
    cfg['timeStepping']['absoluteTolerance'] = 1e-30
    cfg['turbulenceModelSolver']['absoluteTolerance'] = 1e-30
    json.dump(cfg, open(p, 'w'), indent=1)
    return True

def case_list():
    cases = []
    for mesh in MESHES:
        for L in LEVELS:
            for a in ALPHAS:
                d = f"{B}/{mesh}{L}prop_nlf0416_Re4M_a{a}"
                if os.path.exists(d):
                    cases.append((mesh, L, a, d))
    return cases

def main():
    cases = case_list()
    print(f"will rerun {len(cases)} cases (mesh × level × alpha)")
    for mesh, L, a, d in cases:
        ok = reset_case(d)
        print(f"  reset {mesh}{L}_a{a}: {'OK' if ok else 'MISSING'}")
    # Run in parallel on N GPUs (8 GPUs available, run ~6 in parallel to leave headroom)
    N_GPU = 8
    sem = threading.Semaphore(N_GPU)
    lock = threading.Lock()
    results = {}
    def worker(mesh, L, a, d, gpu):
        with sem:
            env, find = make_env()
            env['AI_SA'] = '1'
            env['AI_LAMINAR_SLOWDOWN'] = '0.01'
            tag = f"{mesh}{L}_a{a}"
            t0 = time.time()
            print(f"[{tag}] launching on GPU {gpu}", flush=True)
            try:
                run_solver(d, find, env, gpu=gpu, timeout=10800)
                dt = time.time() - t0
                # Read final residual / forces
                try:
                    last_r = open(f"{d}/nonlinear_residual_v2.csv").readlines()[-1]
                    last_f = open(f"{d}/total_forces_v2.csv").readlines()[-1]
                    with lock:
                        results[tag] = {'dt': dt,
                                        'res': last_r.strip().split(',')[2],
                                        'CL': last_f.strip().split(',')[2],
                                        'CD': last_f.strip().split(',')[3]}
                    print(f"[{tag}] DONE in {dt:.0f}s  res={results[tag]['res']}  CL={results[tag]['CL']}  CD={results[tag]['CD']}", flush=True)
                except Exception as e:
                    print(f"[{tag}] DONE but failed to read forces: {e}", flush=True)
            except Exception as e:
                print(f"[{tag}] FAILED: {e}", flush=True)
                with lock:
                    results[tag] = {'error': str(e)}
    threads = []
    for i, (mesh, L, a, d) in enumerate(cases):
        gpu = i % N_GPU
        t = threading.Thread(target=worker, args=(mesh, L, a, d, gpu))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    json.dump(results, open(f"{B}/rerun_lowsens_results.json", 'w'), indent=1)
    print(f"\nFinished {len(results)}/{len(cases)}")
    for tag, r in sorted(results.items()):
        print(f"  {tag}: {r}")

if __name__ == '__main__':
    main()
