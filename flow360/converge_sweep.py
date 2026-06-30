"""Parallel x_tr-convergence sweep across all 24 NLF α=(0,4,9,15) cases.
Each case is reset from scratch and iterated with converge_by_xtr in 5000-step
batches until x_tr_upper AND x_tr_lower are stable to within tol for 2
consecutive batches (or max_batches reached).
"""
import os, sys, json, shutil, subprocess, threading, time
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")

B = "/home/qiqi/flexcompute/aft-sa/flow360"
MESHES = ['cav', 'str']
LEVELS = ['L0', 'L1', 'L2']
ALPHAS = [0, 4, 9, 15]
BATCH = 5000
TOL = 0.01
MAX_BATCHES = 24       # 24 × 5000 = 120k pseudo-steps cap
SKIP = ('.pvtu','.vtu','.pvd','.gltf','.log','.sock')

def reset(cd):
    """Clean state for fresh start."""
    for f in os.listdir(cd):
        if any(f.endswith(s) for s in SKIP) or f.startswith('ipc') \
                or f.endswith('_v2.csv') or f == 'progress.csv' \
                or f == 'xtr_history.csv':
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
    json.dump(d, open(p, 'w'), indent=1)

def run_case(mesh, L, alpha, gpu, sem, lock, log):
    cd = f"{B}/{mesh}{L}prop_nlf0416_Re4M_a{alpha}"
    tag = f"{mesh}{L}_a{alpha}"
    if not os.path.exists(cd):
        with lock: print(f"[{tag}] SKIP: no such dir", flush=True); return
    with sem:
        with lock: print(f"[{tag}] starting on GPU {gpu}", flush=True)
        # Clean state
        reset(cd)
        # Launch converge_by_xtr.py
        logp = f"/tmp/conv_{tag}.log"
        with open(logp, 'w') as f:
            t0 = time.time()
            try:
                rc = subprocess.run(
                    ["/home/qiqi/flexcompute/compute/.venv/bin/python3",
                     f"{B}/converge_by_xtr.py", cd,
                     "--batch", str(BATCH),
                     "--tol", str(TOL),
                     "--max-batches", str(MAX_BATCHES),
                     "--gpu", str(gpu)],
                    stdout=f, stderr=subprocess.STDOUT, timeout=86400,
                )
                dt = time.time() - t0
                # Parse final x_tr from history file
                hp = f"{cd}/xtr_history.csv"
                if os.path.exists(hp):
                    import csv
                    rows = list(csv.DictReader(open(hp)))
                    if rows:
                        last = rows[-1]
                        with lock:
                            print(f"[{tag}] DONE rc={rc.returncode} step={last['step']} "
                                  f"x_tr_up={float(last['xtr_upper']):.3f} "
                                  f"x_tr_lo={float(last['xtr_lower']):.3f} dt={dt:.0f}s",
                                  flush=True)
                else:
                    with lock: print(f"[{tag}] DONE no history dt={dt:.0f}s", flush=True)
            except subprocess.TimeoutExpired:
                with lock: print(f"[{tag}] TIMEOUT", flush=True)
            except Exception as e:
                with lock: print(f"[{tag}] FAIL {e}", flush=True)

if __name__ == '__main__':
    cases = [(m, L, a) for m in MESHES for L in LEVELS for a in ALPHAS
             if os.path.exists(f"{B}/{m}{L}prop_nlf0416_Re4M_a{a}")]
    print(f"Sweeping {len(cases)} cases, batch={BATCH}, tol={TOL}, max_batches={MAX_BATCHES}",
          flush=True)
    sem = threading.Semaphore(8)
    lock = threading.Lock()
    log = []
    threads = [threading.Thread(target=run_case,
                                args=(m, L, a, i % 8, sem, lock, log))
               for i, (m, L, a) in enumerate(cases)]
    for t in threads: t.start()
    for t in threads: t.join()
    print("\nALL CASES DONE", flush=True)
