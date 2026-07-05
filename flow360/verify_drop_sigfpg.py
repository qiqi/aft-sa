"""Verify dropping sigma_FPG (branch explore/drop-sigma-fpg): rerun a focused NLF
subset with the rebuilt binary (sigma_FPG removed, K_lambda=10 cliff only) and
compare CL/CD to the sigma_FPG-ON baseline. Baseline forces are backed up first
(reset_case wipes them). Hypothesis: ~no change (cliff already dominates FPG).
"""
import os, sys, json, shutil, threading, time
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver

B = "/home/qiqi/flexcompute/aft-sa/flow360"
# Focused FPG-relevant subset: rooftop (a0) + low-alpha (a4), both meshes, L1.
CASES = [("cav", "L1", 0), ("str", "L1", 0), ("cav", "L1", 4), ("str", "L1", 4)]
SKIP = ('.pvtu', '.vtu', '.pvd', '.gltf', '.log', '.sock')

def cd(mesh, L, a): return f"{B}/{mesh}{L}prop_nlf0416_Re4M_a{a}"

def read_last_forces(d):
    p = f"{d}/total_forces_v2.csv"
    if not os.path.exists(p): return None
    r = open(p).readlines()
    if len(r) < 2: return None
    c = r[-1].strip().split(',')
    return (float(c[2]), float(c[3]))  # CL, CD

def backup_baseline(d):
    """Save sigma_FPG-ON forces before the reset wipes them."""
    src = f"{d}/total_forces_v2.csv"; dst = f"{d}/total_forces_v2.csv.sigfpg_on"
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)

def read_baseline(d):
    p = f"{d}/total_forces_v2.csv.sigfpg_on"
    if not os.path.exists(p): return None
    c = open(p).readlines()[-1].strip().split(','); return (float(c[2]), float(c[3]))

def reset_case(d):
    for f in os.listdir(d):
        if any(f.endswith(s) for s in SKIP) or f.startswith('ipc'):
            try: os.remove(os.path.join(d, f))
            except: pass
        elif f in ('restartOutput', 'restart_rank_1_of_1.dmp', 'restart.json',
                   'ipc_data') or f.endswith('_v2.csv'):
            sp = os.path.join(d, f)
            if os.path.isfile(sp): os.remove(sp)
            elif os.path.isdir(sp): shutil.rmtree(sp)
    p = f"{d}/Flow360.json"; cfg = json.load(open(p))
    cfg['runControl']['restart'] = False
    cfg['timeStepping']['maxPseudoSteps'] = 80000
    cfg['timeStepping']['absoluteTolerance'] = 1e-30
    cfg['turbulenceModelSolver']['absoluteTolerance'] = 1e-30
    json.dump(cfg, open(p, 'w'), indent=1)

def main():
    sem = threading.Semaphore(8); lock = threading.Lock(); res = {}
    # backup baselines FIRST (before any reset)
    for mesh, L, a in CASES:
        d = cd(mesh, L, a)
        if os.path.exists(d):
            backup_baseline(d)
            print(f"  baseline {mesh}{L}_a{a}: {read_baseline(d)}", flush=True)
        else:
            print(f"  MISSING {mesh}{L}_a{a}", flush=True)
    def worker(mesh, L, a, gpu):
        d = cd(mesh, L, a); tag = f"{mesh}{L}_a{a}"
        if not os.path.exists(d):
            with lock: res[tag] = {'error': 'missing'}; return
        with sem:
            base = read_baseline(d)
            reset_case(d)
            env, find = make_env(); env['AI_SA'] = '1'; env['AI_LAMINAR_SLOWDOWN'] = '0.01'
            t0 = time.time()
            print(f"[{tag}] launching GPU {gpu} (baseline CL,CD={base})", flush=True)
            try:
                run_solver(d, find, env, gpu=gpu, timeout=10800)
                new = read_last_forces(d); dt = time.time() - t0
                dcl = 100*(new[0]-base[0])/base[0] if base else float('nan')
                dcd = 100*(new[1]-base[1])/base[1] if base else float('nan')
                with lock: res[tag] = {'dt': dt, 'base': base, 'new': new,
                                       'dCL%': round(dcl,2), 'dCD%': round(dcd,2)}
                print(f"[{tag}] DONE {dt:.0f}s  base={base} new={new}  dCL={dcl:+.2f}% dCD={dcd:+.2f}%", flush=True)
            except Exception as e:
                with lock: res[tag] = {'error': str(e)}
                print(f"[{tag}] FAILED: {e}", flush=True)
    threads = [threading.Thread(target=worker, args=(m, L, a, i % 8))
               for i, (m, L, a) in enumerate(CASES)]
    for t in threads: t.start()
    for t in threads: t.join()
    json.dump(res, open(f"{B}/verify_drop_sigfpg_results.json", 'w'), indent=1)
    print("\n=== sigma_FPG OFF vs ON (NLF) ===")
    for tag, r in sorted(res.items()): print(f"  {tag}: {r}")

if __name__ == '__main__':
    main()
