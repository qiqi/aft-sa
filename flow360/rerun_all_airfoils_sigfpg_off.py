"""Re-run ALL airfoil cases (NLF + Eppler, cav+str x L0/L1/L2 x alphas) with the
rebuilt sigma_FPG-off binary, so the paper figures reflect the dropped-sigma_FPG
model. Backs up the sigma_FPG-ON forces first (-> .sigfpg_on) for a full
on-vs-off comparison table. Reuses the rerun_all_nlf_lowsens run pattern.
"""
import os, sys, json, shutil, threading, time
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver

B = "/home/qiqi/flexcompute/aft-sa/flow360"
FAMILIES = [
    ("nlf0416_Re4M",   [0, 4, 9, 15]),
    ("eppler387_Re200k", [0, 2, 5, 7]),
]
LEVELS = ['L0', 'L1', 'L2']; MESHES = ['cav', 'str']
SKIP = ('.pvtu', '.vtu', '.pvd', '.gltf', '.log', '.sock')

def read_forces(p):
    if not os.path.exists(p): return None
    r = open(p).readlines()
    if len(r) < 2: return None
    c = r[-1].strip().split(','); return (float(c[2]), float(c[3]))

def reset_case(d):
    src = f"{d}/total_forces_v2.csv"; bak = f"{d}/total_forces_v2.csv.sigfpg_on"
    if os.path.exists(src) and not os.path.exists(bak): shutil.copy2(src, bak)
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

def case_list():
    out = []
    for fam, alphas in FAMILIES:
        for mesh in MESHES:
            for L in LEVELS:
                for a in alphas:
                    d = f"{B}/{mesh}{L}prop_{fam}_a{a}"
                    if os.path.exists(d): out.append((f"{mesh}{L}_{fam}_a{a}", d))
    return out

def main():
    cases = case_list(); print(f"rerunning {len(cases)} airfoil cases", flush=True)
    sem = threading.Semaphore(8); lock = threading.Lock(); res = {}
    def worker(tag, d, gpu):
        with sem:
            base = read_forces(f"{d}/total_forces_v2.csv.sigfpg_on") or read_forces(f"{d}/total_forces_v2.csv")
            reset_case(d)
            env, find = make_env(); env['AI_SA'] = '1'; env['AI_LAMINAR_SLOWDOWN'] = '0.01'
            t0 = time.time()
            print(f"[{tag}] GPU {gpu} base={base}", flush=True)
            try:
                run_solver(d, find, env, gpu=gpu, timeout=10800)
                new = read_forces(f"{d}/total_forces_v2.csv"); dt = time.time() - t0
                dcd = 100*(new[1]-base[1])/base[1] if (base and new) else float('nan')
                with lock: res[tag] = {'dt': round(dt), 'base': base, 'new': new, 'dCD%': round(dcd, 2)}
                print(f"[{tag}] DONE {dt:.0f}s new={new} dCD={dcd:+.2f}%", flush=True)
            except Exception as e:
                with lock: res[tag] = {'error': str(e)}
                print(f"[{tag}] FAILED: {e}", flush=True)
    threads = [threading.Thread(target=worker, args=(tag, d, i % 8)) for i, (tag, d) in enumerate(cases)]
    for t in threads: t.start()
    for t in threads: t.join()
    json.dump(res, open(f"{B}/rerun_all_airfoils_sigfpg_off_results.json", 'w'), indent=1)
    print("\n=== AIRFOILS sigma_FPG OFF vs ON ===", flush=True)
    worst = max((abs(r.get('dCD%', 0)) for r in res.values() if 'dCD%' in r), default=0)
    for tag, r in sorted(res.items()): print(f"  {tag}: {r}", flush=True)
    print(f"\nworst |dCD| = {worst:.2f}%", flush=True)

if __name__ == '__main__':
    main()
