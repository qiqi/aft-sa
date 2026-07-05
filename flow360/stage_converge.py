"""Generalized staged-fSlow convergence for arbitrary NLF cases (sigma_FPG OFF):
stage 1 fSlow=1 (migrate front fast, BC uncompensated 8.76e-4), then stage 2
fSlow=0.01 (damp the limit cycle, BC compensated 8.76e-6). Restart dump is copied
from restartOutput/ to the case root each stage. Preserves prior forces per stage.
Usage: edit CASES = [(mesh, level, alpha), ...].
"""
import os, sys, json, shutil, threading, time
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver

B = "/home/qiqi/flexcompute/aft-sa/flow360"
CASES = [("str", "L1", 9)]

def cdir(m, L, a): return f"{B}/{m}{L}prop_nlf0416_Re4M_a{a}"

def stage(d, fslow, chi, maxsteps, backup_suffix):
    src = f"{d}/total_forces_v2.csv"
    if os.path.exists(src) and not os.path.exists(f"{src}.{backup_suffix}"):
        shutil.copy2(src, f"{src}.{backup_suffix}")
    for fn in ('restart.json', 'restart_rank_1_of_1.dmp'):
        s = f"{d}/restartOutput/{fn}"
        if os.path.exists(s): shutil.copy2(s, f"{d}/{fn}")
    p = f"{d}/Flow360.json"; c = json.load(open(p))
    tq = {'modelType': 'ModifiedTurbulentViscosityRatio', 'modifiedTurbulentViscosityRatio': chi}
    c['freestream']['turbulenceQuantities'] = tq
    for bn, bc in c.get('boundaries', {}).items():
        if 'farfield' in bn or bc.get('type') == 'Freestream':
            bc['turbulenceQuantities'] = dict(tq)
    c['runControl']['restart'] = True
    c['timeStepping']['maxPseudoSteps'] = maxsteps
    json.dump(c, open(p, 'w'), indent=1)
    env, find = make_env(); env['AI_SA'] = '1'; env['AI_LAMINAR_SLOWDOWN'] = str(fslow)
    run_solver(d, find, env, gpu=0, timeout=10800)
    return open(f"{d}/total_forces_v2.csv").readlines()[-1].strip().split(',')

def main():
    for m, L, a in CASES:
        d = cdir(m, L, a); tag = f"{m}{L}_a{a}"
        try:
            print(f"[{tag}] stage1 fSlow=1 ...", flush=True)
            c1 = stage(d, 1, 8.76e-4, 30000, "presweep")
            print(f"[{tag}] stage1 CD={c1[3]}  -> stage2 fSlow=0.01 ...", flush=True)
            c2 = stage(d, 0.01, 8.76e-6, 40000, "fslow1")
            print(f"[{tag}] DONE  CL={c2[2]} CD={c2[3]}", flush=True)
        except Exception as e:
            print(f"[{tag}] FAILED: {e}", flush=True)
    print("DONE_ALL", flush=True)

if __name__ == '__main__':
    main()
