"""Re-run every sweep + refinement case on the new box (017-v100-dev). The Flow360Solver
binary rsynced from old box hangs in C++ shutdown after writing all outputs (driver 535.288
vs 535.274 minor diff), so we use a log-tail watchdog: detect 'Exporting text outputs
finished' in solver.log, sleep a small grace, then SIGTERM/SIGKILL the solver. Outputs
are already on disk by then. 7-way parallel across GPUs 1..7. Cap 25000 with absTol
early-stop (NS 1e-9 / SA 1e-8) configured in the case JSON."""
import os, sys, json, csv, time, threading, subprocess, shutil, signal
from pathlib import Path
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env

F = "/home/qiqi/flexcompute/aft-sa/flow360"
CAP = 25000
END_MARKER = "Exporting text outputs finished"
GRACE_SEC = 4.0    # after end-marker appears, wait this long then kill solver
MAX_WALL_SEC = 4000  # hard ceiling per case (V100, 25k steps ~ <30 min)

def clean(d):
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if (f.endswith(('.pvtu', '.vtu', '.pvd', '.sock', '.gltf'))
            or f.endswith('_v2.csv')
            or f.endswith('.csv.bk')
            or f.startswith('surface_forces')
            or f.endswith('.log')):
            try: os.remove(p)
            except: pass
        if f in ('ipc_data', 'restartOutput'):
            shutil.rmtree(p, ignore_errors=True)

def run_solver_watchdog(workdir, find, env, gpu):
    """Launch postprocessor + solver; kill solver shortly after it writes end marker."""
    senv = dict(env)
    senv["CUDA_VISIBLE_DEVICES"] = str(gpu)
    senv["OMP_NUM_THREADS"] = "1"
    senv["OMPI_COMM_WORLD_LOCAL_RANK"] = "0"
    senv["OMPI_COMM_WORLD_RANK"] = "0"
    senv["OMPI_COMM_WORLD_SIZE"] = "1"

    workdir = Path(workdir)
    sock = workdir / "ipc_control.sock"
    if sock.exists(): sock.unlink()

    pp_log = open(workdir / "postprocessor.log", "w")
    pp = subprocess.Popen(
        [find("columnarDataProcessor.py"), "--asyncMode",
         "--inputSimulationJson", "simulation.json",
         "--columnarDataProcessorJson", "columnar.json"],
        cwd=str(workdir), env=senv, stdout=pp_log, stderr=subprocess.STDOUT)
    try:
        for _ in range(40):
            if sock.exists(): break
            time.sleep(0.5)
        solver_log = workdir / "solver.log"
        slog = open(solver_log, "w")
        solver = subprocess.Popen([find("Flow360Solver")], cwd=str(workdir), env=senv,
                                  stdout=slog, stderr=subprocess.STDOUT,
                                  preexec_fn=os.setsid)
        end_seen_t = None
        start = time.time()
        try:
            while True:
                ret = solver.poll()
                if ret is not None:
                    return ret
                # tail-check for end marker
                if end_seen_t is None and solver_log.exists():
                    try:
                        # only read tail to avoid mem
                        with open(solver_log, "rb") as f:
                            f.seek(0, 2); sz = f.tell()
                            f.seek(max(0, sz - 4096))
                            tail = f.read().decode(errors="ignore")
                        if END_MARKER in tail:
                            end_seen_t = time.time()
                    except: pass
                if end_seen_t and (time.time() - end_seen_t) > GRACE_SEC:
                    try: os.killpg(os.getpgid(solver.pid), signal.SIGTERM)
                    except: pass
                    try: solver.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try: os.killpg(os.getpgid(solver.pid), signal.SIGKILL)
                        except: pass
                        solver.wait(timeout=5)
                    return 0
                if time.time() - start > MAX_WALL_SEC:
                    try: os.killpg(os.getpgid(solver.pid), signal.SIGKILL)
                    except: pass
                    raise TimeoutError(f"solver hit MAX_WALL_SEC={MAX_WALL_SEC}")
                time.sleep(1.0)
        finally:
            slog.close()
    finally:
        pp.terminate()
        try: pp.wait(timeout=10)
        except subprocess.TimeoutExpired: pp.kill()
        pp_log.close()

res = {}
lock = threading.Lock()
sem = threading.Semaphore(7)

def run(d, gpu):
    with sem:
        wd = f"{F}/{d}"
        if not os.path.isdir(wd):
            print(f"SKIP {d}: no such dir", flush=True); return
        t0 = time.time()
        try:
            clean(wd)
            c = json.load(open(f"{wd}/Flow360.json"))
            c['timeStepping']['maxPseudoSteps'] = CAP
            json.dump(c, open(f"{wd}/Flow360.json", "w"), indent=1)
            env, find = make_env()
            env["AI_SA"] = "1"
            try: run_solver_watchdog(wd, find, env, gpu=gpu)
            except Exception as e: print(f"[{d}] watchdog: {e}", flush=True)
            rr = list(csv.reader(open(f"{wd}/nonlinear_residual_v2.csv")))
            h = [x.strip() for x in rr[0]]
            last = [x for x in rr[-1] if x.strip() != '']
            step = int(float(last[h.index('pseudo_step')]))
            cont = float(last[h.index('0_cont')])
            nuHat = float(last[h.index('5_nuHat')])
            CD = None
            tfp = f"{wd}/total_forces_v2.csv"
            if os.path.exists(tfp):
                tr = list(csv.reader(open(tfp))); th = [x.strip() for x in tr[0]]
                tl = [x for x in tr[-1] if x.strip() != '']
                try: CD = float(tl[th.index('CD')])
                except: CD = None
            with lock:
                res[d] = dict(step=step, cont=cont, nuHat=nuHat, CD=CD,
                              elapsed=round(time.time() - t0, 1), gpu=gpu)
            ok = "OK " if (cont < 1e-9 and nuHat < 1e-8) else "CAP"
            print(f"{ok} {d:40s} step={step:5d} cont={cont:.1e} nuHat={nuHat:.1e} CD={CD} t={time.time()-t0:.0f}s gpu{gpu}", flush=True)
        except Exception as e:
            print(f"FAIL {d}: {e}", flush=True)

sweep = [f"full_{af}_{g}_aftsa_m2_a{a}"
         for af in ['naca0012', 'nlf0416']
         for a in [-2, 0, 2, 4, 6, 8]
         for g in ['cavity', 'ogrid']]
refine = [f"refineA4_cavity_{lv}" for lv in ['coarse', 'med', 'fine', 'xfine', 'xxfine']] + \
         [f"refineA4_struct_{lv}" for lv in ['coarse', 'med', 'fine', 'sxfine', 'sxxfine']]
jobs = sweep + refine
print(f"Launching {len(jobs)} cases ({len(sweep)} sweep + {len(refine)} refinement) at cap={CAP}", flush=True)
ts = [threading.Thread(target=run, args=(d, 1 + (i % 7))) for i, d in enumerate(jobs)]
for t in ts: t.start()
for t in ts: t.join()
json.dump(res, open(f"{F}/run_all_reproduce_results.json", "w"), indent=1)
ok_n = sum(1 for r in res.values() if r['cont'] < 1e-9 and r['nuHat'] < 1e-8)
print(f"\n=== SUMMARY: {ok_n}/{len(res)} converged to NS 1e-9 / SA 1e-8 ===")
caps = [d for d, r in res.items() if not (r['cont'] < 1e-9 and r['nuHat'] < 1e-8)]
for d in caps: r = res[d]; print(f"  CAP {d:40s} step={r['step']} cont={r['cont']:.1e} nuHat={r['nuHat']:.1e}")
