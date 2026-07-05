"""Re-run the two finest meshes with nuHat in volume+slice output, to support BL profile tracing."""
import os, sys, json, csv, time, threading, subprocess, shutil, signal
from pathlib import Path
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env

F = "/home/qiqi/flexcompute/aft-sa/flow360"
END_MARKER = "Exporting text outputs finished"
GRACE_SEC = 4.0
MAX_WALL_SEC = 9000

def run_watchdog(workdir, find, env, gpu):
    senv = dict(env); senv["CUDA_VISIBLE_DEVICES"] = str(gpu)
    senv["OMP_NUM_THREADS"] = "1"
    senv["OMPI_COMM_WORLD_LOCAL_RANK"] = "0"; senv["OMPI_COMM_WORLD_RANK"] = "0"; senv["OMPI_COMM_WORLD_SIZE"] = "1"
    workdir = Path(workdir)
    sock = workdir / "ipc_control.sock"
    if sock.exists(): sock.unlink()
    pp_log = open(workdir / "postprocessor.log", "w")
    pp = subprocess.Popen([find("columnarDataProcessor.py"), "--asyncMode",
        "--inputSimulationJson", "simulation.json",
        "--columnarDataProcessorJson", "columnar.json"],
        cwd=str(workdir), env=senv, stdout=pp_log, stderr=subprocess.STDOUT)
    try:
        for _ in range(40):
            if sock.exists(): break
            time.sleep(0.5)
        slog_path = workdir / "solver.log"
        slog = open(slog_path, "w")
        solver = subprocess.Popen([find("Flow360Solver")], cwd=str(workdir), env=senv,
                                  stdout=slog, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        end_t = None; start = time.time()
        try:
            while True:
                if solver.poll() is not None: return solver.returncode
                if end_t is None and slog_path.exists():
                    try:
                        with open(slog_path, "rb") as f:
                            f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - 4096))
                            tail = f.read().decode(errors="ignore")
                        if END_MARKER in tail: end_t = time.time()
                    except: pass
                if end_t and time.time() - end_t > GRACE_SEC:
                    try: os.killpg(os.getpgid(solver.pid), signal.SIGTERM)
                    except: pass
                    try: solver.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try: os.killpg(os.getpgid(solver.pid), signal.SIGKILL)
                        except: pass
                    return 0
                if time.time() - start > MAX_WALL_SEC:
                    try: os.killpg(os.getpgid(solver.pid), signal.SIGKILL)
                    except: pass
                    raise TimeoutError
                time.sleep(1.0)
        finally: slog.close()
    finally:
        pp.terminate()
        try: pp.wait(timeout=10)
        except subprocess.TimeoutExpired: pp.kill()
        pp_log.close()

def clean(d):
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if f.endswith(('.pvtu','.vtu','.pvd','.sock','.gltf')) or f.endswith('_v2.csv') or f.startswith('surface_forces') or f.endswith('.log'):
            try: os.remove(p)
            except: pass
        if f in ('ipc_data','restartOutput'): shutil.rmtree(p, ignore_errors=True)

def solve_one(d, gpu):
    print(f"solving {d} on GPU {gpu}", flush=True)
    clean(d)
    env, find = make_env(); env["AI_SA"] = "1"
    t0 = time.time()
    try: run_watchdog(d, find, env, gpu=gpu)
    except Exception as e: print(f"wd: {e}", flush=True)
    rr = list(csv.reader(open(f"{d}/nonlinear_residual_v2.csv")))
    h = [x.strip() for x in rr[0]]; last = [x for x in rr[-1] if x.strip() != '']
    step = int(float(last[h.index('pseudo_step')]))
    cont = float(last[h.index('0_cont')]); nuHat = float(last[h.index('5_nuHat')])
    tr = list(csv.reader(open(f"{d}/total_forces_v2.csv")))
    th = [x.strip() for x in tr[0]]; tl = [x for x in tr[-1] if x.strip() != '']
    CD = float(tl[th.index('CD')])
    print(f"  done {d}: step={step}, cont={cont:.1e}, nuHat_res={nuHat:.1e}, CD={CD:.5f}, t={time.time()-t0:.0f}s", flush=True)

ts = []
ts.append(threading.Thread(target=solve_one, args=(f"{F}/proper_refineA4_cavity_xxfine", 1)))
ts.append(threading.Thread(target=solve_one, args=(f"{F}/refineA4_struct_sxxfine", 2)))
for t in ts: t.start()
for t in ts: t.join()
print("DONE")
