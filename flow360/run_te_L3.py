"""Run cavity L3 with TE refinement on 8 GPUs, damped CFL (matches L3 fresh run config)."""
import os, sys, json, csv, time, subprocess, shutil, signal
from pathlib import Path
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env

D = "/home/qiqi/flexcompute/aft-sa/flow360/proper_cav_L3_TE"
NGPU = 8
END_MARKER = "Exporting text outputs finished"
GRACE_SEC = 5.0
MAX_WALL_SEC = 14400

cfg = json.load(open(f"{D}/Flow360.json"))
cfg['runControl']['restart'] = False
cfg['timeStepping']['maxPseudoSteps'] = 50000
cfg['timeStepping']['CFL']['max'] = 2000.0
cfg['turbulenceModelSolver']['CFLMultiplier'] = 1.0
json.dump(cfg, open(f"{D}/Flow360.json", "w"), indent=1)
print(f"Updated Flow360.json: fresh, maxPseudoSteps=50000, CFL.max=2000")

# clean state
for f in os.listdir(D):
    p = f"{D}/{f}"
    if f.endswith(('.pvtu','.vtu','.pvd','.sock','.gltf')) or f.endswith('_v2.csv') or f.endswith('.csv.bk') or f.startswith('surface_forces'):
        try: os.remove(p)
        except: pass
    if f in ('ipc_data','restartOutput'):
        shutil.rmtree(p, ignore_errors=True)
    if 'rank_' in f and '.dmp' in f:
        try: os.remove(p)
        except: pass

env, find = make_env()
env["AI_SA"] = "1"; env["OMP_NUM_THREADS"] = "1"

# partition + process
print(f"\n[1/3] MeshPartitioner --partitions {NGPU}")
r = subprocess.run([find("MeshPartitioner"), "--meshfile", "mesh.cgns",
                    "--partitions", str(NGPU), "--threads", str(NGPU)],
                   cwd=D, env=env, capture_output=True, text=True)
if r.returncode != 0: print(f"FAIL: {r.stderr[-500:]}"); sys.exit(1)
print(f"  partition OK")

print(f"\n[2/3] MeshProcessor mpirun -np {NGPU}")
r = subprocess.run(["mpirun", "-np", str(NGPU), find("MeshProcessor"),
                    "--threads", "1", "mesh.cgns"],
                   cwd=D, env=env, capture_output=True, text=True, timeout=1800)
if r.returncode != 0: print(f"FAIL: {r.stderr[-1000:]}"); sys.exit(1)
print(f"  process OK")

# solver
print(f"\n[3/3] Flow360Solver mpirun -np {NGPU}")
t0 = time.time()
senv = dict(env); senv["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(NGPU))
sock = Path(D)/"ipc_control.sock"
if sock.exists(): sock.unlink()
pp_log = open(f"{D}/postprocessor.log", "w")
pp = subprocess.Popen([find("columnarDataProcessor.py"), "--asyncMode",
                       "--inputSimulationJson", "simulation.json",
                       "--columnarDataProcessorJson", "columnar.json"],
                      cwd=D, env=senv, stdout=pp_log, stderr=subprocess.STDOUT)
try:
    for _ in range(40):
        if sock.exists(): break
        time.sleep(0.5)
    slog = open(f"{D}/solver.log", "w")
    solver = subprocess.Popen(["mpirun", "-np", str(NGPU), find("Flow360Solver")],
                              cwd=D, env=senv, stdout=slog, stderr=subprocess.STDOUT,
                              preexec_fn=os.setsid)
    end_t = None
    while True:
        if solver.poll() is not None:
            print(f"  solver exited rc={solver.returncode}, end_t={end_t}"); break
        try:
            with open(f"{D}/solver.log","rb") as f:
                f.seek(0,2); sz=f.tell(); f.seek(max(0,sz-8192))
                tail = f.read().decode(errors="ignore")
            if END_MARKER in tail and end_t is None:
                end_t = time.time(); print(f"  end-marker at t={time.time()-t0:.0f}s", flush=True)
        except: pass
        if end_t and (time.time()-end_t) > GRACE_SEC:
            try: os.killpg(os.getpgid(solver.pid), signal.SIGTERM)
            except: pass
            try: solver.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(os.getpgid(solver.pid), signal.SIGKILL)
                except: pass
                solver.wait(timeout=5)
            break
        if time.time()-t0 > MAX_WALL_SEC:
            try: os.killpg(os.getpgid(solver.pid), signal.SIGKILL)
            except: pass
            print(f"  TIMEOUT @ {MAX_WALL_SEC}s"); break
        time.sleep(5.0)
    slog.close()
finally:
    pp.terminate()
    try: pp.wait(timeout=10)
    except subprocess.TimeoutExpired: pp.kill()
    pp_log.close()

# report
rr = list(csv.reader(open(f"{D}/nonlinear_residual_v2.csv")))
h = [x.strip() for x in rr[0]]; last = [x for x in rr[-1] if x.strip() != '']
step = int(float(last[h.index('pseudo_step')]))
cont = float(last[h.index('0_cont')]); nu = float(last[h.index('5_nuHat')])
tr = list(csv.reader(open(f"{D}/total_forces_v2.csv")))
th = [x.strip() for x in tr[0]]; tl = [x for x in tr[-1] if x.strip()!='']
CL = float(tl[th.index('CL')]); CD = float(tl[th.index('CD')]); CMy = float(tl[th.index('CMy')])
print(f"\nFINAL L3_TE: step={step}, cont={cont:.2e}, nuHat={nu:.2e}, CL={CL:.4f}, CD={CD:.5f}, CMy={CMy:.4f}")
print(f"elapsed: {time.time()-t0:.0f}s")
