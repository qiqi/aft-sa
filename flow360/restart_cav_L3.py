"""Restart cav L3 from the existing restart files in restartOutput/, extending to 50k steps.
Goal: nuHat < 1e-8, cont < 1e-9.
"""
import os, sys, json, csv, time, subprocess, shutil, signal
from pathlib import Path
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env

D = "/home/qiqi/flexcompute/aft-sa/flow360/proper_cav_L3"
NGPU = 4
END_MARKER = "Exporting text outputs finished"
GRACE_SEC = 5.0
MAX_WALL_SEC = 21600

# 1. Update Flow360.json: enable restart, extend step cap
cfg = json.load(open(f"{D}/Flow360.json"))
cfg['runControl']['restart'] = True
cfg['timeStepping']['maxPseudoSteps'] = 50000
json.dump(cfg, open(f"{D}/Flow360.json", "w"), indent=1)
print(f"Updated Flow360.json: restart=True, maxPseudoSteps=50000")

# 2. Back up previous csvs (preserve historical) — solver will overwrite on restart
ts = time.strftime("%Y%m%d_%H%M%S")
bk = f"{D}/run_step25k_bk_{ts}"
os.makedirs(bk, exist_ok=True)
for f in ['nonlinear_residual_v2.csv','total_forces_v2.csv','linear_iterations_v2.csv',
          'linear_residual_v2.csv','cfl_v2.csv','minmax_state_v2.csv',
          'progress.csv','solver.log','postprocessor.log']:
    p = f"{D}/{f}"
    if os.path.exists(p): shutil.copy(p, f"{bk}/{f}")
print(f"Backed up previous CSVs to {bk}")

# 3. Clean ipc-only state (NOT restartOutput which we need for restart)
for f in os.listdir(D):
    p = f"{D}/{f}"
    if f.endswith(('.pvtu','.vtu','.pvd','.sock','.gltf')):
        try: os.remove(p)
        except: pass
if os.path.exists(f"{D}/ipc_data"):
    shutil.rmtree(f"{D}/ipc_data", ignore_errors=True)

env, find = make_env()
env["AI_SA"] = "1"  # all proper_cav/proper_str cases use SA-AI transition model
env["OMP_NUM_THREADS"] = "1"
print(f"\nStarting solver: ngpu={NGPU}, AI_SA={env.get('AI_SA')}")
t0 = time.time()
senv = dict(env)
senv["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(NGPU))
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
            print(f"  solver exited rc={solver.returncode}, end_t={end_t}")
            break
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
        time.sleep(3.0)
    slog.close()
finally:
    pp.terminate()
    try: pp.wait(timeout=10)
    except subprocess.TimeoutExpired: pp.kill()
    pp_log.close()

# Final result
rr = list(csv.reader(open(f"{D}/nonlinear_residual_v2.csv")))
h = [x.strip() for x in rr[0]]; last = [x for x in rr[-1] if x.strip() != '']
step = int(float(last[h.index('pseudo_step')]))
cont = float(last[h.index('0_cont')]); nuHat = float(last[h.index('5_nuHat')])
tr = list(csv.reader(open(f"{D}/total_forces_v2.csv")))
th = [x.strip() for x in tr[0]]; tl = [x for x in tr[-1] if x.strip()!='']
CL = float(tl[th.index('CL')]); CD = float(tl[th.index('CD')]); CMy = float(tl[th.index('CMy')])
print(f"\nFINAL: step={step}, cont={cont:.2e}, nuHat={nuHat:.2e}, CL={CL:.4f}, CD={CD:.5f}, CMy={CMy:.4f}")
print(f"elapsed: {time.time()-t0:.0f}s")
