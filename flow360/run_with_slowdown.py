"""Run a Flow360 case with custom AI_LAMINAR_SLOWDOWN env var.

Usage:  python run_with_slowdown.py <case_dir> <gpu> <slowdown_value>
"""
import os, sys, json, time, subprocess, shutil
from pathlib import Path
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env

F = "/home/qiqi/flexcompute/aft-sa/flow360"

case = sys.argv[1]
gpu  = int(sys.argv[2])
fSlow = float(sys.argv[3])
d = f"{F}/{case}"
if not os.path.exists(d):
    print(f"no such case dir: {d}"); sys.exit(1)

# Clean previous outputs
for f in os.listdir(d):
    p = os.path.join(d, f)
    if (f.endswith(('.pvtu','.vtu','.pvd','.sock','.gltf','.log','.csv.bk'))
        or f.endswith('_v2.csv') or f.startswith('surface_forces')):
        try: os.remove(p)
        except: pass
    if f in ('ipc_data','restartOutput'): shutil.rmtree(p, ignore_errors=True)
    if 'rank_' in f and '.dmp' in f:
        try: os.remove(p)
        except: pass

env, find = make_env()
env["AI_SA"] = "1"
env["AI_LAMINAR_SLOWDOWN"] = str(fSlow)
env["AI_SLOWWIDTH"] = "4.0"
env["OMP_NUM_THREADS"] = "1"
env["OMPI_COMM_WORLD_LOCAL_RANK"] = "0"

print(f"[{time.strftime('%H:%M:%S')}] {case} on gpu{gpu}, AI_LAMINAR_SLOWDOWN={fSlow}", flush=True)

r = subprocess.run([find("MeshPartitioner"),"--meshfile","mesh.cgns",
                    "--partitions","1","--threads","1"],
                   cwd=d, env=env, capture_output=True, text=True)
if r.returncode != 0:
    print(f"PARTITION FAIL: {r.stderr[-400:]}"); sys.exit(2)

r = subprocess.run([find("MeshProcessor"),"--threads","1","mesh.cgns"],
                   cwd=d, env=env, capture_output=True, text=True, timeout=1800)
if r.returncode != 0:
    print(f"MeshProcessor FAIL: {r.stderr[-400:]}"); sys.exit(3)

senv = dict(env); senv["CUDA_VISIBLE_DEVICES"] = str(gpu)
sock = Path(d)/"ipc_control.sock"
if sock.exists(): sock.unlink()
pp_log = open(f"{d}/postprocessor.log","w")
pp = subprocess.Popen([find("columnarDataProcessor.py"),"--asyncMode",
                      "--inputSimulationJson","simulation.json",
                      "--columnarDataProcessorJson","columnar.json"],
                     cwd=d, env=senv, stdout=pp_log, stderr=subprocess.STDOUT)

print(f"[{time.strftime('%H:%M:%S')}] Flow360Solver {case} on gpu{gpu}...", flush=True)
solver_log = open(f"{d}/solver.log","w")
sp = subprocess.Popen([find("Flow360Solver")], cwd=d, env=senv,
                     stdout=solver_log, stderr=subprocess.STDOUT)
sp.wait(timeout=7200)  # up to 2 hours
pp.wait(timeout=120)
print(f"[{time.strftime('%H:%M:%S')}] done {case}, rc={sp.returncode}", flush=True)
