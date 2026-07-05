"""Run a single Flow360 case dir on a single GPU. Usage:
   python run_one_case.py <case_dir_basename> [gpu_idx]

Mirrors the partition/mesh/solve flow of run_compare_a0.py but for one case.
"""
import os, sys, json, time, subprocess, shutil
from pathlib import Path
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env

F = "/home/qiqi/flexcompute/aft-sa/flow360"

case = sys.argv[1]
gpu  = int(sys.argv[2]) if len(sys.argv) > 2 else 0
d = f"{F}/{case}"
if not os.path.exists(d):
    print(f"no such case dir: {d}"); sys.exit(1)

# clean
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

# CFL/steps
cfg = json.load(open(f"{d}/Flow360.json"))
# cfg['timeStepping']['maxPseudoSteps'] = 50000  # respect file value
cfg['timeStepping']['CFL']['max'] = 2000.0
cfg['turbulenceModelSolver']['CFLMultiplier'] = 1.0
json.dump(cfg, open(f"{d}/Flow360.json","w"), indent=1)

env, find = make_env()
env["AI_SA"] = "1"
env["OMP_NUM_THREADS"] = "1"
env["OMPI_COMM_WORLD_LOCAL_RANK"] = "0"

print(f"[{time.strftime('%H:%M:%S')}] partitioning {case} ...", flush=True)
r = subprocess.run([find("MeshPartitioner"),"--meshfile","mesh.cgns",
                    "--partitions","1","--threads","1"],
                   cwd=d, env=env, capture_output=True, text=True)
if r.returncode != 0:
    print(f"PARTITION FAIL: {r.stderr[-400:]}"); sys.exit(2)

print(f"[{time.strftime('%H:%M:%S')}] mesh processing {case} ...", flush=True)
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

print(f"[{time.strftime('%H:%M:%S')}] Flow360Solver {case} on gpu{gpu} ...", flush=True)
solver_log = open(f"{d}/solver.log","w")
t0 = time.time()
sp = subprocess.Popen([find("Flow360Solver")], cwd=d, env=senv,
                     stdout=solver_log, stderr=subprocess.STDOUT)
sp.wait(timeout=1800)
pp.wait(timeout=120)
dt = time.time() - t0
print(f"[{time.strftime('%H:%M:%S')}] done {case} in {dt:.1f}s; rc={sp.returncode}", flush=True)

# Quick result dump
try:
    last = open(f"{d}/surface_forces.csv").read().splitlines()[-1].split(',')
    cl = float(last[1]); cd = float(last[2]); cdv = float(last[3])
    print(f"  CL={cl:+.4f}  CD={cd:.5f}  CDv={cdv:.5f}")
except Exception as e:
    print(f"  no surface_forces.csv: {e}")
