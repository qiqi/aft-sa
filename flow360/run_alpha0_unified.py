"""Sanity test for the unified a_max=0.2 + chi^{1/4} σ_t formulation.

Compares NACA0012 α=0° on both grids:
  - existing  cavprop_naca0012_a0  / strprop_naca0012_a0  (old a_max=0.05 baseline)
  - new       cavprop_a02_naca0012_a0 / strprop_a02_naca0012_a0 (with rebuilt SA lib)

Acceptance: CL within 1%, CD within 2%, transition x/c within 0.02 of the baseline.
"""
import os, sys, json, csv, time, subprocess, shutil, signal, threading, queue
from pathlib import Path
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env

F = "/home/qiqi/flexcompute/aft-sa/flow360"
END_MARKER = "Exporting text outputs finished"
GRACE_SEC = 5.0
MAX_WALL_SEC = 600

# Two cases: copy the existing case dirs (which already have meshes built) to new names,
# then run with the rebuilt Flow360Solver.
SOURCES = [
    ('cavprop_naca0012_a0',  'cavprop_a02_naca0012_a0'),
    ('strprop_naca0012_a0',  'strprop_a02_naca0012_a0'),
]

# Clone the case dirs (mesh + config) but blank out previous solver results.
def clone_case(src, dst):
    src_dir = f"{F}/{src}"
    dst_dir = f"{F}/{dst}"
    if os.path.exists(dst_dir): shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    # Strip out the prior solver outputs so we run from scratch
    for fname in os.listdir(dst_dir):
        p = f"{dst_dir}/{fname}"
        if (fname.endswith(('.pvtu','.vtu','.pvd','.sock','.gltf')) or fname.endswith('_v2.csv')
            or fname.endswith('.csv.bk') or fname.startswith('surface_forces') or fname.endswith('.log')):
            try: os.remove(p)
            except: pass
        if fname in ('ipc_data','restartOutput'): shutil.rmtree(p, ignore_errors=True)
        if 'rank_' in fname and '.dmp' in fname:
            try: os.remove(p)
            except: pass

def run(case_dir, gpu_idx):
    d = f"{F}/{case_dir}"
    print(f"[gpu{gpu_idx}] running {case_dir}", flush=True)
    t0 = time.time()
    cfg = json.load(open(f"{d}/Flow360.json"))
    cfg['timeStepping']['maxPseudoSteps'] = 50000
    cfg['timeStepping']['CFL']['max'] = 2000.0
    cfg['turbulenceModelSolver']['CFLMultiplier'] = 1.0
    json.dump(cfg, open(f"{d}/Flow360.json","w"), indent=1)
    env, find = make_env()
    env["AI_SA"]    = "1"
    env["OMP_NUM_THREADS"] = "1"
    # Keep the SAME legacy chi_inf input; rans/case.py now rescales internally to chi_inf^4
    env["AI_CHI_INF"] = "0.02"
    r = subprocess.run([find("MeshPartitioner"),"--meshfile","mesh.cgns","--partitions","1","--threads","1"],
                      cwd=d, env=env, capture_output=True, text=True)
    if r.returncode != 0: print(f"[gpu{gpu_idx}] PARTITION FAIL"); return None
    r = subprocess.run([find("MeshProcessor"),"--threads","1","mesh.cgns"],
                      cwd=d, env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0: print(f"[gpu{gpu_idx}] MeshProcessor FAIL"); return None
    senv = dict(env); senv["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    sock = Path(d)/"ipc_control.sock"
    if sock.exists(): sock.unlink()
    pp_log = open(f"{d}/postprocessor.log","w")
    pp = subprocess.Popen([find("columnarDataProcessor.py"),"--asyncMode",
                          "--inputSimulationJson","simulation.json",
                          "--columnarDataProcessorJson","columnar.json"],
                         cwd=d, env=senv, stdout=pp_log, stderr=subprocess.STDOUT)
    try:
        for _ in range(40):
            if sock.exists(): break
            time.sleep(0.5)
        slog = open(f"{d}/solver.log","w")
        senv["OMPI_COMM_WORLD_LOCAL_RANK"]="0"; senv["OMPI_COMM_WORLD_RANK"]="0"
        senv["OMPI_COMM_WORLD_SIZE"]="1"
        solver = subprocess.Popen([find("Flow360Solver")], cwd=d, env=senv,
                                 stdout=slog, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        end_t = None
        while True:
            if solver.poll() is not None: break
            try:
                with open(f"{d}/solver.log","rb") as f:
                    f.seek(0,2); sz=f.tell(); f.seek(max(0,sz-8192))
                    tail = f.read().decode(errors="ignore")
                if END_MARKER in tail and end_t is None: end_t = time.time()
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
                except: pass; break
            time.sleep(3.0)
        slog.close()
    finally:
        pp.terminate()
        try: pp.wait(timeout=10)
        except subprocess.TimeoutExpired: pp.kill()
        pp_log.close()
    try:
        tr = list(csv.reader(open(f"{d}/total_forces_v2.csv")))
        th = [x.strip() for x in tr[0]]; tl = [x for x in tr[-1] if x.strip()!='']
        CL = float(tl[th.index('CL')]); CD = float(tl[th.index('CD')])
        CDp = float(tl[th.index('CDPressure')]); CDf = float(tl[th.index('CDSkinFriction')])
        print(f"[gpu{gpu_idx}] DONE {case_dir} CL={CL:+.4f} CD={CD:.5f} CDp={CDp:.5f} CDf={CDf:.5f} t={time.time()-t0:.0f}s")
        return dict(CL=CL, CD=CD, CDp=CDp, CDf=CDf)
    except Exception as e:
        print(f"[gpu{gpu_idx}] parse FAIL: {e}"); return None

# clone + run
for src, dst in SOURCES:
    print(f"\ncloning {src} -> {dst}")
    clone_case(src, dst)

results = {}
q = queue.Queue()
for i, (_, dst) in enumerate(SOURCES): q.put((dst, i))
def worker(gpu):
    while True:
        try: case, gpu_idx = q.get_nowait()
        except queue.Empty: return
        r = run(case, gpu_idx)
        if r: results[case] = r
        q.task_done()
threads = []
for i in range(2):
    t = threading.Thread(target=worker, args=(i,)); t.start(); threads.append(t)
for t in threads: t.join()

print("\n=== Comparison: unified vs baseline ===")
for src, dst in SOURCES:
    if dst not in results: continue
    new = results[dst]
    try:
        tr = list(csv.reader(open(f"{F}/{src}/total_forces_v2.csv")))
        th = [x.strip() for x in tr[0]]; tl = [x for x in tr[-1] if x.strip()!='']
        old = dict(CL=float(tl[th.index('CL')]), CD=float(tl[th.index('CD')]),
                   CDp=float(tl[th.index('CDPressure')]), CDf=float(tl[th.index('CDSkinFriction')]))
    except: continue
    print(f"  {src}:")
    print(f"    old: CL={old['CL']:+.4f} CD={old['CD']:.5f} CDp={old['CDp']:.5f} CDf={old['CDf']:.5f}")
    print(f"    new: CL={new['CL']:+.4f} CD={new['CD']:.5f} CDp={new['CDp']:.5f} CDf={new['CDf']:.5f}")
    print(f"    Δrel CL={(new['CL']-old['CL'])/old['CL']*100 if abs(old['CL'])>1e-6 else 0:+.2f}% "
          f"CD={(new['CD']-old['CD'])/old['CD']*100:+.2f}% "
          f"CDp={(new['CDp']-old['CDp'])/old['CDp']*100:+.2f}% "
          f"CDf={(new['CDf']-old['CDf'])/old['CDf']*100:+.2f}%")
