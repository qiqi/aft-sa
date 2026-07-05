"""Run a subset of the α=0° comparison cases in parallel across N GPUs.
Set MODEL_TAG env to 'base' or 'uni' to filter."""
import os, sys, json, csv, time, subprocess, shutil, signal, threading, queue
from pathlib import Path
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env

F = "/home/qiqi/flexcompute/aft-sa/flow360"
END_MARKER = "Exporting text outputs finished"
GRACE_SEC = 5.0
MAX_WALL_SEC = 600
NGPU = 8

MODEL_TAG = sys.argv[1] if len(sys.argv) > 1 else 'uni'  # 'base' or 'uni'
CASES = []
for grid in ['cav_L0', 'cav_L1', 'str_L0', 'str_L1']:
    CASES.append(f"{grid}_{MODEL_TAG}_a0")
print(f"running {MODEL_TAG} cases: {CASES}")

def clean(d):
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if (f.endswith(('.pvtu','.vtu','.pvd','.sock','.gltf')) or f.endswith('_v2.csv')
            or f.endswith('.csv.bk') or f.startswith('surface_forces') or f.endswith('.log')):
            try: os.remove(p)
            except: pass
        if f in ('ipc_data','restartOutput'): shutil.rmtree(p, ignore_errors=True)
        if 'rank_' in f and '.dmp' in f:
            try: os.remove(p)
            except: pass

def run_one(case_dir, gpu_idx):
    d = f"{F}/{case_dir}"
    if not os.path.exists(d): return None
    print(f"[{time.strftime('%H:%M:%S')}] [gpu{gpu_idx}] {case_dir}", flush=True)
    t0 = time.time(); clean(d)
    cfg = json.load(open(f"{d}/Flow360.json"))
    cfg['timeStepping']['maxPseudoSteps'] = 50000
    cfg['timeStepping']['CFL']['max'] = 2000.0
    cfg['turbulenceModelSolver']['CFLMultiplier'] = 1.0
    json.dump(cfg, open(f"{d}/Flow360.json","w"), indent=1)
    env, find = make_env(); env["AI_SA"]="1"; env["OMP_NUM_THREADS"]="1"
    # Force the right chi_inf via env too (rans/case.py is back to no-rescaling, but
    # the case dir Flow360.json already has the right value baked in)
    if '_uni_' in case_dir: env["AI_CHI_INF"] = "1.6e-7"  # not actually read here, BC is in JSON
    else: env["AI_CHI_INF"] = "0.02"
    r = subprocess.run([find("MeshPartitioner"),"--meshfile","mesh.cgns","--partitions","1","--threads","1"],
                      cwd=d, env=env, capture_output=True, text=True)
    if r.returncode!=0: print(f"[gpu{gpu_idx}] PARTITION FAIL: {r.stderr[-300:]}"); return None
    r = subprocess.run([find("MeshProcessor"),"--threads","1","mesh.cgns"],
                      cwd=d, env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode!=0: print(f"[gpu{gpu_idx}] MeshProcessor FAIL"); return None
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
        senv["OMPI_COMM_WORLD_LOCAL_RANK"]="0"; senv["OMPI_COMM_WORLD_RANK"]="0"; senv["OMPI_COMM_WORLD_SIZE"]="1"
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
        print(f"[gpu{gpu_idx}] DONE {case_dir} CL={CL:+.4f} CD={CD:.5f} CDp={CDp:.5f} CDf={CDf:.5f} t={time.time()-t0:.0f}s", flush=True)
        return dict(CL=CL, CD=CD, CDp=CDp, CDf=CDf)
    except Exception as e:
        print(f"[gpu{gpu_idx}] parse FAIL: {e}"); return None

q = queue.Queue()
for i, c in enumerate(CASES): q.put((c, i))
results = {}
lock = threading.Lock()

def worker(gpu_idx):
    while True:
        try: case_dir, _ = q.get_nowait()
        except queue.Empty: return
        r = run_one(case_dir, gpu_idx)
        if r:
            with lock: results[case_dir] = r
        q.task_done()

threads = [threading.Thread(target=worker, args=(i,)) for i in range(min(NGPU, len(CASES)))]
for t in threads: t.start()
for t in threads: t.join()

print(f"\nAll {MODEL_TAG} done.")
json.dump(results, open(f"{F}/compare_{MODEL_TAG}_a0_results.json", 'w'), indent=1)
for c, r in sorted(results.items()):
    print(f"  {c}: CL={r['CL']:+.4f} CD={r['CD']:.5f} CDp={r['CDp']:.5f} CDf={r['CDf']:.5f}")
