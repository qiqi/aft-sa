"""Build + run the missing TEprop turbulent cases (NACA0012 at α=-2,2,6,8 and
NLF0416 at α=-2,0,2,4,6,8) so the polar's turb reference exists on the new mesh
for the full sweep. Runs in parallel across 8 GPUs.
"""
import os, sys, json, copy, shutil, time, csv, subprocess, signal, threading, queue
from pathlib import Path
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/flow360")
os.environ["AI_CHI_INF"] = "0.02"
from rans.pipeline import run as pipe_run
from rans.env import make_env
from naca_contour_te import generate_contour_te

F = "/home/qiqi/flexcompute/aft-sa/flow360"
NLF_DAT = "/home/qiqi/flexcompute/aft-sa/external/construct2d/nlf0416.dat"
CFG_NACA = json.load(open(f"{F}/naca0012_re1m.json"))
NLF_BASE = f"{F}/nlf0416_re1m.json"
if os.path.exists(NLF_BASE):
    CFG_NLF = json.load(open(NLF_BASE))
else:
    CFG_NLF = copy.deepcopy(CFG_NACA)
    for el in CFG_NLF.get('elements', []):
        if 'name' in el: el['name'] = 'nlf0416'

MESH = dict(N=400, hwall=0.004, h0=7.0e-6, g=1.1, hmax=2.0, far_r=100.0,
            h_te=7.0e-6, r_te=1.5)

CASES = []
for a in [-2, 2, 6, 8]:        # missing NACA turb
    CASES.append((f"cavprop_naca0012_turb_a{a}", 'naca0012', None,    CFG_NACA, a))
for a in [-2, 0, 2, 4, 6, 8]:  # all NLF turb
    CASES.append((f"cavprop_nlf0416_turb_a{a}",  'nlf0416',  NLF_DAT, CFG_NLF,  a))

# ---------- build ----------
print(f"Building {len(CASES)} turb meshes...")
ncells = {}
for out_name, af, dat_path, base_cfg, alpha in CASES:
    d = f"{F}/{out_name}"
    if os.path.exists(f"{d}/mesh.cgns"):
        print(f"  {out_name}: already built, skipping"); continue
    contour = generate_contour_te(MESH['N'], h_te=MESH['h_te'], r_te=MESH['r_te'],
                                  close_te=False, dat_path=dat_path)
    c = copy.deepcopy(base_cfg)
    if c['elements']:
        c['elements'][0]['contour'] = contour.tolist()
        c['elements'][0]['name'] = af
    c['mesh'] = dict(span=0.1, nspan=1, yplus=0.25,
                     growth=MESH['g'], hwall=MESH['hwall'],
                     h0=MESH['h0'], hmax=MESH['hmax'])
    c['flow']['alpha_deg'] = float(alpha)
    c['solver']['max_steps'] = 50000
    c['farfield'] = {'type': 'circle', 'center': [0.5, 0.0],
                     'radius': float(MESH['far_r']), 'n': 480}
    cfg_path = f"{F}/_{out_name}.json"; json.dump(c, open(cfg_path, 'w'))
    shutil.rmtree(d, ignore_errors=True)
    t = time.time(); pipe_run(cfg_path, d, solve=False)
    print(f"  {out_name}: built in {time.time()-t:.0f}s", flush=True)

# ---------- run in parallel ----------
END_MARKER = "Exporting text outputs finished"
GRACE_SEC = 5.0; MAX_WALL_SEC = 1200
NGPU = 8
results_file = f"{F}/cavprop_alphasweep_results.json"
done = json.load(open(results_file))
results_lock = threading.Lock()

def clean(d):
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if (f.endswith(('.pvtu','.vtu','.pvd','.sock','.gltf'))
            or f.endswith('_v2.csv') or f.endswith('.csv.bk')
            or f.startswith('surface_forces') or f.endswith('.log')):
            try: os.remove(p)
            except: pass
        if f in ('ipc_data','restartOutput'): shutil.rmtree(p, ignore_errors=True)
        if 'rank_' in f and '.dmp' in f:
            try: os.remove(p)
            except: pass

def run_one(case_dir, gpu_idx):
    d = f"{F}/{case_dir}"
    if not os.path.exists(d): return None
    print(f"[{time.strftime('%H:%M:%S')}] [gpu{gpu_idx}] {case_dir} (turb)", flush=True)
    t0 = time.time(); clean(d)
    cfg = json.load(open(f"{d}/Flow360.json"))
    cfg['timeStepping']['maxPseudoSteps'] = 50000
    cfg['timeStepping']['CFL']['max'] = 2000.0
    cfg['turbulenceModelSolver']['CFLMultiplier'] = 1.0
    if 'turbulenceModelSolver' in cfg and 'controlPanel' in cfg['turbulenceModelSolver']:
        cp = cfg['turbulenceModelSolver']['controlPanel']
        if 'freestreamSANuTildeRatio' in cp: cp['freestreamSANuTildeRatio'] = 3.0
    cfg.setdefault('freestream',{})['turbulentViscosityRatio'] = 3.0
    json.dump(cfg, open(f"{d}/Flow360.json","w"), indent=1)
    env, find = make_env(); env["AI_SA"] = "0"; env["OMP_NUM_THREADS"] = "1"
    r = subprocess.run([find("MeshPartitioner"),"--meshfile","mesh.cgns","--partitions","1","--threads","1"],
                      cwd=d, env=env, capture_output=True, text=True)
    if r.returncode!=0: print(f"[gpu{gpu_idx}] PARTITION FAIL"); return None
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
        rr = list(csv.reader(open(f"{d}/nonlinear_residual_v2.csv")))
        h = [x.strip() for x in rr[0]]; last = [x for x in rr[-1] if x.strip()!='']
        step = int(float(last[h.index('pseudo_step')]))
        cont = float(last[h.index('0_cont')]); nuHat = float(last[h.index('5_nuHat')])
        tr = list(csv.reader(open(f"{d}/total_forces_v2.csv")))
        th = [x.strip() for x in tr[0]]; tl = [x for x in tr[-1] if x.strip()!='']
        CL = float(tl[th.index('CL')]); CD = float(tl[th.index('CD')])
        print(f"[gpu{gpu_idx}] DONE {case_dir} CL={CL:.4f} CD={CD:.5f} t={time.time()-t0:.0f}s", flush=True)
        return dict(step=step, cont=cont, nuHat=nuHat, CL=CL, CD=CD, elapsed=time.time()-t0, tag='turb')
    except Exception as e: print(f"[gpu{gpu_idx}] parse FAIL: {e}"); return None

def worker(gpu_idx, q):
    while True:
        try: case_dir, *_ = q.get_nowait()
        except queue.Empty: return
        r = run_one(case_dir, gpu_idx)
        if r:
            with results_lock:
                done[case_dir] = r
                json.dump(done, open(results_file, 'w'), indent=1)
        q.task_done()

q = queue.Queue()
for c in CASES: q.put((c[0],))
threads = []
t_start = time.time()
for gpu in range(min(NGPU, len(CASES))):
    t = threading.Thread(target=worker, args=(gpu, q)); t.start(); threads.append(t)
for t in threads: t.join()
print(f"\nALL DONE in {time.time()-t_start:.0f}s")
