"""Build proper-L1 O-grid (Construct2D) meshes for NACA0012 and NLF(1)-0416,
then dispatch 24 runs in parallel (6 alphas x 2 airfoils x SA-AI + turb).

Outputs land in strprop_{af}_a{a} and strprop_{af}_turb_a{a}.

NACA: reuses the existing proper_struct_L1.p3d (already produced by build_proper_struct.py).
NLF:  runs Construct2D fresh for nlf0416 at the same L1 settings.
"""
import os, sys, json, csv, time, subprocess, shutil, signal, threading, queue, copy
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/flow360")
os.environ["AI_CHI_INF"] = "0.02"
from rans.env import make_env
from rans.config import CaseConfig
from rans import case as _case, mesh as _mesh

F = "/home/qiqi/flexcompute/aft-sa/flow360"
C2D_DIR = "/home/qiqi/flexcompute/aft-sa/external/construct2d"
CONSTRUCT2D = f"{C2D_DIR}/construct2d"
NACA_CFG = f"{F}/naca0012_re1m.json"
NLF_CFG  = f"{F}/nlf0416_re1m.json"
if not os.path.exists(NLF_CFG):
    base = json.load(open(NACA_CFG))
    for el in base.get('elements', []):
        if 'name' in el: el['name'] = 'nlf0416'
    json.dump(base, open(NLF_CFG, 'w'), indent=2)
    print(f"created {NLF_CFG}")

# ---- L1 settings (matched to build_proper_struct.py L1 row) ----
L1 = dict(N=400, jmax=160, h0=7.0e-6, lesp=0.0020, far_r=100.0)
RE = 1e6
def ypls_for_h0(h0): return h0 / 2.13e-5

def script_construct2d(project, lesp, tesp, nsrf, radi, jmax, ypls, recd, work_cwd):
    cmds = [
        f"{project}.dat", "SOPT",
        "NSRF", str(nsrf),
        "LESP", f"{lesp:.6e}",
        "TESP", f"{tesp:.6e}",
        "RADI", f"{radi:.4f}",
        "NWKE", "0",
        "QUIT", "VOPT",
        "JMAX", str(jmax),
        "YPLS", f"{ypls:.6e}",
        "RECD", f"{recd:.1f}",
        "QUIT", "GRID", "SMTH", "QUIT",
    ]
    inp = "\n".join(cmds) + "\n"
    return subprocess.run([CONSTRUCT2D], input=inp, capture_output=True, text=True,
                          cwd=work_cwd, timeout=600)

# ---- build NLF Construct2D L1 ----
nlf_p3d = f"{C2D_DIR}/proper_struct_nlf_L1.p3d"
if not os.path.exists(nlf_p3d):
    print("Building NLF proper_struct_L1 via Construct2D...")
    project = "proper_struct_nlf_L1"
    # Use the nlf0416.dat directly as the input (already in C2D dir, normalized to chord 1)
    shutil.copy(f"{C2D_DIR}/nlf0416.dat", f"{C2D_DIR}/{project}.dat")
    ypls = ypls_for_h0(L1['h0'])
    tesp = L1['lesp'] * 0.063
    t = time.time()
    proc = script_construct2d(project, L1['lesp'], tesp, L1['N'], L1['far_r'],
                               L1['jmax'], ypls, RE, C2D_DIR)
    if not os.path.exists(nlf_p3d):
        print(f"  Construct2D FAIL for NLF L1.")
        print(f"  stdout tail: {proc.stdout[-500:]}")
        print(f"  stderr tail: {proc.stderr[-500:]}")
        sys.exit(1)
    print(f"  built NLF proper_struct_L1 in {time.time()-t:.0f}s")
else:
    print(f"NLF L1 .p3d already exists: {nlf_p3d}")

# ---- p3d → .cgns case dir for each airfoil ----
def build_case_dir(out_dir, p3d, wall_name, base_cfg_path, alpha):
    """Convert .p3d to mesh.cgns and run preprocess (mesh partitioner builds Flow360.json)."""
    os.makedirs(out_dir, exist_ok=True)
    toks = open(p3d).read().split()
    ni, nj = int(toks[0]), int(toks[1])
    vals = np.array(toks[2:2+2*ni*nj], float)
    X = vals[:ni*nj].reshape(nj, ni).T
    Y = vals[ni*nj:2*ni*nj].reshape(nj, ni).T
    Ni = ni - 1; nspan = 1; span = 0.1
    k = lambda i, j: i*nj + j
    P = np.empty((Ni*nj, 2))
    for i in range(Ni):
        for j in range(nj): P[k(i,j)] = (X[i,j], Y[i,j])
    N = Ni * nj
    NL = nspan + 1
    nid = lambda L, kk: L*N + kk + 1
    phys = [(2, 2, wall_name), (2, 3, "farfield"), (2, 4, "symmetry1"), (2, 5, "symmetry2"), (3, 1, "fluid")]
    quads = [(k(i,j), k((i+1)%Ni, j), k((i+1)%Ni, j+1), k(i, j+1)) for i in range(Ni) for j in range(nj-1)]
    wallE = [(k(i, 0), k((i+1)%Ni, 0)) for i in range(Ni)]
    farE = [(k(i, nj-1), k((i+1)%Ni, nj-1)) for i in range(Ni)]
    elems = []; eid = [1]
    def emit(s): elems.append(f"{eid[0]} {s}"); eid[0] += 1
    with open(out_dir + "/mesh.msh", "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n%d\n" % len(phys))
        for d, t_, n in phys: f.write('%d %d "%s"\n' % (d, t_, n))
        f.write("$EndPhysicalNames\n$Nodes\n%d\n" % (NL*N))
        for L in range(NL):
            ys = -span*L/nspan
            for kk in range(N): f.write("%d %.16g %.16g %.16g\n" % (nid(L,kk), P[kk,0], ys, P[kk,1]))
        f.write("$EndNodes\n")
        for a, b, c, d in quads: emit("3 2 4 4 %d %d %d %d" % (nid(0,a), nid(0,b), nid(0,c), nid(0,d)))
        for a, b, c, d in quads: emit("3 2 5 5 %d %d %d %d" % (nid(nspan,a), nid(nspan,b), nid(nspan,c), nid(nspan,d)))
        for a, b in wallE:
            for L in range(nspan): emit("3 2 2 2 %d %d %d %d" % (nid(L,a), nid(L,b), nid(L+1,b), nid(L+1,a)))
        for a, b in farE:
            for L in range(nspan): emit("3 2 3 3 %d %d %d %d" % (nid(L,a), nid(L,b), nid(L+1,b), nid(L+1,a)))
        for L in range(nspan):
            for a, b, c, d in quads: emit("5 2 1 1 %d %d %d %d %d %d %d %d" % (
                nid(L,a), nid(L,b), nid(L,c), nid(L,d), nid(L+1,a), nid(L+1,b), nid(L+1,c), nid(L+1,d)))
        f.write("$Elements\n%d\n" % len(elems)); f.write("\n".join(elems) + "\n$EndElements\n")
    env, find = make_env()
    _mesh.gmsh_to_cgns(out_dir + "/mesh.msh", out_dir + "/mesh.cgns", find("flow360gmshtocgns"), env)
    cfg = CaseConfig.load(base_cfg_path)
    cfg.solver.max_steps = 50000
    cfg.flow.alpha_deg = float(alpha)
    _case.preprocess(out_dir, "mesh.cgns", find, env, cfg=cfg,
                     wall_names=[f"fluid/{wall_name}"],
                     boundary_names=[f"fluid/farfield", f"fluid/{wall_name}",
                                     "fluid/symmetry1", "fluid/symmetry2"],
                     timings={}, sdk_cache_dir=None, sim_builder=_case.build_simulation_json)

ALPHAS = [-2, 0, 2, 4, 6, 8]
NACA_P3D = f"{C2D_DIR}/proper_struct_L1.p3d"

# build dirs
print("\nBuilding case dirs...")
t_b = time.time()
for af, p3d, base_cfg in [('naca0012', NACA_P3D, NACA_CFG),
                          ('nlf0416',  nlf_p3d,  NLF_CFG)]:
    for a in ALPHAS:
        for tag, suffix in [('aftsa', ''), ('turb', '_turb')]:
            d = f"{F}/strprop_{af}{suffix}_a{a}"
            if os.path.exists(f"{d}/mesh.cgns"):
                print(f"  {d}: already built, skip"); continue
            print(f"  building {d} (alpha={a})...", flush=True)
            build_case_dir(d, p3d, af, base_cfg, a)
print(f"Build done in {time.time()-t_b:.0f}s\n")

# ---- run in parallel ----
END_MARKER = "Exporting text outputs finished"
GRACE_SEC = 5.0; MAX_WALL_SEC = 1500
NGPU = 8
results_file = f"{F}/strprop_results.json"
done = {}
if os.path.exists(results_file):
    try: done = json.load(open(results_file))
    except: done = {}
results_lock = threading.Lock()

CASES = []
for af in ['naca0012', 'nlf0416']:
    for a in ALPHAS:
        for tag, suffix in [('aftsa', ''), ('turb', '_turb')]:
            name = f"strprop_{af}{suffix}_a{a}"
            if name in done: continue
            CASES.append((name, tag))
print(f"Pending {len(CASES)} runs: {[c[0] for c in CASES]}")

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

def run_one(case_dir, tag, gpu_idx):
    d = f"{F}/{case_dir}"
    if not os.path.exists(d): return None
    print(f"[{time.strftime('%H:%M:%S')}] [gpu{gpu_idx}] {case_dir} ({tag})", flush=True)
    t0 = time.time(); clean(d)
    cfg = json.load(open(f"{d}/Flow360.json"))
    cfg['timeStepping']['maxPseudoSteps'] = 50000
    cfg['timeStepping']['CFL']['max'] = 2000.0
    cfg['turbulenceModelSolver']['CFLMultiplier'] = 1.0
    if tag == 'turb':
        if 'turbulenceModelSolver' in cfg and 'controlPanel' in cfg['turbulenceModelSolver']:
            cp = cfg['turbulenceModelSolver']['controlPanel']
            if 'freestreamSANuTildeRatio' in cp: cp['freestreamSANuTildeRatio'] = 3.0
        cfg.setdefault('freestream',{})['turbulentViscosityRatio'] = 3.0
    json.dump(cfg, open(f"{d}/Flow360.json","w"), indent=1)
    env, find = make_env(); env["AI_SA"] = "0" if tag == 'turb' else "1"
    env["OMP_NUM_THREADS"] = "1"
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
        cont = float(last[h.index('0_cont')])
        tr = list(csv.reader(open(f"{d}/total_forces_v2.csv")))
        th = [x.strip() for x in tr[0]]; tl = [x for x in tr[-1] if x.strip()!='']
        CL = float(tl[th.index('CL')]); CD = float(tl[th.index('CD')])
        print(f"[gpu{gpu_idx}] DONE {case_dir} CL={CL:.4f} CD={CD:.5f} t={time.time()-t0:.0f}s", flush=True)
        return dict(step=step, cont=cont, CL=CL, CD=CD, elapsed=time.time()-t0, tag=tag)
    except Exception as e: print(f"[gpu{gpu_idx}] parse FAIL: {e}"); return None

def worker(gpu_idx, q):
    while True:
        try: case_dir, tag = q.get_nowait()
        except queue.Empty: return
        r = run_one(case_dir, tag, gpu_idx)
        if r:
            with results_lock:
                done[case_dir] = r
                json.dump(done, open(results_file, 'w'), indent=1)
        q.task_done()

q = queue.Queue()
for c in CASES: q.put(c)
threads = []
t_start = time.time()
for gpu in range(min(NGPU, len(CASES))):
    t = threading.Thread(target=worker, args=(gpu, q)); t.start(); threads.append(t)
for t in threads: t.join()
print(f"\nALL DONE in {time.time()-t_start:.0f}s ({len(done)} cases)")
