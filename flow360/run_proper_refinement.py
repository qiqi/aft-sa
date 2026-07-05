"""Run all proper-refinement cases at α=4°:
  cavity L0..L4 (5 levels)
  struct L0..L2 (3 levels)

Pipeline per case:
  1. MeshPartitioner --partitions N
  2. mpirun -np N MeshProcessor --threads 1
  3. mpirun -np N Flow360Solver  (with watchdog to kill after end-marker)

GPU allocation per level:
  L0:  1 GPU
  L1:  1 GPU
  L2:  2 GPUs
  L3:  4 GPUs (cavity only)
  L4:  8 GPUs (cavity only)
"""
import os, sys, json, csv, time, subprocess, shutil, signal
from pathlib import Path
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env

F = "/home/qiqi/flexcompute/aft-sa/flow360"
END_MARKER = "Exporting text outputs finished"
GRACE_SEC = 5.0
MAX_WALL_SEC = 14400  # 4 hours max per case

NGPU_BY_LEVEL = {'L0': 1, 'L1': 1, 'L2': 2, 'L3': 4, 'L4': 8}

CASES = [  # (dirname, level)
    ('proper_cav_L0', 'L0'),
    ('proper_str_L0', 'L0'),
    ('proper_cav_L1', 'L1'),
    ('proper_str_L1', 'L1'),
    ('proper_cav_L2', 'L2'),
    ('proper_str_L2', 'L2'),
    ('proper_cav_L3', 'L3'),
    ('proper_cav_L4', 'L4'),
]

def clean_run_state(d):
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
    # also clean per-rank dmps if changing partition count
    for f in os.listdir(d):
        if 'rank_' in f and '.dmp' in f:
            try: os.remove(os.path.join(d, f))
            except: pass

def run_pipeline_one_case(case_dir, ngpu):
    d = f"{F}/{case_dir}"
    print(f"\n[{time.strftime('%H:%M:%S')}] === {case_dir} ngpu={ngpu} ===", flush=True)
    t0 = time.time()
    clean_run_state(d)
    env, find = make_env()
    env["AI_SA"] = "1"
    env["OMP_NUM_THREADS"] = "1"

    # 1. partition the mesh
    print(f"  [1/3] MeshPartitioner --partitions {ngpu}", flush=True)
    r = subprocess.run([find("MeshPartitioner"), "--meshfile", "mesh.cgns",
                        "--partitions", str(ngpu), "--threads", str(ngpu)],
                       cwd=d, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    PARTITION FAIL: {r.stderr[-500:]}"); return None
    # 2. process partitions
    print(f"  [2/3] MeshProcessor with mpirun -np {ngpu}", flush=True)
    if ngpu == 1:
        r = subprocess.run([find("MeshProcessor"), "--threads", "1", "mesh.cgns"],
                           cwd=d, env=env, capture_output=True, text=True, timeout=1800)
    else:
        r = subprocess.run(["mpirun", "-np", str(ngpu), find("MeshProcessor"),
                            "--threads", "1", "mesh.cgns"],
                           cwd=d, env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(f"    MeshProcessor FAIL: {r.stderr[-1000:]}"); return None
    # 3. solver with watchdog
    print(f"  [3/3] Flow360Solver with mpirun -np {ngpu}", flush=True)
    senv = dict(env)
    senv["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(ngpu))
    sock = Path(d) / "ipc_control.sock"
    if sock.exists(): sock.unlink()
    pp_log = open(f"{d}/postprocessor.log", "w")
    pp = subprocess.Popen([find("columnarDataProcessor.py"), "--asyncMode",
                            "--inputSimulationJson", "simulation.json",
                            "--columnarDataProcessorJson", "columnar.json"],
                           cwd=d, env=senv, stdout=pp_log, stderr=subprocess.STDOUT)
    try:
        for _ in range(40):
            if sock.exists(): break
            time.sleep(0.5)
        slog = open(f"{d}/solver.log", "w")
        if ngpu == 1:
            # single-rank still needs OMPI env to avoid mpirun issue
            senv["OMPI_COMM_WORLD_LOCAL_RANK"] = "0"
            senv["OMPI_COMM_WORLD_RANK"] = "0"
            senv["OMPI_COMM_WORLD_SIZE"] = "1"
            solver = subprocess.Popen([find("Flow360Solver")], cwd=d, env=senv,
                                      stdout=slog, stderr=subprocess.STDOUT,
                                      preexec_fn=os.setsid)
        else:
            solver = subprocess.Popen(["mpirun", "-np", str(ngpu), find("Flow360Solver")],
                                      cwd=d, env=senv, stdout=slog, stderr=subprocess.STDOUT,
                                      preexec_fn=os.setsid)
        end_t = None
        t_solver = time.time()
        while True:
            if solver.poll() is not None:
                rc = solver.returncode
                if rc != 0 and end_t is None:
                    print(f"    solver exit code {rc}, no end marker — likely error"); break
                break
            try:
                with open(f"{d}/solver.log", "rb") as f:
                    f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - 8192))
                    tail = f.read().decode(errors="ignore")
                if END_MARKER in tail and end_t is None:
                    end_t = time.time()
                    print(f"    end-marker at t={time.time()-t_solver:.0f}s", flush=True)
            except: pass
            if end_t and (time.time() - end_t) > GRACE_SEC:
                try: os.killpg(os.getpgid(solver.pid), signal.SIGTERM)
                except: pass
                try: solver.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(os.getpgid(solver.pid), signal.SIGKILL)
                    except: pass
                    solver.wait(timeout=5)
                break
            if time.time() - t_solver > MAX_WALL_SEC:
                try: os.killpg(os.getpgid(solver.pid), signal.SIGKILL)
                except: pass
                print(f"    TIMEOUT @ {MAX_WALL_SEC}s"); break
            time.sleep(2.0)
        slog.close()
    finally:
        pp.terminate()
        try: pp.wait(timeout=10)
        except subprocess.TimeoutExpired: pp.kill()
        pp_log.close()
    # collect result
    try:
        rr = list(csv.reader(open(f"{d}/nonlinear_residual_v2.csv")))
        h = [x.strip() for x in rr[0]]; last = [x for x in rr[-1] if x.strip() != '']
        step = int(float(last[h.index('pseudo_step')]))
        cont = float(last[h.index('0_cont')]); nuHat = float(last[h.index('5_nuHat')])
        tfp = f"{d}/total_forces_v2.csv"
        CD = None
        if os.path.exists(tfp):
            tr = list(csv.reader(open(tfp))); th = [x.strip() for x in tr[0]]
            tl = [x for x in tr[-1] if x.strip() != '']
            CD = float(tl[th.index('CD')])
        ok = "OK" if (cont < 1e-9 and nuHat < 1e-8) else "CAP"
        print(f"  [{time.strftime('%H:%M:%S')}] {ok} step={step} cont={cont:.1e} nuHat={nuHat:.1e} CD={CD} t={time.time()-t0:.0f}s", flush=True)
        return dict(step=step, cont=cont, nuHat=nuHat, CD=CD, elapsed=time.time()-t0, ngpu=ngpu)
    except Exception as e:
        print(f"  RESULT ERR {e}"); return None

# Set Flow360.json max steps
for case_dir, level in CASES:
    d = f"{F}/{case_dir}"
    if not os.path.exists(f"{d}/Flow360.json"): continue
    c = json.load(open(f"{d}/Flow360.json"))
    c['timeStepping']['maxPseudoSteps'] = 25000
    json.dump(c, open(f"{d}/Flow360.json", "w"), indent=1)

# Sequential execution
results = {}
t_start = time.time()
for case_dir, level in CASES:
    if not os.path.exists(f"{F}/{case_dir}"):
        print(f"SKIP {case_dir}: no dir"); continue
    ngpu = NGPU_BY_LEVEL[level]
    r = run_pipeline_one_case(case_dir, ngpu)
    if r is not None: results[case_dir] = r

json.dump(results, open(f"{F}/proper_refinement_results.json", 'w'), indent=1)
print(f"\n{'='*70}\nALL DONE in {time.time()-t_start:.0f}s")
print("=== summary ===")
for c, r in results.items():
    print(f"  {c}: step={r['step']}, cont={r['cont']:.1e}, nuHat={r['nuHat']:.1e}, CD={r['CD']}, t={r['elapsed']:.0f}s")
