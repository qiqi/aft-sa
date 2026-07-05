"""Run TE-tight (h_TE = h_0) cavity L0-L3 with damped CFL."""
import os, sys, json, csv, time, subprocess, shutil, signal
from pathlib import Path
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env

F = "/home/qiqi/flexcompute/aft-sa/flow360"
END_MARKER = "Exporting text outputs finished"
GRACE_SEC = 5.0
MAX_WALL_SEC = 14400

NGPU_BY_LEVEL = {'L0': 1, 'L1': 1, 'L2': 2, 'L3': 8}
CASES = [('proper_cav_L0_TEtight','L0'),('proper_cav_L1_TEtight','L1'),
         ('proper_cav_L2_TEtight','L2'),('proper_cav_L3_TEtight','L3')]

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

def run_one(case_dir, ngpu):
    d = f"{F}/{case_dir}"
    print(f"\n[{time.strftime('%H:%M:%S')}] === {case_dir} ngpu={ngpu} ===", flush=True)
    t0 = time.time(); clean(d)
    cfg = json.load(open(f"{d}/Flow360.json"))
    cfg['timeStepping']['maxPseudoSteps'] = 50000
    cfg['timeStepping']['CFL']['max'] = 2000.0
    cfg['turbulenceModelSolver']['CFLMultiplier'] = 1.0
    json.dump(cfg, open(f"{d}/Flow360.json", "w"), indent=1)
    env, find = make_env(); env["AI_SA"] = "1"; env["OMP_NUM_THREADS"] = "1"

    r = subprocess.run([find("MeshPartitioner"), "--meshfile", "mesh.cgns",
                        "--partitions", str(ngpu), "--threads", str(ngpu)],
                       cwd=d, env=env, capture_output=True, text=True)
    if r.returncode != 0: print(f"  PARTITION FAIL: {r.stderr[-500:]}"); return None
    if ngpu == 1:
        r = subprocess.run([find("MeshProcessor"), "--threads", "1", "mesh.cgns"],
                           cwd=d, env=env, capture_output=True, text=True, timeout=1800)
    else:
        r = subprocess.run(["mpirun", "-np", str(ngpu), find("MeshProcessor"),
                            "--threads", "1", "mesh.cgns"],
                           cwd=d, env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0: print(f"  MeshProcessor FAIL: {r.stderr[-1000:]}"); return None

    senv = dict(env); senv["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(ngpu))
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
            senv["OMPI_COMM_WORLD_LOCAL_RANK"] = "0"; senv["OMPI_COMM_WORLD_RANK"] = "0"; senv["OMPI_COMM_WORLD_SIZE"] = "1"
            solver = subprocess.Popen([find("Flow360Solver")], cwd=d, env=senv, stdout=slog, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        else:
            solver = subprocess.Popen(["mpirun", "-np", str(ngpu), find("Flow360Solver")], cwd=d, env=senv, stdout=slog, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        end_t = None
        while True:
            if solver.poll() is not None: print(f"  solver exit rc={solver.returncode}"); break
            try:
                with open(f"{d}/solver.log","rb") as f:
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
                break
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
        print(f"  step={step} cont={cont:.1e} nuHat={nuHat:.1e} CL={CL:.4f} CD={CD:.5f} t={time.time()-t0:.0f}s", flush=True)
        return dict(step=step, cont=cont, nuHat=nuHat, CL=CL, CD=CD, elapsed=time.time()-t0, ngpu=ngpu)
    except: return None

results = {}
t_start = time.time()
for case_dir, level in CASES:
    if not os.path.exists(f"{F}/{case_dir}"): print(f"SKIP {case_dir}"); continue
    r = run_one(case_dir, NGPU_BY_LEVEL[level])
    if r: results[case_dir] = r
json.dump(results, open(f"{F}/te_tight_results.json", 'w'), indent=1)
print(f"\nALL DONE in {time.time()-t_start:.0f}s")
for c, r in results.items():
    print(f"  {c}: step={r['step']} cont={r['cont']:.1e} nuHat={r['nuHat']:.1e} CL={r['CL']:.4f} CD={r['CD']:.5f}")
