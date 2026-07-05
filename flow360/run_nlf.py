"""Re-run the NLF(1)-0416 analysis with the CORRECT airfoil (16% thick, ~2.4% camber).
Cavity (pipeline.run) + structured C-grid (Construct2D nlf0416.p3d) sweeps, SA-AI + turb."""
import sys, os, json, shutil, threading, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
os.environ.setdefault("FLOW360_SUPPRESS_BETA_WARNING", "1")
os.environ["AI_CHI_INF"] = "0.02"
from rans.config import CaseConfig
from rans import case as _case, mesh as _mesh
from rans.pipeline import run as pipe_run
from rans.env import make_env
from rans.solve import run_solver, extract_forces
B = "/home/qiqi/flexcompute/aft-sa"; F = f"{B}/flow360"
SKIP = ('.pvtu', '.vtu', '.csv', '.log', '.sock', '.gltf', '.pvd')   # keep .dmp!
ALPHAS = [-2, 0, 2, 4, 6, 8]; span, nspan = 0.1, 1; WALL = "nlf0416"

def build_struct(p3d, OUT):
    os.makedirs(OUT, exist_ok=True)
    toks = open(p3d).read().split(); ni, nj = int(toks[0]), int(toks[1])
    vals = np.array(toks[2:2+2*ni*nj], float); X = vals[:ni*nj].reshape(nj, ni).T; Y = vals[ni*nj:2*ni*nj].reshape(nj, ni).T
    Ni = ni - 1; k = lambda i, j: i*nj + j
    P = np.empty((Ni*nj, 2))
    for i in range(Ni):
        for j in range(nj): P[k(i, j)] = (X[i, j], Y[i, j])
    N = Ni*nj
    quads = [(k(i, j), k((i+1) % Ni, j), k((i+1) % Ni, j+1), k(i, j+1)) for i in range(Ni) for j in range(nj-1)]
    wallE = [(k(i, 0), k((i+1) % Ni, 0)) for i in range(Ni)]; farE = [(k(i, nj-1), k((i+1) % Ni, nj-1)) for i in range(Ni)]
    NL = nspan + 1; nid = lambda L, kk: L*N + kk + 1
    phys = [(2, 2, WALL), (2, 3, "farfield"), (2, 4, "symmetry1"), (2, 5, "symmetry2"), (3, 1, "fluid")]
    elems = []; eid = [1]
    def emit(s): elems.append(f"{eid[0]} {s}"); eid[0] += 1
    with open(OUT + "/mesh.msh", "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n%d\n" % len(phys))
        for d, t, n in phys: f.write('%d %d "%s"\n' % (d, t, n))
        f.write("$EndPhysicalNames\n$Nodes\n%d\n" % (NL*N))
        for L in range(NL):
            ys = -span*L/nspan
            for kk in range(N): f.write("%d %.16g %.16g %.16g\n" % (nid(L, kk), P[kk, 0], ys, P[kk, 1]))
        f.write("$EndNodes\n")
        for a, b, c, d in quads: emit("3 2 4 4 %d %d %d %d" % (nid(0, a), nid(0, b), nid(0, c), nid(0, d)))
        for a, b, c, d in quads: emit("3 2 5 5 %d %d %d %d" % (nid(nspan, a), nid(nspan, b), nid(nspan, c), nid(nspan, d)))
        for a, b in wallE:
            for L in range(nspan): emit("3 2 2 2 %d %d %d %d" % (nid(L, a), nid(L, b), nid(L+1, b), nid(L+1, a)))
        for a, b in farE:
            for L in range(nspan): emit("3 2 3 3 %d %d %d %d" % (nid(L, a), nid(L, b), nid(L+1, b), nid(L+1, a)))
        for L in range(nspan):
            for a, b, c, d in quads: emit("5 2 1 1 %d %d %d %d %d %d %d %d" % (nid(L, a), nid(L, b), nid(L, c), nid(L, d), nid(L+1, a), nid(L+1, b), nid(L+1, c), nid(L+1, d)))
        f.write("$Elements\n%d\n" % len(elems)); f.write("\n".join(elems) + "\n$EndElements\n")
    env, find = make_env()
    _mesh.gmsh_to_cgns(OUT + "/mesh.cgns", OUT + "/mesh.cgns", find("flow360gmshtocgns"), env) if False else _mesh.gmsh_to_cgns(OUT + "/mesh.msh", OUT + "/mesh.cgns", find("flow360gmshtocgns"), env)
    cfg = CaseConfig.load(f"{F}/nlf0416_re1m.json"); cfg.solver.max_steps = 10000; cfg.flow.alpha_deg = 0.0
    _case.preprocess(OUT, "mesh.cgns", find, env, cfg=cfg, wall_names=[f"fluid/{WALL}"],
                     boundary_names=[f"fluid/farfield", f"fluid/{WALL}", "fluid/symmetry1", "fluid/symmetry2"],
                     timings={}, sdk_cache_dir=None, sim_builder=_case.build_simulation_json)

def stage(base, wd, alpha, chi):
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    for f in os.listdir(base):
        if any(f.endswith(x) for x in SKIP) or f in ('ipc_data', 'restartOutput'): continue
        s, d = os.path.join(base, f), os.path.join(wd, f)
        (shutil.copytree if os.path.isdir(s) else shutil.copy)(s, d)
    cf = json.load(open(f"{wd}/Flow360.json")); cf['freestream']['alphaAngle'] = float(alpha)
    cf['freestream']['turbulenceQuantities'] = {'modelType': 'ModifiedTurbulentViscosityRatio', 'modifiedTurbulentViscosityRatio': chi}
    fk = next(k for k in cf['boundaries'] if 'farfield' in k)
    cf['boundaries'][fk]['turbulenceQuantities']['modifiedTurbulentViscosityRatio'] = chi
    json.dump(cf, open(f"{wd}/Flow360.json", "w"), indent=1); return wd

def xtr(out):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{out}/surface_fluid_{WALL}.pvtu"); r.Update()
    g = r.GetOutput(); pts = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    nm = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    a = vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cf')))); v = np.linalg.norm(a, axis=1) if a.ndim > 1 else a
    x, z = pts[:, 0], pts[:, 2]; up = z > 1e-6; xs, cfs = x[up], v[up]; o = np.argsort(xs); xs, cfs = xs[o], cfs[o]
    nb = 50; bins = np.linspace(0, 1, nb + 1); xc = .5*(bins[1:]+bins[:-1])
    cb = np.array([np.median(cfs[(xs >= bins[i]) & (xs < bins[i+1])]) if ((xs >= bins[i]) & (xs < bins[i+1])).any() else np.nan for i in range(nb)])
    ok = np.isfinite(cb); xc, cb = xc[ok], cb[ok]; d = np.diff(cb); dm = .5*(xc[1:]+xc[:-1]); w = (dm > 0.04) & (dm < 0.95)
    if not w.any(): return np.nan
    j = np.argmax(d[w]); return float(dm[w][j])

# build bases
pipe_run(f"{F}/nlf0416_re1m.json", f"{F}/base_nlf_cavity", solve=False); print("built NLF cavity base", flush=True)
build_struct(f"{B}/external/construct2d/nlf0416.p3d", f"{F}/base_nlf_ogrid"); print("built NLF C-grid base", flush=True)

res = {}; lock = threading.Lock(); sem = threading.Semaphore(8)
def solve(tag, wd, aft, gpu):
    with sem:
        try:
            env, find = make_env()
            if aft: env["AI_SA"] = "1"
            run_solver(wd, find, env, gpu=gpu, timeout=2400)
            f = extract_forces(wd); r = dict(CL=f["CL"], CD=f["CD"], xtr=xtr(wd) if aft else 0.0)
            with lock: res[tag] = r
            print("done %-30s CL=%+.4f CD=%.5f xtr=%.3f" % (tag, f["CL"], f["CD"], r["xtr"]), flush=True)
        except Exception as e:
            with lock: res[tag] = dict(err=str(e)[:80]); print("FAIL %s: %s" % (tag, e), flush=True)

jobs = []
for a in ALPHAS:
    jobs.append((f"cav_aftsa_a{a}", stage(f"{F}/base_nlf_cavity", f"{F}/full_nlf0416_cavity_aftsa_m2_a{a}", a, 0.02), True))
    jobs.append((f"cav_turb_a{a}", stage(f"{F}/base_nlf_cavity", f"{F}/full_nlf0416_cavity_turb_a{a}", a, 3.0), False))
    jobs.append((f"og_aftsa_a{a}", stage(f"{F}/base_nlf_ogrid", f"{F}/full_nlf0416_ogrid_aftsa_m2_a{a}", a, 0.02), True))
print("TOTAL NLF JOBS:", len(jobs), flush=True)
ts = [threading.Thread(target=solve, args=(tag, wd, aft, i % 8)) for i, (tag, wd, aft) in enumerate(jobs)]
for t in ts: t.start()
for t in ts: t.join()
json.dump(res, open(f"{F}/run_nlf_results.json", "w"), indent=1)
print("NLF RERUN DONE:", json.dumps({k: (round(v['CD'], 5), v.get('xtr')) if 'CD' in v else 'ERR' for k, v in res.items()}))
