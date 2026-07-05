"""Structured (C-grid) grid-refinement at alpha=0: 3 Construct2D grids
(150x60, 250x100, 400x160) -> Flow360 cases (laminar init) -> SA-AI (m=2)."""
import sys, os, json, threading, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
os.environ.setdefault("FLOW360_SUPPRESS_BETA_WARNING", "1")
os.environ["AI_CHI_INF"] = "0.02"   # laminar interior init (case.py preprocess)
from rans.config import CaseConfig
from rans import case as _case, mesh as _mesh
from rans.env import make_env
from rans.solve import run_solver, extract_forces
B = "/home/qiqi/flexcompute/aft-sa"
GRIDS = [("coarse", f"{B}/external/construct2d/naca0012_sc.p3d"),
         ("med",    f"{B}/external/construct2d/naca0012_sm.p3d"),
         ("fine",   f"{B}/external/construct2d/naca0012_sf.p3d")]
span, nspan = 0.1, 1

def build(p3d, OUT):
    os.makedirs(OUT, exist_ok=True)
    toks = open(p3d).read().split(); ni, nj = int(toks[0]), int(toks[1])
    vals = np.array(toks[2:2+2*ni*nj], float); X = vals[:ni*nj].reshape(nj, ni).T; Y = vals[ni*nj:2*ni*nj].reshape(nj, ni).T
    Ni = ni - 1
    k = lambda i, j: i*nj + j
    P = np.empty((Ni*nj, 2))
    for i in range(Ni):
        for j in range(nj): P[k(i, j)] = (X[i, j], Y[i, j])
    N = Ni*nj
    quads = [(k(i, j), k((i+1) % Ni, j), k((i+1) % Ni, j+1), k(i, j+1)) for i in range(Ni) for j in range(nj-1)]
    wall = [(k(i, 0), k((i+1) % Ni, 0)) for i in range(Ni)]
    far = [(k(i, nj-1), k((i+1) % Ni, nj-1)) for i in range(Ni)]
    NL = nspan + 1; nid = lambda L, kk: L*N + kk + 1
    phys = [(2, 2, "naca0012"), (2, 3, "farfield"), (2, 4, "symmetry1"), (2, 5, "symmetry2"), (3, 1, "fluid")]
    elems = []; eid = [1]
    def emit(s): elems.append(f"{eid[0]} {s}"); eid[0] += 1
    msh = OUT + "/mesh.msh"
    with open(msh, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n%d\n" % len(phys))
        for d, t, n in phys: f.write('%d %d "%s"\n' % (d, t, n))
        f.write("$EndPhysicalNames\n$Nodes\n%d\n" % (NL*N))
        for L in range(NL):
            ys = -span*L/nspan
            for kk in range(N): f.write("%d %.16g %.16g %.16g\n" % (nid(L, kk), P[kk, 0], ys, P[kk, 1]))
        f.write("$EndNodes\n")
        for a, b, c, d in quads: emit("3 2 4 4 %d %d %d %d" % (nid(0, a), nid(0, b), nid(0, c), nid(0, d)))
        for a, b, c, d in quads: emit("3 2 5 5 %d %d %d %d" % (nid(nspan, a), nid(nspan, b), nid(nspan, c), nid(nspan, d)))
        for a, b in wall:
            for L in range(nspan): emit("3 2 2 2 %d %d %d %d" % (nid(L, a), nid(L, b), nid(L+1, b), nid(L+1, a)))
        for a, b in far:
            for L in range(nspan): emit("3 2 3 3 %d %d %d %d" % (nid(L, a), nid(L, b), nid(L+1, b), nid(L+1, a)))
        for L in range(nspan):
            for a, b, c, d in quads: emit("5 2 1 1 %d %d %d %d %d %d %d %d" % (nid(L, a), nid(L, b), nid(L, c), nid(L, d), nid(L+1, a), nid(L+1, b), nid(L+1, c), nid(L+1, d)))
        f.write("$Elements\n%d\n" % len(elems)); f.write("\n".join(elems) + "\n$EndElements\n")
    env, find = make_env()
    _mesh.gmsh_to_cgns(msh, OUT + "/mesh.cgns", find("flow360gmshtocgns"), env)
    cfg = CaseConfig.load(f"{B}/flow360/naca0012_re1m.json"); cfg.solver.max_steps = 10000; cfg.flow.alpha_deg = 0.0
    _case.preprocess(OUT, "mesh.cgns", find, env, cfg=cfg, wall_names=["fluid/naca0012"],
                     boundary_names=["fluid/farfield", "fluid/naca0012", "fluid/symmetry1", "fluid/symmetry2"],
                     timings={}, sdk_cache_dir=None, sim_builder=_case.build_simulation_json)
    return len(quads)  # 2D cell count

def xtr(out):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{out}/surface_fluid_naca0012.pvtu"); r.Update()
    g = r.GetOutput(); pts = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    nm = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    a = vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cf')))); v = np.linalg.norm(a, axis=1) if a.ndim > 1 else a
    x, z = pts[:, 0], pts[:, 2]; up = z > 1e-6; xs, cfs = x[up], v[up]; o = np.argsort(xs); xs, cfs = xs[o], cfs[o]
    nb = 50; bins = np.linspace(0, 1, nb + 1); xc = .5 * (bins[1:] + bins[:-1])
    cb = np.array([np.median(cfs[(xs >= bins[i]) & (xs < bins[i+1])]) if ((xs >= bins[i]) & (xs < bins[i+1])).any() else np.nan for i in range(nb)])
    ok = np.isfinite(cb); xc, cb = xc[ok], cb[ok]; d = np.diff(cb); dm = .5 * (xc[1:] + xc[:-1]); w = (dm > 0.02) & (dm < 0.95)
    if not w.any(): return np.nan
    j = np.argmax(d[w]); return 0.02 if d[w][j] < 0.1 * np.nanmean(cb[(xc > 0.02) & (xc < 0.95)]) else float(dm[w][j])

built = {}
for tag, p3d in GRIDS:
    out = f"{B}/flow360/refine_struct_{tag}"; built[tag] = (out, build(p3d, out))
    print("built struct %-7s ncell2d=%d" % (tag, built[tag][1]), flush=True)

res = {}; lock = threading.Lock()
def worker(tag, gpu):
    out, nc = built[tag]
    try:
        env, find = make_env(); env["AI_SA"] = "1"
        run_solver(out, find, env, gpu=gpu, timeout=2400)
        f = extract_forces(out)
        with lock: res[tag] = dict(CD=f["CD"], CL=f["CL"], xtr=xtr(out), ncell=nc)
        print("solved struct %-7s ncell=%d CD=%.5f xtr=%.3f" % (tag, nc, f["CD"], res[tag]["xtr"]), flush=True)
    except Exception as e:
        with lock: res[tag] = dict(err=str(e)[:90]); print("FAIL struct %s: %s" % (tag, e), flush=True)
ts = [threading.Thread(target=worker, args=(t, i + 3)) for i, (t, _) in enumerate(GRIDS)]  # GPUs 3,4,5
for t in ts: t.start()
for t in ts: t.join()
json.dump(res, open(f"{B}/flow360/refine_struct_results.json", "w"), indent=1)
print("STRUCT REFINE DONE:", json.dumps(res))
