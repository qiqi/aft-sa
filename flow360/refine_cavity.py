"""Cavity-mesh grid-refinement at alpha=0: coarse/med/fine via mesh.hwall/yplus/hmax,
built through rans.pipeline (laminar init baked by AI_CHI_INF), solved SA-AI (m=2)."""
import os, sys, json, threading, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
os.environ["AI_CHI_INF"] = "0.02"   # case.py preprocess -> laminar interior init
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.pipeline import run as pipe_run
from rans.env import make_env
from rans.solve import run_solver, extract_forces
B = "/home/qiqi/flexcompute/aft-sa/flow360"
CFG = json.load(open(f"{B}/naca0012_re1m.json"))
# (tag, hwall, yplus, hmax)  -- ~2x refinement in surface spacing + first cell
MESHES = [("coarse", 0.008, 2.0, 4.0), ("med", 0.004, 1.0, 3.0), ("fine", 0.002, 0.5, 2.0)]

def ncells_2d(out):
    txt = open(f"{out}/mesh2d.vtk").read().split("\n")
    for ln in txt:
        if ln.startswith("CELLS"): return int(ln.split()[1])
    return -1

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

# 1) build all (sequential; CPU + SDK imports not thread-safe)
built = {}
for tag, hwall, yplus, hmax in MESHES:
    c = json.loads(json.dumps(CFG)); c['mesh'].update(hwall=hwall, yplus=yplus, hmax=hmax)
    c['flow']['alpha_deg'] = 0.0; c['solver']['max_steps'] = 10000
    cfgp = f"{B}/refcav_{tag}.json"; json.dump(c, open(cfgp, "w"))
    out = f"{B}/refine_cavity_{tag}"
    pipe_run(cfgp, out, solve=False); built[tag] = (out, ncells_2d(out))
    print("built cavity %-7s ncell2d=%d" % (tag, built[tag][1]), flush=True)

# 2) solve all in parallel
res = {}; lock = threading.Lock()
def worker(tag, gpu):
    out, nc = built[tag]
    try:
        env, find = make_env(); env["AI_SA"] = "1"
        run_solver(out, find, env, gpu=gpu, timeout=2400)
        f = extract_forces(out)
        with lock: res[tag] = dict(CD=f["CD"], CL=f["CL"], xtr=xtr(out), ncell=nc)
        print("solved cavity %-7s ncell=%d CD=%.5f xtr=%.3f" % (tag, nc, f["CD"], res[tag]["xtr"]), flush=True)
    except Exception as e:
        with lock: res[tag] = dict(err=str(e)[:90]); print("FAIL cavity %s: %s" % (tag, e), flush=True)
ts = [threading.Thread(target=worker, args=(t, i)) for i, (t, *_) in enumerate(MESHES)]
for t in ts: t.start()
for t in ts: t.join()
json.dump(res, open(f"{B}/refine_cavity_results.json", "w"), indent=1)
print("CAVITY REFINE DONE:", json.dumps(res))
