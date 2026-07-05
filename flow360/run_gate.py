"""GPU gate: NACA0012 a=0 SA-AI on cavity mesh vs structured O-grid.
Direct test of grid convergence with the new m=2 log-barrier formula."""
import os, sys, json, shutil, threading, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver, extract_forces
B = "/home/qiqi/flexcompute/aft-sa/flow360"
SKIP = ('.pvtu', '.vtu', '.csv', '.log', '.sock', '.gltf', '.pvd')

def stage(wd, base, alpha=0.0, steps=10000):
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    for f in os.listdir(base):
        if any(f.endswith(x) for x in SKIP): continue
        s, d = os.path.join(base, f), os.path.join(wd, f)
        (shutil.copytree if os.path.isdir(s) else shutil.copy)(s, d)
    cf = json.load(open(f"{wd}/Flow360.json"))
    cf['timeStepping']['maxPseudoSteps'] = steps
    cf['freestream']['alphaAngle'] = float(alpha)
    json.dump(cf, open(f"{wd}/Flow360.json", "w"), indent=1)
    return wd

def cf_xy(wd):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{wd}/surface_fluid_naca0012.pvtu"); r.Update()
    g = r.GetOutput(); pts = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    nm = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    a = vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cf'))))
    cf = np.linalg.norm(a, axis=1) if a.ndim > 1 else a
    x, z = pts[:, 0], pts[:, 2]; up = z > 1e-6; xs, cfs = x[up], cf[up]
    o = np.argsort(xs); return xs[o], cfs[o]

def xtr(wd):
    xs, cfs = cf_xy(wd); nb = 50; bins = np.linspace(0, 1, nb + 1); xc = .5 * (bins[1:] + bins[:-1])
    cb = np.array([np.median(cfs[(xs >= bins[i]) & (xs < bins[i + 1])]) if ((xs >= bins[i]) & (xs < bins[i + 1])).any() else np.nan for i in range(nb)])
    ok = np.isfinite(cb); xc, cb = xc[ok], cb[ok]; d = np.diff(cb); dm = .5 * (xc[1:] + xc[:-1])
    w = (dm > 0.02) & (dm < 0.95)
    if not w.any(): return np.nan
    j = np.argmax(d[w])
    if d[w][j] < 0.1 * np.nanmean(cb[(xc > 0.02) & (xc < 0.95)]): return 0.02
    return float(dm[w][j])

res = {}
def worker(tag, base, gpu):
    try:
        wd = stage(f"{B}/gate_{tag}", base)
        env, find = make_env(); env["AI_SA"] = "1"
        run_solver(wd, find, env, gpu=gpu, timeout=2400)
        f = extract_forces(wd)
        res[tag] = dict(CL=f["CL"], CD=f["CD"], xtr=xtr(wd))
        print("done %-8s CL=%.4f CD=%.5f xtr=%.3f" % (tag, f["CL"], f["CD"], res[tag]["xtr"]), flush=True)
    except Exception as e:
        res[tag] = dict(err=str(e)[:100]); print("FAIL %s: %s" % (tag, e), flush=True)

ts = [threading.Thread(target=worker, args=a) for a in
      [("cavity", f"{B}/out_naca0012_lam", 0), ("ogrid", f"{B}/out_naca0012_ogrid", 1)]]
for t in ts: t.start()
for t in ts: t.join()
print("\nGATE RESULTS (NACA0012 a=0 SA-AI, new m=2 formula):")
for k in ("cavity", "ogrid"):
    print("  %-8s %s" % (k, res.get(k)))
