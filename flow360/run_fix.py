"""Fix: re-run the staged sweeps (staging now copies the .dmp mesh partition) and
redo the grid-refinement at alpha=4 (the symmetric a=0 transition is too marginal
to grid-converge). NACA-sm + NLF structured sweeps; refinement a=4 both families."""
import sys, os, json, shutil, threading, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver, extract_forces
F = "/home/qiqi/flexcompute/aft-sa/flow360"
SKIP = ('.pvtu', '.vtu', '.csv', '.log', '.sock', '.gltf', '.pvd')   # NOTE: keep .dmp (mesh partition)!
ALPHAS = [-2, 0, 2, 4, 6, 8]
NCELL = {}  # level -> ncell, from prior result jsons
for fn in ('refine_cavity_results.json', 'refine_struct_results.json', 'run_extras_results.json'):
    try:
        for k, v in json.load(open(f"{F}/{fn}")).items():
            if isinstance(v, dict) and 'ncell' in v: NCELL[k] = v['ncell']
    except Exception: pass

def stage(base, wd, alpha):
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    for f in os.listdir(base):
        if any(f.endswith(x) for x in SKIP) or f in ('ipc_data', 'restartOutput'): continue
        s, d = os.path.join(base, f), os.path.join(wd, f)
        (shutil.copytree if os.path.isdir(s) else shutil.copy)(s, d)
    cf = json.load(open(f"{wd}/Flow360.json")); cf['freestream']['alphaAngle'] = float(alpha)
    json.dump(cf, open(f"{wd}/Flow360.json", "w"), indent=1); return wd

def xtr(out, wall):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{out}/surface_fluid_{wall}.pvtu"); r.Update()
    g = r.GetOutput(); pts = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    nm = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    a = vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cf')))); v = np.linalg.norm(a, axis=1) if a.ndim > 1 else a
    x, z = pts[:, 0], pts[:, 2]; up = z > 1e-6; xs, cfs = x[up], v[up]; o = np.argsort(xs); xs, cfs = xs[o], cfs[o]
    nb = 50; bins = np.linspace(0, 1, nb + 1); xc = .5*(bins[1:]+bins[:-1])
    cb = np.array([np.median(cfs[(xs >= bins[i]) & (xs < bins[i+1])]) if ((xs >= bins[i]) & (xs < bins[i+1])).any() else np.nan for i in range(nb)])
    ok = np.isfinite(cb); xc, cb = xc[ok], cb[ok]; d = np.diff(cb); dm = .5*(xc[1:]+xc[:-1]); w = (dm > 0.02) & (dm < 0.95)
    if not w.any(): return np.nan
    j = np.argmax(d[w]); return 0.02 if d[w][j] < 0.1*np.nanmean(cb[(xc > 0.02) & (xc < 0.95)]) else float(dm[w][j])

res = {}; lock = threading.Lock(); sem = threading.Semaphore(8)
def solve(tag, wd, wall, gpu, ncell=None):
    with sem:
        try:
            env, find = make_env(); env["AI_SA"] = "1"
            run_solver(wd, find, env, gpu=gpu, timeout=3600)
            f = extract_forces(wd); r = dict(CD=f["CD"], CL=f["CL"], xtr=xtr(wd, wall))
            if ncell is not None: r['ncell'] = ncell
            with lock: res[tag] = r
            print("done %-30s CD=%.5f xtr=%.3f" % (tag, f["CD"], r["xtr"]), flush=True)
        except Exception as e:
            with lock: res[tag] = dict(err=str(e)[:90]); print("FAIL %s: %s" % (tag, e), flush=True)

jobs = []
for a in ALPHAS:
    jobs.append((f"naca_str_a{a}", stage(f"{F}/base_ogrid_sm", f"{F}/full_naca0012_ogrid_aftsa_m2_a{a}", a), "naca0012", None))
    jobs.append((f"nlf_str_a{a}", stage(f"{F}/base_ogrid_nlf", f"{F}/full_nlf0416_ogrid_aftsa_m2_a{a}", a), "nlf0416", None))
# refinement at alpha=4
CAVL = [('coarse', 'refine_cavity_coarse'), ('med', 'refine_cavity_med'), ('fine', 'refine_cavity_fine'),
        ('xfine', 'refine_cavity_xfine'), ('xxfine', 'refine_cavity_xxfine')]
STRL = [('coarse', 'refine_struct_coarse'), ('med', 'refine_struct_med'), ('fine', 'refine_struct_fine'),
        ('sxfine', 'refine_struct_sxfine'), ('sxxfine', 'refine_struct_sxxfine')]
ckey = {'coarse': 'coarse', 'med': 'med', 'fine': 'fine', 'xfine': 'cav_xfine', 'xxfine': 'cav_xxfine'}
skey = {'coarse': 'coarse', 'med': 'med', 'fine': 'fine', 'sxfine': 'str_sxfine', 'sxxfine': 'str_sxxfine'}
for lvl, d in CAVL: jobs.append((f"refA4_cav_{lvl}", stage(f"{F}/{d}", f"{F}/refineA4_cavity_{lvl}", 4), "naca0012", NCELL.get(ckey[lvl])))
for lvl, d in STRL: jobs.append((f"refA4_str_{lvl}", stage(f"{F}/{d}", f"{F}/refineA4_struct_{lvl}", 4), "naca0012", NCELL.get(skey[lvl])))

print("TOTAL JOBS:", len(jobs), flush=True)
ts = [threading.Thread(target=solve, args=(tag, wd, wall, i % 8, nc)) for i, (tag, wd, wall, nc) in enumerate(jobs)]
for t in ts: t.start()
for t in ts: t.join()
json.dump(res, open(f"{F}/run_fix_results.json", "w"), indent=1)
print("FIX DONE:", json.dumps({k: (round(v['CD'], 5), v.get('xtr'), v.get('ncell')) if 'CD' in v else 'ERR' for k, v in res.items()}))
