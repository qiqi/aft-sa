"""Build cavity meshes with PROPER refinement of EVERY discretization length scale:
  - bulk surface point count N: doubles per level
  - first-cell height h_0, hwall, hmax: halve per level
  - growth-ratio offset (g-1): halves per level
  - TE contour spacing h_TE = h_0 (halves per level)
  - TE ramp ratio offset (r_TE - 1): halves per level (so cell size at fixed
    chord-distance D from TE refines by 2 per level: dx(D) ≈ D·(r_TE - 1))

Output: proper_cav_L{0,1,2,3}_TEprop directories. MESHES ONLY (no Flow360 setup
beyond what pipe_run gives by default), so we can visualize before running.
"""
import os, sys, json, shutil, copy, time
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/flow360")
os.environ["AI_CHI_INF"] = "0.02"
from rans.pipeline import run as pipe_run
from naca_contour_te import generate_contour_te

B = "/home/qiqi/flexcompute/aft-sa/flow360"
CFG_BASE = json.load(open(f"{B}/naca0012_re1m.json"))

# Per-level discretization length scales — every scale refines by factor 2 per level
LEVELS = [
    # tag    N    hwall    h0       g      hmax  far_r h_te    r_te
    ('L0',  200, 0.008,    14.0e-6, 1.2000, 3.0,  100.0, 14.0e-6,  2.000),
    ('L1',  400, 0.004,     7.0e-6, 1.1000, 2.0,  100.0,  7.0e-6,  1.500),
    ('L2',  800, 0.002,     3.5e-6, 1.0500, 1.5,  100.0,  3.5e-6,  1.250),
    ('L3', 1600, 0.001,    1.75e-6, 1.0250, 1.0,  100.0,  1.75e-6, 1.125),
]

ncells = {}
t0 = time.time()
for tag, N, hwall, h0, g, hmax, far_r, h_te, r_te in LEVELS:
    print(f"\n=== {tag}: N={N}, h0={h0:.2e}, g={g}, h_te={h_te:.2e}, r_te={r_te} "
          f"=> cell at D=0.001c is {0.001*(r_te-1)*1e6:.1f} micron ===", flush=True)
    contour = generate_contour_te(N, h_te=float(h_te), r_te=float(r_te), close_te=False)
    c = copy.deepcopy(CFG_BASE)
    c['elements'][0]['contour'] = contour.tolist()
    c['mesh'] = dict(span=0.1, nspan=1, yplus=0.25,
                     growth=float(g), hwall=float(hwall),
                     h0=float(h0), hmax=float(hmax))
    c['flow']['alpha_deg'] = 4.0
    c['solver']['max_steps'] = 50000
    c['farfield'] = {'type': 'circle', 'center': [0.5, 0.0],
                     'radius': float(far_r), 'n': 480}
    cfg_path = f"{B}/_teprop_{tag}.json"
    json.dump(c, open(cfg_path, 'w'))
    out = f"{B}/proper_cav_{tag}_TEprop"
    shutil.rmtree(out, ignore_errors=True)
    t = time.time()
    try:
        pipe_run(cfg_path, out, solve=False)
    except Exception as e:
        print(f"  FAIL: {e}", flush=True); continue
    nc = 0
    try:
        with open(f"{out}/mesh2d.vtk") as f:
            for ln in f:
                if ln.startswith("CELLS"): nc = int(ln.split()[1]); break
    except: pass
    ncells[tag] = nc
    # also report contour stats
    import numpy as np
    cx, cz = contour[:,0], contour[:,1]
    upper_x = cx[:len(cx)//2 + 1]; idx = np.argsort(upper_x); xu = upper_x[idx]
    dx_te = abs(xu[-1] - xu[-2])
    n_in_te_band = (xu > 0.99).sum()
    print(f"  built: ncells2d = {nc}, contour pts near TE (x>0.99): {n_in_te_band}, "
          f"dx_TE_actual = {dx_te:.2e}, t={time.time()-t:.0f}s")
json.dump(ncells, open(f"{B}/te_proper_ncells.json",'w'), indent=1)
print(f"\nTotal mesh-build time: {time.time()-t0:.0f}s")
print(f"Cell counts: {ncells}")
