"""Build the PROPER cavity refinement ladder (L0..L4): all length scales refine,
including growth ratio approaching 1, and the contour itself refines.
"""
import os, sys, json, shutil, copy
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/flow360")
os.environ["AI_CHI_INF"] = "0.02"
from rans.pipeline import run as pipe_run
from naca_contour import generate_contour

B = "/home/qiqi/flexcompute/aft-sa/flow360"
CFG_BASE = json.load(open(f"{B}/naca0012_re1m.json"))

# Refinement ladder
# r=2 per level: every length scale halves; (g-1) halves; contour N doubles
LEVELS = [
    # (tag,    N,    hwall,    h0,       g,       hmax, farfield_radius)
    ('L0',   200, 0.008,    14.0e-6, 1.2000,  3.0,    100.0),
    ('L1',   400, 0.004,     7.0e-6, 1.1000,  2.0,    100.0),
    ('L2',   800, 0.002,     3.5e-6, 1.0500,  1.5,    100.0),
    ('L3',  1600, 0.001,    1.75e-6, 1.0250,  1.0,    100.0),
    ('L4',  3200, 0.0005, 0.875e-6, 1.0125,  0.75,   100.0),
]

import time, json
ncells_record = {}
t_total = time.time()
for tag, N, hwall, h0, g, hmax, far_r in LEVELS:
    print(f"\n=== building cavity {tag}: N={N}, hwall={hwall}, h0={h0:.3e}, g={g}, hmax={hmax}, R={far_r} ===", flush=True)
    # Generate refined contour
    contour = generate_contour(N)
    # Build config
    c = copy.deepcopy(CFG_BASE)
    c['elements'][0]['contour'] = contour.tolist()
    c['mesh'] = dict(
        span=0.1, nspan=1, yplus=0.25,  # yplus is overridden by explicit h0
        growth=float(g),
        hwall=float(hwall),
        h0=float(h0),
        hmax=float(hmax),
    )
    c['flow']['alpha_deg'] = 4.0
    c['solver']['max_steps'] = 25000
    c['farfield'] = {'type': 'circle', 'center': [0.5, 0.0], 'radius': float(far_r), 'n': 480}
    cfg_path = f"{B}/_proper_cavity_{tag}.json"
    json.dump(c, open(cfg_path, 'w'))
    out = f"{B}/proper_cav_{tag}"
    shutil.rmtree(out, ignore_errors=True)
    t = time.time()
    try:
        pipe_run(cfg_path, out, solve=False)
    except Exception as e:
        print(f"  FAIL {tag}: {e}", flush=True); continue
    # cell count
    nc = 0
    try:
        with open(f"{out}/mesh2d.vtk") as f:
            for ln in f:
                if ln.startswith("CELLS"): nc = int(ln.split()[1]); break
    except: pass
    ncells_record[tag] = nc
    print(f"  built {tag}: ncells2d = {nc}, t={time.time()-t:.0f}s, cum={time.time()-t_total:.0f}s", flush=True)

json.dump(ncells_record, open(f"{B}/proper_cavity_ncells.json", 'w'), indent=1)
print(f"\nTotal time: {time.time()-t_total:.0f}s")
print(f"Cell counts: {ncells_record}")
