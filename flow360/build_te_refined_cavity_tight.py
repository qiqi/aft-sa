"""Build cavity meshes with truly isotropic TE: h_TE = h_0 exactly.
Uses a smoother geometric ramp (r_TE = 1.10) and a longer transition zone so the
mesh2d CDT can absorb the rapid spacing change.

Outputs to proper_cav_L{0,1,2,3}_TEtight directories.
"""
import os, sys, json, shutil, copy, time
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/flow360")
os.environ["AI_CHI_INF"] = "0.02"
from rans.pipeline import run as pipe_run
from naca_contour_te import generate_contour_te

B = "/home/qiqi/flexcompute/aft-sa/flow360"
CFG_BASE = json.load(open(f"{B}/naca0012_re1m.json"))

# Same parameters as the L0-L3 series but with h_te = h0 (truly isotropic at TE)
LEVELS = [
    # (tag,    N,    hwall,    h0,       g,       hmax, far_r, h_te)
    ('L0',   200, 0.008,    14.0e-6, 1.2000,  3.0,   100.0, 14.0e-6),
    ('L1',   400, 0.004,     7.0e-6, 1.1000,  2.0,   100.0,  7.0e-6),
    ('L2',   800, 0.002,     3.5e-6, 1.0500,  1.5,   100.0,  3.5e-6),
    ('L3',  1600, 0.001,    1.75e-6, 1.0250,  1.0,   100.0,  1.75e-6),
]

ncells_record = {}
t_total = time.time()
for tag, N, hwall, h0, g, hmax, far_r, h_te in LEVELS:
    print(f"\n=== building TEtight cavity {tag}: N={N}, h0={h0:.3e}, h_te={h_te:.3e} (ratio {h_te/h0:.1f}) ===", flush=True)
    # use smoother growth ratio for the ramp so the transition is longer
    contour = generate_contour_te(N, h_te=float(h_te), r_te=1.10, close_te=False)
    c = copy.deepcopy(CFG_BASE)
    c['elements'][0]['contour'] = contour.tolist()
    c['mesh'] = dict(span=0.1, nspan=1, yplus=0.25,
                     growth=float(g), hwall=float(hwall),
                     h0=float(h0), hmax=float(hmax))
    c['flow']['alpha_deg'] = 4.0
    c['solver']['max_steps'] = 50000
    c['farfield'] = {'type': 'circle', 'center': [0.5, 0.0], 'radius': float(far_r), 'n': 480}
    cfg_path = f"{B}/_tetight_cavity_{tag}.json"
    json.dump(c, open(cfg_path, 'w'))
    out = f"{B}/proper_cav_{tag}_TEtight"
    shutil.rmtree(out, ignore_errors=True)
    t = time.time()
    try:
        pipe_run(cfg_path, out, solve=False)
    except Exception as e:
        print(f"  FAIL {tag}: {e}", flush=True); continue
    nc = 0
    try:
        with open(f"{out}/mesh2d.vtk") as f:
            for ln in f:
                if ln.startswith("CELLS"): nc = int(ln.split()[1]); break
    except: pass
    ncells_record[tag] = nc
    print(f"  built {tag}: ncells2d = {nc}, t={time.time()-t:.0f}s", flush=True)

json.dump(ncells_record, open(f"{B}/te_tight_cavity_ncells.json", 'w'), indent=1)
print(f"\nTotal time: {time.time()-t_total:.0f}s")
print(f"Cell counts: {ncells_record}")
