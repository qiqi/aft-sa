"""Build the PROPER Eppler 387 cavity refinement ladder (L0..L2).

Mirrors build_nlf_proper_cavity.py.  Key differences from NLF:
  - reads the smooth spline contour from eppler_contour_te.generate_contour_te
  - blunt-TE h_te starts at 1/2 · z_TE_thickness = 0.000833 c at L0
    (E387 z_TE_thickness = 0.001667 c per the NASA TM 4062 modified TE)
  - design Re is the experimental sweet spot: 200k (LSB regime in the paper)
  - fixed 100 c far-field at every level (per the airfoil-mesh-refinement
    practices memory)
"""
import os, sys, json, shutil, copy, time
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/flow360")
os.environ["AFT_CHI_INF"] = "0.02"
from rans.pipeline import run as pipe_run
from eppler_contour_te import generate_contour_te

B = "/home/qiqi/flexcompute/aft-sa/flow360"

# Flow template — patch alpha + airfoil name + Re later.
CFG_BASE = json.load(open(f"{B}/naca0012_re1m.json"))
CFG_BASE['flow']['mach'] = 0.1
CFG_BASE['flow']['reynolds'] = 2.0e5      # E387 test sweet spot (paper Re=200k)
CFG_BASE['elements'][0]['name'] = 'eppler387'

# Proper ladder for blunt-TE Eppler 387 (z_TE_thickness = 0.001667 c):
#   h_te_L0 = 0.5 · 0.001667 = 0.000833, halves per level
#   r_te = 2.0 at L0, (r-1) halves per level, mirroring bulk growth
#   Other length scales match the NLF/NACA ladder; far_r FIXED at 100 c.
LEVELS = [
    # (tag,   N,    hwall,    h0,         g,      hmax, far_r, h_te,      r_te)
    ('L0',  200, 0.008,    14.0e-6,  1.2000, 3.0,  100.0,  8.33e-4,  2.000),
    ('L1',  400, 0.004,     7.0e-6,  1.1000, 2.0,  100.0,  4.17e-4,  1.500),
    ('L2',  800, 0.002,     3.5e-6,  1.0500, 1.5,  100.0,  2.08e-4,  1.250),
]

ncells_record = {}
t_total = time.time()
for tag, N, hwall, h0, g, hmax, far_r, h_te, r_te in LEVELS:
    print(f"\n=== building E387 cav {tag}: N={N}, h0={h0:.3e}, h_te={h_te:.3e}, "
          f"r_te={r_te}, far_r={far_r} ===", flush=True)
    contour = generate_contour_te(N, h_te=h_te, r_te=r_te)
    c = copy.deepcopy(CFG_BASE)
    c['elements'][0]['contour'] = contour.tolist()
    c['mesh'] = dict(span=0.1, nspan=1, yplus=0.25,
                     growth=float(g), hwall=float(hwall),
                     h0=float(h0), hmax=float(hmax))
    c['flow']['alpha_deg'] = 0.0
    c['solver']['max_steps'] = 1
    c['farfield'] = {'type': 'circle', 'center': [0.5, 0.0],
                     'radius': float(far_r), 'n': 480}
    cfg_path = f"{B}/_eppler_cavity_proper_{tag}.json"
    json.dump(c, open(cfg_path, 'w'))
    out = f"{B}/epprop_cav_{tag}"
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
    print(f"  built {tag}: ncells2d = {nc}, t={time.time()-t:.0f}s, "
          f"cum={time.time()-t_total:.0f}s", flush=True)

json.dump(ncells_record, open(f"{B}/eppler_cavity_proper_ncells.json", 'w'), indent=1)
print(f"\nTotal time: {time.time()-t_total:.0f}s")
print(f"E387 cav cell counts: {ncells_record}")
