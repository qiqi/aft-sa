"""Build the PROPER NLF(1)-0416 cavity refinement ladder (L0..L2 for now).
Mirrors flow360/build_proper_cavity.py for NACA, but:
  - reads the smooth spline contour from nlf_contour_te.generate_contour_te
  - blunt-TE h_te starts at 1/2·z_TE_thickness = 0.00125 c at L0, halves per level
  - r_te varies per level (~2.0 at L0, ~1.5 at L1, ~1.25 at L2 etc.)
  - fixed 100 c far-field at every level (no domain change)
"""
import os, sys, json, shutil, copy, time
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/flow360")
os.environ["AI_CHI_INF"] = "0.02"
from rans.pipeline import run as pipe_run
from nlf_contour_te import generate_contour_te

B = "/home/qiqi/flexcompute/aft-sa/flow360"

# Use the existing NACA Re=1M config as a flow template, then patch alpha + airfoil name
CFG_BASE = json.load(open(f"{B}/naca0012_re1m.json"))

# NLF Re=4M operating condition will be set in the per-case Flow360 setup later;
# here we just build the MESHES.  The cfg flow.{mach,reynolds} fed to pipe_run
# only affects yplus-derived spacing.  Set to NLF conditions (Re=4M, M=0.1).
CFG_BASE['flow']['mach'] = 0.1
CFG_BASE['flow']['reynolds'] = 4.0e6
CFG_BASE['elements'][0]['name'] = 'nlf0416'

# Proper ladder for blunt-TE NLF(1)-0416 (z_TE_thickness = 0.0025 c):
#   h_te_L0 = 0.5 · 0.0025 = 0.00125, halves per level
#   r_te = 2.0 at L0, (r-1) halves per level, mirroring the bulk growth column
#   All other length scales (h_wall, h0, growth, h_max) halve as in build_proper_cavity.py
#   farfield_radius FIXED at 100 c.
LEVELS = [
    # (tag,   N,    hwall,    h0,         g,      hmax, far_r, h_te,      r_te)
    ('L0',  200, 0.008,    14.0e-6,  1.2000, 3.0,  100.0,  1.250e-3,  2.000),
    ('L1',  400, 0.004,     7.0e-6,  1.1000, 2.0,  100.0,  6.250e-4,  1.500),
    ('L2',  800, 0.002,     3.5e-6,  1.0500, 1.5,  100.0,  3.125e-4,  1.250),
]

ncells_record = {}
t_total = time.time()
for tag, N, hwall, h0, g, hmax, far_r, h_te, r_te in LEVELS:
    print(f"\n=== building NLF cav {tag}: N={N}, h0={h0:.3e}, h_te={h_te:.3e}, r_te={r_te}, far_r={far_r} ===", flush=True)
    contour = generate_contour_te(N, h_te=h_te, r_te=r_te)
    c = copy.deepcopy(CFG_BASE)
    c['elements'][0]['contour'] = contour.tolist()
    c['mesh'] = dict(span=0.1, nspan=1, yplus=0.25,
                     growth=float(g), hwall=float(hwall),
                     h0=float(h0), hmax=float(hmax))
    c['flow']['alpha_deg'] = 0.0     # geometry only; alpha doesn't affect the mesh
    c['solver']['max_steps'] = 1     # no solve in this script
    c['farfield'] = {'type': 'circle', 'center': [0.5, 0.0],
                     'radius': float(far_r), 'n': 480}
    cfg_path = f"{B}/_nlf_cavity_proper_{tag}.json"
    json.dump(c, open(cfg_path, 'w'))
    out = f"{B}/nlfprop_cav_{tag}"
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
    print(f"  built {tag}: ncells2d = {nc}, t={time.time()-t:.0f}s, cum={time.time()-t_total:.0f}s", flush=True)

json.dump(ncells_record, open(f"{B}/nlf_cavity_proper_ncells.json", 'w'), indent=1)
print(f"\nTotal time: {time.time()-t_total:.0f}s")
print(f"NLF cav cell counts: {ncells_record}")
