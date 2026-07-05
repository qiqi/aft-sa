"""Build α-sweep cavity meshes with TEprop methodology (new mesher metric + closed contour
+ h_TE=h_0). Replaces the legacy proper_naca0012_cavity_a* and full_nlf0416_cavity_*
meshes whose tangential metric ignored the contour h_TE.

Settings: L1_TEprop-equivalent resolution (~100k cells; the convergence study showed
this is already within 1% of L2 in CL on NACA0012).

Outputs:
  cavprop_naca0012_a{-2,0,2,4,6,8}        — SA-AI configs
  cavprop_naca0012_turb_a{0,4}            — turbulent SA reference
  cavprop_nlf0416_a{-2,0,2,4,6,8}         — SA-AI configs
"""
import os, sys, json, shutil, copy, time
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/flow360")
os.environ["AI_CHI_INF"] = "0.02"
from rans.pipeline import run as pipe_run
from naca_contour_te import generate_contour_te

B = "/home/qiqi/flexcompute/aft-sa/flow360"
CFG_BASE_NACA = json.load(open(f"{B}/naca0012_re1m.json"))
NLF_DAT = "/home/qiqi/flexcompute/aft-sa/external/construct2d/nlf0416.dat"
# Look for an NLF base config if it exists; otherwise we'll derive from NACA
NLF_BASE = f"{B}/nlf0416_re1m.json"
if os.path.exists(NLF_BASE):
    CFG_BASE_NLF = json.load(open(NLF_BASE))
else:
    print(f"warning: no {NLF_BASE}; deriving NLF base config from NACA")
    CFG_BASE_NLF = copy.deepcopy(CFG_BASE_NACA)
    # The NLF cases share Re, M, mu but the airfoil name differs
    for el in CFG_BASE_NLF.get('elements', []):
        if 'name' in el: el['name'] = 'nlf0416'

# Mesh settings: L1_TEprop (~100k cells, h_TE=h_0, monotonic convergence demonstrated)
MESH = dict(N=400, hwall=0.004, h0=7.0e-6, g=1.1, hmax=2.0, far_r=100.0,
            h_te=7.0e-6, r_te=1.5)

ALPHAS = [-2, 0, 2, 4, 6, 8]

# Build the case list: (out_dir, airfoil, dat_path, base_cfg, alpha, model_tag)
# model_tag is 'aftsa' (uses AI_SA env=1, default) or 'turb' (we'll set chi_inf=3 at run time)
CASES = []
for a in ALPHAS:
    CASES.append((f"cavprop_naca0012_a{a}",       'naca0012', None,    CFG_BASE_NACA, a, 'aftsa'))
for a in ALPHAS:
    CASES.append((f"cavprop_nlf0416_a{a}",        'nlf0416',  NLF_DAT, CFG_BASE_NLF,  a, 'aftsa'))
for a in [0, 4]:
    CASES.append((f"cavprop_naca0012_turb_a{a}",  'naca0012', None,    CFG_BASE_NACA, a, 'turb'))

ncells = {}
t_total = time.time()
for out_name, af, dat_path, base_cfg, alpha, tag in CASES:
    print(f"\n=== {out_name} (af={af}, alpha={alpha}, tag={tag}) ===", flush=True)
    contour = generate_contour_te(MESH['N'], h_te=MESH['h_te'], r_te=MESH['r_te'],
                                  close_te=False, dat_path=dat_path)
    c = copy.deepcopy(base_cfg)
    if c['elements']:
        c['elements'][0]['contour'] = contour.tolist()
        c['elements'][0]['name'] = af
    c['mesh'] = dict(span=0.1, nspan=1, yplus=0.25,
                     growth=MESH['g'], hwall=MESH['hwall'],
                     h0=MESH['h0'], hmax=MESH['hmax'])
    c['flow']['alpha_deg'] = float(alpha)
    c['solver']['max_steps'] = 50000
    c['farfield'] = {'type': 'circle', 'center': [0.5, 0.0],
                     'radius': float(MESH['far_r']), 'n': 480}
    cfg_path = f"{B}/_{out_name}.json"
    json.dump(c, open(cfg_path, 'w'))
    out = f"{B}/{out_name}"
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
    ncells[out_name] = nc
    print(f"  built {out_name}: ncells2d = {nc}, t={time.time()-t:.0f}s", flush=True)

json.dump(ncells, open(f"{B}/cavprop_alphasweep_ncells.json", 'w'), indent=1)
print(f"\nTotal: {time.time()-t_total:.0f}s")
for k, v in ncells.items(): print(f"  {k}: {v}")
