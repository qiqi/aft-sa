"""Rebuild cavity meshes with explicit h0 so the LE y+ actually matches target.

Root cause: rans.config.wall_h0() derives first-cell height from the
turbulent-flat-plate Cf 0.026/Re^(1/7); the laminar LE Cf peak on NACA0012 is
~2.7x this, so u_tau at the LE is ~1.6x what the formula expects and y+_LE
overshoots the nominal yplus by the same factor. Setting mesh.h0 explicitly
bypasses the formula.

Production cavity:  h0 = 3.5e-6  -> y+_LE ~ 0.25 (matches O-grid resolution)
Refinement series at alpha=4 with isotropic 2x scaling:
  coarse  h0 = 1.4e-5,  hwall = 0.004    (y+_LE ~ 1.0)
  med     h0 = 7.0e-6,  hwall = 0.002    (y+_LE ~ 0.5)
  fine    h0 = 3.5e-6,  hwall = 0.001    (y+_LE ~ 0.25)
  xfine   h0 = 1.75e-6, hwall = 0.0005   (y+_LE ~ 0.125)
  xxfine  h0 = 8.75e-7, hwall = 0.00025  (y+_LE ~ 0.0625)
"""
import os, sys, json, copy, shutil
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
os.environ["AI_CHI_INF"] = "0.02"
from rans.pipeline import run as pipe_run

B = "/home/qiqi/flexcompute/aft-sa/flow360"
CFG_NACA = json.load(open(f"{B}/naca0012_re1m.json"))
CFG_NLF  = json.load(open(f"{B}/nlf0416_re1m.json"))

# Production cavity: y+_LE ~ 0.25 -- match O-grid baseline resolution.
PROD_MESH = dict(hwall=0.001, h0=3.5e-6, hmax=1.5, growth=1.2, span=0.1, nspan=1, yplus=0.25)

# Cavity refinement at alpha=4, isotropic 2x per step
REFINE = [
    ("coarse",  dict(hwall=0.004,   h0=1.40e-5,  hmax=3.0, growth=1.2)),
    ("med",     dict(hwall=0.002,   h0=7.00e-6,  hmax=2.0, growth=1.2)),
    ("fine",    dict(hwall=0.001,   h0=3.50e-6,  hmax=1.5, growth=1.2)),
    ("xfine",   dict(hwall=0.0005,  h0=1.75e-6,  hmax=1.0, growth=1.2)),
    ("xxfine",  dict(hwall=0.00025, h0=8.75e-7,  hmax=0.75, growth=1.2)),
]

def build_one(cfgname, cfg_template, mesh_override, alpha_deg, outdir):
    c = copy.deepcopy(cfg_template)
    c['mesh'].update(mesh_override)
    c['mesh'].setdefault('span', 0.1); c['mesh'].setdefault('nspan', 1)
    c['mesh'].setdefault('yplus', 0.25)
    c['flow']['alpha_deg'] = float(alpha_deg)
    c['solver']['max_steps'] = 25000
    cfgp = f"{B}/_rebuild_{cfgname}.json"
    json.dump(c, open(cfgp, 'w'))
    # rebuild from scratch
    shutil.rmtree(outdir, ignore_errors=True)
    pipe_run(cfgp, outdir, solve=False)
    # mesh2d.vtk cell count
    nc = 0
    with open(f"{outdir}/mesh2d.vtk") as f:
        for ln in f:
            if ln.startswith("CELLS"): nc = int(ln.split()[1]); break
    return nc

JOBS = []
# 1) production: NACA + NLF, 6 alphas each
for af, cfg in [('naca0012', CFG_NACA), ('nlf0416', CFG_NLF)]:
    for a in [-2, 0, 2, 4, 6, 8]:
        out = f"{B}/proper_{af}_cavity_a{a}"
        JOBS.append((f"prod_{af}_a{a}", cfg, PROD_MESH, a, out))
# 2) refinement at alpha=4
for tag, m in REFINE:
    out = f"{B}/proper_refineA4_cavity_{tag}"
    JOBS.append((f"refA4_{tag}", CFG_NACA, m, 4, out))

print(f"Building {len(JOBS)} new cavity meshes (production={len(JOBS)-len(REFINE)}, refinement={len(REFINE)})", flush=True)
import time
ncells = {}
t0 = time.time()
for name, cfg, m, a, out in JOBS:
    t = time.time()
    try:
        nc = build_one(name, cfg, m, a, out)
        ncells[name] = nc
        print(f"  built {name:30s} ncells2d={nc:7d}  t={time.time()-t:.1f}s  cum={time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"  FAIL  {name}: {e}", flush=True)
json.dump(ncells, open(f"{B}/proper_cavity_ncells.json", 'w'), indent=1)
print(f"BUILD DONE  total t={time.time()-t0:.0f}s")
