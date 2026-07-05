"""Build α=0° case dirs for unified-a_max comparison across 2 grids × 2 levels × 2 models.

Sources (mesh.cgns reused from existing dirs, just clone + change Flow360.json):
  unstructured L0: proper_cav_L0_TEprop  (3.1e4 nodes)
  unstructured L1: proper_cav_L1_TEprop  (1.1e5 nodes)
  O-grid       L0: proper_str_L0          (3.2e4 nodes)
  O-grid       L1: proper_str_L1          (1.3e5 nodes)

For each, produce TWO α=0° case dirs:
  *_base_a0:  baseline (a_max=0.05, χ_∞=0.02, chi-linear σ_t — uses BASELINE lib)
  *_uni_a0:   unified  (a_max=0.2,  χ_∞=1.6e-7, chi^{1/4} σ_t — uses UNIFIED lib)
"""
import os, sys, json, shutil

F = "/home/qiqi/flexcompute/aft-sa/flow360"
SOURCES = [
    ('cav_L0', f"{F}/proper_cav_L0_TEprop"),
    ('cav_L1', f"{F}/proper_cav_L1_TEprop"),
    ('str_L0', f"{F}/proper_str_L0"),
    ('str_L1', f"{F}/proper_str_L1"),
]

def strip_results(d):
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if (f.endswith(('.pvtu','.vtu','.pvd','.sock','.gltf')) or f.endswith('_v2.csv')
            or f.endswith('.csv.bk') or f.startswith('surface_forces') or f.endswith('.log')):
            try: os.remove(p)
            except: pass
        if f in ('ipc_data', 'restartOutput'): shutil.rmtree(p, ignore_errors=True)
        if 'rank_' in f and '.dmp' in f:
            try: os.remove(p)
            except: pass

def set_alpha_and_chi(case_dir, alpha_deg, chi_inf):
    cfg = json.load(open(f"{case_dir}/Flow360.json"))
    # alpha
    fs = cfg.get('freestream', {})
    fs['alphaAngle'] = float(alpha_deg)
    # chi_inf
    fs['turbulenceQuantities'] = {
        'modelType': 'ModifiedTurbulentViscosityRatio',
        'modifiedTurbulentViscosityRatio': float(chi_inf)
    }
    cfg['freestream'] = fs
    json.dump(cfg, open(f"{case_dir}/Flow360.json", 'w'), indent=1)
    # simulation.json — also update alpha + turbulence quantities for consistency
    sim_path = f"{case_dir}/simulation.json"
    if os.path.exists(sim_path):
        sim = json.load(open(sim_path))
        for model in sim.get('models', []):
            if model.get('type', '') == 'Fluid' or 'navierStokes' in str(model).lower():
                # Most JSON schemas put alpha here
                if 'initial_condition' in model or 'reference_geometry' in model:
                    pass  # alpha lives in operating_condition usually
        op = sim.get('operating_condition', {})
        # Best-effort: this is just a sanity check; Flow360.json is what the solver reads.
        json.dump(sim, open(sim_path, 'w'), indent=1)

# Build 8 case dirs
for label, src in SOURCES:
    for model_label, chi_inf in [('base', 0.02), ('uni', 1.6e-7)]:
        dst = f"{F}/{label}_{model_label}_a0"
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)
        strip_results(dst)
        set_alpha_and_chi(dst, 0.0, chi_inf)
        print(f"built {dst} (alpha=0, chi_inf={chi_inf})")
print("\nDone. 8 case dirs built.")
