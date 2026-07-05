"""Set up 12 NLF cav (unstructured) cases at α∈{0,4,9,15} from the L0/L1/L2 meshes.
Mirrors what build_nlf_struct_cases.py does for str, but the meshes were already
built by build_nlf_proper_cavity.py — we just clone the case dir for each alpha
and patch Flow360.json. Output: cavL{0,1,2}prop_nlf0416_Re4M_a{0,4,9,15}.
"""
import os, sys, json, shutil

OUT_BASE = "/home/qiqi/flexcompute/aft-sa/flow360"
LEVELS = ['L0', 'L1', 'L2']
ALPHAS = [0.0, 4.0, 9.0, 15.0]
RES_FIELDS = ['residualTurbulence', 'residualNavierStokes', 'nuHat', 'wallDistance', 'vorticityMagnitude']
CHI_INPUT = 8.76e-6

def patch_flow360(p, alpha):
    d = json.load(open(p))
    d['freestream']['Mach'] = 0.1
    d['freestream']['muRef'] = 2.5e-8
    d['freestream']['alphaAngle'] = alpha
    tq = d['freestream'].setdefault('turbulenceQuantities', {})
    tq['modelType'] = 'ModifiedTurbulentViscosityRatio'
    tq['modifiedTurbulentViscosityRatio'] = CHI_INPUT
    for bn, bc in d.get('boundaries', {}).items():
        if 'farfield' in bn or bc.get('type') == 'Freestream':
            btq = bc.setdefault('turbulenceQuantities', {})
            btq['modelType'] = 'ModifiedTurbulentViscosityRatio'
            btq['modifiedTurbulentViscosityRatio'] = CHI_INPUT
    vol = d.setdefault('volumeOutput', {}).setdefault('outputFields', [])
    for f in RES_FIELDS:
        if f not in vol: vol.append(f)
    for sn, sc in d.get('sliceOutput', {}).get('slices', {}).items():
        sf = sc.setdefault('outputFields', [])
        for f in RES_FIELDS:
            if f not in sf: sf.append(f)
    d.setdefault('fluidProperties', {})['sutherlandConstantDim'] = 110.4
    d['runControl']['restart'] = False
    d['timeStepping']['maxPseudoSteps'] = 80000
    d['timeStepping']['absoluteTolerance'] = 1e-30
    d.setdefault('turbulenceModelSolver', {})['absoluteTolerance'] = 1e-30
    json.dump(d, open(p, 'w'), indent=1)

for tag in LEVELS:
    src = f"{OUT_BASE}/nlfprop_cav_{tag}"
    if not os.path.exists(src):
        print(f"SKIP {tag}: source not found"); continue
    for alpha in ALPHAS:
        dst = f"{OUT_BASE}/cavL{tag[1:]}prop_nlf0416_Re4M_a{int(alpha)}"
        if os.path.exists(dst): shutil.rmtree(dst)
        os.makedirs(dst)
        for f in os.listdir(src):
            if f.endswith(('.pvtu','.pvd','.gltf','.log','.sock')) or f.startswith('ipc'):
                continue
            sp = os.path.join(src, f)
            if os.path.isfile(sp):
                shutil.copy2(sp, dst)
        patch_flow360(f"{dst}/Flow360.json", alpha)
        print(f"built {os.path.basename(dst)}: α={alpha}, chi_in={CHI_INPUT}")
