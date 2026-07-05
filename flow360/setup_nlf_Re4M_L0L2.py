"""Set up NLF(1)-0416 L0 (coarse) and L2 (fine cav-only) cases at
Re = 4×10⁶, M = 0.1, chi_inf = 8.76×10⁻⁶ (compensated for AI_LAMINAR_SLOWDOWN=0.01).

For each case: copy source dir, patch Flow360.json with new freestream,
add residualTurbulence + residualNavierStokes outputs, clear stale outputs.

L0 cav: full_nlf0416_cavity_aftsa_m2_aX → cavL0_nlf0416_Re4M_aX
L0 str: full_nlf0416_ogrid_aftsa_m2_aX  → strL0_nlf0416_Re4M_aX
L2 cav: proper_nlf0416_cavity_aX        → cavL2_nlf0416_Re4M_aX
"""
import json, os, shutil

CHI_INPUT = 8.76e-6   # compensated for fSlow=0.01 -> actual chi_inf=8.76e-4 (N_crit=9)
ALPHAS = [0.0, 4.0]
RES_FIELDS = ['residualTurbulence','residualNavierStokes','nuHat','wallDistance','vorticityMagnitude']

def walk(o, fn):
    if isinstance(o, dict):
        for k, v in list(o.items()):
            fn(o, k)
            walk(v, fn)
    elif isinstance(o, list):
        for x in o: walk(x, fn)

def patch_case(src, new, alpha):
    if os.path.exists(new):
        shutil.rmtree(new)
    # copy excluding sockets / stale outputs
    os.makedirs(new)
    for f in os.listdir(src):
        sp = os.path.join(src, f)
        if f == 'ipc_control.sock' or f == 'ipc_data':
            continue
        if f.endswith(('.pvtu','.pvd','.gltf','.log','.sock','_v2.csv','_proc0.vtu')):
            continue
        if os.path.isdir(sp):
            continue  # skip restartOutput etc — start fresh
        shutil.copy2(sp, new)

    pf = f'{new}/Flow360.json'
    if not os.path.exists(pf): return False
    d = json.load(open(pf))
    d['freestream']['Mach'] = 0.1
    d['freestream']['muRef'] = 2.5e-8
    d['freestream']['alphaAngle'] = alpha
    # Patch chi_inf to compensated value
    def set_chi(o, k):
        if k in ('modifiedTurbulentViscosityRatio',):
            o[k] = CHI_INPUT
    walk(d, set_chi)
    # Add residual fields to volume + per-slice outputs
    vol_fields = d['volumeOutput']['outputFields']
    for f in RES_FIELDS:
        if f not in vol_fields: vol_fields.append(f)
    for slice_name, slice_cfg in d.get('sliceOutput', {}).get('slices', {}).items():
        sfields = slice_cfg['outputFields']
        for f in RES_FIELDS:
            if f not in sfields: sfields.append(f)
    # surface output: leave as is
    # disable restart
    d['runControl']['restart'] = False
    # Ensure compressible NS model
    json.dump(d, open(pf,'w'), indent=1)

    # Also patch simulation.json mildly (alpha, chi_inf) for completeness
    ps = f'{new}/simulation.json'
    if os.path.exists(ps):
        s = json.load(open(ps))
        def set_chi_alpha(o, k):
            if k in ('modifiedTurbulentViscosityRatio','modified_turbulent_viscosity_ratio'):
                o[k] = CHI_INPUT
            if k in ('alphaAngle','angle_of_attack'):
                v = o[k]
                if isinstance(v, dict) and 'value' in v:
                    v['value'] = alpha
                else:
                    o[k] = alpha
        walk(s, set_chi_alpha)
        json.dump(s, open(ps,'w'), indent=1)
    return True

mappings = [
    ('full_nlf0416_cavity_aftsa_m2_a{}', 'cavL0_nlf0416_Re4M_a{}p0'),  # L0 cav
    ('full_nlf0416_ogrid_aftsa_m2_a{}',  'strL0_nlf0416_Re4M_a{}p0'),  # L0 str
    ('proper_nlf0416_cavity_a{}',        'cavL2_nlf0416_Re4M_a{}p0'),  # L2 cav
]

count = 0
for src_tmpl, new_tmpl in mappings:
    for alpha in ALPHAS:
        ai = int(alpha) if alpha == int(alpha) else alpha
        src = src_tmpl.format(ai)
        new = new_tmpl.format(ai)
        if not os.path.exists(src):
            print(f'  src missing: {src}'); continue
        ok = patch_case(src, new, alpha)
        if ok:
            count += 1
            print(f'  set up: {new}  (source: {src}, M=0.1, Re=4M, alpha={alpha}, chi_inf_input={CHI_INPUT})')
        else:
            print(f'  FAIL: {new}')

print(f'\nSet up {count} cases')
