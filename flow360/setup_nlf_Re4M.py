"""Set up NLF(1)-0416 cases at Re = 4×10⁶, M = 0.1, χ_∞ = 8.76×10⁻⁴.
Source: existing cavprop_nlf0416_a0 / strprop_nlf0416_a0 (same mesh, just
update freestream + chi_inf + outputs)."""
import json, os, shutil

CHI = 8.76e-4
ALPHAS = [0.0, 2.5]

def walk(o, fn):
    if isinstance(o, dict):
        for k, v in list(o.items()):
            fn(o, k)
            walk(v, fn)
    elif isinstance(o, list):
        for x in o: walk(x, fn)

def set_chi(o, k):
    if k in ('modifiedTurbulentViscosityRatio', 'modified_turbulent_viscosity_ratio'):
        o[k] = CHI

def set_alpha(target):
    def fn(o, k):
        if k == 'alphaAngle':
            o[k] = target
        if k == 'angle_of_attack':
            o[k] = target
    return fn

for base in ['cavprop_nlf0416', 'strprop_nlf0416']:
    src = f'{base}_a0'
    if not os.path.exists(src):
        print(f'src missing: {src}'); continue
    for alpha in ALPHAS:
        aname = str(alpha).replace('.', 'p')
        new = f'{base}_Re4M_a{aname}'
        if os.path.exists(new): shutil.rmtree(new)
        shutil.copytree(src, new)
        # Patch Flow360.json
        cfg = json.load(open(f'{new}/Flow360.json'))
        cfg['freestream']['Mach'] = 0.1
        cfg['freestream']['muRef'] = 2.5e-8
        cfg['freestream']['alphaAngle'] = alpha
        walk(cfg, set_chi)
        walk(cfg, set_alpha(alpha))
        # Outputs
        for fo in [cfg.get('volumeOutput', {}),
                   cfg.get('sliceOutput', {}).get('slices', {}).get('centerSpan', {})]:
            fld = fo.get('outputFields', [])
            for v in ['nuHat', 'wallDistance', 'vorticityMagnitude']:
                if v not in fld: fld.append(v)
        json.dump(cfg, open(f'{new}/Flow360.json', 'w'), indent=1)
        # simulation.json
        sim = f'{new}/simulation.json'
        if os.path.exists(sim):
            s = json.load(open(sim))
            walk(s, set_chi)
            walk(s, set_alpha(alpha))
            json.dump(s, open(sim, 'w'), indent=1)
        # Clean prior outputs
        for f in os.listdir(new):
            if f.endswith(('.pvtu','.vtu','.pvd','.sock','.gltf','.log','.csv.bk')) or f.endswith('_v2.csv'):
                try: os.remove(f'{new}/{f}')
                except: pass
            if f in ('ipc_data','restartOutput'):
                shutil.rmtree(f'{new}/{f}', ignore_errors=True)
        print(f'{new}: M=0.1 Re=4e6 alpha={alpha} chi_inf={CHI}')
