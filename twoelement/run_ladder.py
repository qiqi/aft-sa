"""Mesh + solve the adopted two-element article on the L0/L1/L2 ladder.

Same refinement structure as the paper's airfoil campaign (three levels,
halving the wall spacing and tightening the growth ratio, far field fixed at
100 c), at Re = 1e6, M = 0.1, alpha = -1 deg, SA-AI with the N_crit = 9 seed.

Seed: chi_inf = c_v1 exp(-9) = 7.1 * e^-9 = 8.76e-4, set through
AFT_CHI_INF, which rans/case.py maps onto the Flow360
freestream.turbulenceQuantities.modifiedTurbulentViscosityRatio -- the same
field the Daedalus campaign case carries (8.76e-6 there, its quieter seed).

Environment this machine needs (019): the compute binaries were built against
an openmpi at a path that does not exist here.

    export LD_LIBRARY_PATH=/local_data_2/shared_data/third_party/openmpi-4.1.4/lib:\
/home/qiqi/flexcompute/compute/install/release/lib:$LD_LIBRARY_PATH
    export OPAL_PREFIX=/local_data_2/shared_data/third_party/openmpi-4.1.4

Run:  python3 run_ladder.py [L0 L1 L2]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/home/qiqi/flexcompute/flexfoil/rans')

RE, MACH, ALPHA = 1.0e6, 0.10, -1.0
CHI_INF = 7.1*np.exp(-9.0)          # N_crit = 9

# TRAILING-EDGE BASE, as a fraction of the TOTAL chord. 0.003 c ~ 2 delta* at
# the fore TE (Re=1e6), so it adds no wake feature the layer did not already
# carry, and gives the two prism stacks a face to land on instead of a point.
TE_BASE = 0.003

# (tag, yplus, growth, hwall, hmax, max_steps, n_surface_panels)
# n_surface_panels is now part of the ladder: contours.py writes the points
# VERBATIM, so surface and volume refinement were previously decoupled and
# L0/L1 shared an identical wall discretisation (11.7 deg LE turning on both).
# WAKE BAND on the flap suction side (mesher wall index 1: walls are the
# wallFlag=1 curves in order, so 0=fore, 1=flap). The plain nearest-wall
# Spalding metric relaxes to hmax a few percent off any surface, so the fore
# element's wake -- which rides ABOVE the flap rather than along it -- was
# unresolved. (band_d, band_h) cap the spacing within band_d of the flap.
# h_te is an ABSOLUTE arc-length step at the TE corner (total-chord units);
# each element divides by its own chord. Halves per level, and (r_te - 1)
# halves per level, following flow360/build_eppler_proper_cavity.py.
# (tag, yplus, growth, hwall, hmax, steps, N_surf, band_d, band_h, h_te, r_te)
LEVELS = [
    ('L0', 1.00, 1.20, 0.008, 3.0,  8000,  600, 0.060, 0.0040, 1.40e-3, 2.00),
    ('L1', 0.50, 1.10, 0.004, 2.0, 12000, 1200, 0.060, 0.0020, 7.00e-4, 1.50),
    ('L2', 0.30, 1.05, 0.002, 1.5, 16000, 2400, 0.060, 0.0010, 3.50e-4, 1.25),
]
BAND_WALL = 1          # flap
import os as _os
# 1 = isotropic inside the band, 0 = wall-normal-only (rank-1)
BAND_ISO = int(_os.environ.get('BAND_ISO', '0'))


def make_cfg(tag, yplus, growth, hwall, hmax, steps, npan, hte, rte):
    import json as _json
    from contour_te import blunt_contour
    cfgA = _json.load(open('twoelement_adopted.json'))
    F, FL = cfgA['fore'], cfgA['flap']
    ch_f, ch_a = cfgA['chord'], FL['chord']
    n1, nf1 = blunt_contour(npan, hte/ch_f, rte, TE_BASE/ch_f, ch_f,
                            F['inc'], 0.0, 0.0, F['m'], F['p'], F['t'],
                            modified=True, xm=F['xm'], ik=6.0, km=F['km'])
    n2, nf2 = blunt_contour(npan, hte/ch_a, rte, TE_BASE/ch_a, ch_a,
                            FL['inc'], FL['le'][0], FL['le'][1],
                            FL['m'], FL['p'], FL['t'])
    print('   blunt-face edges: fore %d, flap %d' % (nf1, nf2), flush=True)
    els = []
    for k, nd in enumerate((n1, n2)):
        np.savetxt('%s_elem%d.dat' % (tag, k+1), nd, fmt='%12.8f')
        els.append(dict(name=['fore', 'flap'][k],
                        contour=[[float(x), float(z)] for x, z in nd[:-1]],
                        is_wall=True))
    allp = np.vstack([np.array(e['contour']) for e in els])
    cx = 0.5*(allp[:, 0].min() + allp[:, 0].max())
    cz = 0.5*(allp[:, 1].min() + allp[:, 1].max())
    cfg = dict(elements=els,
               farfield=dict(type='circle', center=[float(cx), float(cz)],
                             radius=100.0, n=360),
               flow=dict(reynolds=RE, mach=MACH, alpha_deg=ALPHA,
                         temperature=288.15),
               mesh=dict(span=0.1, nspan=1, yplus=yplus, growth=growth,
                         hwall=hwall, hmax=hmax, h0=0.0),
               solver=dict(max_steps=steps))
    p = 'case_%s.json' % tag
    json.dump(cfg, open(p, 'w'), indent=1)
    return p


if __name__ == '__main__':
    want = sys.argv[1:] or ['L0', 'L1', 'L2']
    os.environ['AFT_CHI_INF'] = '%.6e' % CHI_INF
    print('SA-AI seed chi_inf = %.4e  (N_crit = 9)' % CHI_INF)
    print('Re = %.1e  M = %.2f  alpha = %+.1f deg\n' % (RE, MACH, ALPHA))
    # env.py hardcodes compute/src/.../Flow360Core/Scripts onto PYTHONPATH.
    # On this machine that source tree is NEWER than the venv (it needs
    # flow360_schema.DEFAULT_SCHMIDT_NUMBER, which the venv's copy lacks), so
    # the entry point loads a translator inconsistent with its schema. The venv
    # ships self-consistent flow360scripts/flow360translator/flow360_schema --
    # drop the source path and let those win.
    import rans.env as _renv
    from rans import pipeline as _rp, case as _rc
    _orig_make_env = _renv.make_env

    def _make_env(compute_root=None):
        env, find = _orig_make_env(compute_root)
        pp = [q for q in env.get('PYTHONPATH', '').split(':')
              if q and 'Flow360Core/Scripts' not in q]
        env['PYTHONPATH'] = ':'.join(pp)
        return env, find

    _need = False
    try:
        import flow360_schema.models.simulation.models.material as _mat
        _need = not hasattr(_mat, 'DEFAULT_SCHMIDT_NUMBER')
    except Exception:
        _need = False
    if _need:
        print('  (stripping source Scripts from PYTHONPATH: venv schema is older)')
        for _mod in (_renv, _rp, _rc):
            if hasattr(_mod, 'make_env'):
                _mod.make_env = _make_env
    from rans.pipeline import run as pipe_run

    results = {}
    for tag, yp, g, hw, hm, st, npan, bd, bh, hte, rte in LEVELS:
        if tag not in want:
            continue
        cfgp = make_cfg(tag, yp, g, hw, hm, st, npan, hte, rte)

        # append the band spec to contours.txt (rans/contours.py is left
        # untouched so other users of the pipeline are unaffected)
        import rans.contours as _rcont
        from rans import pipeline as _rp2
        _orig_wc = getattr(_rcont, '_orig_write_contours', _rcont.write_contours)
        _rcont._orig_write_contours = _orig_wc

        def _wc(cfg_, path_, _bd=bd, _bh=bh):
            info = _orig_wc(cfg_, path_)
            with open(path_, 'a') as fh:
                fh.write('NBANDS 1\n%d %.8e %.8e %d\n'
                         % (BAND_WALL, _bd, _bh, BAND_ISO))
            info['band'] = dict(wall=BAND_WALL, d=_bd, h=_bh)
            return info
        _rcont.write_contours = _wc
        if hasattr(_rp2, 'write_contours'):
            _rp2.write_contours = _wc
        out = 'case_%s' % tag
        t0 = time.time()
        print('=== %s : y+ %.2f  growth %.2f  hwall %.3f  hmax %.1f  steps %d'
              '  N %d  TE base %.4f  h_te %.2e  r_te %.2f  band d<%.3f h<=%.4f'
              % (tag, yp, g, hw, hm, st, npan, TE_BASE, hte, rte, bd, bh),
              flush=True)
        try:
            r = pipe_run(cfgp, out, solve=True, gpu=0, flow_field=True)
            results[tag] = r
            print('  done in %.1f min' % ((time.time()-t0)/60), flush=True)
            for k in ('CL', 'CD', 'CM', 'cells', 'ncells'):
                if isinstance(r, dict) and k in r:
                    print('   %s = %s' % (k, r[k]), flush=True)
        except Exception as e:                                   # noqa: BLE001
            print('  FAILED: %s' % str(e)[:300], flush=True)
            for attr in ('stderr', 'stdout', 'output'):
                v = getattr(e, attr, None)
                if v:
                    print('  --- %s ---' % attr, flush=True)
                    print('\n'.join(str(v).splitlines()[-25:]), flush=True)
    json.dump({k: {kk: str(vv)[:200] for kk, vv in v.items()}
               for k, v in results.items() if isinstance(v, dict)},
              open('ladder_results.json', 'w'), indent=2)
    print('\nwrote ladder_results.json')
