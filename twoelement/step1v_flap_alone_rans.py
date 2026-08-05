"""STEP 1v (RANS) -- the flap ALONE, solved with the same SA-AI setup as the
two-element ladder, so the streamline through the future fore-element leading
edge can be traced in the real viscous field.

mfoil (step1v_flap_viscous.py) already says the flap's circulation is only ~81%
of its inviscid value at Re_flap = 3e5. But a mean line has to be traced through
a FIELD, and reconstructing an equivalent-inviscid displacement body from delta*
introduces its own modelling choices. Solving the flap alone with the same
solver, transition model and Reynolds number as the final two-element case gives
the field directly, and the circulation deficit it contains is exactly the one
the two-element run will see.

The mesh carries an ISOTROPIC refinement band out to one chord from the flap:
the plain Spalding metric relaxes toward hmax and would leave the region the
streamline crosses (x = 0 to 0.7, which is up to 0.7 chord ahead of the flap)
far too coarse to integrate a streamline through.

Run:  python3 step1v_flap_alone_rans.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/home/qiqi/flexcompute/flexfoil/rans')

from contour_te import blunt_contour                          # noqa: E402

import sys as _sys
RE, MACH = 1.0e6, 0.10
# anchoring incidence for the streamline trace -- the tuning knob for slot size.
# It need NOT equal the operating incidence.
ALPHA = float(_sys.argv[1]) if len(_sys.argv) > 1 else -1.0
TAG = 'A%+05.1f' % ALPHA
CHI_INF = 7.1*np.exp(-9.0)               # N_crit = 9, same seed as the ladder
TE_BASE = 0.003

FLAP = dict(m=0.09, p=0.40, t=0.16, chord=0.30, inc=-8.0, le=(0.70, 0.04))

# (tag, yplus, growth, hwall, hmax, steps, N_surf, band_d, band_h, h_te, r_te)
LEVEL = (TAG, 0.50, 1.10, 0.004, 2.0, 12000, 1200, 1.00, 0.012, 7.00e-4, 1.50)


def make_cfg(tag, yplus, growth, hwall, hmax, steps, npan, hte, rte):
    ch = FLAP['chord']
    nd, nface = blunt_contour(npan, hte/ch, rte, TE_BASE/ch, ch,
                              FLAP['inc'], FLAP['le'][0], FLAP['le'][1],
                              FLAP['m'], FLAP['p'], FLAP['t'])
    print('   blunt-face edges: flap %d' % nface, flush=True)
    np.savetxt('%s_flap.dat' % tag, nd, fmt='%12.8f')
    els = [dict(name='flap',
                contour=[[float(x), float(z)] for x, z in nd[:-1]],
                is_wall=True)]
    allp = np.array(els[0]['contour'])
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
    os.environ['AFT_CHI_INF'] = '%.6e' % CHI_INF
    tag, yp, g, hw, hm, st, npan, bd, bh, hte, rte = LEVEL
    print('FLAP ALONE, SA-AI seed chi_inf = %.4e (N_crit = 9)' % CHI_INF)
    print('Re = %.1e per unit length, flap chord %.2f -> Re_flap = %.2e'
          % (RE, FLAP['chord'], RE*FLAP['chord']))
    print('M = %.2f  alpha = %+.1f deg\n' % (MACH, ALPHA))

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
        print('  (stripping source Scripts from PYTHONPATH: venv schema older)')
        for _mod in (_renv, _rp, _rc):
            if hasattr(_mod, 'make_env'):
                _mod.make_env = _make_env
    from rans.pipeline import run as pipe_run

    cfgp = make_cfg(tag, yp, g, hw, hm, st, npan, hte, rte)

    import rans.contours as _rcont
    from rans import pipeline as _rp2
    _orig_wc = getattr(_rcont, '_orig_write_contours', _rcont.write_contours)
    _rcont._orig_write_contours = _orig_wc

    def _wc(cfg_, path_, _bd=bd, _bh=bh):
        info = _orig_wc(cfg_, path_)
        with open(path_, 'a') as fh:
            # wall 0 is the flap (the only wall); iso = 1
            fh.write('NBANDS 1\n0 %.8e %.8e 1\n' % (_bd, _bh))
        info['band'] = dict(wall=0, d=_bd, h=_bh, iso=1)
        return info
    _rcont.write_contours = _wc
    if hasattr(_rp2, 'write_contours'):
        _rp2.write_contours = _wc

    t0 = time.time()
    print('=== %s : y+ %.2f growth %.2f hwall %.3f hmax %.1f steps %d N %d'
          '  h_te %.2e r_te %.2f  ISO band d<%.2f h<=%.4f'
          % (tag, yp, g, hw, hm, st, npan, hte, rte, bd, bh), flush=True)
    r = pipe_run(cfgp, 'case_%s' % tag, solve=True, gpu=0, flow_field=True)
    print('  done in %.1f min' % ((time.time() - t0)/60), flush=True)
    json.dump({k: str(v)[:400] for k, v in r.items()},
              open('flap_alone_result_%s.json' % tag, 'w'), indent=2)
    for k in ('forces', 'mesh'):
        if k in r:
            print('  %s: %s' % (k, str(r[k])[:300]), flush=True)
