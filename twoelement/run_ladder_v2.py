"""Mesh + solve the ADOPTED two-element geometry (step7v) on the L0/L1 ladder.

Carries over every meshing lesson from the first ladder:

  * mesh2d.cpp's tangential metric reads the LOCAL surface spacing through
    edges[minEdgeIdx], not points[minEdgeIdx]. That bug made the upper surfaces
    5x over-refined and the lower ones coarse; fixed and rebuilt on 017.
  * ANISOTROPIC wake band on the flap (mesher wall index 1), rank-1 along the
    flap normal with a graded release, to resolve the fore wake riding above it.
  * h_te is ABSOLUTE (total-chord units); each element divides by its own chord
    so both trailing edges get the same physical spacing.
  * Blunt trailing edge, TE_BASE = 0.003 total chord, with EXPLICIT face points
    so the two prism stacks land on a face rather than a point.
  * Surface refinement is part of the ladder: N_surf doubles per level, so the
    wall discretisation is not shared between levels.
  * Contours come from contour_cst, which fixes the degenerate leading-edge
    cluster (2.4e-6 segments) and the misplaced clustering of the old generator.

Known and NOT fixed here: measured y+ ran 2-4x the nominal ladder value on the
previous geometry. The yplus column below is therefore a request, not a result;
it is reported again after the solve.

Run:  python3 run_ladder_v2.py [L0 L1]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/home/qiqi/flexcompute/flexfoil/rans')

import panel2e as M                                           # noqa: E402
import step2v_build as B                                      # noqa: E402
import step1v_flap_viscous as V                                # noqa: E402
import flap_displacement as FD                                 # noqa: E402
from contour_cst import blunt_contour_from_surface, report     # noqa: E402

RE, MACH = 1.0e6, 0.10
# Operating incidence. Overridable so an alpha sweep reuses one geometry.
ALPHA = float(os.environ.get('ALPHA', '-1.0'))
CHI_INF = 7.1*np.exp(-9.0)          # N_crit = 9
TE_BASE = 0.003

# (tag, yplus, growth, hwall, hmax, steps, N_surf, band_d, band_h, h_te, h_le)
# h_le is new: the leading-edge arc spacing. It halves per level like h_te, and
# together with the curvature cap in contour_cst it gives 7.1 deg of leading-edge
# turning at L0 and 3.6 deg at L1 -- a genuine 2x per level, where the old
# generator gave 11.7 deg at BOTH because surface and volume refinement were
# decoupled.
LEVELS = [
    ('L0', 1.00, 1.20, 0.008, 3.0,  8000,  600, 0.060, 0.0040, 1.40e-3, 5.0e-4),
    ('L1', 0.50, 1.10, 0.004, 2.0, 12000, 1200, 0.060, 0.0020, 7.00e-4, 2.5e-4),
    ('L2', 0.30, 1.05, 0.002, 1.5, 16000, 2400, 0.060, 0.0010, 3.50e-4, 1.25e-4),
]
BAND_WALL = 1                       # the flap
BAND_ISO = int(os.environ.get('BAND_ISO', '0'))   # 0 = anisotropic (adopted)
# Scale the requested y+ (hence the first-layer thickness). Measured y+ came out
# 3-4x the nominal request on both geometries, so halving this is the direct
# lever on the first layer. Case directories get a suffix so runs do not
# overwrite each other.
# y+ scale 0.5 is the ADOPTED setting: it puts measured y+ at ~2 on L0 and ~1
# on L1, which is the intent. The boundary-layer state is insensitive to it
# (transition moved by <0.001 chord) but the forces are not (CL -5%, CD +7%),
# so the halved-layer numbers are the ones to quote.
YPLUS_SCALE = float(os.environ.get('YPLUS_SCALE', '1.0'))
# Multiplier on the per-level step count. Every run in the sweep hit max_steps
# with a momentum residual of 3e-9 to 3e-8 -- ABOVE the solver's own 1e-9
# absolute tolerance -- so none of them are converged by that criterion, and
# laminar cases in this family are known to need ~4e4 steps.
STEP_SCALE = float(os.environ.get('STEP_SCALE', '1.0'))
# Which GPU to solve on. 017 has 8; pinning this lets an extra sweep run
# alongside a long one instead of queueing behind it.
GPU = int(os.environ.get('GPU', '0'))
SUFFIX = ('' if YPLUS_SCALE == 1.0 else '_yp%g' % YPLUS_SCALE) \
    + ('' if ALPHA == -1.0 else '_a%+05.1f' % ALPHA) \
    + ('' if STEP_SCALE == 1.0 else '_s%g' % STEP_SCALE)


def fore_surface(n=4000):
    """Dense fore-element loop, lower TE -> LE -> upper TE, blunt TE already
    opened to TE_BASE by fore_nodes' te_half term."""
    d = np.load('step7v_final.npz')
    nd, xs, zc, th = B.fore_nodes(d['thk'], d['cam_par'], float(d['x_end']),
                                  npts=n//2 + 1)
    return nd


def flap_surface(n=4000):
    """Dense TRUE flap section (NACA 9416) placed in the global frame. The
    displacement body was a device for the inviscid tuning only -- the RANS
    resolves the boundary layer itself, so the real section is meshed."""
    return M.place(M.airfoil_nodes(n, V.FLAP['m'], V.FLAP['p'], V.FLAP['t'],
                                   modified=False, te_thick=V.TE_THICK),
                   FD.FLAP_CHORD, FD.FLAP_INC, *FD.FLAP_LE)


def make_cfg(tag, yplus, growth, hwall, hmax, steps, npan, hte, hle):
    d = np.load('step7v_final.npz')
    ch_f = float(d['x_end'])
    ch_a = FD.FLAP_CHORD
    n1, nf1 = blunt_contour_from_surface(fore_surface(), npan, hte, 1.5,
                                         TE_BASE, h_le=hle)
    n2, nf2 = blunt_contour_from_surface(flap_surface(), npan, hte, 1.5,
                                         TE_BASE, h_le=hle)
    print('   blunt-face edges: fore %d, flap %d' % (nf1, nf2), flush=True)
    report(n1, 'fore')
    report(n2, 'flap')
    els = []
    for k, nd in enumerate((n1, n2)):
        np.savetxt('%s_v2%s_elem%d.dat' % (tag, SUFFIX, k+1), nd,
                   fmt='%12.8f')
        els.append(dict(name=['fore', 'flap'][k],
                        contour=[[float(x), float(z)] for x, z in nd],
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
    p = 'case_%s_v2%s.json' % (tag, SUFFIX)
    json.dump(cfg, open(p, 'w'), indent=1)
    return p


if __name__ == '__main__':
    want = sys.argv[1:] or ['L0', 'L1']
    os.environ['AFT_CHI_INF'] = '%.6e' % CHI_INF
    print('ADOPTED geometry (step7v), SA-AI seed chi_inf = %.4e (N_crit = 9)'
          % CHI_INF)
    print('Re = %.1e  M = %.2f  alpha = %+.1f deg   band: wall %d, iso=%d\n'
          % (RE, MACH, ALPHA, BAND_WALL, BAND_ISO))

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

    results = {}
    for tag, yp, g, hw, hm, st, npan, bd, bh, hte, hle in LEVELS:
        if tag not in want:
            continue
        cfgp = make_cfg(tag, yp*YPLUS_SCALE, g, hw, hm,
                        int(round(st*STEP_SCALE)), npan, hte, hle)

        import rans.contours as _rcont
        from rans import pipeline as _rp2
        _orig_wc = getattr(_rcont, '_orig_write_contours', _rcont.write_contours)
        _rcont._orig_write_contours = _orig_wc

        def _wc(cfg_, path_, _bd=bd, _bh=bh):
            info = _orig_wc(cfg_, path_)
            with open(path_, 'a') as fh:
                fh.write('NBANDS 1\n%d %.8e %.8e %d\n'
                         % (BAND_WALL, _bd, _bh, BAND_ISO))
            info['band'] = dict(wall=BAND_WALL, d=_bd, h=_bh, iso=BAND_ISO)
            return info
        _rcont.write_contours = _wc
        if hasattr(_rp2, 'write_contours'):
            _rp2.write_contours = _wc

        out = 'case_%s_v2%s' % (tag, SUFFIX)
        t0 = time.time()
        print('=== %s : y+ %.3f (scale %g) growth %.2f hwall %.3f hmax %.1f '
              'steps %d N %d  h_te %.2e h_le %.2e  band d<%.3f h<=%.4f'
              % (tag, yp*YPLUS_SCALE, YPLUS_SCALE, g, hw, hm,
                 int(round(st*STEP_SCALE)), npan, hte, hle, bd, bh),
              flush=True)
        try:
            r = pipe_run(cfgp, out, solve=True, gpu=GPU, flow_field=True)
            results[tag] = r
            print('  done in %.1f min' % ((time.time()-t0)/60), flush=True)
            for k in ('forces', 'mesh'):
                if k in r:
                    print('   %s: %s' % (k, str(r[k])[:260]), flush=True)
        except Exception as e:                                   # noqa: BLE001
            print('  FAILED: %s' % str(e)[:400], flush=True)
            for attr in ('stderr', 'stdout', 'output'):
                v = getattr(e, attr, None)
                if v:
                    print('\n'.join(str(v).splitlines()[-25:]), flush=True)
    json.dump({k: {kk: str(vv)[:200] for kk, vv in v.items()}
               for k, v in results.items() if isinstance(v, dict)},
              open('ladder_results_v2%s.json' % SUFFIX, 'w'), indent=2)
    print('\nwrote ladder_results_v2.json')
