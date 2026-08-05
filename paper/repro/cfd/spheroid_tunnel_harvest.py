"""Harvest the spheroid TUNNEL-CONDITION campaign and compare to measurement.

The cases staged by spheroid/build_tunnel_cases.py: matched tunnel Mach, and
each condition at BOTH the facility-calibrated seed and the facility-MEASURED
hot-wire seed.  Reuses the validated instruments rather than reimplementing
them (facet-re-based analytic-normal rays, near-wall chi band, sub-cell front
crossing) -- see spheroid_uniformity_profiles / _fullbody_check / _a0_physics.

Per case: the near-wall chi = 1 and chi = c_v1 fronts on a meridian sweep at
each azimuth where a measurement exists, then the signed comparison.

Run from paper/:  python3 repro/cfd/spheroid_tunnel_harvest.py
Writes repro/cfd/figs_explore/spheroid_tunnel_harvest.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'paper', 'repro'))
FIGD = os.path.join(HERE, 'figs_explore')
DATA = os.path.join(REPO, 'paper', 'data')

import vtk                                                      # noqa: E402
from spheroid_uniformity_profiles import sweep_case             # noqa: E402
from spheroid_unstruct_a0_verdict import load_wall_facets       # noqa: E402
from spheroid_a0_physics import front_crossing                  # noqa: E402
from lib.calibrate_kernel import C_V1                           # noqa: E402

ROOT = os.environ.get('SAAI_SPH_ROOT', '/local_data/qiqi/sa-ai/spheroid_fv1')
XS = np.linspace(0.05, 0.97, 47)          # station grid for the front search
PHI_CURVE = np.arange(5.0, 180.0, 7.5)    # dense azimuth grid for the front curve

# tag -> (alpha, Re, facility, measured file, measured key)
CASES = {
    'a0_gcal':     (0.0,  7.20e6, 'NWG', 'stock2006_fig14a_digitized.json', None),
    'a0_gmeas':    (0.0,  7.20e6, 'NWG', 'stock2006_fig14a_digitized.json', None),
    'a10hi_gcal':  (10.0, 6.56e6, 'NWG', 'stock2006_fig15a_digitized.json',
                    're_6p56e6_alpha10_circles'),
    'a10hi_gmeas': (10.0, 6.56e6, 'NWG', 'stock2006_fig15a_digitized.json',
                    're_6p56e6_alpha10_circles'),
    'a10f1_fcal':  (10.0, 6.56e6, 'F1',  'stock2006_fig17a_digitized.json',
                    'measured_re6p56e6_squares'),
    'a10f1_fmeas': (10.0, 6.56e6, 'F1',  'stock2006_fig17a_digitized.json',
                    'measured_re6p56e6_squares'),
    'a10lo_gcal':  (10.0, 1.52e6, 'NWG', 'stock2006_fig15a_digitized.json',
                    're_1p52e6_alpha10_squares'),
    'a10lo_gmeas': (10.0, 1.52e6, 'NWG', 'stock2006_fig15a_digitized.json',
                    're_1p52e6_alpha10_squares'),
    'a2p5_gcal':   (2.5,  7.20e6, 'NWG', 'stock2006_fig14b_digitized.json',
                    'measured_squares'),
    'a5lo_gcal':   (5.0,  1.52e6, 'NWG', 'stock2006_fig14c_digitized.json',
                    'measured_re1p52e6_squares'),
    'a5hi_gcal':   (5.0,  6.49e6, 'NWG', 'stock2006_fig14c_digitized.json',
                    'measured_re6p49e6_circles'),
    'a2p5_gmeas':  (2.5,  7.20e6, 'NWG', 'stock2006_fig14b_digitized.json',
                    'measured_squares'),
    'a5lo_gmeas':  (5.0,  1.52e6, 'NWG', 'stock2006_fig14c_digitized.json',
                    'measured_re1p52e6_squares'),
    'a5hi_gmeas':  (5.0,  6.49e6, 'NWG', 'stock2006_fig14c_digitized.json',
                    'measured_re6p49e6_circles'),
    # Closes the low-Re band.  At this incidence the whole leeward half is open
    # free-vortex-layer separation, so the measured line is the separation line;
    # Stock digitized no computed curve for this panel, so his columns are '--'.
    'a29p7_gcal':  (29.7, 1.53e6, 'NWG', 'stock2006_fig16c_digitized.json',
                    'measured_re1p53e6_squares'),
    'a29p7_gmeas': (29.7, 1.53e6, 'NWG', 'stock2006_fig16c_digitized.json',
                    'measured_re1p53e6_squares'),
}


def measured(fn, key):
    d = json.load(open(os.path.join(DATA, fn)))
    if key is None:                       # alpha = 0: uniform front
        for k, v in d.items():
            if isinstance(v, list) and v and isinstance(v[0], dict) \
                    and 'xL' in v[0]:
                pts = [(p.get('phi_deg', 90.0), p['xL']) for p in v]
                return sorted(pts, key=lambda t: -t[0])
        return []
    return sorted([(p['phi_deg'], p['xL']) for p in d[key]],
                  key=lambda t: -t[0])


def load_grid(case):
    pv = os.path.join(case, 'volume.pvtu')
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(pv)
    r.Update()
    return r.GetOutput()


def harvest(tag):
    alpha, Re, fac, mfile, mkey = CASES[tag]
    case = os.path.join(ROOT, f'case_ogrid_L1_tun_{tag}')
    fj = json.load(open(os.path.join(case, 'Flow360.json')))
    mu_ref = float(fj['freestream']['muRef'])
    mach = float(fj['freestream']['Mach'])
    chi_inf = float(fj['freestream']['turbulenceQuantities']
                    ['modifiedTurbulentViscosityRatio'])
    n_crit = float(np.log(C_V1 / chi_inf))
    grid = load_grid(case)
    facets = load_wall_facets(os.path.realpath(os.path.join(case, 'mesh.cgns')))

    meas = measured(mfile, mkey)
    rows = []
    for phi, x_meas in meas:
        sw = sweep_case(grid, mu_ref, facets, phi_deg=phi, xs=XS)
        chim = np.array([s['chimax_nearwall'] for s in sw], float)
        f1 = front_crossing(XS, chim, 1.0)
        fc = front_crossing(XS, chim, C_V1)
        rows.append(dict(phi=phi, x_meas=x_meas, chi1=f1, cv1=fc,
                         d_chi1=f1 - x_meas))
        print(f'   phi {phi:6.1f}  meas {x_meas:.3f}   chi1 {f1:6.3f} '
              f'cv1 {fc:6.3f}   d(chi1-meas) {f1 - x_meas:+.3f}')
    # dense azimuth sweep -> the front as a curve, for the comparison figures
    curve = []
    for phi in PHI_CURVE:
        sw = sweep_case(grid, mu_ref, facets, phi_deg=float(phi), xs=XS)
        chim = np.array([s['chimax_nearwall'] for s in sw], float)
        f1 = front_crossing(XS, chim, 1.0)
        fc = front_crossing(XS, chim, C_V1)
        curve.append(dict(phi=float(phi),
                          chi1=(None if f1 != f1 else float(f1)),
                          cv1=(None if fc != fc else float(fc))))
    n_ok = sum(1 for c in curve if c['chi1'] is not None)
    print(f'   front curve: {n_ok}/{len(curve)} azimuths resolved')

    d = np.array([r['d_chi1'] for r in rows], float)
    ok = np.isfinite(d)
    stat = dict(n=int(ok.sum()),
                mean=float(d[ok].mean()) if ok.any() else None,
                rms=float(np.sqrt((d[ok] ** 2).mean())) if ok.any() else None,
                worst=float(d[ok][np.argmax(abs(d[ok]))]) if ok.any() else None)
    print(f'   -> vs measured: n={stat["n"]} mean {stat["mean"]:+.3f} '
          f'rms {stat["rms"]:.3f} worst {stat["worst"]:+.3f}')
    return dict(tag=tag, alpha=alpha, Re=Re, facility=fac, mach=mach,
                chi_inf=chi_inf, n_crit=n_crit, measured_source=mfile,
                measured_key=mkey, rows=rows, stat=stat, curve=curve)


def main():
    # --only TAG [TAG ...] harvests just those cases and MERGES them into the
    # existing JSON, leaving every other entry byte-identical.  Without it the
    # whole matrix is re-harvested, which is correct but re-derives numbers that
    # are already published -- use --only when adding a case.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=None, metavar='TAG')
    args = ap.parse_args()
    p_out = os.path.join(FIGD, 'spheroid_tunnel_harvest.json')
    out = {}
    if args.only:
        assert all(t in CASES for t in args.only), \
            f'unknown tag(s): {[t for t in args.only if t not in CASES]}'
        if os.path.exists(p_out):
            out = json.load(open(p_out))
            print(f'merging into {len(out)} existing entries')
    todo = args.only if args.only else list(CASES)
    for tag in todo:
        case = os.path.join(ROOT, f'case_ogrid_L1_tun_{tag}')
        if not os.path.exists(os.path.join(case, 'volume.pvtu')):
            print(f'-- {tag}: no volume.pvtu yet, skipped')
            continue
        print(f'== {tag}')
        try:
            out[tag] = harvest(tag)
        except Exception as e:                       # keep going, report
            print(f'   FAILED: {type(e).__name__}: {e}')
    os.makedirs(FIGD, exist_ok=True)
    json.dump(out, open(p_out, 'w'), indent=1)
    print(f'wrote {p_out} ({len(out)} entries)')


if __name__ == '__main__':
    main()
