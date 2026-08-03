"""Oil-flow / max-chi pairs for the measured-seed spheroid tunnel cases.

-> paper/figs/spheroid_oilflow_<tag>.pdf, one per case

Two panels side by side, both on the surface unrolled into (x/L, phi) with
phi = 0 the windward symmetry line:

  LEFT   the oil-flow analogue: skin-friction LINES, integrated from the
         surface shear vector itself (not inviscid streamlines, which is what
         Stock's Figs. 14-17 draw), over light filled contours of |c_f|.
         Convergence of the streaks marks separation.
  RIGHT  max(chi) over a short wall-normal segment shot from each surface
         point -- the 3D form of the near-wall chi probe used for the airfoils
         and the drag-crisis cylinder -- with the chi = 1 and chi = c_v1
         crossings drawn as the model-native front.

Both panels carry the measured DFVLR transition points, so the oil-flow
structure and the model front can be read against the same data.

The probe data comes from `spheroid/surface_map.py` (run once per case; it
caches surface_map_<tag>.npz).  Sequential fills are single-hue light-to-dark
per panel, and the two panels use different hues so a reader never has to
compare a magnitude across them; the measured symbols stay red, which
separates from both ramps.

Run from paper/:  python3 repro/cfd/regen_spheroid_tunnel_oilflow.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(PAPER, 'data')
FIGD = os.path.join(PAPER, 'figs')
ROOT = os.environ.get('SAAI_SPH_ROOT', '/local_data/qiqi/sa-ai/spheroid_fv1')

C_V1 = 7.1
A_AX, B_AX = 0.5, 1.0 / 12.0        # spheroid semi-axes / L (surface_map.py)
C_MEAS = '#D62728'

# tag -> (label, measured file, measured key or None)
CASES = [
    ('a0_gmeas',    r'$\alpha=0^\circ$, $Re_L=7.2\times10^6$',
     'stock2006_fig14a_digitized.json', None),
    ('a2p5_gmeas',  r'$\alpha=2.5^\circ$, $Re_L=7.2\times10^6$',
     'stock2006_fig14b_digitized.json', 'measured_squares'),
    ('a5lo_gmeas',  r'$\alpha=5^\circ$, $Re_L=1.52\times10^6$',
     'stock2006_fig14c_digitized.json', 'measured_re1p52e6_squares'),
    ('a5hi_gmeas',  r'$\alpha=5^\circ$, $Re_L=6.49\times10^6$',
     'stock2006_fig14c_digitized.json', 'measured_re6p49e6_circles'),
    ('a10lo_gmeas', r'$\alpha=10^\circ$, $Re_L=1.52\times10^6$',
     'stock2006_fig15a_digitized.json', 're_1p52e6_alpha10_squares'),
    ('a10hi_gmeas', r'$\alpha=10^\circ$, $Re_L=6.56\times10^6$',
     'stock2006_fig15a_digitized.json', 're_6p56e6_alpha10_circles'),
    ('a10f1_fmeas', r'$\alpha=10^\circ$, $Re_L=6.56\times10^6$, ONERA F1',
     'stock2006_fig17a_digitized.json', 'measured_re6p56e6_squares'),
]


def measured(fn, key):
    d = json.load(open(os.path.join(DATA, fn)))
    if key is None:
        for v in d.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) \
                    and 'xL' in v[0]:
                return [(p.get('phi_deg', 90.0), p['xL']) for p in v]
        return []
    return [(p['phi_deg'], p['xL']) for p in d[key]]


def one(tag, label, mfn, mkey):
    case = os.path.join(ROOT, f'case_ogrid_L1_tun_{tag}')
    npz = os.path.join(case, f'surface_map_case_ogrid_L1_tun_{tag}.npz')
    if not os.path.exists(npz):
        print(f'-- {tag}: no probe npz yet, skipped')
        return False
    d = np.load(npz)
    xl, phd = d['xl'], d['phi_deg']
    cf, chimax, us, up = d['cf'], d['chimax'], d['us'], d['up']
    meas = measured(mfn, mkey)

    fig, (aL, aR) = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True,
                                 constrained_layout=True)

    # ---- LEFT: oil flow -------------------------------------------------
    cfm = cf * 1e3
    top = float(np.nanpercentile(cfm, 99.0))
    im = aL.contourf(xl, phd, cfm, levels=np.linspace(0.0, top, 25),
                     cmap='Greys', extend='max', zorder=1, alpha=0.55)
    cb = fig.colorbar(im, ax=aL, pad=0.02, fraction=0.055)
    cb.set_label(r'$c_f\times10^3$', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    # circumferential shear -> dphi/dt needs the local ring radius
    XL = np.meshgrid(xl, phd)[0]
    rad = B_AX * np.sqrt(np.clip(1.0 - ((-A_AX + XL) / A_AX) ** 2, 1e-6, None))
    dphi = np.degrees(up / np.maximum(rad, 1e-9))
    # Light 3-point smoothing of the integrand before tracing.  The wall shear
    # carries cell-level noise, and where the circumferential component is
    # near zero (the whole surface at zero incidence) an unsmoothed integrand
    # makes the traces oscillate about the true straight streak.
    def smooth(a):
        b = a.astype(float).copy()
        for _ in range(2):
            b[1:-1, :] = 0.25 * b[:-2, :] + 0.5 * b[1:-1, :] + 0.25 * b[2:, :]
            b[:, 1:-1] = 0.25 * b[:, :-2] + 0.5 * b[:, 1:-1] + 0.25 * b[:, 2:]
        return b
    us, dphi = smooth(us), smooth(dphi)
    # A CONTROLLED set of streaks, seeded like Stock's ~20 printed lines,
    # rather than matplotlib's automatic density (which at zero incidence
    # fills the panel with indistinguishable horizontal lines).
    seed_phi = np.arange(2.0, 179.0, 7.5)
    starts = np.column_stack([np.full_like(seed_phi, 0.055), seed_phi])
    aL.streamplot(xl, phd, us, dphi, start_points=starts, color='0.15',
                  linewidth=0.7, density=35, arrowsize=0, integration_direction='forward',
                  broken_streamlines=False, zorder=3)
    aL.set_title('oil flow: skin-friction lines over $|c_f|$', fontsize=9)

    # ---- RIGHT: max chi over the wall-normal segment --------------------
    lg = np.log10(np.maximum(chimax, 1e-8))
    im2 = aR.contourf(xl, phd, lg, levels=np.linspace(-4, 2, 25), cmap='Blues',
                      extend='both', zorder=1)
    cb2 = fig.colorbar(im2, ax=aR, pad=0.02, fraction=0.055,
                       ticks=[-4, -3, -2, -1, 0, 1, 2])
    cb2.set_label(r'$\log_{10}\max_n\chi$', fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    for lvl, lw in ((1.0, 1.9), (C_V1, 1.3)):
        aR.contour(xl, phd, chimax, levels=[lvl], colors='k',
                   linewidths=lw, zorder=3)
    aR.set_title(r'$\max_n \chi$ on a wall-normal segment;'
                 r' black $\chi=1$, $c_{v1}$', fontsize=9)

    for a in (aL, aR):
        if meas:
            a.plot([m[1] for m in meas], [m[0] for m in meas], 's',
                   mfc='none', mec=C_MEAS, mew=1.4, ms=5.5, zorder=5)
        a.set_xlim(0, 1); a.set_ylim(0, 180)
        a.set_yticks([0, 45, 90, 135, 180])
        a.set_xlabel('$x/L$', fontsize=9)
        a.tick_params(labelsize=8)
    aL.set_ylabel(r'$\phi$ [deg]  (0 = windward)', fontsize=9)

    out = os.path.join(FIGD, f'spheroid_oilflow_{tag}.pdf')
    os.makedirs(FIGD, exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print('wrote', out)
    return True


def overlay(tag, label, mfn, mkey):
    """Single panel: the two views superposed, for the inclined conditions.

    The chi fill is deliberately truncated to the LIGHT half of the ramp so a
    dark friction line reads over every part of it -- stacking two saturated
    sequential fields, or drawing dark lines over a full light-to-dark ramp,
    is what makes this kind of composite unreadable.  Separation is left to
    the streak convergence itself rather than given its own coloured locus:
    that convergence is the oil-flow signature, and a fourth colour would
    collide with either the measured red or the front black.
    """
    import matplotlib.colors as mcolors
    case = os.path.join(ROOT, f'case_ogrid_L1_tun_{tag}')
    npz = os.path.join(case, f'surface_map_case_ogrid_L1_tun_{tag}.npz')
    if not os.path.exists(npz):
        print(f'-- {tag}: no probe npz, skipped'); return False
    d = np.load(npz)
    xl, phd = d['xl'], d['phi_deg']
    cf, chimax, us, up = d['cf'], d['chimax'], d['us'], d['up']

    fig, a = plt.subplots(figsize=(6.6, 4.1), constrained_layout=True)
    light = mcolors.LinearSegmentedColormap.from_list(
        'BluesLight', plt.get_cmap('Blues')(np.linspace(0.04, 0.60, 256)))
    im = a.contourf(xl, phd, np.log10(np.maximum(chimax, 1e-8)),
                    levels=np.linspace(-4, 2, 25), cmap=light, extend='both',
                    zorder=1)
    cb = fig.colorbar(im, ax=a, pad=0.02, fraction=0.05,
                      ticks=[-4, -3, -2, -1, 0, 1, 2])
    cb.set_label(r'$\log_{10}\max_n\chi$', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    XL = np.meshgrid(xl, phd)[0]
    rad = B_AX * np.sqrt(np.clip(1.0 - ((-A_AX + XL) / A_AX) ** 2, 1e-6, None))
    dphi = np.degrees(up / np.maximum(rad, 1e-9))

    def smooth(v):
        b = v.astype(float).copy()
        for _ in range(2):
            b[1:-1, :] = 0.25 * b[:-2, :] + 0.5 * b[1:-1, :] + 0.25 * b[2:, :]
            b[:, 1:-1] = 0.25 * b[:, :-2] + 0.5 * b[:, 1:-1] + 0.25 * b[:, 2:]
        return b
    seed_phi = np.arange(2.0, 179.0, 7.5)
    starts = np.column_stack([np.full_like(seed_phi, 0.055), seed_phi])
    a.streamplot(xl, phd, smooth(us), smooth(dphi), start_points=starts,
                 color='0.20', linewidth=0.6, density=35, arrowsize=0,
                 integration_direction='forward', broken_streamlines=False,
                 zorder=3)
    for lvl, lw in ((1.0, 2.2), (C_V1, 1.4)):
        a.contour(xl, phd, chimax, levels=[lvl], colors='k', linewidths=lw,
                  zorder=4)
    meas = measured(mfn, mkey)
    if meas:
        a.plot([m[1] for m in meas], [m[0] for m in meas], 's', mfc='none',
               mec=C_MEAS, mew=1.6, ms=6.0, zorder=5)
    a.set_xlim(0, 1); a.set_ylim(0, 180)
    a.set_yticks([0, 45, 90, 135, 180])
    a.set_xlabel('$x/L$', fontsize=9)
    a.set_ylabel(r'$\phi$ [deg]  (0 = windward)', fontsize=9)
    a.tick_params(labelsize=8)
    out = os.path.join(FIGD, f'spheroid_overlay_{tag}.pdf')
    fig.savefig(out); plt.close(fig)
    print('wrote', out)
    return True


def main():
    n = sum(one(*c) for c in CASES)
    print(f'{n}/{len(CASES)} pair figures written')
    # overlays for the INCLINED conditions only: at zero incidence both views
    # are functions of x/L alone and the superposition adds nothing.
    m = sum(overlay(*c) for c in CASES if not c[0].startswith('a0_'))
    print(f'{m} overlay figures written')


if __name__ == '__main__':
    main()
