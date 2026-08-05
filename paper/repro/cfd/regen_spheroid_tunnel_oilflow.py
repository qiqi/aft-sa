"""Oil-flow / max-chi views for the measured-seed spheroid tunnel cases.

-> paper/figs/spheroid_oilflow_<tag>.pdf   (pair: oil flow | max chi)
-> paper/figs/spheroid_overlay_<tag>.pdf   (the two superposed, inclined only)

LINE ART ONLY -- no filled contours, no colour bars, black on white, so the
figures stay legible on an e-ink reader and match the paper's existing
labeled-black-contour style for these maps.  Both views are the surface
unrolled into (x/L, phi), phi = 0 the windward symmetry line.

  oil flow   skin-friction LINES integrated from the wall shear vector itself
             (not the inviscid streamlines of Stock's Figs. 14-17), thin solid;
             drawn GREY so they are never confused with the black contour
             families; |c_f| as labeled black dashed contours.  Streak
             convergence is the separation signature.
  max chi    max(chi) over a short wall-normal segment shot from each surface
             point -- the 3D form of the near-wall probe used for the airfoils
             and the drag-crisis cylinder.  Sub-unity decades dashed,
             chi = 1 heavy, chi = c_v1 and the supercritical decades solid,
             following the convention of the paper's other spheroid maps.

Measured DFVLR transition points are white-filled black squares, so they
occlude the line art underneath rather than competing with it.

Probe data: `spheroid/surface_map.py` caches surface_map_<tag>.npz in each
case dir.  Those caches are also kept in repro/cfd/cache_spheroid_surface/ so
these figures regenerate WITHOUT the 42 GB case tree.

Run from paper/:  python3 repro/cfd/regen_spheroid_tunnel_oilflow.py
"""
import json
import os

import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(PAPER, 'data')
FIGD = os.path.join(PAPER, 'figs')
CACHE = os.path.join(HERE, 'cache_spheroid_surface')
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'analytic')))
ROOT = os.environ.get('SAAI_SPH_ROOT', '/local_data/qiqi/sa-ai/spheroid_fv1')

C_V1 = 7.1
A_AX, B_AX = 0.5, 1.0 / 12.0        # spheroid semi-axes / L (surface_map.py)
CF_LEV = [1.0, 2.0, 3.0]                    # c_f x 1e3; three only --
#   the field carries cell-level noise, and a fine level set turns the
#   turbulent plateau into a jagged tangle that buries the streaks
CHI_SUB = [0.1, 0.3]                        # sub-unity: dashed
CHI_SUP = [C_V1, 30.0, 100.0]               # supercritical: solid

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
    ('a29p7_gmeas', r'$\alpha=29.7^\circ$, $Re_L=1.53\times10^6$',
     'stock2006_fig16c_digitized.json', 'measured_re1p53e6_squares'),
    ('a10hi_gmeas', r'$\alpha=10^\circ$, $Re_L=6.56\times10^6$',
     'stock2006_fig15a_digitized.json', 're_6p56e6_alpha10_circles'),
    ('a10f1_fmeas', r'$\alpha=10^\circ$, $Re_L=6.56\times10^6$, ONERA F1',
     'stock2006_fig17a_digitized.json', 'measured_re6p56e6_squares'),
]


def load(tag):
    name = f'surface_map_case_ogrid_L1_tun_{tag}.npz'
    for p in (os.path.join(CACHE, name),
              os.path.join(ROOT, f'case_ogrid_L1_tun_{tag}', name)):
        if os.path.exists(p):
            return np.load(p)
    return None


def measured(fn, key):
    d = json.load(open(os.path.join(DATA, fn)))
    if key is None:
        for v in d.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) \
                    and 'xL' in v[0]:
                return [(p.get('phi_deg', 90.0), p['xL']) for p in v]
        return []
    return [(p['phi_deg'], p['xL']) for p in d[key]]


def smooth(v):
    """Two 3-point passes.  The wall shear carries cell-level noise, and where
    its circumferential component is near zero (the whole surface at zero
    incidence) an unsmoothed integrand makes the traces oscillate about the
    true straight streak."""
    b = v.astype(float).copy()
    for _ in range(2):
        b[1:-1, :] = 0.25 * b[:-2, :] + 0.5 * b[1:-1, :] + 0.25 * b[2:, :]
        b[:, 1:-1] = 0.25 * b[:, :-2] + 0.5 * b[:, 1:-1] + 0.25 * b[:, 2:]
    return b


SEED_TOTAL = 24            # ~ the printed trace count of Stock Figs. 14-17
SWEEP_SCALE = 3.0          # Lambda at which the windward rake takes over
SEED_X_NOSE = 0.055        # nose-ring station
SEED_PHI_WIND = 1.2        # windward-rake azimuth, deg (0 would never sweep)
SEED_X_RAKE = (0.03, 0.85)


def sweep_profile(alpha_deg, xl):
    """Cumulative azimuthal sweep Lambda(x) of a surface streamline, from the
    exact potential field.

    A surface streamline obeys d(phi)/ds = u_phi/(r0 u_s), and with the O(alpha)
    potential result u_phi = U alpha g sin(phi), g = T/b constant, this
    separates: the variable u = ln tan(phi/2) simply TRANSLATES,

        du/ds = alpha g / (r0 f0),      Lambda(x) = integral du.

    So Lambda is the natural measure of how far a streamline is carried around
    the body between two stations, and it is what sets the seeding.  See
    repro/analytic/spheroid_potential.py.
    """
    from spheroid_potential import surface_fields, metrics, XI0, B_AX
    eta = np.clip(2.0 * xl - 1.0, -1 + 1e-9, 1 - 1e-9)
    f0, _, g = surface_fields(eta)
    _, h_eta, _ = metrics(XI0, eta)
    r0 = B_AX * np.sqrt(1.0 - eta**2)
    dds_dx = 2.0 * h_eta                       # eta = 2x - 1
    rate = np.radians(alpha_deg) * g / np.maximum(r0 * f0, 1e-12) * dds_dx
    return np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1])
                                            * np.diff(xl))])


def seeds(alpha_deg, xl):
    """Two rakes, split by how much the flow sweeps in azimuth.

    At alpha = 0 the surface flow is purely meridional, so a ring of seeds near
    the nose crosses every streamline and nothing else is needed.  As alpha
    grows the streamlines are swept windward -> leeward, the nose ring's traces
    all pile onto the leeward side, and the windward surface aft of the nose is
    left bare -- which is what made the alpha = 10 and 29.7 panels unusable.
    So a second rake is laid along the windward line itself, and the split
    between the two follows the total sweep Lambda:

        w = Lambda / (Lambda + 3),   n_wind = round(w * 24),  n_nose = 24 - n_wind

    giving all-nose at alpha = 0 and mostly-windward by alpha = 29.7.  The
    windward seeds are spaced uniformly in REMAINING sweep rather than in x, so
    their traces peel off the windward line at evenly spread azimuths instead of
    bunching where the sweep is fastest.
    """
    L = sweep_profile(alpha_deg, xl)
    lam = float(L[-1] - L[0])
    w = lam / (lam + SWEEP_SCALE)
    n_wind = int(round(w * SEED_TOTAL))
    n_nose = SEED_TOTAL - n_wind
    pts = []
    if n_nose > 0:
        for ph in np.linspace(2.0, 178.0, n_nose):
            pts.append((SEED_X_NOSE, ph))
    if n_wind > 0:
        m = (xl >= SEED_X_RAKE[0]) & (xl <= SEED_X_RAKE[1])
        xs, Ls = xl[m], L[m]
        psi = Ls[-1] - Ls                      # sweep still to come
        for target in np.linspace(psi[0], psi[-1], n_wind + 2)[1:-1]:
            pts.append((float(np.interp(-target, -psi, xs)), SEED_PHI_WIND))
    return np.array(pts), lam, n_nose, n_wind


def streaks(ax, xl, phd, us, up, alpha_deg):
    XL = np.meshgrid(xl, phd)[0]
    rad = B_AX * np.sqrt(np.clip(1.0 - ((-A_AX + XL) / A_AX) ** 2, 1e-6, None))
    dphi = np.degrees(up / np.maximum(rad, 1e-9))
    pts, lam, n_nose, n_wind = seeds(alpha_deg, xl)
    print(f'    seeds: Lambda={lam:5.2f} rad -> {n_nose} nose ring + '
          f'{n_wind} windward rake')
    # Integrate BOTH ways from each seed: tracing only downstream left the nose
    # region blank.
    ax.streamplot(xl, phd, smooth(us), smooth(dphi),
                  start_points=np.column_stack([pts[:, 0], pts[:, 1]]),
                  color='0.55', linewidth=0.6, density=35, arrowsize=0,
                  integration_direction='both', broken_streamlines=False,
                  zorder=3)


def cf_contours(ax, xl, phd, cf):
    cs = ax.contour(xl, phd, smooth(cf) * 1e3, levels=CF_LEV, colors='k',
                    linewidths=0.8, linestyles='dashed', zorder=2)
    ax.clabel(cs, fmt='%.0f', fontsize=6.5, inline=True, inline_spacing=6)


def chi_contours(ax, xl, phd, chimax, heavy=2.2):
    lo, hi = float(np.nanmin(chimax)), float(np.nanmax(chimax))
    sub = [l for l in CHI_SUB if lo < l < hi]
    sup = [l for l in CHI_SUP if lo < l < hi]
    if sub:
        cs = ax.contour(xl, phd, chimax, levels=sub, colors='k',
                        linewidths=0.7, linestyles='dashed', zorder=3)
        ax.clabel(cs, fmt='%g', fontsize=6, inline=True, inline_spacing=2)
    if sup:
        cs = ax.contour(xl, phd, chimax, levels=sup, colors='k',
                        linewidths=0.9, zorder=3)
        ax.clabel(cs, fmt='%g', fontsize=6, inline=True, inline_spacing=2)
    if lo < 1.0 < hi:
        ax.contour(xl, phd, chimax, levels=[1.0], colors='k',
                   linewidths=heavy, zorder=4)


def mark(ax, meas):
    if meas:
        ax.plot([m[1] for m in meas], [m[0] for m in meas], 's', mfc='white',
                mec='k', mew=1.5, ms=6.0, zorder=6)


def frame(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 180)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel('$x/L$', fontsize=9)
    ax.tick_params(labelsize=8)


def one(tag, label, mfn, mkey):
    d = load(tag)
    if d is None:
        print(f'-- {tag}: no probe cache, skipped')
        return False
    xl, phd = d['xl'], d['phi_deg']
    meas = measured(mfn, mkey)
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True,
                                 constrained_layout=True)
    cf_contours(aL, xl, phd, d['cf'])
    streaks(aL, xl, phd, d['us'], d['up'], float(d['alpha']))
    aL.set_title(r'oil flow: skin-friction lines; dashed $c_f\times10^3$',
                 fontsize=9)
    chi_contours(aR, xl, phd, d['chimax'])
    aR.set_title(r'$\max_n\chi$; heavy $\chi=1$, dashed sub-unity', fontsize=9)
    for a in (aL, aR):
        mark(a, meas)
        frame(a)
    aL.set_ylabel(r'$\phi$ [deg]  (0 = windward)', fontsize=9)
    out = os.path.join(FIGD, f'spheroid_oilflow_{tag}.pdf')
    os.makedirs(FIGD, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print('wrote', out)
    return True


def overlay(tag, label, mfn, mkey):
    d = load(tag)
    if d is None:
        print(f'-- {tag}: no probe cache, skipped')
        return False
    xl, phd = d['xl'], d['phi_deg']
    fig, a = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    streaks(a, xl, phd, d['us'], d['up'], float(d['alpha']))
    chi_contours(a, xl, phd, d['chimax'], heavy=2.6)
    mark(a, measured(mfn, mkey))
    frame(a)
    a.set_ylabel(r'$\phi$ [deg]  (0 = windward)', fontsize=9)
    out = os.path.join(FIGD, f'spheroid_overlay_{tag}.pdf')
    fig.savefig(out)
    plt.close(fig)
    print('wrote', out)
    return True


def main():
    n = sum(one(*c) for c in CASES)
    print(f'{n}/{len(CASES)} pair figures')
    # Zero incidence is included so the section carries one overlay per
    # condition; there it is simply the degenerate case, both families
    # functions of x/L alone.
    m = sum(overlay(*c) for c in CASES)
    print(f'{m} overlay figures')


if __name__ == '__main__':
    main()
