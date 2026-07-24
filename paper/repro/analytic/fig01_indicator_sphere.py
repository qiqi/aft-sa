"""fig:model -> paper/figs/indicator_sphere.pdf.

The RP^2 indicator sphere (orthographic x-ray view): curvature indicator Z
vertical, shear-minus-velocity horizontal (velocity pole left, shear pole
right, +/- curvature poles top and bottom). Five representative profiles run
as thin curves from the wall point (centre, X=Y, Z=0) toward the freestream
(velocity pole); the near hemisphere is drawn solid, the far side dotted
(x-ray); NO in-figure text (all identification lives in the caption:
poles by triangle markers, profiles by color). The THICK segment on each curve marks the wall-normal band where the
most-amplified Orr-Sommerfeld mode's Reynolds-stress production density
p = (alpha/2) |Im(phi' phi*)| |U'| exceeds half its peak -- where the
disturbance actually grows (attached profiles evaluated at Re_theta = 500,
where the favorable layer is still subcritical and so carries no band; the
reversed-flow (Stewartson lower-branch, beta=-0.19, H~4.9) profile at
Re_theta = 200). Grey lines: contours of the rate
coordinate S_hat*g at {0.025, 0.2, 0.4, 0.6, 0.8, 1.0}; the lowest hugs the
neutral locus S_hat*g = 0 (the parabola great circle g = 0 plus the
shear-free meridian).

Run from paper/: python3 repro/analytic/fig01_indicator_sphere.py
"""
import _saai  # noqa: F401
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from explore_lsb_frozen_profile import build_profile
from explore_wavepacket_regions import os_mode, production, contiguous

SQ2 = np.sqrt(2.0)

PROFILES = [
    (0.30,   None,  'C0', 'favorable',            500.0),
    (0.0,    None,  'C2', 'Blasius',              500.0),
    (-0.10,  None,  'C1', 'adverse',              500.0),
    (-0.16,  None,  'orangered', 'strong adverse',  500.0),
    (-0.19,  None,  'goldenrod', 'near-separation (attached)', 500.0),
    (-0.1988, None, 'C3', 'incipient separation', 500.0),
    (-0.19, -0.03,  'C4', 'separated (reversed)', 200.0),
]

LEVELS = [0.025, 0.2, 0.4, 0.6, 0.8, 1.0]


def sphere_coords(u, up, upp, eta):
    X0 = u; Y0 = eta*up; Z0 = 0.5*eta**2*upp
    R = np.sqrt(X0*X0 + Y0*Y0 + Z0*Z0) + 1e-30
    X, Y, Z = X0/R, Y0/R, Z0/R
    h = (Y - X)/SQ2
    v = Z
    w = (X + Y)/SQ2      # depth toward viewer; >=0 is the near hemisphere
    return h, v, w


def band_eta(pr, Re_th):
    """eta-range where OS production > half its peak; None if stable."""
    try:
        yh, phi, al, c, unstable = os_mode(pr, Re_th)
    except Exception:
        return None
    if not unstable:
        return None
    Th = pr['Theta']
    Up = np.interp(yh*Th, pr['eta'], pr['up'])*Th
    p = production(yh, phi, al, Up)
    jp = int(np.argmax(p))
    mask = contiguous(p > 0.5*p[jp], jp)
    et = yh[mask]*Th
    return float(et.min()), float(et.max())


def draw_segments(ax, h, v, w, col, lw, band=None, eta=None):
    """Plot solid where near (w>=0), dotted where far; thick over the band."""
    near = w >= 0
    def _plot(mask, style, width, z):
        m = np.where(mask)[0]
        if len(m) == 0:
            return
        # split into runs so line breaks at hemisphere crossings
        splits = np.where(np.diff(m) > 1)[0]
        for run in np.split(m, splits + 1):
            ax.plot(h[run], v[run], style, color=col, lw=width, zorder=z)
    _plot(near, '-', lw, 4)
    _plot(~near, ':', lw*0.9, 3)
    if band is not None and eta is not None:
        bm = (eta >= band[0]) & (eta <= band[1])
        _plot(bm & near, '-', lw*3.2, 5)
        _plot(bm & ~near, ':', lw*3.2, 5)


def main():
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    # silhouette + poles
    tt = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(tt), np.sin(tt), '-', color='0.75', lw=1.0, zorder=1)
    ax.plot([-1/SQ2], [0], marker='<', color='0.3', ms=9, zorder=6)
    ax.plot([+1/SQ2], [0], marker='>', color='0.3', ms=9, zorder=6)
    ax.plot([0], [+1], marker='^', color='0.3', ms=9, zorder=6)
    ax.plot([0], [-1], marker='v', color='0.3', ms=9, zorder=6)

    # grey S_hat*g contours over the near hemisphere
    hg = np.linspace(-0.999, 0.999, 601)
    vg = np.linspace(-0.999, 0.999, 601)
    Hm, Vm = np.meshgrid(hg, vg)
    disk = Hm**2 + Vm**2 < 1.0
    Wm = np.sqrt(np.clip(1.0 - Hm**2 - Vm**2, 0, None))
    X = (Wm - Hm)/SQ2; Y = (Wm + Hm)/SQ2; Z = Vm
    Shat = Y/np.sqrt(X*X + Y*Y + 1e-30)
    G = Y - X - Z
    Psg = np.where(disk, Shat*G, np.nan)
    cs = ax.contour(Hm, Vm, Psg, levels=LEVELS, colors='0.6', linewidths=0.8,
                    zorder=2)

    for beta, guess, col, lab, Re_th in PROFILES:
        pr = build_profile(beta, guess)
        eta, u, up, upp = pr['eta'], pr['u'], pr['up'], pr['upp']
        h, v, w = sphere_coords(u, up, upp, eta)
        band = band_eta(pr, Re_th)
        draw_segments(ax, h, v, w, col, 1.4, band=band, eta=eta)
        print(f"beta={beta:+.3f} ({lab}): H={pr['H']:.2f}, "
              f"band={'%.2f-%.2f theta-units' % band if band else 'none (stable)'}",
              flush=True)

    ax.plot([0], [0], 'k.', ms=5, zorder=6)
    ax.set_aspect('equal'); ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.2, 1.15)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('figs/indicator_sphere.pdf')
    plt.savefig('repro/analytic/figs_explore/indicator_sphere_new.png', dpi=140)
    print('wrote figs/indicator_sphere.pdf')


if __name__ == '__main__':
    main()
