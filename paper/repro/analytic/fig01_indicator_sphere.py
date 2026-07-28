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
coordinate Omega_hat*I_hat at {0.025, 0.2, 0.4, 0.6, 0.8, 1.0}; the lowest hugs the
neutral locus Omega_hat*I_hat = 0 (the parabola great circle I_hat = 0 plus the
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


def shade_sphere(rgb, cx, cy, r):
    """Light-marble shading: per-pixel hemisphere normals inside the
    silhouette circle; Lambertian diffuse composited by MULTIPLY (white
    interior rounds into a body, dark ink stays dark), tight Blinn-Phong
    specular + Fresnel rim composited by ADD (the glints). Kept very
    light: ambient floor 0.82."""
    ny_, nx_ = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]].astype(float)
    nx = (nx_ - cx)/r
    ny = (cy - ny_)/r                       # image rows grow downward
    rr2 = nx*nx + ny*ny
    inside = rr2 < 1.0
    nz = np.sqrt(np.clip(1.0 - rr2, 0.0, None))
    L = np.array([-0.45, 0.55, 0.70]); L /= np.linalg.norm(L)
    H = L + np.array([0.0, 0.0, 1.0]);  H /= np.linalg.norm(H)
    ndl = np.clip(nx*L[0] + ny*L[1] + nz*L[2], 0.0, None)
    ndh = np.clip(nx*H[0] + ny*H[1] + nz*H[2], 0.0, None)
    diffuse = 0.82 + 0.18*ndl               # multiply term, floor 0.82
    spec = 0.20 * ndh**42                   # tight upper-left glint
    rim = 0.09 * (1.0 - nz)**3              # steel-like edge light
    # soft edge so the shading fades over the last ~1.5 px of radius
    w = np.zeros_like(nz)
    w[inside] = np.clip((1.0 - np.sqrt(rr2[inside]))*r/1.5, 0.0, 1.0)
    mult = 1.0 + w*(diffuse - 1.0)
    add = w*(spec + rim)
    return np.clip(rgb*mult[..., None] + add[..., None], 0.0, 1.0)


def main():
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    # silhouette + poles
    tt = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(tt), np.sin(tt), '-', color='0.75', lw=1.0, zorder=1)
    ax.plot([-1/SQ2], [0], marker='<', color='0.3', ms=9, zorder=6)
    ax.plot([+1/SQ2], [0], marker='>', color='0.3', ms=9, zorder=6)
    ax.plot([0], [+1], marker='^', color='0.3', ms=9, zorder=6)
    ax.plot([0], [-1], marker='v', color='0.3', ms=9, zorder=6)

    # grey Omega_hat*I_hat contours over the near hemisphere
    hg = np.linspace(-0.999, 0.999, 601)
    vg = np.linspace(-0.999, 0.999, 601)
    Hm, Vm = np.meshgrid(hg, vg)
    disk = Hm**2 + Vm**2 < 1.0
    Wm = np.sqrt(np.clip(1.0 - Hm**2 - Vm**2, 0, None))
    X = (Wm - Hm)/SQ2; Y = (Wm + Hm)/SQ2; Z = Vm
    Omega_hat = Y/np.sqrt(X*X + Y*Y + 1e-30)
    G = Y - X - Z
    Psg = np.where(disk, Omega_hat*G, np.nan)
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

    # rasterize the line drawing, shade it as a lit marble hemisphere,
    # then rebuild the figure as raster + crisp vector pole labels on top
    DPI = 350
    fig.canvas.draw()
    cx, cy_top = ax.transData.transform((0.0, 0.0))
    ex, _ = ax.transData.transform((1.0, 0.0))
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI)
    buf.seek(0)
    img = plt.imread(buf)                    # RGBA float in [0,1]
    s = DPI/fig.dpi                          # canvas->saved-png pixel scale
    cxp, cyp, rp = cx*s, img.shape[0] - cy_top*s, (ex - cx)*s
    shaded = img.copy()
    shaded[..., :3] = shade_sphere(img[..., :3], cxp, cyp, rp)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7.2, 7.2))
    ax2.imshow(shaded, extent=[-1.25, 1.25, -1.2, 1.15], zorder=1,
               interpolation='lanczos')
    for x, y, txt, ha, va in [(-1/SQ2 - 0.05, 0.05, '$X$', 'right', 'bottom'),
                              (+1/SQ2 + 0.05, 0.05, '$Y$', 'left', 'bottom'),
                              (0.06, 1.03, '$+Z$', 'left', 'bottom'),
                              (0.06, -1.06, '$-Z$', 'left', 'top')]:
        ax2.text(x, y, txt, fontsize=13, color='0.15', ha=ha, va=va, zorder=5)
    ax2.set_aspect('equal'); ax2.set_xlim(-1.25, 1.25); ax2.set_ylim(-1.2, 1.15)
    ax2.axis('off')
    plt.tight_layout()
    fig2.savefig('figs/indicator_sphere.pdf', dpi=DPI)
    fig2.savefig('repro/analytic/figs_explore/indicator_sphere_new.png', dpi=140)
    print('wrote figs/indicator_sphere.pdf (marble-shaded)')


if __name__ == '__main__':
    main()
