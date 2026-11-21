"""fig:model -> paper/figs/indicator_sphere.pdf.

The RP^2 indicator sphere (orthographic x-ray view): curvature indicator Z
vertical, shear-minus-velocity horizontal (velocity pole left, shear pole
right, +/- curvature poles top and bottom). Five representative profiles run
as thin curves from the wall point (centre, X=Y, Z=0) toward the freestream
(velocity pole); the near hemisphere is drawn solid, the far side dotted
(x-ray). In-figure text is limited to the pole labels and one label on each
grey contour; the profiles are identified by color in the caption. The THICK segment on each curve marks the wall-normal band where the
most-amplified Orr-Sommerfeld mode's Reynolds-stress production density
p = (alpha/2) |Im(phi' phi*)| |U'|, multiplied by local velocity over local
vorticity, exceeds half its peak. The weight u/omega converts a spatial
amplification into the model's dimensionless rate (d ln nu~/dx = a omega/u);
p u/omega is the wave's Reynolds shear stress times the local velocity (attached profiles evaluated at Re_theta = 500,
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
from fs_frozen_profile import build_profile
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
# One on-contour label per level, at a point clear of the coloured curves: the
# 0.025 contour is labelled on its left branch, the others down the right side.
LABEL_AT = [(-0.60, 0.45), (0.50, 0.51), (0.575, 0.41), (0.60, 0.25),
            (0.60, 0.03), (0.66, -0.20)]
# 15 pt on the 7.2 in canvas is 9.2 pt at the paper's 0.68\textwidth.
LABEL_FS = 15


def sphere_coords(u, up, upp, eta):
    X0 = u; Y0 = eta*up; Z0 = 0.5*eta**2*upp
    R = np.sqrt(X0*X0 + Y0*Y0 + Z0*Z0) + 1e-30
    X, Y, Z = X0/R, Y0/R, Z0/R
    h = (Y - X)/SQ2
    v = Z
    w = (X + Y)/SQ2      # depth toward viewer; >=0 is the near hemisphere
    return h, v, w


def band_eta(pr, Re_th):
    """eta-range where OS production * u / omega > half its peak; None if stable."""
    try:
        yh, phi, al, c, unstable = os_mode(pr, Re_th)
    except Exception:
        return None
    if not unstable:
        return None
    Th = pr['Theta']
    Up = np.interp(yh*Th, pr['eta'], pr['up'])*Th
    p = production(yh, phi, al, Up)
    # Convert the production to the model's dimensionless rate. The model has
    # d ln(nu~)/dx = a*omega/u, so a = (spatial rate) * u / omega: weight the
    # production by local velocity over local vorticity. Since the production
    # is Reynolds stress times shear, this is the wave's Reynolds shear
    # stress times the local velocity.
    Uh = np.interp(yh*Th, pr['eta'], pr['u'])
    ok = np.abs(Up) > 1e-6*np.abs(Up).max()
    p = np.where(ok, p*np.abs(Uh)/np.where(ok, np.abs(Up), 1.0), 0.0)
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
    diffuse = 0.66 + 0.31*ndl               # multiply term: floor 0.66,
    # lit side tops out at 0.97 so even the bright side reads off-white
    spec = 0.26 * ndh**42                   # tight upper-left glint
    rim = 0.12 * (1.0 - nz)**3              # steel-like edge light
    # soft edge so the shading fades over the last ~1.5 px of radius
    w = np.zeros_like(nz)
    w[inside] = np.clip((1.0 - np.sqrt(rr2[inside]))*r/1.5, 0.0, 1.0)
    mult = 1.0 + w*(diffuse - 1.0)
    add = w*(spec + rim)
    return np.clip(rgb*mult[..., None] + add[..., None], 0.0, 1.0)


def pole_cone(ax, n, a=0.0425, h=0.15, color='0.35', zorder=6):
    """A cone standing on the sphere at pole direction n, drawn in projection.

    The base is a circle of angular radius arcsin(a) lying ON the sphere, so
    its rim meets the silhouette exactly where the sphere does; the apex sits
    at radius 1+h along n. Everything is projected with the figure's own map
    (H = (Y-X)/sqrt2, V = Z), so each pole is foreshortened by its own
    orientation: the +/-Z cones come out edge-on (the base collapses to a
    chord of the silhouette, giving a clean triangle) and the X and Y cones,
    whose axes lie at 45 deg to the view, keep an elliptical base. That is
    what makes them read as cones ON the sphere rather than flat arrowheads
    beside it.

    h has to be generous: the X and Y axes lie at 45 deg to the view, so their
    cones are foreshortened by cos45 and a height that looks right on the
    edge-on Z poles collapses those two into blobs.
    """
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    # any orthonormal pair spanning the base plane
    tmp = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    def proj(P):
        X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
        return np.stack([(Y - X)/SQ2, Z], axis=-1)

    phi = np.linspace(0, 2*np.pi, 241)
    rim = (np.sqrt(1.0 - a*a)*n[None, :]
           + a*(np.cos(phi)[:, None]*e1[None, :] + np.sin(phi)[:, None]*e2[None, :]))
    apex = (1.0 + h)*n
    R = proj(rim)
    A = proj(apex[None, :])[0]
    # Silhouette = convex hull of the rim plus the apex. It must be a REAL
    # hull: angle-sorting about the centroid traces the base ellipse and then
    # detours to the apex, which fills as an ellipse with a zero-width spike --
    # the X and Y cones came out as blobs that way, because their apex is only
    # ~2 base-radii from the base centre once foreshortened.
    from scipy.spatial import ConvexHull
    pts = np.vstack([R, A[None, :]])
    hull = pts[ConvexHull(pts).vertices]
    ax.fill(hull[:, 0], hull[:, 1], color=color, lw=0, zorder=zorder)
    ax.plot(R[:, 0], R[:, 1], '-', color=color, lw=0.8, zorder=zorder + 0.1)


def main():
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    # silhouette + poles
    tt = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(tt), np.sin(tt), '-', color='0.75', lw=1.0, zorder=1)
    for n in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)):
        pole_cone(ax, n)

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
    # clabel cuts a gap in each contour for its label. Keep the gaps, record
    # where the labels go, and remove the text so it is not rasterized with
    # the line drawing; it is redrawn as vector text on the final figure.
    labs = ax.clabel(cs, manual=LABEL_AT, inline=True, inline_spacing=6,
                     fmt='%g', fontsize=LABEL_FS)
    contour_labels = [(t.get_position(), t.get_text(), t.get_rotation())
                      for t in labs]
    for t in labs:
        t.remove()

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
    # Data coordinates of the canvas corners. The raster below is the whole
    # canvas, so it must be laid over exactly this extent: any other extent
    # stretches the sphere and moves it off the vector labels.
    inv = ax.transData.inverted()
    (ext_x0, ext_y0) = inv.transform((0.0, 0.0))
    (ext_x1, ext_y1) = inv.transform((fig.bbox.width, fig.bbox.height))
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
    ax2.imshow(shaded, extent=[ext_x0, ext_x1, ext_y0, ext_y1], zorder=1,
               interpolation='lanczos')
    for x, y, txt, ha, va in [(-1/SQ2 - 0.05, 0.09, '$X$', 'right', 'bottom'),
                              (+1/SQ2 + 0.05, 0.09, '$Y$', 'left', 'bottom'),
                              (0.06, 1.03, '$+Z$', 'left', 'bottom'),
                              (0.06, -1.06, '$-Z$', 'left', 'top')]:
        ax2.text(x, y, txt, fontsize=LABEL_FS, color='0.15', ha=ha, va=va,
                 zorder=5)
    for (x, y), txt, rot in contour_labels:
        ax2.text(x, y, txt, fontsize=LABEL_FS, color='0.3', ha='center',
                 va='center', rotation=rot, zorder=5)
    ax2.set_aspect('equal'); ax2.set_xlim(-1.25, 1.25); ax2.set_ylim(-1.2, 1.15)
    ax2.axis('off')
    plt.tight_layout()
    fig2.savefig('figs/indicator_sphere.pdf', dpi=DPI)
    fig2.savefig('repro/analytic/figs_explore/indicator_sphere_new.png', dpi=140)
    print('wrote figs/indicator_sphere.pdf (marble-shaded)')


if __name__ == '__main__':
    main()
