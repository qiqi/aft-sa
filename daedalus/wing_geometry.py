"""Daedalus wing geometry: DAE-11/21/31 sections blended along an approximate
planform. Shared by both L0 mesh generators (stacked O-grid and Flynn360
cavity/tet).

Geometry model (documented approximations, replace with measured data when
available):
  - half-span s = 17.07 m, root chord 0.917 m (Daedalus-88: b = 34.14 m,
    S = 30.84 m^2, AR = 37.8; mean chord 0.903 m ~= root chord, so the
    planform is nearly rectangular with a short tapered tip panel).
  - chord: constant 0.917 m to eta = 0.88, linear taper to 0.35 m at the tip
    (gives S ~= 30.1 m^2).
  - sections: DAE-11 to eta = 0.30, blend to DAE-21 by 0.60, blend to DAE-31
    by 0.90, DAE-31 outboard. Blending is y-wise at matched cosine-clustered
    x stations (all three share LE at x=0, sharp TE at x=1).
  - no twist; sections aligned on the quarter-chord line (unswept), z = 0
    chord plane. Span axis = y, chordwise = x, thickness = z.

Coordinate convention matches the 2D cases: chord along +x, z up, y spanwise
(root plane y = 0).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HALF_SPAN = 17.07        # m
C_ROOT = 0.917           # m
C_TIP = 0.35             # m
TAPER_START = 0.88       # eta where the tip taper begins
XQC = 0.25 * C_ROOT      # quarter-chord x position (all sections aligned here)

# (eta of pure section, name); blends are linear between consecutive entries.
SECTION_STATIONS = [(0.00, 'dae11'), (0.30, 'dae11'), (0.60, 'dae21'),
                    (0.90, 'dae31'), (1.00, 'dae31')]


def _load_upper_lower(name):
    """Selig loop (TE->upper->LE->lower->TE) -> (xu, zu), (xl, zl), LE->TE."""
    pts = np.array([[float(t) for t in ln.split()]
                    for ln in open(f'{HERE}/{name}_selig.dat').readlines()[1:]])
    ile = int(np.argmin(pts[:, 0]))
    up = pts[:ile + 1][::-1]     # LE -> TE
    lo = pts[ile:]               # LE -> TE
    return up, lo


def _resample_half(half, xs):
    """Monotone-x half surface resampled at xs (PCHIP keeps the nose shape)."""
    from scipy.interpolate import PchipInterpolator
    x, z = half[:, 0], half[:, 1]
    keep = np.concatenate([[True], np.diff(x) > 1e-12])
    return PchipInterpolator(x[keep], z[keep])(xs)


class SectionFamily:
    """All three DAE sections resampled on one shared cosine x grid."""

    def __init__(self, n_per_side=96):
        # Cosine clustering at LE and TE; identical for upper and lower so the
        # zero-thickness slit (tip pinch) has mirror points exactly coincident.
        th = np.linspace(0.0, np.pi, n_per_side + 1)
        self.xs = 0.5 * (1.0 - np.cos(th))          # 0 (LE) .. 1 (TE)
        self.upper = {}
        self.lower = {}
        for name in ('dae11', 'dae21', 'dae31'):
            up, lo = _load_upper_lower(name)
            self.upper[name] = _resample_half(up, self.xs)
            self.lower[name] = _resample_half(lo, self.xs)

    def _blend(self, eta):
        """(zu, zl) at unit chord for the blended section at eta."""
        st = SECTION_STATIONS
        for (e0, n0), (e1, n1) in zip(st, st[1:]):
            if eta <= e1 or (e1 == st[-1][0]):
                if eta <= e0:
                    w = 0.0
                elif eta >= e1:
                    w = 1.0
                else:
                    w = (eta - e0) / (e1 - e0)
                zu = (1 - w) * self.upper[n0] + w * self.upper[n1]
                zl = (1 - w) * self.lower[n0] + w * self.lower[n1]
                if eta <= e1:
                    return zu, zl
        return zu, zl

    def contour(self, eta, thickness_scale=1.0):
        """Closed physical contour at eta, walked TE -> upper -> LE -> lower -> TE
        (first point TE, not repeated at the end). Returns (N, 2) array (x, z),
        N = 2*n_per_side. thickness_scale=0 collapses to the camber slit."""
        c = chord(eta)
        zu, zl = self._blend(eta)
        cam = 0.5 * (zu + zl)
        zu = cam + thickness_scale * (zu - cam)
        zl = cam + thickness_scale * (zl - cam)
        xs = self.xs
        # TE(=x=1) .. upper .. LE, then LE+1 .. lower .. TE-1 (loop closes TE->TE)
        xw = np.concatenate([xs[::-1], xs[1:-1]])
        zw = np.concatenate([zu[::-1], zl[1:-1]])
        x_le = XQC - 0.25 * c
        return np.column_stack([x_le + c * xw, c * zw])


def chord(eta):
    eta = np.asarray(eta, dtype=float)
    c = np.where(eta <= TAPER_START, C_ROOT,
                 C_ROOT + (C_TIP - C_ROOT) * (eta - TAPER_START) / (1 - TAPER_START))
    return float(c) if c.ndim == 0 else c


def wing_surface_stl(path, n_per_side=64, n_span=80, full_span=True):
    """Watertight wing surface triangulation as binary STL (for Flynn360).
    Sections at cosine-clustered span stations; tip closed by collapsing the
    last ring to the camber slit over the final station gap (knife edge) and
    fanning the slit. Root: mirrored to -y when full_span (watertight body,
    no symmetry-plane special casing)."""
    fam = SectionFamily(n_per_side)
    # Span stations clustered toward the tip; last station is the pinch (slit).
    t = np.linspace(0, 1, n_span + 1)
    etas = 1.0 - (1.0 - t**1.5) * (1.0 - 0.0)     # mild clustering at tip
    etas = np.sin(0.5 * np.pi * t)                # cosine-type: dense at tip
    ys = HALF_SPAN * etas
    rings = []
    for k, eta in enumerate(etas):
        ts = 1.0 if k < n_span else 0.0           # pinch only the last ring
        rings.append(fam.contour(eta, thickness_scale=ts))
    N = rings[0].shape[0]

    tris = []                                      # (a, b, c) as xyz triples

    def emit_strip(r0, y0, r1, y1):
        for i in range(N):
            j = (i + 1) % N
            a = (r0[i, 0], y0, r0[i, 1]); b = (r0[j, 0], y0, r0[j, 1])
            c = (r1[i, 0], y1, r1[i, 1]); d = (r1[j, 0], y1, r1[j, 1])
            tris.append((a, d, b)); tris.append((a, c, d))

    for k in range(n_span):
        emit_strip(rings[k], ys[k], rings[k + 1], ys[k + 1])
    # Tip slit: the pinched ring is degenerate (mirror points coincide), the
    # strip triangles above already close the knife edge watertight; the slit
    # itself has zero area, nothing more to emit.
    if full_span:
        mirrored = [t[::-1] for t in
                    [tuple((x, -y, z) for (x, y, z) in tri) for tri in tris]]
        tris += mirrored
    else:
        # close the root ring with a fan (cap on the y=0 plane)
        r0 = rings[0]
        ctr = (float(r0[:, 0].mean()), 0.0, float(r0[:, 1].mean()))
        for i in range(N):
            j = (i + 1) % N
            tris.append(((r0[j, 0], 0.0, r0[j, 1]), (r0[i, 0], 0.0, r0[i, 1]), ctr))

    import struct
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(tris)))
        for (a, b, c) in tris:
            u = np.subtract(b, a); v = np.subtract(c, a)
            n = np.cross(u, v); nn = np.linalg.norm(n)
            n = n / nn if nn > 0 else np.array([0., 0., 1.])
            f.write(struct.pack('<3f', *n))
            for p in (a, b, c):
                f.write(struct.pack('<3f', *p))
            f.write(struct.pack('<H', 0))
    return len(tris)


if __name__ == '__main__':
    fam = SectionFamily()
    for eta in (0.0, 0.45, 0.75, 1.0):
        r = fam.contour(eta)
        th = r[:, 1].max() - r[:, 1].min()
        print(f'eta={eta:4.2f} chord={chord(eta):.3f} m  N={len(r)}  '
              f'max thickness={th:.4f} m ({100*th/chord(eta):.1f}% c)')
    n = wing_surface_stl(f'{HERE}/wing_L0.stl')
    print(f'wing_L0.stl: {n} triangles')
