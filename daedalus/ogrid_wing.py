"""Stacked-O-grid L0 mesh for the Daedalus half-wing.

Topology: one 2D O-grid (N_s surface points x N_j+1 layers) per spanwise
station, identical (i, j) dimensions everywhere, connected into hexahedra.
  - Stations 0..K-1: blended DAE sections (wing_geometry.SectionFamily).
  - Station K (the pinch, y = half-span): the section collapses to its camber
    slit (thickness 0); mirror surface points coincide and are MERGED, so the
    tip closes as a knife edge (the [K-1, K] wall quads are the closeout).
  - Stations K+1..: the same slit sections, but the slit is interior (sewn by
    the same node merge, no wall quads emitted) -- the classic collapsed-O-grid
    extension of the far field beyond the tip.
All planes share one far-field circle (center at the quarter-chord, radius
R_FF), so the outer boundary is a cylinder with an end-cap disk at y_max and
the symmetry plane at y = 0.

March per plane: wall-normal geometric extrusion (smoothed normals, window
growing with j) through the boundary-layer block, then straight blend to the
far-circle anchor assigned by normalized arclength.

Output: Gmsh 2.2 ASCII (all hexes + boundary quads) -> flow360gmshtocgns.
Boundaries: fluid/wing (wall), fluid/symmetry (root plane), fluid/farfield
(outer cylinder + end cap).

Usage: python3 ogrid_wing.py [out_prefix]
"""
import sys
import numpy as np
from wing_geometry import SectionFamily, chord, HALF_SPAN, C_ROOT, XQC

# ---- L0 resolution knobs ----------------------------------------------------
N_PER_SIDE = 96          # -> N_s = 192 surface points per section
N_J = 64                 # wall-normal layers
H0 = 3.7e-5              # first-layer height [m] (~y+ <= 1 at Re ~ 5e5)
R_FF = 100.0 * C_ROOT    # far-field radius [m], shared cylinder
N_WING = 40              # spanwise intervals root -> last finite section
N_BEYOND = 16            # slit stations beyond the tip (geometric growth)
CENTER = np.array([XQC, 0.0])


def solve_growth(h0, n, target):
    """growth g with h0*(g^n - 1)/(g - 1) = target."""
    lo, hi = 1.01, 2.0
    for _ in range(80):
        g = 0.5 * (lo + hi)
        if h0 * (g**n - 1) / (g - 1) < target:
            lo = g
        else:
            hi = g
    return 0.5 * (lo + hi)


BLEND_D1 = 0.06   # [m] blend start: pure normal extrusion below this distance
BLEND_D2 = 0.60   # [m] blend end: pure radial-to-anchor beyond this distance


def blend_weight(d):
    """Smoothstep in log distance from BLEND_D1 to BLEND_D2. Blending the
    normal-extrusion grid into the radial-anchor grid over many layers keeps
    the direction change C1-smooth; a hard switch at one layer dumps the whole
    direction mismatch into a single sheared cell layer."""
    x = (np.log(np.maximum(d, 1e-30)) - np.log(BLEND_D1)) \
        / (np.log(BLEND_D2) - np.log(BLEND_D1))
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def anchors_from_directions(P, dirs, d_ref=1.0):
    """Far-circle anchors aligned with the extrusion rays: take the angular
    position (about CENTER) of each column's ray at distance d_ref, monotonize
    along the loop, and rescale the sweep to one full turn. Anchors assigned by
    arclength instead put the radial grid laterally offset from the extrusion
    grid, and the blend band absorbs that offset as cell shear."""
    ring = P + d_ref * dirs
    ang = np.unwrap(np.arctan2(ring[:, 1] - CENTER[1], ring[:, 0] - CENTER[0]))
    n = len(ang)
    min_inc = 0.15 * 2.0 * np.pi / n
    for i in range(1, n):
        if ang[i] < ang[i - 1] + min_inc:
            ang[i] = ang[i - 1] + min_inc
    ang = ang[0] + (ang - ang[0]) * (2.0 * np.pi * (1.0 - 1.0 / n)) \
        / max(ang[-1] - ang[0], 1e-12)
    return CENTER + R_FF * np.column_stack([np.cos(ang), np.sin(ang)])


def march_plane(P, d, rot_sign, j_bl):
    """O-grid plane from closed contour P (N_s, 2). d: cumulative layer
    distances (N_j+1). Returns (N_s, N_j+1, 2)."""
    N = len(P)
    # tangents (loop-aware), outward normals via fixed rotation sign
    t = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    tl = np.linalg.norm(t, axis=1)
    t = t / np.maximum(tl, 1e-30)[:, None]
    n_raw = rot_sign * np.column_stack([t[:, 1], -t[:, 0]])
    # Sharp-TE fan: across the TE wrap (i = 0), surface normals jump from the
    # lower side (-z-ish) to the upper (+z-ish). Replace the directions in a
    # small window with an angular sweep through the downstream bisector so
    # the wrap column cannot fold.
    m = 4
    a_up = np.arctan2(n_raw[m, 1], n_raw[m, 0])
    a_lo = np.arctan2(n_raw[-m, 1], n_raw[-m, 0])
    delta = (a_up - a_lo + np.pi) % (2.0 * np.pi) - np.pi   # short arc
    for step, idx in enumerate(range(-m, m + 1)):
        ang = a_lo + delta * step / (2 * m)
        n_raw[idx] = (np.cos(ang), np.sin(ang))

    def smooth(v, w):
        if w < 1:
            return v
        k = 2 * w + 1
        ker = np.ones(k) / k
        out = np.empty_like(v)
        for c in range(2):
            out[:, c] = np.convolve(np.concatenate([v[-w:, c], v[:, c], v[:w, c]]),
                                    ker, mode='valid')
        return out / np.maximum(np.linalg.norm(out, axis=1), 1e-30)[:, None]

    # anchors follow the fully-smoothed normal field so the radial grid is
    # nearly parallel to the extrusion grid through the blend band
    A = anchors_from_directions(P, smooth(n_raw, N // 8))

    nodes = np.empty((N, len(d), 2))
    nodes[:, 0] = P
    for j in range(1, len(d)):
        w = int(round((j - 1) * (N // 8) / max(j_bl, 1)))  # 0 at the wall
        nj = smooth(n_raw, min(w, N // 8))
        E = P + d[j] * nj                          # normal extrusion
        R = P + (d[j] / d[-1]) * (A - P)           # radial ray to the anchor
        s = blend_weight(d[j])
        nodes[:, j] = (1.0 - s) * E + s * R
    nodes[:, -1] = A                               # land exactly on the circle
    return nodes


def slit_plane(x0, x1, d, n_s):
    """O-grid around the straight slit [x0, x1] at z = 0: rays from cosine-
    spaced slit points along elliptic-family directions
      u(th) ~ (sinh(mu*)cos(th), cosh(mu*)sin(th)),
    which rotate monotonically (fold-free) and turn into +-x at the slit ends
    (no 4 nm confocal crowding: distances along each ray are the shared d_j).
    Beyond d_bl the rays blend linearly to far-circle anchors at angle th.
    Mirror points (i, N_s - i) on the slit coincide exactly for the seam merge."""
    a = 0.5 * (x1 - x0)
    xm = 0.5 * (x0 + x1)
    th = 2.0 * np.pi * np.arange(n_s) / n_s          # i=0 at TE (x1)
    # UNIFORM x-spacing along the slit (not cosine): the slit is interior, and
    # cosine clustering puts the last point ~1e-4 m from the seam end, making
    # the end-fan cells razor slivers (corner sine 0.03) that blow up the
    # SA-AI kinematic indicators. Mirror points still coincide exactly.
    half = n_s // 2
    i_arr = np.arange(n_s)
    frac = np.where(i_arr <= half, i_arr / half, (n_s - i_arr) / half)
    P = np.column_stack([x1 - (x1 - x0) * frac, np.zeros(n_s)])
    # Ray directions from the elliptic family AT THE POSITION's ellipse
    # parameter (arccos of the normalized slit coordinate), not at the index
    # parameter: with uniform positions an index-based fan keeps near-end rays
    # almost parallel to the end ray and they cross (inverted cells for the
    # first ~10 layers). The elliptic fan never self-crosses.
    MU = 0.05
    xi = np.clip((P[:, 0] - xm) / a, -1.0, 1.0)
    th_pos = np.where(i_arr <= half, np.arccos(xi), 2.0 * np.pi - np.arccos(xi))
    u = np.column_stack([np.sinh(MU) * np.cos(th_pos), np.cosh(MU) * np.sin(th_pos)])
    u /= np.linalg.norm(u, axis=1)[:, None]
    # End rays stay ON the slit axis (the exact elliptic fan, provably
    # fold-free). The end nodes' least-squares rank problem is fixed by the
    # alternating z-offset of the beyond-tip planes in build(), not by
    # bending rays (any bend either folds the fan or goes concave).
    A = anchors_from_directions(P, u)
    nodes = np.empty((n_s, len(d), 2))
    nodes[:, 0] = P
    for j in range(1, len(d)):
        E = P + d[j] * u                           # ray-fan extrusion
        R = P + (d[j] / d[-1]) * (A - P)           # radial ray to the anchor
        s = blend_weight(d[j])
        nodes[:, j] = (1.0 - s) * E + s * R
    nodes[:, -1] = A
    return nodes


def build(out_prefix='wing_ogrid_L0'):
    fam = SectionFamily(N_PER_SIDE)
    N_s = 2 * N_PER_SIDE

    growth = solve_growth(H0, N_J, R_FF)
    d = H0 * (growth**np.arange(N_J + 1) - 1) / (growth - 1)
    j_bl = int(np.argmin(np.abs(d - 0.12)))      # BL block ~0.12 m
    print(f'growth={growth:.4f}  d_bl={d[j_bl]:.3f} m (j_bl={j_bl})  '
          f'd_max={d[-1]:.1f} m')

    # spanwise stations: one smooth tip-clustered sequence over [0, HALF_SPAN]
    # whose LAST station is the pinch plane, so the closeout interval continues
    # the wing clustering instead of jumping (no separate tip gap).
    t = np.linspace(0, 1, N_WING + 2)
    ys = list(HALF_SPAN * np.sin(0.5 * np.pi * t))
    K = len(ys) - 1                               # index of the pinch plane
    dy0 = ys[-1] - ys[-2]                         # closeout interval (~13 mm)
    # beyond-tip: N_BEYOND geometric steps covering R_FF exactly
    lo, hi = 1.05, 3.0
    for _ in range(60):
        g = 0.5 * (lo + hi)
        total = dy0 * g * (g**N_BEYOND - 1) / (g - 1)
        lo, hi = (g, hi) if total < R_FF else (lo, g)
    dy = dy0
    for _ in range(N_BEYOND):
        dy *= g
        ys.append(ys[-1] + dy)
    ys = np.array(ys)
    print(f'closeout interval {dy0*1e3:.1f} mm; beyond-tip growth {g:.3f}')
    print(f'{K} wing planes + pinch + {N_BEYOND} beyond-tip; '
          f'y_max={ys[-1]:.1f} m (tip at {HALF_SPAN} m)')

    # rotation sign from the root section: normal at topmost point must be +z
    P0 = fam.contour(0.0)
    tt = np.roll(P0, -1, axis=0) - np.roll(P0, 1, axis=0)
    itop = int(np.argmax(P0[:, 1]))
    rot_sign = 1.0 if -tt[itop, 0] > 0 else -1.0   # n_z = rot_sign * (-t_x)

    # straight slit for the pinch and beyond: chord line of the tip section
    c_tip = chord(1.0)
    x0_slit, x1_slit = XQC - 0.25 * c_tip, XQC + 0.75 * c_tip

    h0_eff = H0
    planes = np.empty((len(ys), N_s, N_J + 1, 2))
    for k, y in enumerate(ys):
        if k < K:
            P = fam.contour(min(y / HALF_SPAN, 1.0))
            planes[k] = march_plane(P, d, rot_sign, j_bl)
        else:
            # The seam is interior, not a wall: its first spacing does not
            # need h0. Grow it with the spanwise spacing (constant shear
            # between adjacent planes), capped at 2 mm -- wall-fine seam
            # cells make |omega| enormous on the wake sheet and blow up the
            # SA-AI kinematic indicators.
            m = k - K
            h_m = h0_eff if m == 0 else min(H0 * g**m, 2.0e-3)
            g_m = solve_growth(h_m, N_J, R_FF)
            d_m = h_m * (g_m**np.arange(N_J + 1) - 1) / (g_m - 1)
            planes[k] = slit_plane(x0_slit, x1_slit, d_m, N_s)
            if m > 0:
                # alternating z-offset: gives the seam-END nodes (whose ray
                # and slit edges are collinear in x) first-order z-offset
                # spanwise neighbors with NON-collinear directions, making
                # their least-squares gradient stencils rank 3. A LINEAR
                # tilt does not work: it keeps the stencil affinely planar.
                planes[k][:, :, 1] += (1 if m % 2 else -1) * 0.1 * (ys[k] - ys[k - 1])
                # parabolic z-arc of the seam (zero AT the ends, decaying
                # away from it): the END-CAP plane's seam-end node has
                # spanwise neighbors on one side only, so the tilt alone
                # leaves its stencil rank-2; the arc gives every seam node
                # an off-axis IN-PLANE slit neighbor.
                L = x1_slit - x0_slit
                xw = planes[k][:, :, 0]
                bump = np.clip(4.0 * (xw - x0_slit) * (x1_slit - xw) / L**2, 0.0, None)
                decay = np.exp(-d_m / 0.05)[None, :]
                planes[k][:, :, 1] += 0.01 * L * bump * decay
        if k % 10 == 0:
            print(f'  plane {k}/{len(ys)-1} marched', flush=True)

    def layer_skew(pl):
        """Per-layer worst corner sine (1 = orthogonal, 0 = degenerate)."""
        ei = np.roll(pl, -1, axis=0) - pl               # i-edges (N, J+1, 2)
        ej = pl[:, 1:] - pl[:, :-1]                     # j-edges (N, J, 2)
        worst = []
        for j in range(pl.shape[1] - 1):
            a, b = ei[:, j], ej[:, j]
            cr = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
            sin = cr / np.maximum(np.linalg.norm(a, axis=1)
                                  * np.linalg.norm(b, axis=1), 1e-300)
            worst.append(sin.min())
        return np.array(worst)
    for name, pl in (('root', planes[0]), ('slit', planes[K])):
        w = layer_skew(pl)
        print(f'  {name} plane worst corner sine by layer: min={w.min():.3f} '
              f'at j={int(w.argmin())} (1=orthogonal); '
              f'layers below 0.3: {(w < 0.3).sum()}')

    # ---- global node ids with slit merging (vectorized) ---------------------
    NP = N_J + 1
    NK = len(ys)
    half = N_s // 2
    keep = np.ones((NK, N_s, NP), dtype=bool)
    keep[K:, half + 1:, 0] = False               # merged slit wall-ring nodes
    nid = (np.cumsum(keep.ravel()).reshape(NK, N_s, NP) - 1).astype(np.int64)
    ii_m = np.arange(half + 1, N_s)
    nid[K:, ii_m, 0] = nid[K:, N_s - ii_m, 0]    # sew the slit wall ring
    coords = np.column_stack([
        planes[:, :, :, 0][keep],
        np.broadcast_to(ys[:, None, None], keep.shape)[keep],
        planes[:, :, :, 1][keep]])
    print(f'nodes: {len(coords)} (merged '
          f'{(NK - K) * (half - 1)} slit duplicates)')

    # ---- elements (vectorized, same (k, i, j) ordering as before) -----------
    nid1 = np.roll(nid, -1, axis=1)              # i+1 with wrap
    def corners(kk0, kk1):
        return [nid[kk0, :, :-1], nid1[kk0, :, :-1], nid1[kk0, :, 1:],
                nid[kk0, :, 1:], nid[kk1, :, :-1], nid1[kk1, :, :-1],
                nid1[kk1, :, 1:], nid[kk1, :, 1:]]
    hexes = np.stack(corners(slice(0, NK - 1), slice(1, NK)),
                     axis=-1).reshape(-1, 8)
    p0 = coords[hexes[len(hexes) // 2]]
    if np.linalg.det(np.array([p0[1] - p0[0], p0[3] - p0[0],
                               p0[4] - p0[0]])) < 0:
        hexes = hexes[:, [4, 5, 6, 7, 0, 1, 2, 3]]
        print('flipped hex orientation')

    wall = np.stack([nid[:K, :, 0], nid1[:K, :, 0],
                     nid1[1:K + 1, :, 0], nid[1:K + 1, :, 0]],
                    axis=-1).reshape(-1, 4)
    far_ring = np.stack([nid[:-1, :, N_J], nid1[:-1, :, N_J],
                         nid1[1:, :, N_J], nid[1:, :, N_J]],
                        axis=-1).reshape(-1, 4)
    sym = np.stack([nid[0, :, :-1], nid1[0, :, :-1],
                    nid1[0, :, 1:], nid[0, :, 1:]], axis=-1).reshape(-1, 4)
    far_cap = np.stack([nid[-1, :, :-1], nid1[-1, :, :-1],
                        nid1[-1, :, 1:], nid[-1, :, 1:]],
                       axis=-1).reshape(-1, 4)
    far = np.vstack([far_ring, far_cap])
    print(f'hexes: {len(hexes)}  wall quads: {len(wall)}  '
          f'sym: {len(sym)}  far: {len(far)}')

    # ---- Gmsh 2.2 (chunked numpy writes) -------------------------------------
    msh = f'{out_prefix}.msh'
    with open(msh, 'w') as f:
        f.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
        f.write('$PhysicalNames\n4\n')
        f.write('2 2 "wing"\n2 3 "symmetry"\n2 4 "farfield"\n3 1 "fluid"\n')
        f.write('$EndPhysicalNames\n')
        f.write(f'$Nodes\n{len(coords)}\n')
        node_ids = np.arange(1, len(coords) + 1)
        for a, b in _chunks(len(coords)):
            np.savetxt(f, np.column_stack([node_ids[a:b], coords[a:b]]),
                       fmt='%d %.16g %.16g %.16g')
        f.write('$EndNodes\n')
        ne = len(wall) + len(sym) + len(far) + len(hexes)
        f.write(f'$Elements\n{ne}\n')
        eid = 1
        for tag, quads in ((2, wall), (3, sym), (4, far)):
            n = len(quads)
            arr = np.empty((n, 9), dtype=np.int64)
            arr[:, 0] = np.arange(eid, eid + n)
            arr[:, 1] = 3; arr[:, 2] = 2; arr[:, 3] = tag; arr[:, 4] = tag
            arr[:, 5:] = quads + 1
            for a, b in _chunks(n):
                np.savetxt(f, arr[a:b], fmt='%d')
            eid += n
        n = len(hexes)
        arr = np.empty((n, 13), dtype=np.int64)
        arr[:, 0] = np.arange(eid, eid + n)
        arr[:, 1] = 5; arr[:, 2] = 2; arr[:, 3] = 1; arr[:, 4] = 1
        arr[:, 5:] = hexes + 1
        for a, b in _chunks(n):
            np.savetxt(f, arr[a:b], fmt='%d')
        f.write('$EndElements\n')
    print(f'wrote {msh}')

    # quality: min corner jacobian over all 8 corners of each hex
    H = np.asarray(hexes)
    p = coords[H]                                 # (n, 8, 3)
    # corner (a; b,c,e) triplets of the gmsh hex, right-handed
    CORNERS = [(0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
               (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3)]
    minj = np.full(len(H), np.inf)
    for (a, b, c, e) in CORNERS:
        d1 = p[:, b] - p[:, a]; d2 = p[:, c] - p[:, a]; d3 = p[:, e] - p[:, a]
        det = np.einsum('ij,ij->i', np.cross(d1, d2), d3)
        minj = np.minimum(minj, det)
    neg = np.where(minj <= 0)[0]
    print(f'min corner jacobian: {minj.min():.3e}; '
          f'{len(neg)} of {len(H)} hexes have a non-positive corner')
    if len(neg):
        # decode (k, i, j) from append order: idx = (k*N_s + i)*N_J + j
        ks = neg // (N_s * N_J); ii = (neg // N_J) % N_s; jj = neg % N_J
        import collections
        print('  by span plane k:', dict(collections.Counter(ks.tolist())))
        print('  i histogram (surface index):',
              dict(collections.Counter((ii // 8 * 8).tolist())))
        print('  j histogram (layer):', dict(collections.Counter(jj.tolist())))
    return msh


def _chunks(n, size=2_000_000):
    for a in range(0, n, size):
        yield a, min(a + size, n)


# Refinement ladder: every length scale halves per level (surface spacing,
# first-layer height, growth-ratio offset via the solve, spanwise spacing).
LEVELS = {
    0: dict(N_PER_SIDE=96, N_J=64, H0=3.7e-5, N_WING=40, N_BEYOND=16),
    1: dict(N_PER_SIDE=192, N_J=128, H0=1.85e-5, N_WING=80, N_BEYOND=20),
    2: dict(N_PER_SIDE=384, N_J=256, H0=9.25e-6, N_WING=160, N_BEYOND=24),
}


if __name__ == '__main__':
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    globals().update(LEVELS[lvl])
    out = sys.argv[2] if len(sys.argv) > 2 else f'wing_ogrid_L{lvl}'
    build(out)
