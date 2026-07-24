"""Structured O-grid mesh for the 6:1 prolate spheroid half-model
(transition-validation campaign), modeled on daedalus/ogrid_wing.py.

Geometry: analytic prolate spheroid, length L = 1.0 grid units, diameter
D = L/6, CENTERED AT THE ORIGIN: x in [-L/2, +L/2], surface
    x^2/A^2 + r^2/B^2 = 1,   A = L/2, B = L/12,  r = sqrt(y^2 + z^2).

Topology: meridional (i) x circumferential (p) x wall-normal (j).
One 2D grid is built in the meridian half-plane (x, r): the meridian arc
from the nose pole (theta = 0, x = -A) to the tail pole (theta = pi,
x = +A), extruded wall-normal (geometric growth, ogrid_wing's
solve_growth) and blended -- with ogrid_wing's log-smoothstep
blend_weight -- into radial rays landing on far-field anchors assigned
by ray angle (anchors_from_directions logic, open-arc version). That
half-plane grid is revolved over phi in [0, pi]:
    y = r sin(phi),  z = r cos(phi),
so the SYMMETRY PLANE IS y = 0 (both phi = 0 and phi = pi sheets); the
body axis is x and the incidence vector lies in the x-z plane
(phi = 0 is +z, the leeward side at positive alpha; phi = pi is -z).

Far field: a SPHERE of radius R_FF = 30 L centered at the body center
(the revolved far half-circle), closed at the two on-axis points.

Pole degeneracy (the wing-tip-pinch analogue): at the nose and tail the
surface normal is axial, so the whole j-column lies ON the axis and is
independent of phi -- all phi copies are MERGED into a single column
(same node-merge machinery as the wing's slit seam).  The cells of the
first/last meridional interval therefore collapse in phi and are emitted
as PRISMS (Gmsh type 6), not degenerate hexes, so every element has
strictly positive corner Jacobians; the adjacent wall/farfield faces are
emitted as triangles.

Refinement ladder (Daedalus convention -- every spacing halves per
level): L0 150x40x60, L1 300x80x90, L2 600x160x130 (meridional x
circumferential-half x normal intervals) ~ 0.37M / 2.2M / 12.6M nodes.
Meridional spacing is a solved uniform+cosine blend clustered at both
poles so the pole spacing matches the paper's around-the-section ladder
(tab:daemesh LE column) scaled by the local cross dimension D = L/6:
    ds_pole = 4.5e-3 D / 2.25e-3 D / 1.125e-3 D  at L0/L1/L2.
First wall spacing: --h0 gives the L0 value (default 5e-6 L for the
Re_L = 1.5e6 ladder; use --h0 1.5e-6 for the Re_L = 6.5e6 cases) and is
halved per level like every other spacing; pass --h0-fixed to use the
given value verbatim at any level.

Output: Gmsh 2.2 ASCII -> flow360gmshtocgns -> CGNS (HDF5), boundaries
fluid/wall (spheroid), fluid/symmetry (both y=0 half-plane sheets),
fluid/farfield (outer sphere).

Usage:
    python3 ogrid_spheroid.py L1 [--h0 5e-6] [--out spheroid/mesh_L1.cgns]
    python3 ogrid_spheroid.py L1 --summary        # metrics only, no files
"""
import argparse
import os
import sys
import numpy as np

# ---- geometry (grid units: L = 1) -------------------------------------------
L = 1.0
A = 0.5 * L                  # semi-major axis (body axis = x)
B = L / 12.0                 # semi-minor axis (max radius); diameter D = L/6
D = L / 6.0
R_FF = 30.0 * L              # far-field sphere radius, centered at the origin

# Refinement ladder: intervals meridional / circumferential(half) / normal,
# and the pole meridional spacing target (units of D, paper tab:daemesh LE
# ladder: 4.5e-3 / 2.25e-3 / 1.125e-3 -- exact halving).
LEVELS = {
    0: dict(N_M=150, N_C=40,  N_J=60,  DS_POLE_OVER_D=4.5e-3),
    1: dict(N_M=300, N_C=80,  N_J=90,  DS_POLE_OVER_D=2.25e-3),
    2: dict(N_M=600, N_C=160, N_J=130, DS_POLE_OVER_D=1.125e-3),
}

BLEND_D1 = 0.05 * L   # blend start: pure normal extrusion below this distance
BLEND_D2 = 0.50 * L   # blend end: pure radial-to-anchor beyond this distance


def solve_growth(h0, n, target):
    """growth g with h0*(g^n - 1)/(g - 1) = target.  (ogrid_wing.py)"""
    lo, hi = 1.01, 2.0
    for _ in range(80):
        g = 0.5 * (lo + hi)
        if h0 * (g**n - 1) / (g - 1) < target:
            lo = g
        else:
            hi = g
    return 0.5 * (lo + hi)


def blend_weight(d):
    """Smoothstep in log distance from BLEND_D1 to BLEND_D2 (ogrid_wing.py):
    blending the normal-extrusion grid into the radial-anchor grid over many
    layers keeps the direction change C1-smooth."""
    x = (np.log(np.maximum(d, 1e-30)) - np.log(BLEND_D1)) \
        / (np.log(BLEND_D2) - np.log(BLEND_D1))
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ---- meridian discretization -------------------------------------------------
def meridian_arclength(n_dense=200_001):
    """Dense (theta, s) table along the half-ellipse meridian
    P(theta) = (-A cos theta, B sin theta), theta in [0, pi]."""
    th = np.linspace(0.0, np.pi, n_dense)
    x, r = -A * np.cos(th), B * np.sin(th)
    seg = np.hypot(np.diff(x), np.diff(r))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return th, s


def meridian_points(n_m, ds_pole):
    """n_m+1 meridian points, symmetric uniform+cosine arclength blend
    s(u) = S[(1-beta) u + beta (1-cos(pi u))/2], beta solved so the first
    (= last) interval equals ds_pole."""
    th_d, s_d = meridian_arclength()
    S = s_d[-1]
    u = np.arange(n_m + 1) / n_m

    def first_ds(beta):
        return S * ((1 - beta) * u[1] + beta * 0.5 * (1 - np.cos(np.pi * u[1])))

    if first_ds(1.0) > ds_pole:
        beta = 1.0                       # even pure cosine is coarser: take it
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            beta = 0.5 * (lo + hi)
            lo, hi = (beta, hi) if first_ds(beta) > ds_pole else (lo, beta)
        beta = 0.5 * (lo + hi)
    s = S * ((1 - beta) * u + beta * 0.5 * (1 - np.cos(np.pi * u)))
    th = np.interp(s, s_d, th_d)
    th[0], th[-1] = 0.0, np.pi           # poles exact
    P = np.column_stack([-A * np.cos(th), B * np.sin(th)])
    P[0] = (-A, 0.0)
    P[-1] = (A, 0.0)
    # outward normals of the ellipse: grad(x^2/A^2 + r^2/B^2)
    n = np.column_stack([-np.cos(th) / A, np.sin(th) / B])
    n /= np.linalg.norm(n, axis=1)[:, None]
    n[0] = (-1.0, 0.0)                   # axial at the poles, exactly
    n[-1] = (1.0, 0.0)
    return P, n, s, beta


def anchors_open_arc(P, dirs, n_m, d_ref=1.0):
    """Far-circle anchors aligned with the extrusion rays (open-arc version
    of ogrid_wing.anchors_from_directions): angular position about the origin
    of each column's ray at distance d_ref, monotonized (decreasing pi -> 0)
    and pinned exactly to the axis at both ends."""
    ring = P + d_ref * dirs
    ang = np.arctan2(ring[:, 1], ring[:, 0])       # in [0, pi], r >= 0
    ang[0], ang[-1] = np.pi, 0.0
    min_dec = 0.15 * np.pi / n_m
    for i in range(1, len(ang)):                   # enforce strict decrease
        if ang[i] > ang[i - 1] - min_dec:
            ang[i] = ang[i - 1] - min_dec
    ang = (ang - ang[-1]) * np.pi / (ang[0] - ang[-1])   # exact [pi, 0] sweep
    return R_FF * np.column_stack([np.cos(ang), np.sin(ang)])


def march_meridian(P, n, d, n_m):
    """2D half-plane grid (n_m+1, n_j+1, 2): wall-normal geometric extrusion
    blended into radial rays to the far anchors (ogrid_wing.march_plane; no
    normal smoothing needed -- the meridian is convex with no corners)."""
    Af = anchors_open_arc(P, n, n_m)
    nodes = np.empty((len(P), len(d), 2))
    nodes[:, 0] = P
    for j in range(1, len(d)):
        E = P + d[j] * n                         # normal extrusion
        R = P + (d[j] / d[-1]) * (Af - P)        # radial ray to the anchor
        s = blend_weight(d[j])
        nodes[:, j] = (1.0 - s) * E + s * R
    nodes[:, -1] = Af                            # land exactly on the circle
    nodes[0, :, 1] = 0.0                         # pole columns exactly on axis
    nodes[-1, :, 1] = 0.0
    return nodes


def plane_skew(pl):
    """Per-layer worst corner sine of the 2D half-plane grid (open in i);
    1 = orthogonal, 0 = degenerate.  (ogrid_wing layer_skew, open-curve.)"""
    ei = pl[1:] - pl[:-1]                        # i-edges (N, J+1, 2)
    ej = pl[:, 1:] - pl[:, :-1]                  # j-edges (N+1, J, 2)
    worst = []
    for j in range(pl.shape[1] - 1):
        a, b = ei[:, j], ej[:-1, j]
        cr = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        sin = cr / np.maximum(np.linalg.norm(a, axis=1)
                              * np.linalg.norm(b, axis=1), 1e-300)
        worst.append(sin.min())
    return np.array(worst)


# ---- analytic references ------------------------------------------------------
def analytic():
    e = np.sqrt(1.0 - (B / A)**2)
    S_wall_half = 0.5 * 2.0 * np.pi * B**2 * (1.0 + (A / (B * e)) * np.arcsin(e))
    S_far_half = 2.0 * np.pi * R_FF**2
    S_sym = np.pi * R_FF**2 - np.pi * A * B         # both sheets together
    V_half = (2.0 * np.pi / 3.0) * (R_FF**3 - A * B**2)
    return S_wall_half, S_far_half, S_sym, V_half


# ---- element helpers ----------------------------------------------------------
def _chunks(n, size=2_000_000):
    for a in range(0, n, size):
        yield a, min(a + size, n)


HEX_CORNERS = [(0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
               (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3)]


def hex_min_jacobian(coords, hexes):
    minj = np.inf
    for a0, b0 in _chunks(len(hexes)):
        p = coords[hexes[a0:b0]]
        mj = np.full(b0 - a0, np.inf)
        for (a, b, c, e) in HEX_CORNERS:
            d1 = p[:, b] - p[:, a]
            d2 = p[:, c] - p[:, a]
            d3 = p[:, e] - p[:, a]
            det = np.einsum('ij,ij->i', np.cross(d1, d2), d3)
            mj = np.minimum(mj, det)
        minj = min(minj, mj.min())
    return minj


def prism_corner_jacobians(coords, prisms):
    """Min corner Jacobian over the 6 corners of each prism (bottom tri
    0,1,2; top tri 3,4,5; k connected to k+3)."""
    p = coords[prisms]
    mj = np.full(len(prisms), np.inf)
    for k in range(3):
        k1, k2 = (k + 1) % 3, (k + 2) % 3
        det = np.einsum('ij,ij->i',
                        np.cross(p[:, k1] - p[:, k], p[:, k2] - p[:, k]),
                        p[:, k + 3] - p[:, k])
        mj = np.minimum(mj, det)
        det = np.einsum('ij,ij->i',
                        np.cross(p[:, k2 + 3] - p[:, k + 3],
                                 p[:, k1 + 3] - p[:, k + 3]),
                        p[:, k] - p[:, k + 3])
        mj = np.minimum(mj, det)
    return mj


def hex_volumes_sum(coords, hexes):
    """Sum of hex volumes via the 6-tet (long-diagonal 0-6) decomposition."""
    TETS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
            (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    tot = 0.0
    vmin = np.inf
    for a0, b0 in _chunks(len(hexes)):
        p = coords[hexes[a0:b0]]
        v = np.zeros(b0 - a0)
        for (a, b, c, e) in TETS:
            v += np.einsum('ij,ij->i',
                           np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a]),
                           p[:, e] - p[:, a]) / 6.0
        tot += v.sum()
        vmin = min(vmin, v.min())
    return tot, vmin


def prism_volumes(coords, prisms):
    p = coords[prisms]
    TETS = [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]
    v = np.zeros(len(prisms))
    for (a, b, c, e) in TETS:
        v += np.einsum('ij,ij->i',
                       np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a]),
                       p[:, e] - p[:, a]) / 6.0
    return v


def tri_area_sum(coords, tris):
    p = coords[tris]
    return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0],
                                         p[:, 2] - p[:, 0]), axis=1).sum()


def quad_area_sum(coords, quads):
    tot = 0.0
    for a0, b0 in _chunks(len(quads)):
        p = coords[quads[a0:b0]]
        tot += 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0],
                                             p[:, 2] - p[:, 0]), axis=1).sum()
        tot += 0.5 * np.linalg.norm(np.cross(p[:, 2] - p[:, 0],
                                             p[:, 3] - p[:, 0]), axis=1).sum()
    return tot


# ---- build --------------------------------------------------------------------
def build(level, h0_l0, h0_fixed, out_cgns, summary_only):
    cfg = LEVELS[level]
    NM, NC, NJ = cfg['N_M'], cfg['N_C'], cfg['N_J']
    ds_pole = cfg['DS_POLE_OVER_D'] * D
    h0 = h0_l0 if h0_fixed else h0_l0 * 0.5**level

    growth = solve_growth(h0, NJ, R_FF)
    d = h0 * (growth**np.arange(NJ + 1) - 1) / (growth - 1)
    P, n, s, beta = meridian_points(NM, ds_pole)
    ds = np.diff(s)
    i_mid = NM // 2
    # circumferential arc spacings
    dphi = np.pi / NC
    arc_max = B * dphi                          # at max radius (x = 0)
    r1 = P[1, 1]                                # first off-pole ring radius
    arc_pole = r1 * dphi

    print(f'== spheroid O-grid L{level}: {NM} x {NC} x {NJ} intervals '
          f'(meridional x circumferential-half x normal)')
    print(f'   h0/L = {h0:.3e}  growth = {growth:.4f}  R_ff = {R_FF:.0f} L  '
          f'beta = {beta:.4f}')

    def summary():
        n_nodes = (NM - 1) * (NC + 1) * (NJ + 1) + 2 * (NJ + 1)
        n_hex = (NM - 2) * NC * NJ
        n_pri = 2 * NC * NJ
        rows = [
            ('nodes', f'{n_nodes:,}'),
            ('cells (hex + prism)', f'{n_hex + n_pri:,} '
             f'({n_hex:,} + {n_pri:,})'),
            ('meridional x circumf x normal', f'{NM} x {NC} x {NJ}'),
            ('ds @ nose pole /L (/D)', f'{ds[0]:.3e} ({ds[0] / D:.3e})'),
            ('ds @ mid-body /L (/D)', f'{ds[i_mid]:.3e} ({ds[i_mid] / D:.3e})'),
            ('ds @ tail pole /L (/D)', f'{ds[-1]:.3e} ({ds[-1] / D:.3e})'),
            ('circumf arc @ mid-body /L', f'{arc_max:.3e}'),
            ('circumf arc @ 1st pole ring /L', f'{arc_pole:.3e}'),
            ('first wall spacing h0/L', f'{h0:.3e}'),
            ('wall-normal growth ratio', f'{growth:.4f}'),
            ('cell height @ d=0.01L /L', f'{np.interp(0.01, d[:-1], np.diff(d)):.3e}'),
            ('far-field radius /L', f'{R_FF:.0f} (sphere)'),
        ]
        w = max(len(r[0]) for r in rows)
        print(f'-- mesh metrics, L{level} ' + '-' * (w + 14))
        for k, v in rows:
            print(f'  {k:<{w}}  {v}')

    if summary_only:
        summary()
        return

    plane = march_meridian(P, n, d, NM)
    w = plane_skew(plane)
    print(f'   half-plane worst corner sine by layer: min={w.min():.3f} at '
          f'j={int(w.argmin())} (1=orthogonal); layers below 0.3: '
          f'{(w < 0.3).sum()}')

    # ---- 3D node ids with pole-column merging (wing slit-merge machinery) ----
    phi = np.pi * np.arange(NC + 1) / NC
    keep = np.ones((NM + 1, NC + 1, NJ + 1), dtype=bool)
    keep[0, 1:, :] = False                       # merged nose pole column
    keep[-1, 1:, :] = False                      # merged tail pole column
    nid = (np.cumsum(keep.ravel()).reshape(NM + 1, NC + 1, NJ + 1) - 1
           ).astype(np.int64)
    nid[0, :, :] = nid[0, 0, :]
    nid[-1, :, :] = nid[-1, 0, :]

    X = np.broadcast_to(plane[:, None, :, 0], keep.shape)
    Rr = plane[:, None, :, 1]
    Y = Rr * np.sin(phi)[None, :, None]
    Z = Rr * np.cos(phi)[None, :, None]
    Y = np.broadcast_to(Y, keep.shape)
    Z = np.broadcast_to(Z, keep.shape)
    coords = np.column_stack([X[keep], Y[keep], Z[keep]])
    print(f'   nodes: {len(coords):,} (merged {2 * NC * (NJ + 1):,} pole '
          f'duplicates)')

    # ---- elements -------------------------------------------------------------
    a_i, b_i = slice(1, NM - 1), slice(2, NM)    # interior meridional intervals
    hexes = np.stack([
        nid[a_i, :-1, :-1], nid[b_i, :-1, :-1], nid[b_i, 1:, :-1],
        nid[a_i, 1:, :-1],
        nid[a_i, :-1, 1:], nid[b_i, :-1, 1:], nid[b_i, 1:, 1:],
        nid[a_i, 1:, 1:]], axis=-1).reshape(-1, 8)
    p0 = coords[hexes[len(hexes) // 2]]
    if np.linalg.det(np.array([p0[1] - p0[0], p0[3] - p0[0],
                               p0[4] - p0[0]])) < 0:
        hexes = hexes[:, [4, 5, 6, 7, 0, 1, 2, 3]]
        print('   flipped hex orientation')

    def pole_prisms(pole_col, ring):             # ring: nid[1 or NM-1]
        bb = np.broadcast_to(pole_col[None, :-1], (NC, NJ))
        tt = np.broadcast_to(pole_col[None, 1:], (NC, NJ))
        pr = np.stack([bb, ring[:-1, :-1], ring[1:, :-1],
                       tt, ring[:-1, 1:], ring[1:, 1:]],
                      axis=-1).reshape(-1, 6)
        if np.median(prism_corner_jacobians(coords, pr[:64])) < 0:
            pr = pr[:, [0, 2, 1, 3, 5, 4]]
        return pr

    prisms = np.vstack([pole_prisms(nid[0, 0], nid[1]),
                        pole_prisms(nid[-1, 0], nid[NM - 1])])

    def shell(j):
        quads = np.stack([nid[a_i, :-1, j], nid[b_i, :-1, j],
                          nid[b_i, 1:, j], nid[a_i, 1:, j]],
                         axis=-1).reshape(-1, 4)
        t_nose = np.stack([np.broadcast_to(nid[0, 0, j], (NC,)),
                           nid[1, :-1, j], nid[1, 1:, j]],
                          axis=-1).reshape(-1, 3)
        t_tail = np.stack([np.broadcast_to(nid[-1, 0, j], (NC,)),
                           nid[NM - 1, 1:, j], nid[NM - 1, :-1, j]],
                          axis=-1).reshape(-1, 3)
        return quads, np.vstack([t_nose, t_tail])

    wall_q, wall_t = shell(0)
    far_q, far_t = shell(NJ)
    sym = np.vstack([
        np.stack([nid[:-1, p, :-1], nid[1:, p, :-1],
                  nid[1:, p, 1:], nid[:-1, p, 1:]], axis=-1).reshape(-1, 4)
        for p in (0, NC)])
    print(f'   hexes: {len(hexes):,}  prisms: {len(prisms):,}  wall: '
          f'{len(wall_q):,}q+{len(wall_t)}t  sym: {len(sym):,}q  far: '
          f'{len(far_q):,}q+{len(far_t)}t')

    # ---- validity checks --------------------------------------------------------
    S_wall, S_far, S_sym, V_half = analytic()
    assert not np.isnan(coords).any(), 'NaN in coordinates'
    minj_h = hex_min_jacobian(coords, hexes)
    minj_p = prism_corner_jacobians(coords, prisms).min()
    vol_h, vmin_h = hex_volumes_sum(coords, hexes)
    vol_p = prism_volumes(coords, prisms)
    a_wall = quad_area_sum(coords, wall_q) + tri_area_sum(coords, wall_t)
    a_far = quad_area_sum(coords, far_q) + tri_area_sum(coords, far_t)
    a_sym = quad_area_sum(coords, sym)
    vol = vol_h + vol_p.sum()
    print('   -- validity --')
    print(f'   no NaNs; min hex corner jac = {minj_h:.3e}, '
          f'min prism corner jac = {minj_p:.3e} (both must be > 0)')
    print(f'   min hex volume = {vmin_h:.3e}, min prism volume = '
          f'{vol_p.min():.3e}')
    print(f'   wall area  {a_wall:.6f} vs analytic {S_wall:.6f} '
          f'({(a_wall / S_wall - 1) * 100:+.3f}%)')
    print(f'   farfield   {a_far:.1f} vs analytic {S_far:.1f} '
          f'({(a_far / S_far - 1) * 100:+.3f}%)  [sphere closed at poles]')
    print(f'   symmetry   {a_sym:.1f} vs analytic {S_sym:.1f} '
          f'({(a_sym / S_sym - 1) * 100:+.3f}%)')
    print(f'   volume     {vol:.1f} vs analytic {V_half:.1f} '
          f'({(vol / V_half - 1) * 100:+.3f}%)  [closure]')
    ok = (minj_h > 0 and minj_p > 0 and vmin_h > 0 and vol_p.min() > 0
          and abs(a_wall / S_wall - 1) < 0.01 and abs(a_far / S_far - 1) < 0.01
          and abs(vol / V_half - 1) < 0.02)
    print(f'   validity: {"PASS" if ok else "FAIL"}')
    if not ok:
        sys.exit('validity check FAILED -- not writing mesh')

    summary()

    # ---- Gmsh 2.2 (chunked numpy writes, ogrid_wing conventions) ---------------
    msh = os.path.splitext(out_cgns)[0] + '.msh'
    os.makedirs(os.path.dirname(os.path.abspath(msh)), exist_ok=True)
    with open(msh, 'w') as f:
        f.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
        f.write('$PhysicalNames\n4\n')
        f.write('2 2 "wall"\n2 3 "symmetry"\n2 4 "farfield"\n3 1 "fluid"\n')
        f.write('$EndPhysicalNames\n')
        f.write(f'$Nodes\n{len(coords)}\n')
        node_ids = np.arange(1, len(coords) + 1)
        for a0, b0 in _chunks(len(coords)):
            np.savetxt(f, np.column_stack([node_ids[a0:b0], coords[a0:b0]]),
                       fmt='%d %.16g %.16g %.16g')
        f.write('$EndNodes\n')
        groups = [(2, 2, np.vstack([wall_t])), (3, 2, wall_q),
                  (2, 3, None), (3, 3, sym),
                  (2, 4, far_t), (3, 4, far_q),
                  (6, 1, prisms), (5, 1, hexes)]
        groups = [(ty, tag, el) for (ty, tag, el) in groups
                  if el is not None and len(el)]
        ne = sum(len(el) for _, _, el in groups)
        f.write(f'$Elements\n{ne}\n')
        eid = 1
        for ty, tag, el in groups:
            m = len(el)
            arr = np.empty((m, 5 + el.shape[1]), dtype=np.int64)
            arr[:, 0] = np.arange(eid, eid + m)
            arr[:, 1] = ty
            arr[:, 2] = 2
            arr[:, 3] = tag
            arr[:, 4] = tag
            arr[:, 5:] = el + 1
            for a0, b0 in _chunks(m):
                np.savetxt(f, arr[a0:b0], fmt='%d')
            eid += m
        f.write('$EndElements\n')
    print(f'   wrote {msh}')

    # ---- convert to CGNS (flow360gmshtocgns via flexfoil rans helpers) ---------
    try:
        sys.path.insert(0, '/home/qiqi/flexcompute/flexfoil/rans')
        from rans import mesh as _mesh
        from rans.env import make_env
        env, find = make_env()
        _mesh.gmsh_to_cgns(msh, out_cgns, find('flow360gmshtocgns'), env)
        print(f'   wrote {out_cgns} ({os.path.getsize(out_cgns):,} bytes)')
        os.remove(msh)
    except Exception as exc:                      # keep the .msh for retry
        print(f'   CGNS conversion failed ({exc}); .msh kept at {msh}')
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('level', help='L0 | L1 | L2 (or 0/1/2)')
    ap.add_argument('--h0', type=float, default=5e-6,
                    help='first wall spacing /L at L0 (halved per level); '
                         'default 5e-6 (Re_L=1.5e6), use 1.5e-6 for Re_L=6.5e6')
    ap.add_argument('--h0-fixed', action='store_true',
                    help='use --h0 verbatim at any level (no ladder halving)')
    ap.add_argument('--out', default=None,
                    help='output CGNS path (default spheroid/mesh_<level>.cgns)')
    ap.add_argument('--summary', action='store_true',
                    help='print the mesh-metrics table only; no mesh files')
    args = ap.parse_args()
    level = int(args.level.lstrip('Ll'))
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, f'mesh_L{level}.cgns')
    build(level, args.h0, args.h0_fixed, out, args.summary)


if __name__ == '__main__':
    main()
