"""UNSTRUCTURED mesh family for the 6:1 prolate spheroid -- the Daedalus
cavity-family pipeline (watertight triangulated skin -> Flynn360, the
production in-house mesher: anisotropic BL prisms + Tetgen glue + auto
octree far field) applied to the transition-validation spheroid.

Geometry and orientation are IDENTICAL to ogrid_spheroid.py: analytic
prolate spheroid, L = 1 grid unit, D = L/6, centered at the origin, body
axis = x.  This family is a FULL-CIRCUMFERENCE body (user directive: no
y = 0 symmetry plane; the O-grid family is the half-model).  The azimuth
convention matches the O-grid's revolve:
    y = r sin(phi),  z = r cos(phi),   phi in [0, 2*pi),
so phi = 0 is +z (the LEEWARD side at positive alpha; incidence vector in
the x-z plane) -- alpha > 0 cases on this family are directly comparable
to the O-grid's phi convention.

Surface triangulation (latitude rings):
  - the meridian point distribution IS the O-grid ladder's
    (ogrid_spheroid.meridian_points: solved uniform+cosine blend with the
    pole spacing ds_pole = tab:daemesh LE ladder x D), so pole clustering
    matches the O-grid level exactly;
  - each interior meridian station becomes a full ring of nodes; the
    circumferential spacing target is min(ARC_MAX, local meridian
    spacing): ARC_MAX = B*pi/N_C is the O-grid's mid-body circumferential
    arc at the same level, and the isotropy cap keeps the pole fans
    equilateral at ds_pole scale (the O-grid instead lets its
    circumferential arc shrink linearly with r toward the poles);
  - rings are stitched with the classic two-pointer angular walk
    (watertight by construction, verified: every edge shared by exactly
    two triangles with opposite traversal, Euler V - E + F = 2), poles
    are triangle fans;
  - all triangle normals oriented outward via the ellipsoid gradient
    (convex body: per-triangle test is exact).

Prism layers: firstLayerThickness / growthRate match the O-grid re65/re72
ladder AT THE SAME LEVEL (build log runlogs/spheroid_mesh65_*.log):
    L0: h0 = 1.5e-6 L, g = 1.2969;  L1: h0 = 7.5e-7 L, g = 1.1927;
    L2: h0 = 3.75e-7 L, g = 1.1325.
numBoundaryLayers is left at the mesher default (-1 = grow until the
stack reaches the local isotropic scale), exactly like the wing cavity
family.  Far field: Flynn360 "auto" octree with relativeSize = 60 (box
faces at +-30 L, the O-grid's far-field distance; the O-grid's is a
sphere of radius 30 L).

Usage:
    python3 unstruct_spheroid.py L1 [--h0 X --growth G] [--stl-only]
        [--out /local_data/qiqi/sa-ai/spheroid_meshes/mesh_unstr_L1.cgns]
Writes <out-dir>/unstr_L<k>.stl, Flynn360_L<k>.json, and (unless
--stl-only) runs Flynn360 -> CGNS; mesher log lands next to the CGNS.
"""
import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ogrid_spheroid import (A, B, D, L, LEVELS, meridian_points,  # noqa: E402
                            tri_area_sum, analytic)

# h0 / growth per level = the O-grid re65/re72 ladder's realized values
# (runlogs/spheroid_mesh65_L0.log, ..._L1L2.log)
BL_LADDER = {0: (1.5e-6, 1.2969), 1: (7.5e-7, 1.1927), 2: (3.75e-7, 1.1325)}
RELATIVE_SIZE = 60.0          # auto-octree box side / body length -> +-30 L
FLYNN = '/home/qiqi/flexcompute/compute/install/release/bin/Flynn360'
RLIB = '/home/qiqi/flexcompute/compute/install/release/lib'


# ---- surface triangulation ---------------------------------------------------
def build_rings(level):
    cfg = LEVELS[level]
    NM, NC = cfg['N_M'], cfg['N_C']
    ds_pole = cfg['DS_POLE_OVER_D'] * D
    arc_max = B * np.pi / NC                    # O-grid mid-body circumf arc
    P, _, s, beta = meridian_points(NM, ds_pole)
    ds = np.diff(s)

    pts = [np.array([-A, 0.0, 0.0])]
    rings = []                                  # (start_index, n_i, phi array)
    for i in range(1, NM):
        x, r = P[i]
        dsl = 0.5 * (ds[i - 1] + ds[i])
        dc = min(arc_max, dsl)
        ni = max(6, int(np.ceil(2.0 * np.pi * r / dc)))
        off = 0.5 * (2.0 * np.pi / ni) * (i % 2)          # stagger
        phi = off + 2.0 * np.pi * np.arange(ni) / ni
        rings.append((len(pts), ni, phi))
        pts.extend(np.column_stack([np.full(ni, x),
                                    r * np.sin(phi), r * np.cos(phi)]))
    i_tail = len(pts)
    pts.append(np.array([A, 0.0, 0.0]))
    return np.array(pts), rings, i_tail, s, ds, arc_max, beta


def stitch(idx1, phi1, idx2, phi2):
    """Triangle strip between two rings: two-pointer walk over the unwrapped
    angles (each ring closed by repeating its first node at phi+2*pi)."""
    a = np.concatenate([idx1, [idx1[0]]])
    pa = np.concatenate([phi1, [phi1[0] + 2.0 * np.pi]])
    b = np.concatenate([idx2, [idx2[0]]])
    pb = np.concatenate([phi2, [phi2[0] + 2.0 * np.pi]])
    tris, i, j = [], 0, 0
    while i < len(a) - 1 or j < len(b) - 1:
        take_a = (j == len(b) - 1) or (i < len(a) - 1 and pa[i + 1] <= pb[j + 1])
        if take_a:
            tris.append((a[i], a[i + 1], b[j]))
            i += 1
        else:
            tris.append((b[j + 1], b[j], a[i]))
            j += 1
    return tris


def triangulate(level):
    pts, rings, i_tail, s, ds, arc_max, beta = build_rings(level)
    tris = []
    # nose fan
    i0, n0, _ = rings[0]
    for k in range(n0):
        tris.append((0, i0 + k, i0 + (k + 1) % n0))
    # ring stitches
    for (ia, na, pa), (ib, nb, pb) in zip(rings[:-1], rings[1:]):
        tris.extend(stitch(np.arange(ia, ia + na), pa,
                           np.arange(ib, ib + nb), pb))
    # tail fan
    il, nl, _ = rings[-1]
    for k in range(nl):
        tris.append((i_tail, il + (k + 1) % nl, il + k))
    tris = np.array(tris, dtype=np.int64)

    # outward orientation via the ellipsoid gradient at the centroid
    p = pts[tris]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    c = p.mean(axis=1)
    grad = np.column_stack([c[:, 0] / A**2, c[:, 1] / B**2, c[:, 2] / B**2])
    flip = np.einsum('ij,ij->i', nrm, grad) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return pts, tris, rings, s, ds, arc_max, beta


# ---- validity ------------------------------------------------------------------
def check_watertight_oriented(pts, tris):
    """Every directed edge must appear exactly once and its reverse exactly
    once (closed, consistently oriented, manifold)."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    key = e[:, 0] * len(pts) + e[:, 1]
    rkey = e[:, 1] * len(pts) + e[:, 0]
    ks, cnt = np.unique(key, return_counts=True)
    assert cnt.max() == 1, 'duplicated directed edge (non-manifold/inverted)'
    assert np.array_equal(ks, np.unique(rkey)), 'unpaired edge (not watertight)'
    V, F, E = len(pts), len(tris), len(ks) // 2
    assert V - E + F == 2, f'Euler check failed: V-E+F = {V - E + F}'
    return E


def tri_quality(pts, tris):
    p = pts[tris]
    e0 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    e1 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    e2 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0],
                                         p[:, 2] - p[:, 0]), axis=1)
    lmax = np.maximum(e0, np.maximum(e1, e2))
    # min altitude / max edge (1 = equilateral-ish scale, small = sliver)
    q = 2.0 * area / (lmax * lmax)
    ar = lmax / np.minimum(e0, np.minimum(e1, e2))
    return area, q, ar


# ---- STL ----------------------------------------------------------------------
def write_stl_binary(path, pts, tris):
    p = pts[tris].astype(np.float32)
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-30)
    rec = np.zeros((len(tris),), dtype=np.dtype([('n', '<f4', 3),
                                                 ('v', '<f4', (3, 3)),
                                                 ('attr', '<u2')]))
    rec['n'] = n
    rec['v'] = p
    with open(path, 'wb') as f:
        f.write(b'unstruct spheroid (Flynn360 cavity family)'.ljust(80, b' '))
        f.write(struct.pack('<I', len(tris)))
        f.write(rec.tobytes())


# ---- build --------------------------------------------------------------------
def build(level, h0, growth, out_cgns, stl_only, re_l=7.2e6):
    pts, tris, rings, s, ds, arc_max, beta = triangulate(level)
    E = check_watertight_oriented(pts, tris)
    area, q, ar = tri_quality(pts, tris)
    S_wall_half = analytic()[0]
    S_wall = 2.0 * S_wall_half
    a_tot = area.sum()
    n_ring = np.array([r[1] for r in rings])
    i_mid = len(rings) // 2

    # y+ estimate at the run condition (laminar Blasius mid-body + a
    # turbulent-aft bound), u_tau/U = sqrt(Cf/2), y+ = h0 * Re_L * u_tau/U
    cf_lam = 0.664 / np.sqrt(re_l * 0.5)
    cf_turb = 0.0576 / (re_l * 0.9)**0.2
    yp_lam = h0 * re_l * np.sqrt(cf_lam / 2.0)
    yp_turb = h0 * re_l * np.sqrt(cf_turb / 2.0)
    # prism stack forecast to the isotropic stop at the mid-body size
    n_stop = int(np.ceil(np.log(arc_max / h0) / np.log(growth)))
    h_stack = h0 * (growth**n_stop - 1.0) / (growth - 1.0)

    print(f'== spheroid UNSTRUCTURED (cavity family) L{level}: full body')
    rows = [
        ('surface nodes / triangles', f'{len(pts):,} / {len(tris):,}'),
        ('meridian stations (rings)', f'{len(rings)} (+2 poles)'),
        ('ring size min/mid/max', f'{n_ring.min()} / {n_ring[i_mid]} / '
                                  f'{n_ring.max()}'),
        ('ds @ poles /L', f'{ds[0]:.3e} / {ds[-1]:.3e}'),
        ('ds @ mid-body /L', f'{ds[len(ds) // 2]:.3e}'),
        ('circumf arc cap /L', f'{arc_max:.3e} (O-grid mid-body arc)'),
        ('tri quality 2A/lmax^2 min/p1', f'{q.min():.3f} / '
                                         f'{np.percentile(q, 1):.3f}'),
        ('tri aspect max/p99', f'{ar.max():.2f} / {np.percentile(ar, 99):.2f}'),
        ('wall area vs analytic', f'{a_tot:.6f} vs {S_wall:.6f} '
                                  f'({(a_tot / S_wall - 1) * 100:+.3f}%)'),
        ('first layer h0 /L', f'{h0:.3e}'),
        ('BL growth rate', f'{growth:.4f}'),
        ('y+ @ h0 (lam mid / turb aft)', f'{yp_lam:.3f} / {yp_turb:.3f}'),
        ('prism stack forecast', f'~{n_stop} layers to isotropic '
                                 f'{arc_max:.2e}, height ~{h_stack:.3e} L'),
        ('far field', f'auto octree, relativeSize {RELATIVE_SIZE:.0f} '
                      f'(box faces at +-{RELATIVE_SIZE / 2:.0f} L)'),
    ]
    w = max(len(r[0]) for r in rows)
    for k, v in rows:
        print(f'  {k:<{w}}  {v}')
    assert not np.isnan(pts).any()
    assert abs(a_tot / S_wall - 1) < 0.01, 'wall area off by > 1%'
    print(f'  watertight + consistently oriented: PASS ({E:,} edges, '
          f'V-E+F=2)')

    out_dir = os.path.dirname(os.path.abspath(out_cgns))
    os.makedirs(out_dir, exist_ok=True)
    stl = os.path.join(out_dir, f'unstr_L{level}.stl')
    write_stl_binary(stl, pts, tris)
    print(f'  wrote {stl} ({os.path.getsize(stl):,} bytes)')

    cfgp = os.path.join(out_dir, f'Flynn360_L{level}.json')
    with open(cfgp, 'w') as f:
        json.dump({'farfield': {'type': 'auto', 'relativeSize': RELATIVE_SIZE},
                   'volume': {'firstLayerThickness': h0,
                              'boundaryLayerGrowthRate': growth},
                   'surface': {'boundaryIds': []}}, f, indent=4)
    print(f'  wrote {cfgp}')
    if stl_only:
        return

    env = dict(os.environ, LD_LIBRARY_PATH=RLIB, OMP_NUM_THREADS='16',
               OMPI_COMM_WORLD_LOCAL_RANK='0', OMPI_COMM_WORLD_RANK='0',
               OMPI_COMM_WORLD_SIZE='1')
    log = os.path.splitext(out_cgns)[0] + '_flynn360.log'
    with open(log, 'w') as lf:
        rc = subprocess.run([FLYNN, '-g', stl, '-m', cfgp, '-p', '16',
                             '-o', os.path.abspath(out_cgns)],
                            cwd=out_dir, env=env, stdout=lf,
                            stderr=subprocess.STDOUT).returncode
    print(f'  Flynn360 rc = {rc}; log {log}')
    if rc != 0:
        sys.exit(f'Flynn360 FAILED (rc {rc}) -- see {log}')
    print(f'  wrote {out_cgns} ({os.path.getsize(out_cgns):,} bytes)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('level', help='L0 | L1 | L2 (or 0/1/2)')
    ap.add_argument('--h0', type=float, default=None,
                    help='first layer thickness /L (default: O-grid '
                         're65/re72 ladder value at this level)')
    ap.add_argument('--growth', type=float, default=None,
                    help='BL growth rate (default: ladder value)')
    ap.add_argument('--out', default=None,
                    help='output CGNS (default /local_data/qiqi/sa-ai/'
                         'spheroid_meshes/mesh_unstr_L<k>.cgns)')
    ap.add_argument('--stl-only', action='store_true',
                    help='write STL + config only, skip the mesher')
    args = ap.parse_args()
    level = int(args.level.lstrip('Ll'))
    h0_d, g_d = BL_LADDER[level]
    out = args.out or (f'/local_data/qiqi/sa-ai/spheroid_meshes/'
                       f'mesh_unstr_L{level}.cgns')
    build(level, args.h0 or h0_d, args.growth or g_d, out, args.stl_only)


if __name__ == '__main__':
    main()
