"""Structured O-grid mesh for the 6:1 prolate spheroid FULL BODY
(alpha=0 mean-flow anomaly discriminator; half-model: ogrid_spheroid.py).

Identical meridian half-plane 2D grid as ogrid_spheroid.py (same meridional
ladder, same first wall spacing / growth / far-field blend), but revolved
over the FULL circumference phi in [0, 2*pi):
    y = r sin(phi),  z = r cos(phi),
with PERIODIC CLOSURE BY NODE IDENTIFICATION at phi = 0 = 2*pi -- the
azimuthal index simply wraps, there is no seam boundary and there are NO
SYMMETRY SHEETS. Azimuthal interval count N_P = 2 x the half-model's
circumferential count, so every cell size matches the half-model level
exactly. Boundaries: fluid/wall + fluid/farfield only.

Purpose (agent-paper-review/2026-07-28-0105-spheroid-a0-meanflow.md): the
half-model alpha=0 RANS laminar BL carries a cell-locked ONE-AZIMUTHAL-CELL
staggering mode (du/u p2p 2.6% at 0.15 delta99 on an azimuthally uniform
flow) and an anomalously full profile (H 2.49-2.44 vs marched 2.56-2.63).
The half-model has symmetry sheets at y = 0 (both phi = 0 and phi = pi);
this mesh removes them: if the azimuthal mode and/or the H deficit survive
on the full body, the symmetry sheets are exonerated and the O-grid
truncation/low-Mach-dissipation candidates remain.

Pole treatment: identical to the half-model -- at the nose and tail the
whole j-column lies ON the axis; all phi copies are merged into a single
column and the first/last meridional intervals are emitted as PRISMS all
around (2 * N_P * N_J of them), wall/farfield pole caps as triangles.

Node/cell bookkeeping vs the half-model at the same level:
    nodes = (N_M - 1) * N_P * (N_J + 1) + 2 * (N_J + 1)
    cells = (N_M - 2) * N_P * N_J hexes + 2 * N_P * N_J prisms
          = exactly 2 x the half-model cell count (L1: 4,320,000).

Usage:
    python3 ogrid_spheroid_full.py L1 --h0 1.5e-6 \
        --out /local_data/qiqi/sa-ai/spheroid_meshes/mesh_re65full_L1.cgns
    (--h0 1.5e-6 matches the Re_L = 6.5e6/7.2e6 campaign ladder: L1 h0/L
     = 7.5e-7, byte-identical meridian to mesh_re65_L1.cgns)
    python3 ogrid_spheroid_full.py L1 --summary     # metrics only
"""
import argparse
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# meridian machinery + element/validity helpers: reuse the half-model's
# verbatim (the meridian half-plane grid must be IDENTICAL).
from ogrid_spheroid import (                                   # noqa: E402
    A, B, D, L, R_FF, LEVELS, solve_growth, meridian_points, march_meridian,
    plane_skew, _chunks, hex_min_jacobian, prism_corner_jacobians,
    hex_volumes_sum, prism_volumes, tri_area_sum, quad_area_sum)


def analytic_full():
    e = np.sqrt(1.0 - (B / A)**2)
    S_wall = 2.0 * np.pi * B**2 * (1.0 + (A / (B * e)) * np.arcsin(e))
    S_far = 4.0 * np.pi * R_FF**2
    V = (4.0 * np.pi / 3.0) * (R_FF**3 - A * B**2)
    return S_wall, S_far, V


def build(level, h0_l0, h0_fixed, out_cgns, summary_only):
    cfg = LEVELS[level]
    NM, NJ = cfg['N_M'], cfg['N_J']
    NP = 2 * cfg['N_C']                     # FULL circumference, 2x half count
    ds_pole = cfg['DS_POLE_OVER_D'] * D
    h0 = h0_l0 if h0_fixed else h0_l0 * 0.5**level

    growth = solve_growth(h0, NJ, R_FF)
    d = h0 * (growth**np.arange(NJ + 1) - 1) / (growth - 1)
    P, n, s, beta = meridian_points(NM, ds_pole)
    ds = np.diff(s)
    i_mid = NM // 2
    dphi = 2.0 * np.pi / NP                 # SAME arc as the half-model level
    arc_max = B * dphi
    r1 = P[1, 1]
    arc_pole = r1 * dphi

    print(f'== spheroid FULL-BODY O-grid L{level}: {NM} x {NP} x {NJ} '
          f'intervals (meridional x circumferential-FULL x normal)')
    print(f'   h0/L = {h0:.3e}  growth = {growth:.4f}  R_ff = {R_FF:.0f} L  '
          f'beta = {beta:.4f}  dphi = {np.degrees(dphi):.4f} deg')

    def summary():
        n_nodes = (NM - 1) * NP * (NJ + 1) + 2 * (NJ + 1)
        n_hex = (NM - 2) * NP * NJ
        n_pri = 2 * NP * NJ
        rows = [
            ('nodes', f'{n_nodes:,}'),
            ('cells (hex + prism)', f'{n_hex + n_pri:,} '
             f'({n_hex:,} + {n_pri:,})'),
            ('vs half-model cells', f'{(n_hex + n_pri) // 2:,} x 2'),
            ('meridional x circumf x normal', f'{NM} x {NP} x {NJ}'),
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
        print(f'-- mesh metrics, L{level} (full body) ' + '-' * w)
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

    # ---- 3D node ids: FULL revolve, periodic in phi, pole columns merged ----
    phi = 2.0 * np.pi * np.arange(NP) / NP          # phi = 2*pi node == phi = 0
    keep = np.ones((NM + 1, NP, NJ + 1), dtype=bool)
    keep[0, 1:, :] = False                          # merged nose pole column
    keep[-1, 1:, :] = False                         # merged tail pole column
    nid = (np.cumsum(keep.ravel()).reshape(NM + 1, NP, NJ + 1) - 1
           ).astype(np.int64)
    nid[0, :, :] = nid[0, 0, :]
    nid[-1, :, :] = nid[-1, 0, :]
    # periodic closure BY NODE IDENTIFICATION: azimuthal neighbor of the last
    # column is column 0 -- append a wrapped copy so the half-model's slicing
    # connectivity applies verbatim (no seam nodes, no seam boundary).
    nid_x = np.concatenate([nid, nid[:, :1, :]], axis=1)   # (NM+1, NP+1, NJ+1)

    X = np.broadcast_to(plane[:, None, :, 0], keep.shape)
    Rr = plane[:, None, :, 1]
    Y = Rr * np.sin(phi)[None, :, None]
    Z = Rr * np.cos(phi)[None, :, None]
    Y = np.broadcast_to(Y, keep.shape)
    Z = np.broadcast_to(Z, keep.shape)
    coords = np.column_stack([X[keep], Y[keep], Z[keep]])
    print(f'   nodes: {len(coords):,} (merged {2 * (NP - 1) * (NJ + 1):,} '
          f'pole duplicates; periodic wrap adds none)')

    # ---- elements (half-model connectivity on the wrapped index array) -----
    a_i, b_i = slice(1, NM - 1), slice(2, NM)
    hexes = np.stack([
        nid_x[a_i, :-1, :-1], nid_x[b_i, :-1, :-1], nid_x[b_i, 1:, :-1],
        nid_x[a_i, 1:, :-1],
        nid_x[a_i, :-1, 1:], nid_x[b_i, :-1, 1:], nid_x[b_i, 1:, 1:],
        nid_x[a_i, 1:, 1:]], axis=-1).reshape(-1, 8)
    p0 = coords[hexes[len(hexes) // 2]]
    if np.linalg.det(np.array([p0[1] - p0[0], p0[3] - p0[0],
                               p0[4] - p0[0]])) < 0:
        hexes = hexes[:, [4, 5, 6, 7, 0, 1, 2, 3]]
        print('   flipped hex orientation')

    def pole_prisms(pole_col, ring):                # ring: nid_x[1 or NM-1]
        bb = np.broadcast_to(pole_col[None, :-1], (NP, NJ))
        tt = np.broadcast_to(pole_col[None, 1:], (NP, NJ))
        pr = np.stack([bb, ring[:-1, :-1], ring[1:, :-1],
                       tt, ring[:-1, 1:], ring[1:, 1:]],
                      axis=-1).reshape(-1, 6)
        if np.median(prism_corner_jacobians(coords, pr[:64])) < 0:
            pr = pr[:, [0, 2, 1, 3, 5, 4]]
        return pr

    prisms = np.vstack([pole_prisms(nid[0, 0], nid_x[1]),
                        pole_prisms(nid[-1, 0], nid_x[NM - 1])])

    def shell(j):
        quads = np.stack([nid_x[a_i, :-1, j], nid_x[b_i, :-1, j],
                          nid_x[b_i, 1:, j], nid_x[a_i, 1:, j]],
                         axis=-1).reshape(-1, 4)
        t_nose = np.stack([np.broadcast_to(nid[0, 0, j], (NP,)),
                           nid_x[1, :-1, j], nid_x[1, 1:, j]],
                          axis=-1).reshape(-1, 3)
        t_tail = np.stack([np.broadcast_to(nid[-1, 0, j], (NP,)),
                           nid_x[NM - 1, 1:, j], nid_x[NM - 1, :-1, j]],
                          axis=-1).reshape(-1, 3)
        return quads, np.vstack([t_nose, t_tail])

    wall_q, wall_t = shell(0)
    far_q, far_t = shell(NJ)
    print(f'   hexes: {len(hexes):,}  prisms: {len(prisms):,}  wall: '
          f'{len(wall_q):,}q+{len(wall_t)}t  far: {len(far_q):,}q+'
          f'{len(far_t)}t  (no symmetry boundary)')

    # ---- validity checks ----------------------------------------------------
    S_wall, S_far, V_full = analytic_full()
    assert not np.isnan(coords).any(), 'NaN in coordinates'
    # every node id must be referenced-consistent: ids in range, wrap column
    # closed (max id == len(coords) - 1)
    assert hexes.max() < len(coords) and hexes.min() >= 0
    assert int(nid.max()) == len(coords) - 1, 'node id/coordinate mismatch'
    minj_h = hex_min_jacobian(coords, hexes)
    minj_p = prism_corner_jacobians(coords, prisms).min()
    vol_h, vmin_h = hex_volumes_sum(coords, hexes)
    vol_p = prism_volumes(coords, prisms)
    a_wall = quad_area_sum(coords, wall_q) + tri_area_sum(coords, wall_t)
    a_far = quad_area_sum(coords, far_q) + tri_area_sum(coords, far_t)
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
    print(f'   volume     {vol:.1f} vs analytic {V_full:.1f} '
          f'({(vol / V_full - 1) * 100:+.3f}%)  [closure: watertight only if '
          f'the periodic wrap is seamless]')
    ok = (minj_h > 0 and minj_p > 0 and vmin_h > 0 and vol_p.min() > 0
          and abs(a_wall / S_wall - 1) < 0.01 and abs(a_far / S_far - 1) < 0.01
          and abs(vol / V_full - 1) < 0.02)
    print(f'   validity: {"PASS" if ok else "FAIL"}')
    if not ok:
        sys.exit('validity check FAILED -- not writing mesh')

    summary()

    # ---- Gmsh 2.2 (chunked numpy writes, half-model conventions) ------------
    msh = os.path.splitext(out_cgns)[0] + '.msh'
    os.makedirs(os.path.dirname(os.path.abspath(msh)), exist_ok=True)
    with open(msh, 'w') as f:
        f.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
        f.write('$PhysicalNames\n3\n')
        f.write('2 2 "wall"\n2 4 "farfield"\n3 1 "fluid"\n')
        f.write('$EndPhysicalNames\n')
        f.write(f'$Nodes\n{len(coords)}\n')
        node_ids = np.arange(1, len(coords) + 1)
        for a0, b0 in _chunks(len(coords)):
            np.savetxt(f, np.column_stack([node_ids[a0:b0], coords[a0:b0]]),
                       fmt='%d %.16g %.16g %.16g')
        f.write('$EndNodes\n')
        groups = [(2, 2, wall_t), (3, 2, wall_q),
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
                         'use 1.5e-6 for the Re_L=6.5e6/7.2e6 ladder')
    ap.add_argument('--h0-fixed', action='store_true',
                    help='use --h0 verbatim at any level (no ladder halving)')
    ap.add_argument('--out', default=None,
                    help='output CGNS path (default spheroid/mesh_full_<level>.cgns)')
    ap.add_argument('--summary', action='store_true',
                    help='print the mesh-metrics table only; no mesh files')
    args = ap.parse_args()
    level = int(args.level.lstrip('Ll'))
    out = args.out or os.path.join(HERE, f'mesh_full_L{level}.cgns')
    build(level, args.h0, args.h0_fixed, out, args.summary)


if __name__ == '__main__':
    main()
