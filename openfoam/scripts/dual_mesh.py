#!/usr/bin/env python3
"""Median-dual mesh converter: quasi-2D gmsh triangle-prism mesh -> OpenFOAM polyMesh.

Reads a gmsh 2.2 ASCII quasi-2D mesh (2D triangulation extruded one cell in y),
builds the median dual of the 2D triangulation (one dual polygon per primal
node), verifies exact area conservation, extrudes the dual polygons over the
original span, and writes an OpenFOAM polyMesh (ASCII).

2D triangulation source: the surface triangles of the symmetry patch lying on
the smaller-y node plane. Boundary edges (farfield / nlf0416) are derived from
the quad faces of those physical surfaces: each quad bridges the two span
planes; its two nodes on the smaller-y plane form one 2D boundary edge.

Dual polygons:
  interior node: alternating midpoints of incident edges / centroids of
                 incident triangles, ordered by a topological fan walk, CCW.
  boundary node: same fan from one boundary-edge midpoint to the other,
                 closed through the node itself.

Internal dual faces (2D): one per primal edge:
  interior primal edge (tris T1,T2): polyline c(T1) - m(edge) - c(T2)
  boundary primal edge (tri T):      polyline m(edge) - c(T)
Boundary dual faces (2D): per boundary primal edge (i,j): segments
  node_i - m(edge) (cell i) and m(edge) - node_j (cell j), patch of the edge.
Plus the dual polygons themselves on the two span planes (empty patches).
"""

import sys
from collections import defaultdict
import numpy as np

MSH = '/local_data/qiqi/openfoam-sa-ai/cases/nlf_cavL0_a0/mesh.msh'
OUT = '/local_data/qiqi/openfoam-sa-ai/cases/nlf_cavdualL0_a0/constant/polyMesh'

# gmsh physical-surface ids (from $PhysicalNames)
PHYS_NAME = {2: 'farfield', 3: 'nlf0416', 4: 'symmetry1', 5: 'symmetry2'}


def parse_msh(path):
    lines = open(path).read().split('\n')
    i = lines.index('$Nodes')
    nn = int(lines[i + 1])
    nid = np.empty(nn, dtype=np.int64)
    xyz = np.empty((nn, 3))
    for k, l in enumerate(lines[i + 2:i + 2 + nn]):
        p = l.split()
        nid[k] = int(p[0])
        xyz[k] = [float(p[1]), float(p[2]), float(p[3])]
    assert np.all(nid == np.arange(1, nn + 1)), 'non-contiguous node ids'
    j = lines.index('$Elements')
    ne = int(lines[j + 1])
    tris = defaultdict(list)   # phys -> [(n1,n2,n3)]
    quads = defaultdict(list)  # phys -> [(n1..n4)]
    for l in lines[j + 2:j + 2 + ne]:
        p = l.split()
        etype, ntags = int(p[1]), int(p[2])
        phys = int(p[3])
        nds = tuple(int(x) for x in p[3 + ntags:])
        if etype == 2:
            tris[phys].append(nds)
        elif etype == 3:
            quads[phys].append(nds)
    return xyz, tris, quads


def build_2d(xyz, tris, quads):
    """Extract the 2D triangulation (x,z coords) on the smaller-y plane."""
    yvals = np.unique(xyz[:, 1])
    assert len(yvals) == 2, yvals
    y_lo, y_hi = yvals[0], yvals[1]
    on_lo = np.isclose(xyz[:, 1], y_lo)
    lo_ids = np.nonzero(on_lo)[0] + 1          # 1-based gmsh ids on lower plane
    g2l = {g: k for k, g in enumerate(lo_ids)} # gmsh id -> local 2D index
    pts2d = xyz[lo_ids - 1][:, [0, 2]]         # (x, z)

    # symmetry patch whose triangles live on the lower plane
    sym_lo = None
    for phys in (4, 5):
        if on_lo[tris[phys][0][0] - 1]:
            sym_lo = phys
    assert sym_lo is not None
    T = np.array([[g2l[n] for n in t] for t in tris[sym_lo]], dtype=np.int64)

    # enforce consistent CCW orientation in the (x,z) plane
    a, b, c = pts2d[T[:, 0]], pts2d[T[:, 1]], pts2d[T[:, 2]]
    s = np.cross(b - a, c - a)
    flip = s < 0
    T[flip] = T[flip][:, [0, 2, 1]]
    assert np.all(np.abs(s) > 0), 'degenerate triangle'

    # boundary edges from quad faces of farfield / nlf0416
    bedges = {}
    for phys in (2, 3):
        for q in quads[phys]:
            e = tuple(sorted(g2l[n] for n in q if on_lo[n - 1]))
            assert len(e) == 2
            bedges[e] = PHYS_NAME[phys]
    return pts2d, T, bedges, (y_lo, y_hi)


def median_dual(pts, T, bedges):
    """Return per-node dual polygons and supporting structures.

    Dual vertices carry tags: ('c', tri), ('m', edge-key), ('n', node).
    Each polygon is a CCW list of tags."""
    nt = len(T)
    edge_tris = defaultdict(list)          # sorted edge -> [tri indices]
    for t in range(nt):
        for k in range(3):
            e = tuple(sorted((T[t, k], T[t, (k + 1) % 3])))
            edge_tris[e].append(t)
    # sanity: boundary edges from quads == edges with a single triangle
    single = {e for e, ts in edge_tris.items() if len(ts) == 1}
    assert single == set(bedges), 'boundary-edge mismatch'

    node_tris = defaultdict(list)
    for t in range(nt):
        for n in T[t]:
            node_tris[n].append(t)

    def other_edge(t, i, e):
        """the other edge of triangle t at node i (not e)."""
        for n in T[t]:
            if n != i:
                e2 = tuple(sorted((i, n)))
                if e2 != e:
                    return e2
        raise AssertionError

    polys = []
    for i in range(len(pts)):
        inc = node_tris[i]
        b = [e for e in ((tuple(sorted((i, int(n)))) )
                         for t in inc for n in T[t] if n != i)
             if e in single]
        b = sorted(set(b))
        boundary = len(b) > 0
        if boundary:
            assert len(b) == 2, f'node {i}: {len(b)} boundary edges'
            e = b[0]
        else:
            e = tuple(sorted((i, int(next(n for n in T[inc[0]] if n != i)))))
        # fan walk
        tags = [('m', e)]
        used = set()
        t = next(t for t in edge_tris[e] if t not in used)
        while True:
            used.add(t)
            tags.append(('c', t))
            e = other_edge(t, i, e)
            tags.append(('m', e))
            nxt = [tt for tt in edge_tris[e] if tt not in used]
            if not nxt:
                break
            t = nxt[0]
        assert len(used) == len(inc), f'node {i}: fan walk incomplete'
        if boundary:
            assert tags[-1] == ('m', b[1]) or tags[-1] == ('m', b[0])
            tags.append(('n', i))
        else:
            assert tags[-1] == tags[0]
            tags.pop()
        polys.append(tags)

    # dual-vertex coordinates
    def coord(tag):
        kind, k = tag
        if kind == 'c':
            return pts[T[k]].mean(axis=0)
        if kind == 'm':
            return 0.5 * (pts[k[0]] + pts[k[1]])
        return pts[k]

    # enforce CCW, verify simple, and compute areas
    areas = np.empty(len(polys))
    for i, tags in enumerate(polys):
        P = np.array([coord(t) for t in tags])
        s = 0.5 * np.sum(np.cross(P, np.roll(P, -1, axis=0)))
        if s < 0:
            tags.reverse()
            s = -s
        assert s > 0, f'node {i}: zero-area dual polygon'
        areas[i] = s
    return polys, edge_tris, coord, areas


def tri_area(pts, T):
    a, b, c = pts[T[:, 0]], pts[T[:, 1]], pts[T[:, 2]]
    return 0.5 * np.abs(np.cross(b - a, c - a))


def build_polymesh(pts, T, bedges, polys, edge_tris, coord, y_lo, y_hi,
                   sym_lo_name, sym_hi_name):
    """Assemble OpenFOAM points/faces/owner/neighbour/patches."""
    npts = len(pts)
    # dual vertex numbering: one index per (tag, plane)
    vidx = {}
    points = []

    def vid(tag, plane):
        key = (tag, plane)
        if key not in vidx:
            x, z = coord(tag)
            y = y_lo if plane == 0 else y_hi
            vidx[key] = len(points)
            points.append((x, y, z))
        return vidx[key]

    def lateral_face(polyline_tags, ref):
        """Extruded face from 2D polyline; oriented so its area vector has
        positive dot product with the 2D reference direction `ref`."""
        lo = [vid(t, 0) for t in polyline_tags]
        hi = [vid(t, 1) for t in polyline_tags]
        face = lo + hi[::-1]
        # exact area-vector sign via Newell on the actual 3D points
        F = np.array([points[p] for p in face])
        nv = np.zeros(3)
        for k in range(len(F)):
            a, b = F[k], F[(k + 1) % len(F)]
            nv += np.cross(a, b)
        if nv[0] * ref[0] + nv[2] * ref[1] < 0:
            face.reverse()
        return face

    internal = []  # (owner, neighbour, facepoints)
    for e, ts in sorted(edge_tris.items()):
        i, j = e
        o, n = (i, j) if i < j else (j, i)
        m2d = coord(('m', e))
        for t in ts:
            # one PLANAR quad per median segment m(ij)-c(t); the segment
            # passes through the midpoint of ij, and nodes i,j lie strictly
            # on opposite sides of its line, so sign(normal . (x_n - m))
            # is an exact out-of-owner test
            pl = [('m', e), ('c', t)]
            internal.append((o, n, lateral_face(pl, pts[n] - m2d)))
    internal.sort(key=lambda x: (x[0], x[1]))

    patches = {'farfield': [], 'nlf0416': [],
               sym_lo_name: [], sym_hi_name: []}
    for e, name in bedges.items():
        i, j = e
        m2d = coord(('m', e))
        c2d = coord(('c', edge_tris[e][0]))  # centroid of the edge's triangle
        for cell in (i, j):
            pl = [('n', cell), ('m', e)]
            # face lies on the boundary line; outward points from the
            # (strictly interior) triangle centroid toward the edge midpoint
            patches[name].append((cell, lateral_face(pl, m2d - c2d)))

    # span-plane (empty) faces: the dual polygons themselves
    for i, tags in enumerate(polys):
        lo = [vid(t, 0) for t in tags]      # CCW in (x,z)
        hi = [vid(t, 1) for t in tags]
        # CCW in (x,z) => area vector along -y? check via Newell once:
        # orient lo-plane face outward (-y), hi-plane outward (+y)
        F = np.array([points[p] for p in lo])
        ny = sum(np.cross(F[k], F[(k + 1) % len(F)])[1] for k in range(len(F)))
        f_lo = lo if ny < 0 else lo[::-1]
        f_hi = hi[::-1] if ny < 0 else hi
        patches[sym_lo_name].append((i, f_lo))
        patches[sym_hi_name].append((i, f_hi))

    for name in patches:
        patches[name].sort(key=lambda x: x[0])
    return points, internal, patches


def write_polymesh(outdir, points, internal, patches, patch_types):
    import os
    os.makedirs(outdir, exist_ok=True)

    def header(cls, obj, note=None):
        s = ('FoamFile\n{\n    version     2.0;\n    format      ascii;\n'
             f'    class       {cls};\n')
        if note:
            s += f'    note        "{note}";\n'
        s += f'    object      {obj};\n}}\n\n'
        return s

    nint = len(internal)
    faces = [f for _, _, f in internal]
    owner = [o for o, _, _ in internal]
    neigh = [n for _, n, _ in internal]
    starts = {}
    for name in patch_types:
        starts[name] = len(faces)
        for o, f in patches[name]:
            faces.append(f)
            owner.append(o)
    ncells = max(owner) + 1
    note = (f'nPoints:{len(points)}  nCells:{ncells}  nFaces:{len(faces)}'
            f'  nInternalFaces:{nint}')

    with open(f'{outdir}/points', 'w') as f:
        f.write(header('vectorField', 'points'))
        f.write(f'{len(points)}\n(\n')
        f.writelines(f'({p[0]:.16g} {p[1]:.16g} {p[2]:.16g})\n' for p in points)
        f.write(')\n')
    with open(f'{outdir}/faces', 'w') as f:
        f.write(header('faceList', 'faces'))
        f.write(f'{len(faces)}\n(\n')
        f.writelines(f'{len(fc)}({" ".join(map(str, fc))})\n' for fc in faces)
        f.write(')\n')
    with open(f'{outdir}/owner', 'w') as f:
        f.write(header('labelList', 'owner', note))
        f.write(f'{len(owner)}\n(\n')
        f.writelines(f'{o}\n' for o in owner)
        f.write(')\n')
    with open(f'{outdir}/neighbour', 'w') as f:
        f.write(header('labelList', 'neighbour', note))
        f.write(f'{len(neigh)}\n(\n')
        f.writelines(f'{n}\n' for n in neigh)
        f.write(')\n')
    with open(f'{outdir}/boundary', 'w') as f:
        f.write(header('polyBoundaryMesh', 'boundary'))
        f.write(f'{len(patch_types)}\n(\n')
        for name, ptype in patch_types.items():
            extra = ('        inGroups        1(wall);\n'
                     if ptype == 'wall' else '')
            f.write(f'    {name}\n    {{\n'
                    f'        type            {ptype};\n{extra}'
                    f'        nFaces          {len(patches[name])};\n'
                    f'        startFace       {starts[name]};\n    }}\n')
        f.write(')\n')


def main():
    xyz, tris, quads = parse_msh(MSH)
    pts, T, bedges, (y_lo, y_hi) = build_2d(xyz, tris, quads)
    print(f'2D: {len(pts)} nodes, {len(T)} triangles, '
          f'{len(bedges)} boundary edges; span y in [{y_lo}, {y_hi}]')

    polys, edge_tris, coord, areas = median_dual(pts, T, bedges)
    A_tri = tri_area(pts, T).sum()
    A_dual = areas.sum()
    rel = abs(A_dual - A_tri) / A_tri
    print(f'area check: tri {A_tri:.15e}  dual {A_dual:.15e}  '
          f'rel err {rel:.3e}')
    assert rel < 1e-10, 'area conservation failed'
    assert len(polys) == len(pts)
    print(f'dual cells: {len(polys)} == 2D nodes: OK')

    # which named symmetry patch is on which plane
    yq = xyz[tris[4][0][0] - 1, 1]
    sym_lo_name, sym_hi_name = (('symmetry1', 'symmetry2')
                                if np.isclose(yq, y_lo)
                                else ('symmetry2', 'symmetry1'))
    points, internal, patches = build_polymesh(
        pts, T, bedges, polys, edge_tris, coord, y_lo, y_hi,
        sym_lo_name, sym_hi_name)
    patch_types = {'farfield': 'patch', 'nlf0416': 'wall',
                   'symmetry1': 'empty', 'symmetry2': 'empty'}
    write_polymesh(OUT, points, internal, patches, patch_types)
    print(f'wrote {OUT}: {len(points)} points, '
          f'{len(internal)} internal faces, '
          + ', '.join(f'{k}:{len(v)}' for k, v in patches.items()))


if __name__ == '__main__':
    main()
