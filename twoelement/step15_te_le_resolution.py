"""STEP 15 -- how blunt should the trailing edges be, and how well is the
leading-edge curvature resolved?

(1) TRAILING-EDGE THICKNESS. A sharp TE is both unbuildable and badly meshed
    (the upper and lower prism stacks collide at a point). Physically the TE
    thickness wants to be of the order of the boundary-layer scale there, so it
    is compared against theta and delta* at Re = 1e6.

(2) LEADING-EDGE RESOLUTION. Measure the TURNING ANGLE between adjacent
    surface segments on the meshes actually generated, near each element's
    leading edge. For a circular nose of radius r discretised at spacing ds the
    turning angle is ds/r, so the angle is the direct measure of how well the
    curvature is resolved. CFD practice wants <~ 5 deg, ideally 2-3 deg at a
    laminar leading edge where the pressure peak lives.

Run:  python3 step15_te_le_resolution.py case_L0 case_L1
"""
import sys

import numpy as np

import plot_solution_mesh as PM

RE = 1.0e6
CHORD_TOTAL = 1.0


def boundary_loops(pts, tris):
    """Ordered boundary loops of a triangulation (edges used by one triangle)."""
    from collections import defaultdict
    cnt = defaultdict(int)
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            cnt[(min(a, b), max(a, b))] += 1
    edges = [e for e, c in cnt.items() if c == 1]
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        loop, cur, prev = [start], start, None
        seen.add(start)
        while True:
            nxt = [v for v in adj[cur] if v != prev]
            nxt = [v for v in nxt if v not in seen or v == start]
            if not nxt:
                break
            v = nxt[0]
            if v == start:
                break
            loop.append(v); seen.add(v); prev, cur = cur, v
        if len(loop) > 8:
            loops.append(np.array(loop))
    return loops


def turning_angles(P):
    """Angle (deg) between consecutive segments of a closed polyline."""
    d = np.diff(np.vstack([P, P[:1]]), axis=0)
    n = np.linalg.norm(d, axis=1)
    d = d/np.maximum(n, 1e-30)[:, None]
    c = np.clip((d[:-1]*d[1:]).sum(axis=1), -1, 1)
    return np.degrees(np.arccos(c)), n


if __name__ == '__main__':
    cases = sys.argv[1:] or ['case_L0', 'case_L1']

    print('=== (1) trailing-edge thickness scales at Re = %.0e ===' % RE)
    print('%-22s %10s %10s %10s' % ('station', 'theta/c', 'delta*/c', 'Re_theta'))
    for name, x in (('fore TE  (x=0.70)', 0.70), ('flap TE  (x=1.00)', 1.00)):
        th = 0.664*np.sqrt(x/RE)
        print('%-22s %10.5f %10.5f %10.0f' % (name, th, 2.59*th, RE*x*th/x))
    print('  candidate TE thicknesses, as a fraction of TOTAL chord:')
    for f in (0.002, 0.003, 0.005):
        th_fore = 0.664*np.sqrt(0.70/RE)
        print('    %.3f c  = %5.2f x theta(fore TE) = %5.2f x delta*(fore TE)'
              % (f, f/th_fore, f/(2.59*th_fore)))

    print('\n=== (2) leading-edge turning angle on the generated meshes ===')
    print('%-9s %-6s %8s %8s %8s %9s %9s'
          % ('case', 'elem', 'ds_LE/c', 'ang_LE', 'ang_max', 'ang_p95', 'n_edges'))
    for case in cases:
        try:
            gm, pm, _ = PM.read_vtu('%s/mesh2d.vtk' % case)
        except Exception as e:                                  # noqa: BLE001
            print('%-9s  (no mesh: %s)' % (case, str(e)[:50]))
            continue
        from vtk.util.numpy_support import vtk_to_numpy
        cm = vtk_to_numpy(gm.GetCells().GetConnectivityArray())
        om = vtk_to_numpy(gm.GetCells().GetOffsetsArray())
        mt = np.array([cm[a:b] for a, b in zip(om[:-1], om[1:]) if b - a == 3])
        nun = [len(np.unique(np.round(pm[:, i], 9))) for i in range(3)]
        ax2 = [i for i in range(3) if nun[i] > 2][:2]
        P2 = pm[:, ax2]
        loops = boundary_loops(P2, mt)
        loops = [L for L in loops
                 if np.ptp(P2[L][:, 0]) < 3.0]          # drop the farfield
        loops.sort(key=lambda L: P2[L][:, 0].min())
        for k, L in enumerate(loops[:2]):
            Q = P2[L]
            ang, seg = turning_angles(Q)
            i_le = int(np.argmin(Q[:, 0]))
            w = np.arange(i_le - 6, i_le + 7) % len(ang)
            print('%-9s %-6s %8.5f %8.2f %8.2f %9.2f %9d'
                  % (case, ['fore', 'flap'][k], seg[w].mean(),
                     ang[w].max(), ang.max(), np.percentile(ang, 95), len(L)))
    print('\n  ang_LE = worst turning angle within +/-6 segments of the nose.')
    print('  Target for a laminar leading edge: <= 2-3 deg.')
