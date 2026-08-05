"""Build the RANS case config for the two-element wake-interaction article.

Geometry from steps 1-4 (all public NACA definitions, so the case is fully
reproducible):

  fore element   mean line   NACA 4-digit, m = -3.5% at p = 70%, mounted at the
                             incidence fitting the flap-alone streamline
                 thickness   NACA 4-digit MODIFIED, t/c = 0.08, x_m = 0.50,
                             I = 6, K_m = 0.7
                 chord 0.68, LE at the origin
  aft element    NACA 9416, chord 0.30, LE (0.70, 0.04), inc -8 deg

Purpose: the fore element runs laminar on BOTH surfaces and sheds a laminar
wake that passes close over the flap, where transition matters. That is the
SA-AI blindspot the case exists to expose.

Writes twoelement_case.json in the rans CaseConfig schema.
Run:  python3 make_case.py
"""
import json
import numpy as np

import panel2e as M
import step4_coupled as S
from step1_flap_and_camber import flap_nodes, FORE_CHORD, ALPHA
from step2_questions import refit_inc

RE = 1.0e6
MACH = 0.10
NPTS = 400          # contour points per element handed to the mesher


def contours():
    n2 = flap_nodes()
    Pf, _ = M.solve_elements([n2], ALPHA)
    xs, zs = M.streamline_camber(Pf, 0.0, 0.0, FORE_CHORD, n=400)
    inc = refit_inc(S.M_CAM, S.P_CAM, xs, zs)
    n1 = M.place(M.airfoil_nodes(NPTS, S.M_CAM, S.P_CAM, 0.08, modified=True,
                                 xm=S.XM, ik=S.IK, km=S.KM, le_blend=0.15),
                 FORE_CHORD, inc, 0.0, 0.0)
    n2 = M.place(M.airfoil_nodes(NPTS, 0.09, 0.40, 0.16), 0.30, -8.0,
                 0.70, 0.04)
    return n1, n2, inc


if __name__ == '__main__':
    n1, n2, inc = contours()

    # the mesher closes the loop itself -- drop the duplicated last point
    def cl(a):
        return [[float(x), float(z)] for x, z in a[:-1]]

    allpts = np.vstack([n1, n2])
    cx = 0.5*(allpts[:, 0].min() + allpts[:, 0].max())
    cz = 0.5*(allpts[:, 1].min() + allpts[:, 1].max())

    cfg = dict(
        elements=[dict(name='fore', contour=cl(n1), is_wall=True),
                  dict(name='flap', contour=cl(n2), is_wall=True)],
        farfield=dict(type='circle', center=[float(cx), float(cz)],
                      radius=100.0, n=360),
        flow=dict(reynolds=RE, mach=MACH, alpha_deg=ALPHA,
                  temperature=288.15),
        # y+ ~ 0.3 as in the paper's airfoil campaign; the anisotropic BL metric
        # is built per element and intersected by the mesher
        mesh=dict(span=0.1, nspan=1, yplus=0.3, growth=1.10, hwall=0.003,
                  hmax=3.0, h0=0.0),
        solver=dict(max_steps=12000),
    )
    json.dump(cfg, open('twoelement_case.json', 'w'), indent=1)
    print('fore: %d pts, x %.4f..%.4f' % (len(cfg['elements'][0]['contour']),
                                          n1[:, 0].min(), n1[:, 0].max()))
    print('flap: %d pts, x %.4f..%.4f' % (len(cfg['elements'][1]['contour']),
                                          n2[:, 0].min(), n2[:, 0].max()))
    print('fore mounted at %+.2f deg;  Re=%.1e  M=%.2f  alpha=%.1f'
          % (inc, RE, MACH, ALPHA))
    # slot metrics
    d = np.hypot(n1[:, 0][:, None] - n2[:, 0][None, :],
                 n1[:, 1][:, None] - n2[:, 1][None, :])
    print('min gap between elements: %.5f c' % d.min())
    print('wrote twoelement_case.json')
