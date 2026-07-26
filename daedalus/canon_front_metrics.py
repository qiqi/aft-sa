"""Transition-front and bubble metrics from the canon Daedalus solutions,
for the Sec. daedalus text: per case, on the upper surface --
  front x/c where near-wall max chi crosses 1 (from chi_surface.npz), and
  bubble extent from the contiguous Cf_x < 0 run (surface pvtu),
at eta = 0.31 and as the median over the constant-chord panel
(0.1 < eta < 0.8), plus the e^N strip reference at the same stations.

Usage: python3 canon_front_metrics.py
"""
import os
import pickle
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import sectional_compare as SC
from wing_geometry import chord, HALF_SPAN, XQC

HERE = os.path.dirname(os.path.abspath(__file__))
# SAAI_DAE_ROOT overrides the case tree (fv1 recomputation)
ROOT = os.environ.get('SAAI_DAE_ROOT', HERE)
SURF = {'ogrid': 'surface_fluid_wing.pvtu', 'cavity': 'surface_farfield_body.pvtu'}
ETAS = np.arange(0.10, 0.801, 0.035)


def upper_mask(p):
    c_loc = chord(np.clip(np.abs(p[:, 1]) / HALF_SPAN, 0, 1))
    xc = np.clip((p[:, 0] - (XQC - 0.25 * c_loc)) / c_loc, 0, 1)
    return xc, p[:, 2] >= np.interp(xc, SC._CAM_X, SC._CAM_Z) * c_loc


def station(case, fam, eta, band=0.008):
    d = np.load(f'{ROOT}/{case}/chi_surface.npz')
    w, chi = d['wall'], d['chi']
    xcw, upw = upper_mask(w)
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f'{ROOT}/{case}/{SURF[fam]}')
    r.Update()
    g = r.GetOutput()
    p = vtk_to_numpy(g.GetPoints().GetData())
    cfx = vtk_to_numpy(g.GetPointData().GetArray('CfVec'))[:, 0]
    xcp, upp = upper_mask(p)
    y0 = eta * HALF_SPAN
    out = {}
    for tag, pts, xc, up, val in (('chi', w, xcw, upw, chi),
                                  ('cfx', p, xcp, upp, cfx)):
        m = up & (np.abs(np.abs(pts[:, 1]) - y0) < band * HALF_SPAN)
        if m.sum() < 10:                     # O-grid planes: widen to nearest
            dy = np.abs(np.abs(pts[:, 1]) - y0)
            m = up & (dy < np.partition(dy[up], 400)[400] + 1e-9)
        o = np.argsort(xc[m])
        out[tag] = (xc[m][o], val[m][o])
    x, c = out['chi']
    front = np.nan
    above = c >= 1.0
    if above.any() and not above[0]:
        j = int(np.argmax(above))
        x0, x1, c0, c1 = x[j - 1], x[j], c[j - 1], c[j]
        front = x0 + (x1 - x0) * (np.log(1 / c0) / np.log(c1 / c0))
    x, f = out['cfx']
    # bubble: longest contiguous run of (bin-averaged) Cf_x < 0 before 0.9c
    xb = np.linspace(0.02, 0.95, 120)
    fb = np.interp(xb, x, f)
    neg = fb < 0
    runs, i = [], 0
    while i < len(neg):
        if neg[i]:
            j = i
            while j < len(neg) and neg[j]:
                j += 1
            runs.append((xb[i], xb[j - 1]))
            i = j
        else:
            i += 1
    bub = max(runs, key=lambda r: r[1] - r[0]) if runs else (np.nan, np.nan)
    return front, bub


def strips_ref(a, eta):
    s = pickle.load(open('/home/qiqi/flexcompute/sa-ai/flow360_ai/'
                         'flexfoil_daedalus_strips.pkl', 'rb'))[float(a)]
    i = int(np.argmin(np.abs(np.asarray(s['eta']) - eta)))
    st = s['stations'][i]
    if st is None:
        return np.nan, (np.nan, np.nan)
    xc = np.asarray(st['xc'], float)
    n = np.asarray(st['n'], float)
    xtr = xc[~np.isfinite(n)][0] if (~np.isfinite(n)).any() else np.nan
    cf = np.asarray(st['cf'], float)
    neg = np.where((cf < 0) & (xc < 0.98))[0]
    bub = (xc[neg[0]], xc[neg[-1]]) if len(neg) else (np.nan, np.nan)
    return xtr, bub


if __name__ == '__main__':
    for a in (4, 5, 6):
        for fam, lv in (('ogrid', 'L1'), ('ogrid', 'L2'), ('cavity', 'L1'),
                        ('cavity', 'L2')):
            case = f'case_{fam}_{lv}_saai_a{a}'
            if not os.path.exists(f'{ROOT}/{case}/chi_surface.npz'):
                continue
            f31, b31 = station(case, fam, 0.31)
            fronts = [station(case, fam, e)[0] for e in ETAS]
            f_med = float(np.nanmedian(fronts))
            print(f'a{a} {fam:6s} {lv}: front(0.31)={f31:.3f} '
                  f'panel-median={f_med:.3f} bubble(0.31)={b31[0]:.3f}-{b31[1]:.3f}',
                  flush=True)
        xr, br = strips_ref(a, 0.31)
        print(f'a{a} e^N strips : front(0.31)={xr:.3f} '
              f'bubble(0.31)={br[0]:.3f}-{br[1]:.3f}', flush=True)
