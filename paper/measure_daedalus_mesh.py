"""Measure the around-the-airfoil resolution rows for tab:daemesh, at the
eta=0.30 section of all six wing meshes: tangential spacing Delta-s/c at
LE / 0.05c / 0.25c / 0.98c / TE (upper surface), first off-wall spacing
h0/c, and off-wall cell size at 0.001c and 0.5c above the 0.25c point.
Prints a table; values are hand-copied into main.tex."""
import os
import sys
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree

D = '/home/qiqi/flexcompute/sa-ai/scripts/daedalus'
sys.path.insert(0, D)
import sectional_compare as SC                     # noqa: E402
from wing_geometry import chord, HALF_SPAN, XQC    # noqa: E402

CASES = {('cav', 0): 'case_cavity_saai_a5', ('cav', 1): 'case_cavity_L1_saai_a5',
         ('cav', 2): 'case_cavity_L2_saai_a5'}
SURF = {'str': 'surface_fluid_wing.pvtu', 'cav': 'surface_farfield_body.pvtu'}
SPAN_DY = {('str', 0): 0.654, ('str', 1): 0.331, ('str', 2): 0.167,
           ('cav', 0): 0.0223, ('cav', 1): 0.0112, ('cav', 2): 0.0056}
ETA0 = 0.30
XT = {'LE': 0.0, '0.05c': 0.05, '0.25c': 0.25, '0.98c': 0.98, 'TE': 1.0}


def load(path):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def section_strip(p, fam, lv):
    y0 = ETA0 * HALF_SPAN
    # both families carry discrete spanwise stations (planes / STL rings):
    ys = np.unique(np.round(p[:, 1], 5))
    y0 = float(ys[np.argmin(np.abs(ys - y0))])
    band = 5e-4
    m = np.abs(p[:, 1] - y0) < band
    return m, y0


def main():
    c0 = float(chord(ETA0))
    for (fam, lv), case in CASES.items():
        g = load(f'{D}/{case}/{SURF[fam]}')
        p = vtk_to_numpy(g.GetPoints().GetData())
        m, y0 = section_strip(p, fam, lv)
        q = p[m]
        c_loc = c0
        xc = np.clip((q[:, 0] - (XQC - 0.25 * c_loc)) / c_loc, 0, 1)
        up = q[:, 2] >= np.interp(xc, SC._CAM_X, SC._CAM_Z) * c_loc
        # order the upper-surface strip by xc, measure local spacing in s
        qq = q[up][np.argsort(xc[up])]
        xs = np.sort(xc[up])
        ds = np.hypot(np.diff(qq[:, 0]), np.diff(qq[:, 2])) / c_loc
        xmid = 0.5 * (xs[1:] + xs[:-1])
        row = {}
        for lab, xt in XT.items():
            w = np.abs(xmid - xt) < 0.02 + 0.02 * (fam == 'cav')
            row[lab] = np.median(ds[w]) if w.sum() else np.nan
        # off-wall sizes from the volume mesh near (0.25c, eta0)
        vg = load(f'{D}/{case}/volume.pvtu')
        vp = vtk_to_numpy(vg.GetPoints().GetData())
        x25 = (XQC - 0.25 * c_loc) + 0.25 * c_loc
        zs = float(np.interp(0.25, SC._CAM_X, np.interp(0.25, SC._CAM_X, SC._CAM_Z) * np.ones_like(SC._CAM_X)))  # placeholder
        zs = float(np.interp(0.25, SC._CAM_X, SC._CAM_Z)) * c_loc  # camber z
        # use actual wall z at 0.25c from the surface strip
        iz = np.argmin(np.abs(xs - 0.25))
        zwall = qq[iz, 2]
        box = (np.abs(vp[:, 1] - y0) < 0.3) & (np.abs(vp[:, 0] - x25) < 0.15) \
            & (vp[:, 2] > zwall - 0.02) & (vp[:, 2] < zwall + 0.7)
        vb = vp[box]
        tree = cKDTree(vb)
        out = {}
        for lab, dz in (('h0', 0.0), ('0.001c', 0.001 * c_loc), ('0.5c', 0.5 * c_loc)):
            probe = np.array([x25, y0, zwall + dz])
            dd, ii = tree.query(probe, k=8)
            if lab == 'h0':
                zoff = np.abs(vb[ii, 2] - zwall)
                zoff = zoff[zoff > 1e-7]
                out[lab] = zoff.min() / c_loc if len(zoff) else np.nan
            else:
                nn, _ = tree.query(vb[ii[0]], k=2)
                out[lab] = nn[1] / c_loc
        print(f"{fam} L{lv}: ds/c LE={row['LE']*1e3:.2f} 0.05c={row['0.05c']*1e3:.2f} "
              f"0.25c={row['0.25c']*1e3:.2f} 0.98c={row['0.98c']*1e3:.2f} "
              f"TE={row['TE']*1e3:.2f} (x1e-3) | h0/c={out['h0']*1e6:.1f}e-6 "
              f"cell@0.001c={out['0.001c']*1e6:.0f}e-6 cell@0.5c={out['0.5c']:.4f}",
              flush=True)


if __name__ == '__main__':
    main()
