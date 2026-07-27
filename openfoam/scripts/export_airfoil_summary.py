"""Export the OpenFOAM airfoil cases to a compact JSON the paper overlays
(paper/data/openfoam_airfoil_summary.json).

Forces come from the cross-solver campaign's own deliverable,
openfoam/sweep_results.csv (staged-protocol runs). Transition fronts are
re-extracted from the VTK fields at the PAPER's convention -- the
near-wall chi=1 crossing -- because the CSV's fronts use chi=c_v1, which
sits up to 0.05c aft on the slow negative-alpha pressure-side front (and
~0.01-0.03c aft generally). The extraction walks per-x-bin wall-adjacent
cells (z within `band` of the local surface, per side), so it does not
read the lifted shear layer over the Eppler bubble the way a plain
kd-tree near-wall band does (~0.07c forward bias, observed). Validated:
nlf am8 L2 chi=1 = 0.567 vs the paper's 0.559-0.561.

Upper/lower is decided against the airfoil's own midline z_c(x) (from
the wall patch), NOT sign(z) or a global z threshold: the Eppler 387
lower surface rises ABOVE z=0 aft of x ~ 0.58, so a sign(z) split lets
below-airfoil fluid into the upper mask there (its min-z "wall" then
hugs the LOWER surface), and a global z threshold truncates the aft
upper surface before the alpha=0 reattachment (both observed). NLF rows
are unaffected by construction (its fluid never crosses z=0 inside the
chord band) and were verified unchanged after the fix.

Per case (eppler|nlf)_str{L0,L1,L2}_{a,am}N:
  cl, cd    : the CSV's final coefficient.dat sample (steady; the
              median over the final 20% of rows agrees to <=0.5%)
  xtr_up/lo : near-wall chi=1 crossing per side (chi = nuTilda*Re; U=1,
              c=1); None when the side stays laminar to the TE
  ls, tr    : Eppler upper-surface signed-Cfx zero crossings (separation /
              reattachment, the fig:eppbubble convention), from the VTK

Run from openfoam/scripts/: python3 export_airfoil_summary.py
"""
import csv
import glob
import json
import os
import re

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_DIR = os.path.join(HERE, '..', 'data', 'cases')
SWEEP = os.path.join(HERE, '..', 'sweep_results.csv')
OUT = '/home/qiqi/flexcompute/sa-ai/paper/data/openfoam_airfoil_summary.json'


def read_any(path):
    if path.endswith('.vtp'):
        r = vtk.vtkXMLPolyDataReader()
    else:
        r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def cell_centers(g):
    cc = vtk.vtkCellCenters()
    cc.SetInputData(g)
    cc.Update()
    return vtk_to_numpy(cc.GetOutput().GetPoints().GetData())


def latest_vtk(case, name):
    ds = sorted(glob.glob(f'{case}/VTK/*_[0-9]*/'),
                key=lambda d: int(d.rstrip('/').rsplit('_', 1)[-1]))
    assert ds, f'no VTK export in {case}'
    return ds[-1] + name


def airfoil_patch(case):
    vd = os.path.dirname(latest_vtk(case, 'x'))
    cands = [p for p in glob.glob(f'{vd}/boundary/*.vtp')
             if not re.search(r'far|inout|inlet|outlet|front|back|empty',
                              os.path.basename(p))]
    assert cands, f'no airfoil patch in {vd}'
    return max(cands, key=os.path.getsize)


def midline(case, nbins=400):
    """z_c(x) interpolant of the airfoil midline from the wall patch:
    per x-bin, the midpoint of the two surfaces' z. The side split must
    use this, not sign(z) -- see the module docstring (Eppler aft lower
    surface sits above z=0)."""
    surf = read_any(airfoil_patch(case))
    sp = cell_centers(surf)
    xs, zs = sp[:, 0], sp[:, 2]
    bins = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(xs, bins) - 1, 0, nbins - 1)
    xc, zc = [], []
    for i in range(nbins):
        k = idx == i
        if not k.any():
            continue
        xc.append(0.5 * (bins[i] + bins[i + 1]))
        zc.append(0.5 * (zs[k].min() + zs[k].max()))
    return np.array(xc), np.array(zc)


def chi1_fronts(case, re, band=0.003, x0=0.005, x1=0.995, nbins=240):
    """Near-wall chi=1 crossing per side (upper, lower), paper convention.

    Per x-bin and side, 'wall-adjacent' means within `band` of the
    side's extreme z in that bin (the cells hugging the surface), which
    excludes the lifted shear layer over a separation bubble.
    """
    vol = read_any(latest_vtk(case, 'internal.vtu'))
    p = cell_centers(vol)
    chi = vtk_to_numpy(vol.GetCellData().GetArray('nuTilda')) * re
    x, z = p[:, 0], p[:, 2]
    xc_w, zc_w = midline(case)
    z_c = np.interp(x, xc_w, zc_w)
    bins = np.linspace(x0, x1, nbins + 1)
    out = []
    for upper in (True, False):
        side = (z > z_c) if upper else (z < z_c)
        st = prev = prevx = None
        for i in range(nbins):
            k = side & (x >= bins[i]) & (x < bins[i + 1])
            if not k.any():
                continue
            zw = z[k].min() if upper else z[k].max()
            kk = k & (np.abs(z - zw) < band)
            c = chi[kk].max()
            xc = 0.5 * (bins[i] + bins[i + 1])
            if st is None and prev is not None and prev <= 1.0 < c:
                st = prevx + (1.0 - prev) / (c - prev) * (xc - prevx)
            if st is None and prev is None and c > 1.0:
                st = 0.0  # already turbulent at the first valid bin (LE front)
            prev, prevx = c, xc
        out.append(None if st is None else round(float(st), 4))
    return out


def bubble_stations(case):
    surf = read_any(airfoil_patch(case))
    spts = cell_centers(surf)
    tau = vtk_to_numpy(surf.GetCellData().GetArray('wallShearStress'))
    x, z = spts[:, 0], spts[:, 2]
    # OpenFOAM wallShearStress is the stress ON the wall: attached
    # forward flow gives tau_x < 0. Negate for the paper's C_{f,x}.
    cfx = -2.0 * tau[:, 0]
    # Side split against the local midline (a global z threshold truncates
    # the aft upper surface before the alpha=0 reattachment; see docstring).
    xc_w, zc_w = midline(case)
    upper = z > np.interp(x, xc_w, zc_w)
    o = np.argsort(x[upper])
    xu, cu = x[upper][o], cfx[upper][o]
    m = (xu > 0.05) & (xu < 0.995)
    xu, cu = xu[m], cu[m]
    ls = tr = None
    for i in range(1, len(xu)):
        if cu[i] < 0 and cu[i-1] >= 0 and ls is None:
            f = -cu[i-1] / (cu[i] - cu[i-1])
            ls = float(xu[i-1] + f * (xu[i] - xu[i-1]))
    neg = np.where(cu < 0)[0]
    if len(neg) and neg[-1] + 1 < len(xu):
        i = neg[-1]
        f = -cu[i] / (cu[i+1] - cu[i])
        tr = float(xu[i] + f * (xu[i+1] - xu[i]))
    return (None if ls is None else round(ls, 4),
            None if tr is None else round(tr, 4))


RE = {'eppler': 2e5, 'nlf': 4e6}


def main():
    res = {}
    for row in csv.DictReader(open(SWEEP)):
        if not row['of_cl'].strip():
            continue  # Flow360-only reference row (their L2 not run yet)
        name = row['case']
        af = row['airfoil']
        case = os.path.join(CASES_DIR, name)
        try:
            up, lo = chi1_fronts(case, RE[af])
        except (AssertionError, OSError) as e:
            print(f'{name}: no VTK yet, SKIP ({e})')
            continue
        rec = dict(alpha=int(float(row['alpha'])), level=f"L{row['level']}",
                   cl=round(float(row['of_cl']), 4),
                   cd=round(float(row['of_cd']), 5),
                   xtr_up=up, xtr_lo=lo)
        if af == 'eppler':
            try:
                rec['ls'], rec['tr'] = bubble_stations(case)
            except Exception as e:
                print(f'{name}: no bubble stations ({e})')
                rec['ls'] = rec['tr'] = None
        res[name] = rec
        print(name, rec)
    json.dump({'source': 'OpenFOAM v2412 SpalartAllmarasAI, structured '
                         '(Construct2D) family. Forces from '
                         'openfoam/sweep_results.csv (staged-protocol '
                         'campaign, final steady force sample); fronts '
                         're-extracted from the VTK fields at the paper '
                         'convention (near-wall chi=1 crossing; None = '
                         'laminar to the TE); Eppler bubble stations '
                         '(ls, tr) from the VTK signed-Cfx zero crossings.',
               'cases': res}, open(OUT, 'w'), indent=1)
    print('wrote', OUT, f'({len(res)} cases)')


if __name__ == '__main__':
    main()
