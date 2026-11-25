"""Stations where the largest chi near the wall crosses 1 and c_v1, every airfoil case.

Section III defines a transition location as the station where the largest chi near
the wall crosses one, and quotes how much farther aft chi crosses c_v1. This script
measures both crossings on the upper and the lower surface of every NLF(1)-0416 and
Eppler 387 solution (both grid families, levels L0 to L2, all Reynolds numbers) and
writes data/chi_crossings_airfoil.json. repro/tools/check_results_prose.py asserts
the quoted ranges from that file.

"Near the wall" is a wall distance of at most 0.01 c, as in the figures. The chord is
divided into 400 stations of 0.0025 c; the crossing is interpolated in ln(chi) between
the two stations that bracket it. The chi = c_v1 crossing is searched from the chi = 1
crossing aft. A null entry means that chi does not reach the level on that surface.

Each case directory needs Flow360.json, slice_centerSpan.pvtu and surface_fluid_*.pvtu.

Run from paper/ with a python that has vtk:
    SAAI_CFD_ROOT=/path/to/flow360_fv1 python repro/cfd/chi_crossings_airfoil.py
"""
import glob
import json
import os
import re
from multiprocessing import Pool

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fv1")
OUT = os.path.join("data", "chi_crossings_airfoil.json")
C_V1, NB, DMAX = 7.1, 400, 0.01
CASES = re.compile(
    r"^(cav|str)L[012]prop_(eppler387_Re200k|nlf0416_Re4M)_a(m?[0-9p]+x?)$"
    r"|^sweep_(cav|str)(L[012])?_Re[0-9]+k_a(m?[0-9p]+)$"
    r"|^nlfsweep_(cav|str)L[012]_Re[0-9]M_a(m?[0-9]+)$")


def read(f):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f)
    r.Update()
    return r.GetOutput()


def surfaces(d):
    """Upper and lower surface, each as (x, z) sorted by x."""
    p = vtk_to_numpy(read(glob.glob(os.path.join(d, "surface_fluid_*.pvtu"))[0])
                     .GetPoints().GetData())
    X, Z = p[:, 0], p[:, 2]
    te = X > X.max() - 1e-4
    chord = np.interp(X, [X.min(), X.max()],
                      [Z[np.argmin(X)], 0.5*(Z[te].max() + Z[te].min())])
    up = Z >= chord

    def side(m):
        o = np.argsort(X[m])
        return X[m][o], Z[m][o]
    return side(up), side(~up)


def cross(x, c, level, x_from=-1.0):
    ok = ~np.isnan(c) & (x >= x_from)
    x, c = x[ok], c[ok]
    j = np.where(c >= level)[0]
    if not len(j):
        return None
    i = j[0]
    if i == 0:
        return float(x[0])
    f = (np.log(level) - np.log(c[i-1]))/(np.log(c[i]) - np.log(c[i-1]))
    return float(x[i-1] + f*(x[i] - x[i-1]))


def one(name):
    d = os.path.join(ROOT, name)
    nu = float(json.load(open(os.path.join(d, "Flow360.json")))["freestream"]["muRef"])
    g = read(os.path.join(d, "slice_centerSpan.pvtu"))
    pd = g.GetPointData()
    p = vtk_to_numpy(g.GetPoints().GetData())
    chi = vtk_to_numpy(pd.GetArray("nuHat"))/nu
    wd = vtk_to_numpy(pd.GetArray("wallDistance"))
    (xu, zu), (xl, zl) = surfaces(d)
    x, z = p[:, 0], p[:, 2]
    near = (x >= 0) & (x <= 1) & (wd > 0) & (wd <= DMAX)
    zc = 0.5*(np.interp(x, xu, zu) + np.interp(x, xl, zl))
    edges = np.linspace(0, 1, NB + 1)
    xc = 0.5*(edges[:-1] + edges[1:])
    ib = np.clip(np.digitize(x, edges) - 1, 0, NB - 1)
    out = {}
    for key, m in (("upper", near & (z > zc)), ("lower", near & (z < zc))):
        mx = np.full(NB, np.nan)
        np.fmax.at(mx, ib[m], chi[m])
        x1 = cross(xc, mx, 1.0)
        xv = cross(xc, mx, C_V1, x1) if x1 is not None else None
        out[key] = [None if v is None else round(v, 4) for v in (x1, xv)]
    return name, out


def main():
    names = sorted(n for n in os.listdir(ROOT) if CASES.match(n))
    with Pool(min(24, os.cpu_count())) as pool:
        res = dict(pool.map(one, names))
    doc = {"_doc": "x/c where the largest chi within 0.01 c of the wall crosses 1 and "
                   "c_v1 = 7.1: {case: {surface: [x(chi=1), x(chi=c_v1)]}}. Written by "
                   "repro/cfd/chi_crossings_airfoil.py.",
           "cases": res}
    with open(OUT + ".tmp", "w") as f:
        json.dump(doc, f, indent=0)
        f.write("\n")
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, "with", len(res), "cases")


if __name__ == "__main__":
    main()
