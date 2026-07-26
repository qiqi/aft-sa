#!/usr/bin/env python3
"""Model-to-model flat-plate comparison: OpenFOAM SpalartAllmarasAI vs the
Flow360 SA-AI reference runs (flow360_fr/flatplate_sphere_Tu*).

Same metric on both: Re_theta at the chi = max_z(nuHat/nu) = 1 crossing, and
Re_x at the Cf minimum (transition onset). Flow360 conventions per
paper/regen_flatplate_flow360.py: velocity in c_inf units (divide by M=0.1),
NU = 1e-6, quasi-2D single span slice.

Run with the compute venv python (needs vtk):
  /home/qiqi/flexcompute/compute/.venv/bin/python scripts/compare_flow360.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_transition import (AGS_Reth, CASE_ROOT, NU, TU_LIST, profiles)

F360_ROOT = "/home/qiqi/flexcompute/sa-ai/flow360_fr"
MACH = 0.1
PLATE_END, MARGIN = 6.0, 0.5


def f360_profiles(cd):
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f"{cd}/volume.pvtu")
    r.Update()
    g = r.GetOutput()
    pd = g.GetPointData()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    u = vtk_to_numpy(pd.GetArray("velocity"))[:, 0] / MACH
    chi = vtk_to_numpy(pd.GetArray("nuHat")) / NU
    # single span slice
    ys = np.unique(pts[:, 1])
    keep = np.abs(pts[:, 1] - ys[0]) < 1e-12
    x, z, u, chi = pts[keep, 0], pts[keep, 2], u[keep], chi[keep]
    # column-by-column (structured nodes share exact x)
    xs = np.unique(x)
    xs = xs[(xs > 0) & (xs < PLATE_END - MARGIN)]
    Re_theta = np.empty(len(xs))
    maxchi = np.empty(len(xs))
    Cf = np.empty(len(xs))
    for i, xv in enumerate(xs):
        col = np.abs(x - xv) < 1e-12
        zc, uc, cc = z[col], u[col], chi[col]
        o = np.argsort(zc)
        zc, uc, cc = zc[o], uc[o], cc[o]
        uu = np.clip(uc, 0.0, 1.0)
        Re_theta[i] = np.trapz(uu*(1.0 - uu), zc)/NU
        maxchi[i] = cc.max()
        Cf[i] = 2.0*NU*uc[1]/zc[1] if zc[0] == 0 else 2.0*NU*uc[0]/zc[0]
    return xs, Re_theta, Cf, maxchi


def crossing(Re_theta, maxchi):
    ix = np.where(maxchi >= 1.0)[0]
    if not len(ix) or ix[0] == 0:
        return np.nan
    i = ix[0]
    w = (1.0 - maxchi[i-1])/(maxchi[i] - maxchi[i-1])
    return Re_theta[i-1] + w*(Re_theta[i] - Re_theta[i-1])


def cfmin_rex(x, Cf):
    i = int(np.argmin(Cf + 1e9*(x < 0.05)))
    return x[i]*1e6


def main():
    print(f"{'Tu%':>5} | {'OF Reth':>8} {'F360 Reth':>9} {'dRe%':>6} | "
          f"{'OF Rex_tr':>9} {'F360 Rex_tr':>11} {'dRex%':>6} | {'AGS':>6}")
    for Tu in TU_LIST:
        tag = f"{int(round(Tu*1000)):04d}"
        of = profiles(os.path.join(CASE_ROOT, f"flatplate_Tu{tag}"))
        if of is None:
            continue
        xo, Ro, Cfo, mo = of
        ko = xo < PLATE_END - MARGIN
        r_of = crossing(Ro[ko], mo[ko])
        rex_of = cfmin_rex(xo[ko], Cfo[ko])
        cd = f"{F360_ROOT}/flatplate_sphere_Tu{tag}"
        if os.path.isdir(cd):
            xf, Rf, Cff, mf = f360_profiles(cd)
            r_f6 = crossing(Rf, mf)
            rex_f6 = cfmin_rex(xf, Cff)
        else:
            r_f6 = rex_f6 = np.nan
        print(f"{Tu:5.2f} | {r_of:8.1f} {r_f6:9.1f} "
              f"{100*(r_of-r_f6)/r_f6:6.1f} | {rex_of:9.3e} {rex_f6:11.3e} "
              f"{100*(rex_of-rex_f6)/rex_f6:6.1f} | {AGS_Reth(Tu):6.1f}")


if __name__ == "__main__":
    main()
