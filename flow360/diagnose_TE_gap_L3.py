"""TE Cp gap (Kutta indicator) for all proper-refinement cases including cav L3."""
import numpy as np, vtk, os
from vtk.util.numpy_support import vtk_to_numpy

def contour_walk(d, af='naca0012'):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f'{d}/surface_fluid_{af}.pvtu'); r.Update()
    g = r.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    arrs = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)) for i in range(pd.GetNumberOfArrays())}
    cf = arrs['Cf']
    cp = arrs['Cp']; cp = cp if cp.ndim == 1 else cp[:,0]
    x, y, z = p[:,0], p[:,1], p[:,2]
    s = np.abs(y - y.min()) < 1e-6
    X, Z, CF, CP = x[s], z[s], cf[s], cp[s]
    n = len(X); pts = np.column_stack([X, Z])
    st = int(np.argmin(X)); o = [st]; u = np.zeros(n, bool); u[st] = True
    for _ in range(n-1):
        c = o[-1]; dd = np.sum((pts - pts[c])**2, 1); dd[u] = 1e9
        nx = int(np.argmin(dd)); o.append(nx); u[nx] = True
    o = np.array(o); xo, zo, cfo, cpo = X[o], Z[o], CF[o], CP[o]
    te = int(np.argmax(xo)); b1, b2 = slice(0, te+1), slice(te, n)
    up = b1 if zo[b1].mean() >= zo[b2].mean() else b2
    lo = b2 if up == b1 else b1
    def srt(sl):
        xs = xo[sl]; oo = np.argsort(xs)
        return xs[oo], cfo[sl][oo], cpo[sl][oo], zo[sl][oo]
    return srt(up), srt(lo)

CASES = ['proper_cav_L0','proper_cav_L1','proper_cav_L2','proper_cav_L3',
         'proper_str_L0','proper_str_L1','proper_str_L2']

print("=== Cp at last on-airfoil panel before TE singularity ===")
print(f"{'case':>14s}  {'x_up_last':>10s}  {'Cp_up_last':>12s}  {'x_lo_last':>10s}  {'Cp_lo_last':>12s}  {'ΔCp_TE':>9s}")
for d in CASES:
    full = f"/home/qiqi/flexcompute/aft-sa/flow360/{d}"
    if not os.path.exists(f"{full}/surface_fluid_naca0012.pvtu"):
        print(f"  {d}: SKIP"); continue
    (xu, cfu, cpu, zu), (xl, cfl, cpl, zl) = contour_walk(full)
    if abs(xu[-1] - 1.0) < 1e-6 and abs(zu[-1]) < 1e-6:
        i_u = -2; i_l = -2
    else:
        i_u = -1; i_l = -1
    print(f"  {d:12s}  {xu[i_u]:.6f}  {cpu[i_u]:+12.6f}  {xl[i_l]:.6f}  {cpl[i_l]:+12.6f}  {cpu[i_u]-cpl[i_l]:+9.5f}")
