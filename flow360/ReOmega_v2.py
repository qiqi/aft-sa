"""Cleaner Re_Ω vs Γ decomposition using VTK's vtkImplicitPolyDataDistance
to compute proper signed distance to the airfoil polyline."""
import numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

NU_REF = 0.2 / 1e6

def load_with_vorticity(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/slice_centerSpan.pvtu"); r.Update()
    g = r.GetOutput()
    gf = vtk.vtkGradientFilter(); gf.SetInputData(g); gf.SetInputScalars(0, "velocity")
    gf.SetComputeVorticity(True); gf.SetVorticityArrayName("vorticity"); gf.Update()
    out = gf.GetOutput(); pd = out.GetPointData()
    arrs = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)) for i in range(pd.GetNumberOfArrays())}
    return vtk_to_numpy(out.GetPoints().GetData()), arrs

def build_airfoil_polyline(d):
    """Build a closed polyline of the airfoil contour at the slice plane (centerSpan).
    Returns a vtkPolyData for use with vtkImplicitPolyDataDistance."""
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/surface_fluid_naca0012.pvtu"); r.Update()
    g = r.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData())
    x, y, z = p[:,0], p[:,1], p[:,2]
    s = np.abs(y - y.min()) < 1e-6
    X, Z = x[s], z[s]
    # contour-walk to get ordered points
    n = len(X); pts2 = np.column_stack([X, Z])
    st = int(np.argmin(X)); o = [st]; u = np.zeros(n, bool); u[st] = True
    for _ in range(n-1):
        c = o[-1]; dd = np.sum((pts2 - pts2[c])**2, 1); dd[u] = 1e9; nx = int(np.argmin(dd)); o.append(nx); u[nx] = True
    o = np.array(o)
    pts_ordered = np.column_stack([X[o], np.zeros(n), Z[o]])  # build at y=0
    # build PolyData with line segments
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    for i in range(n):
        points.InsertNextPoint(pts_ordered[i,0], 0.0, pts_ordered[i,2])
    polydata.SetPoints(points)
    lines = vtk.vtkCellArray()
    for i in range(n):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, (i+1) % n)
        lines.InsertNextCell(line)
    polydata.SetLines(lines)
    return polydata, X[o], Z[o]

def project_signed(pts, polydata, x_contour, z_contour):
    """Compute (x_wall, signed_d) for each slice point by point-to-segment distance
    along the airfoil polyline (vectorized in numpy). Sign is positive above the
    upper surface (z > z_wall_at_nearest_x)."""
    N = len(pts)
    xz = np.column_stack([pts[:,0], pts[:,2]])  # (N, 2)
    Ncont = len(x_contour)
    # build segments
    A = np.column_stack([x_contour, z_contour])  # (Ncont, 2) — segment start
    B = np.roll(A, -1, axis=0)                   # (Ncont, 2) — segment end
    seg = B - A                                  # (Ncont, 2)
    seg_len2 = (seg**2).sum(axis=1)              # (Ncont,)
    seg_len2[seg_len2 == 0] = 1e-30
    # for each pt, find nearest segment
    d_min = np.full(N, np.inf); proj_x = np.empty(N); proj_z = np.empty(N)
    # chunk over segments to keep memory bounded
    chunk = 1000
    for s in range(0, Ncont, chunk):
        e = min(s+chunk, Ncont)
        AP = xz[:, None, :] - A[None, s:e, :]   # (N, ch, 2)
        t = (AP * seg[None, s:e, :]).sum(axis=2) / seg_len2[None, s:e]  # (N, ch)
        t = np.clip(t, 0.0, 1.0)
        proj = A[None, s:e, :] + t[:, :, None] * seg[None, s:e, :]      # (N, ch, 2)
        dd2 = ((xz[:, None, :] - proj)**2).sum(axis=2)                  # (N, ch)
        # find min within this chunk
        i_loc = np.argmin(dd2, axis=1)
        d_loc = dd2[np.arange(N), i_loc]
        better = d_loc < d_min
        d_min[better] = d_loc[better]
        proj_x[better] = proj[better, i_loc[better], 0]
        proj_z[better] = proj[better, i_loc[better], 1]
    d = np.sqrt(d_min)
    # sign: positive if point is above the proj (z_pt > proj_z)
    sign = np.where(pts[:,2] > proj_z, 1.0, -1.0)
    return proj_x, d * sign

DCAV = "proper_refineA4_cavity_xxfine"; DSTR = "refineA4_struct_sxxfine"
print("loading + gradients ...", flush=True)
pc, ac = load_with_vorticity(DCAV)
ps, as_ = load_with_vorticity(DSTR)
print("building airfoil polylines ...", flush=True)
pdc, xc_cav, zc_cav = build_airfoil_polyline(DCAV)
pds, xc_str, zc_str = build_airfoil_polyline(DSTR)
print(f"projecting cavity points (this may take a while, N={len(pc)})...", flush=True)
xw_cav, d_cav = project_signed(pc, pdc, xc_cav, zc_cav)
print(f"  done; |d| stats: min={np.abs(d_cav).min():.5f}, max={np.abs(d_cav).max():.5f}", flush=True)
print(f"projecting struct points (N={len(ps)})...", flush=True)
xw_str, d_str = project_signed(ps, pds, xc_str, zc_str)
print(f"  done", flush=True)

def quantities(pts, arrs, xw_arr, d_arr, xt, dx=0.005, max_d=0.02):
    vel = arrs['velocity']; vort = arrs['vorticity']; nuHat = arrs.get('nuHat', None)
    u_mag = np.sqrt(vel[:,0]**2 + vel[:,2]**2); omega = np.abs(vort[:,1])
    # upper side (d>0 above airfoil), tight x-projected band
    m = (np.abs(xw_arr - xt) < dx) & (d_arr > 0) & (d_arr < max_d)
    if m.sum() < 5: return None
    d = d_arr[m]; un = u_mag[m]; on = omega[m]; nh = nuHat[m] if nuHat is not None else None
    ix = np.argsort(d)
    d = d[ix]; un = un[ix]; on = on[ix]; nh = nh[ix] if nh is not None else None
    Re_Om = d**2 * on / NU_REF; od = on*d
    Gamma = 2.0*od*od / (un*un + od*od + 1e-30)
    return d, un, on, Re_Om, Gamma, nh

print("\n=== quantities at upper-surface stations (signed-distance projection) ===")
print(f"{'x_w':>5s} {'grid':>7s} {'δ99':>8s} {'ω_wall':>9s} {'(d²ω)_pk':>10s} {'Γ_pk':>7s} {'|u|@RΩpk':>10s}")
X = [0.03, 0.05, 0.07, 0.10, 0.13, 0.20]
for xt in X:
    cav = quantities(pc, ac, xw_cav, d_cav, xt, dx=0.005)
    sst = quantities(ps, as_, xw_str, d_str, xt, dx=0.005)
    if cav is None or sst is None: continue
    for lbl, (dz, un, on, RO, GA, nh) in [('cavity', cav), ('O-grid', sst)]:
        u_e = un.max()
        above = (un >= 0.99*u_e) & (dz > 1e-4)
        d99 = dz[above].min() if above.any() else 0.005
        # peak ω at wall (innermost point)
        # smallest d gives most-wall-adjacent point
        ow = on[0] if len(on) > 0 else 0
        bl = dz < 1.5*d99
        if bl.any():
            iRO = int(np.argmax(RO[bl]))
            print(f"{xt:5.2f} {lbl:>7s} {d99:8.4f} {ow:9.2e} {RO[bl].max():10.1f} {GA[bl].max():7.3f} {un[bl][iRO]/u_e:10.3f}")

# plot: Re_Omega and Gamma profiles
fig, axs = plt.subplots(2, 3, figsize=(9.0, 6.0), sharey='row')
for i, xt in enumerate([0.05, 0.07, 0.10]):
    cav = quantities(pc, ac, xw_cav, d_cav, xt, dx=0.005); sst = quantities(ps, as_, xw_str, d_str, xt, dx=0.005)
    if cav is None or sst is None: continue
    dc, uc, oc, ROc, GAc, nhc = cav; ds, us, os_, ROs, GAs, nhs = sst
    u_ec = uc.max(); u_es = us.max()
    above_c = (uc >= 0.99*u_ec) & (dc > 1e-4); above_s = (us >= 0.99*u_es) & (ds > 1e-4)
    d99c = dc[above_c].min() if above_c.any() else 0.005
    d99s = ds[above_s].min() if above_s.any() else 0.005
    # bin by d/δ99 and take median to smooth cavity scatter
    bins = np.linspace(0, 2.5, 40)
    def binned(d, val, d99):
        rn = d/d99; med = np.full(len(bins)-1, np.nan)
        for j in range(len(bins)-1):
            mb = (rn >= bins[j]) & (rn < bins[j+1])
            if mb.any(): med[j] = np.median(val[mb])
        return 0.5*(bins[:-1]+bins[1:]), med
    rc, ROc_b = binned(dc, ROc, d99c); rs, ROs_b = binned(ds, ROs, d99s)
    _, GAc_b = binned(dc, GAc, d99c); _, GAs_b = binned(ds, GAs, d99s)
    axs[0,i].plot(ROc_b, rc, 'k-', lw=1.4, label='cavity')
    axs[0,i].plot(ROs_b, rs, 'r--', lw=1.4, label='O-grid')
    axs[0,i].axvline(1000, color='0.5', ls=':', lw=0.8)
    axs[0,i].set_xlim(0, 3000); axs[0,i].set_ylim(0, 2.5)
    axs[0,i].set_xlabel(r'$Re_\Omega = d^2\omega/\nu$')
    if i==0: axs[0,i].set_ylabel(r'$d/\delta_{99}$')
    axs[0,i].set_title(f'$x/c={xt:.2f}$', fontsize=10); axs[0,i].grid(alpha=0.3)
    axs[1,i].plot(GAc_b, rc, 'k-', lw=1.4)
    axs[1,i].plot(GAs_b, rs, 'r--', lw=1.4)
    axs[1,i].axvline(1.04, color='0.5', ls=':', lw=0.8)
    axs[1,i].set_xlim(0, 2.0); axs[1,i].set_ylim(0, 2.5)
    axs[1,i].set_xlabel(r'$\Gamma$')
    if i==0: axs[1,i].set_ylabel(r'$d/\delta_{99}$')
    axs[1,i].grid(alpha=0.3)
axs[0,0].legend(fontsize=8, frameon=False, loc='upper right')
plt.tight_layout(pad=0.4)
plt.savefig('/tmp/ReOmega_v2.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/ReOmega_Gamma_decomp.pdf')
print("\nwrote /tmp/ReOmega_v2.png + paper figs/ReOmega_Gamma_decomp.pdf")
