"""Decomposition of cavity-vs-O-grid Re_Ω and Γ at the same BL location.

The trick: use distance-to-wall and wall-projected chord position as the coordinates,
so the analysis is topology-agnostic. For each slice point, compute its (x_wall, d_wall)
by finding the nearest point on the airfoil contour. Then group by x_wall and plot
profiles vs d_wall.
"""
import numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

NU_REF = 0.2 / 1e6   # Flow360 nondim: muRef = M/Re

def load_with_vorticity(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/slice_centerSpan.pvtu"); r.Update()
    g = r.GetOutput()
    gf = vtk.vtkGradientFilter(); gf.SetInputData(g); gf.SetInputScalars(0, "velocity")
    gf.SetComputeVorticity(True); gf.SetVorticityArrayName("vorticity"); gf.Update()
    out = gf.GetOutput(); pd = out.GetPointData()
    arrs = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)) for i in range(pd.GetNumberOfArrays())}
    return vtk_to_numpy(out.GetPoints().GetData()), arrs

def load_airfoil_upper_contour(d):
    """Get the airfoil upper-surface contour (x, z) by reading the surface VTU and
    contour-walking, then keep upper points."""
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/surface_fluid_naca0012.pvtu"); r.Update()
    g = r.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData())
    x, y, z = p[:,0], p[:,1], p[:,2]
    s = np.abs(y - y.min()) < 1e-6
    X, Z = x[s], z[s]
    n = len(X); pts = np.column_stack([X, Z])
    st = int(np.argmin(X)); o = [st]; u = np.zeros(n, bool); u[st] = True
    for _ in range(n-1):
        c = o[-1]; dd = np.sum((pts - pts[c])**2, 1); dd[u] = 1e9; nx = int(np.argmin(dd)); o.append(nx); u[nx] = True
    o = np.array(o); xo, zo = X[o], Z[o]; te = int(np.argmax(xo))
    b1, b2 = slice(0, te+1), slice(te, n); up = b1 if zo[b1].mean() >= zo[b2].mean() else b2
    xu, zu = xo[up], zo[up]; ix = np.argsort(xu)
    return xu[ix], zu[ix]

def project_to_wall(pts, xc, zc, side='upper'):
    """For each (x,z) slice point, find nearest airfoil contour point (x_w, z_w),
    return wall-projected x (x_w), perpendicular distance d (sign>0 if above wall).
    The 'above wall' side is z > z_w for the upper surface."""
    x, z = pts[:,0], pts[:,2]
    # nearest contour point for each slice point
    # vectorized: chunk to avoid OOM
    N = len(x); chunk = 50000
    x_w = np.empty(N); z_w = np.empty(N); d = np.empty(N)
    for s in range(0, N, chunk):
        e = min(s+chunk, N)
        xs = x[s:e][:, None]; zs = z[s:e][:, None]
        dd = (xs - xc[None,:])**2 + (zs - zc[None,:])**2
        idx = np.argmin(dd, axis=1)
        x_w[s:e] = xc[idx]; z_w[s:e] = zc[idx]
        d[s:e] = np.sqrt(dd[np.arange(e-s), idx])
    # sign: positive if above upper surface; for the contour stored, "above" means z > z_w
    sign = np.where(z > z_w, 1.0, -1.0)
    return x_w, d * sign

DCAV = "proper_refineA4_cavity_xxfine"
DSTR = "refineA4_struct_sxxfine"
print("loading + computing gradients ...", flush=True)
p_cav, a_cav = load_with_vorticity(DCAV)
p_str, a_str = load_with_vorticity(DSTR)
print("loading contour ...", flush=True)
xc_cav, zc_cav = load_airfoil_upper_contour(DCAV)
xc_str, zc_str = load_airfoil_upper_contour(DSTR)
print(f"cavity contour: N={len(xc_cav)}, struct contour: N={len(xc_str)}", flush=True)

print("projecting points to wall coords ...", flush=True)
xw_cav, d_cav = project_to_wall(p_cav, xc_cav, zc_cav)
xw_str, d_str = project_to_wall(p_str, xc_str, zc_str)
print(f"  done", flush=True)

def quantities(pts, arrs, xw_arr, d_arr, xt, dx=0.002, max_d=0.02):
    """Extract upper-BL profile at wall-projected x = xt."""
    vel = arrs['velocity']; vort = arrs['vorticity']; nuHat = arrs.get('nuHat', None)
    u_mag = np.sqrt(vel[:,0]**2 + vel[:,2]**2); omega = np.abs(vort[:,1])
    m = (np.abs(xw_arr - xt) < dx) & (d_arr > 0) & (d_arr < max_d)
    if m.sum() < 5: return None
    d = d_arr[m]; un = u_mag[m]; on = omega[m]; nh = nuHat[m] if nuHat is not None else None
    ix = np.argsort(d)
    d = d[ix]; un = un[ix]; on = on[ix]
    if nh is not None: nh = nh[ix]
    Re_Om = d**2 * on / NU_REF
    od = on * d
    Gamma = 2.0*od*od / (un*un + od*od + 1e-30)
    return d, un, on, Re_Om, Gamma, nh

print("\n=== quantities at upper-surface stations (after wall-projection) ===")
print(f"{'x_w':>6s}  {'grid':>7s}  {'δ99':>8s}  {'ω_pk':>10s}  {'ReΩ_pk':>10s}  {'Γ_pk':>8s}  {'(ωd)_pk':>10s}  {'|u|@pk':>8s}")
X = [0.03, 0.05, 0.07, 0.10, 0.13, 0.20]
for xt in X:
    cav = quantities(p_cav, a_cav, xw_cav, d_cav, xt)
    sst = quantities(p_str, a_str, xw_str, d_str, xt)
    if cav is None or sst is None: continue
    dc, uc, oc, ROc, GAc, nhc = cav
    ds, us, os_, ROs, GAs, nhs = sst
    # δ99
    uec = uc.max(); ues = us.max()
    d99c = dc[(uc >= 0.99*uec) & (dc > 1e-4)].min() if ((uc >= 0.99*uec) & (dc > 1e-4)).any() else 0.005
    d99s = ds[(us >= 0.99*ues) & (ds > 1e-4)].min() if ((us >= 0.99*ues) & (ds > 1e-4)).any() else 0.005
    # location of Re_Ω peak
    bl_c = dc < 1.5*d99c; bl_s = ds < 1.5*d99s
    if not bl_c.any() or not bl_s.any(): continue
    iROc = int(np.argmax(ROc[bl_c])); iROs = int(np.argmax(ROs[bl_s]))
    od_c_pk = (oc * dc)[bl_c][iROc]; od_s_pk = (os_ * ds)[bl_s][iROs]
    print(f"{xt:6.2f}  {'cavity':>7s}  {d99c:8.4f}  {oc[bl_c].max():10.2e}  {ROc[bl_c].max():10.1f}  {GAc[bl_c].max():8.3f}  {od_c_pk:10.2e}  {uc[bl_c][iROc]:8.3f}")
    print(f"{xt:6.2f}  {'O-grid':>7s}  {d99s:8.4f}  {os_[bl_s].max():10.2e}  {ROs[bl_s].max():10.1f}  {GAs[bl_s].max():8.3f}  {od_s_pk:10.2e}  {us[bl_s][iROs]:8.3f}")

# plot 3 panels: Re_Omega(d), Gamma(d), nuHat(d) at 3 x stations
fig, axs = plt.subplots(3, 3, figsize=(9.0, 8.0), sharey='row')
for i, xt in enumerate([0.05, 0.07, 0.10]):
    cav = quantities(p_cav, a_cav, xw_cav, d_cav, xt)
    sst = quantities(p_str, a_str, xw_str, d_str, xt)
    if cav is None or sst is None: continue
    dc, uc, oc, ROc, GAc, nhc = cav
    ds, us, os_, ROs, GAs, nhs = sst
    uec = uc.max(); ues = us.max()
    d99c = dc[(uc >= 0.99*uec) & (dc > 1e-4)].min() if ((uc >= 0.99*uec) & (dc > 1e-4)).any() else 0.005
    d99s = ds[(us >= 0.99*ues) & (ds > 1e-4)].min() if ((us >= 0.99*ues) & (ds > 1e-4)).any() else 0.005
    # normalize d by δ99 of the GRID (cavity by cav δ99, struct by struct δ99)
    ax = axs[0,i]
    ax.plot(ROc, dc/d99c, 'k.-', ms=2, lw=1.0, label='cavity')
    ax.plot(ROs, ds/d99s, 'r.--', ms=2, lw=1.0, label='O-grid')
    ax.set_xlim(0, 5000); ax.set_ylim(0, 2.5)
    ax.set_xlabel(r'$Re_\Omega = d^2\omega/\nu$')
    if i==0: ax.set_ylabel(r'$d/\delta_{99}$')
    ax.set_title(f"$x/c={xt:.2f}$", fontsize=10); ax.grid(alpha=0.3)
    ax = axs[1,i]
    ax.plot(GAc, dc/d99c, 'k.-', ms=2, lw=1.0, label='cavity')
    ax.plot(GAs, ds/d99s, 'r.--', ms=2, lw=1.0, label='O-grid')
    ax.set_xlim(0, 2.0); ax.set_ylim(0, 2.5)
    ax.set_xlabel(r'$\Gamma$')
    if i==0: ax.set_ylabel(r'$d/\delta_{99}$')
    ax.grid(alpha=0.3); ax.axvline(1.04, color='0.5', ls=':', lw=0.8)
    ax = axs[2,i]
    if nhc is not None and nhs is not None:
        ax.semilogx(nhc + 1e-12, dc/d99c, 'k.-', ms=2, lw=1.0)
        ax.semilogx(nhs + 1e-12, ds/d99s, 'r.--', ms=2, lw=1.0)
        ax.set_xlim(1e-10, 1e-4); ax.set_ylim(0, 2.5)
        ax.set_xlabel(r'$\hat\nu$')
        if i==0: ax.set_ylabel(r'$d/\delta_{99}$')
        ax.grid(alpha=0.3, which='both')
axs[0,0].legend(fontsize=8, frameon=False)
plt.tight_layout(pad=0.5)
plt.savefig('/tmp/ReOmega_Gamma_decomp.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/ReOmega_Gamma_decomp.pdf')
print("\nwrote /tmp/ReOmega_Gamma_decomp.png + paper figs/ReOmega_Gamma_decomp.pdf")
