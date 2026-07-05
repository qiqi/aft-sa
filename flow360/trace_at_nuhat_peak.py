"""For each x station along the chord, find d* where ν̂ peaks in the BL on each
grid. Then trace Re_Omega, Gamma, omega, u, d, and a·ω AT that d*.

This is what actually drives ν̂'s growth: dν̂/dt at d* tells us how fast the
peak of ν̂ is amplifying.
"""
import numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

NU_REF = 0.2 / 1e6
# AFT constants
A_MAX = 0.05; S_PARAM = 35.0; RE0 = 1000.0; K_PARAM = 50.0; G_C = 1.04
RE_NA = 5.0; M_EXP = 2.0

def activation(Re, G):
    a = 0.0
    if Re > RE_NA:
        z = S_PARAM*(np.log10(Re/RE0)/K_PARAM + G - G_C) + M_EXP*np.log(max(1 - RE_NA/Re, 1e-30))
        a = A_MAX / (1.0 + np.exp(-z))
    return a

def load_with_vorticity(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/slice_centerSpan.pvtu"); r.Update()
    g = r.GetOutput()
    gf = vtk.vtkGradientFilter(); gf.SetInputData(g); gf.SetInputScalars(0, "velocity")
    gf.SetComputeVorticity(True); gf.SetVorticityArrayName("vorticity"); gf.Update()
    out = gf.GetOutput(); pd = out.GetPointData()
    arrs = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)) for i in range(pd.GetNumberOfArrays())}
    return vtk_to_numpy(out.GetPoints().GetData()), arrs

def airfoil_polyline(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/surface_fluid_naca0012.pvtu"); r.Update()
    g = r.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData())
    x, y, z = p[:,0], p[:,1], p[:,2]
    s = np.abs(y - y.min()) < 1e-6
    X, Z = x[s], z[s]
    n = len(X); pts2 = np.column_stack([X, Z])
    st = int(np.argmin(X)); o = [st]; u = np.zeros(n, bool); u[st] = True
    for _ in range(n-1):
        c = o[-1]; dd = np.sum((pts2 - pts2[c])**2, 1); dd[u] = 1e9; nx = int(np.argmin(dd)); o.append(nx); u[nx] = True
    o = np.array(o)
    return X[o], Z[o]

def project_signed(pts, x_contour, z_contour):
    N = len(pts); xz = np.column_stack([pts[:,0], pts[:,2]])
    Ncont = len(x_contour)
    A = np.column_stack([x_contour, z_contour]); B = np.roll(A, -1, axis=0); seg = B - A
    seg_len2 = (seg**2).sum(axis=1); seg_len2[seg_len2 == 0] = 1e-30
    d_min = np.full(N, np.inf); proj_x = np.empty(N); proj_z = np.empty(N)
    chunk = 1000
    for s in range(0, Ncont, chunk):
        e = min(s+chunk, Ncont)
        AP = xz[:, None, :] - A[None, s:e, :]
        t = (AP * seg[None, s:e, :]).sum(axis=2) / seg_len2[None, s:e]; t = np.clip(t, 0.0, 1.0)
        proj = A[None, s:e, :] + t[:, :, None] * seg[None, s:e, :]
        dd2 = ((xz[:, None, :] - proj)**2).sum(axis=2)
        i_loc = np.argmin(dd2, axis=1); d_loc = dd2[np.arange(N), i_loc]
        better = d_loc < d_min
        d_min[better] = d_loc[better]
        proj_x[better] = proj[better, i_loc[better], 0]; proj_z[better] = proj[better, i_loc[better], 1]
    d = np.sqrt(d_min); sign = np.where(pts[:,2] > proj_z, 1.0, -1.0)
    return proj_x, d * sign

def extract_at_x(pts, arrs, xw_arr, d_arr, xt, dx=0.003, max_d=0.012):
    """Return (d, u, omega, nuHat) profiles at x_target."""
    vel = arrs['velocity']; vort = arrs['vorticity']; nuHat = arrs['nuHat']
    u_mag = np.sqrt(vel[:,0]**2 + vel[:,2]**2); omega = np.abs(vort[:,1])
    m = (np.abs(xw_arr - xt) < dx) & (d_arr > 0) & (d_arr < max_d)
    if m.sum() < 5: return None
    ix = np.argsort(d_arr[m])
    return d_arr[m][ix], u_mag[m][ix], omega[m][ix], nuHat[m][ix]

def bin_smooth(d, val, nbins=40, max_d=0.01):
    bins = np.linspace(0, max_d, nbins+1); db = 0.5*(bins[:-1]+bins[1:])
    out = np.full(nbins, np.nan)
    for j in range(nbins):
        m = (d >= bins[j]) & (d < bins[j+1])
        if m.any(): out[j] = np.median(val[m])
    return db, out

DCAV = "proper_refineA4_cavity_xxfine"; DSTR = "refineA4_struct_sxxfine"
print("loading + gradients ...", flush=True)
pc, ac = load_with_vorticity(DCAV); ps, as_ = load_with_vorticity(DSTR)
xc_cav, zc_cav = airfoil_polyline(DCAV); xc_str, zc_str = airfoil_polyline(DSTR)
print("wall distances ...", flush=True)
xw_cav, d_cav = project_signed(pc, xc_cav, zc_cav)
xw_str, d_str = project_signed(ps, xc_str, zc_str)

# scan x
xs = np.linspace(0.01, 0.30, 30)
results = {'cav': {}, 'str': {}}
for grid, pts, arrs, xw, dw in [('cav', pc, ac, xw_cav, d_cav), ('str', ps, as_, xw_str, d_str)]:
    rows = []
    for xt in xs:
        prof = extract_at_x(pts, arrs, xw, dw, xt)
        if prof is None: continue
        d, u, om, nu = prof
        db, ub = bin_smooth(d, u); _, ob = bin_smooth(d, om); _, nb = bin_smooth(d, nu)
        # u_edge
        u_e = np.nanmax(ub) if np.isfinite(np.nanmax(ub)) else 1.0
        above = (ub/u_e >= 0.99) & (db > 1e-4)
        d99 = db[np.where(above)[0][0]] if above.any() else 0.005
        # find d* where binned nuHat peaks within BL
        bl = (db > 5e-5) & (db < 1.5*d99) & np.isfinite(nb)
        if not bl.any(): continue
        i_peak = int(np.argmax(nb[bl]))
        d_star = db[bl][i_peak]
        nu_star = float(nb[bl][i_peak])
        # interpolate u, omega at d_star from binned profile
        u_star = float(np.interp(d_star, db, ub))
        om_star = float(np.interp(d_star, db, ob))
        Re_star = d_star*d_star * om_star / NU_REF
        od_star = om_star * d_star
        G_star = 2*od_star*od_star / (u_star*u_star + od_star*od_star + 1e-30)
        a_star = activation(Re_star, G_star)
        growth_star = a_star * om_star
        rows.append(dict(x=xt, d_star=d_star, nu_star=nu_star, u_star=u_star,
                         om_star=om_star, Re_star=Re_star, G_star=G_star,
                         a_star=a_star, growth=growth_star, d99=d99, u_e=u_e))
    results[grid] = rows

# table
print(f"\n{'x':>5s} {'grid':>4s} {'d*':>8s} {'d*/δ99':>7s} {'ν̂*':>10s} {'ω*':>8s} {'Re_Ω*':>8s} {'Γ*':>6s} {'a*':>7s} {'a·ω':>7s} {'u*':>6s}")
n = max(len(results['cav']), len(results['str']))
for i in range(n):
    rc = results['cav'][i] if i < len(results['cav']) else None
    rs = results['str'][i] if i < len(results['str']) else None
    for tag, r in [('cav', rc), ('str', rs)]:
        if r is None: continue
        print(f"{r['x']:5.3f} {tag:>4s} {r['d_star']:8.4f} {r['d_star']/r['d99']:7.3f} {r['nu_star']:10.3e} {r['om_star']:8.2f} {r['Re_star']:8.1f} {r['G_star']:6.3f} {r['a_star']:7.4f} {r['growth']:7.2f} {r['u_star']:6.3f}")

# also build a 4-panel plot: each quantity at d* vs x, both grids overlaid
fig, axs = plt.subplots(2, 3, figsize=(12, 6.5))
xc_arr = np.array([r['x'] for r in results['cav']])
xs_arr = np.array([r['x'] for r in results['str']])
def pull(grid_rows, key):
    return np.array([r[key] for r in grid_rows])

# (a) ν̂*
axs[0,0].semilogy(xc_arr, pull(results['cav'], 'nu_star'), 'k-', lw=1.5, label='cavity')
axs[0,0].semilogy(xs_arr, pull(results['str'], 'nu_star'), 'r--', lw=1.5, label='O-grid')
axs[0,0].set_xlabel('$x/c$'); axs[0,0].set_ylabel(r'$\hat\nu^{\max}$ (peak in BL)')
axs[0,0].set_title('(a) amplification factor at $d^*$', fontsize=10)
axs[0,0].grid(alpha=0.3, which='both'); axs[0,0].legend(fontsize=8, frameon=False)

# (b) Re_Omega at d*
axs[0,1].plot(xc_arr, pull(results['cav'], 'Re_star'), 'k-', lw=1.5)
axs[0,1].plot(xs_arr, pull(results['str'], 'Re_star'), 'r--', lw=1.5)
axs[0,1].axhline(RE0, color='0.5', ls=':', lw=0.8)
axs[0,1].set_xlabel('$x/c$'); axs[0,1].set_ylabel(r'$Re_\Omega$ at $d^*$')
axs[0,1].set_title(r'(b) $Re_\Omega$ at $d^*$ (where $\hat\nu$ peaks)', fontsize=10)
axs[0,1].grid(alpha=0.3)

# (c) Gamma at d*
axs[0,2].plot(xc_arr, pull(results['cav'], 'G_star'), 'k-', lw=1.5)
axs[0,2].plot(xs_arr, pull(results['str'], 'G_star'), 'r--', lw=1.5)
axs[0,2].axhline(G_C, color='0.5', ls=':', lw=0.8)
axs[0,2].set_xlabel('$x/c$'); axs[0,2].set_ylabel(r'$\Gamma$ at $d^*$')
axs[0,2].set_title(r'(c) $\Gamma$ at $d^*$', fontsize=10)
axs[0,2].grid(alpha=0.3)

# (d) d*  and δ99
axs[1,0].plot(xc_arr, pull(results['cav'], 'd_star'), 'k-', lw=1.5, label=r'cavity $d^*$')
axs[1,0].plot(xs_arr, pull(results['str'], 'd_star'), 'r--', lw=1.5, label=r'O-grid $d^*$')
axs[1,0].plot(xc_arr, pull(results['cav'], 'd99'), 'k:', lw=1.0, label=r'cavity $\delta_{99}$')
axs[1,0].plot(xs_arr, pull(results['str'], 'd99'), 'r:', lw=1.0, label=r'O-grid $\delta_{99}$')
axs[1,0].set_xlabel('$x/c$'); axs[1,0].set_ylabel(r'$d^*$ and $\delta_{99}$')
axs[1,0].set_title(r'(d) wall-normal locations', fontsize=10)
axs[1,0].grid(alpha=0.3); axs[1,0].legend(fontsize=7, frameon=False)

# (e) a*
axs[1,1].plot(xc_arr, pull(results['cav'], 'a_star'), 'k-', lw=1.5)
axs[1,1].plot(xs_arr, pull(results['str'], 'a_star'), 'r--', lw=1.5)
axs[1,1].axhline(A_MAX, color='0.5', ls=':', lw=0.8)
axs[1,1].set_xlabel('$x/c$'); axs[1,1].set_ylabel(r'$a(Re_\Omega, \Gamma)$ at $d^*$')
axs[1,1].set_title(r'(e) activation rate at $d^*$', fontsize=10)
axs[1,1].grid(alpha=0.3)
axs[1,1].set_ylim(0, A_MAX*1.1)

# (f) growth rate a*·ω*
axs[1,2].plot(xc_arr, pull(results['cav'], 'growth'), 'k-', lw=1.5)
axs[1,2].plot(xs_arr, pull(results['str'], 'growth'), 'r--', lw=1.5)
axs[1,2].set_xlabel('$x/c$'); axs[1,2].set_ylabel(r'$a\cdot\omega$ at $d^*$')
axs[1,2].set_title(r'(f) growth rate of $\hat\nu^{\max}$', fontsize=10)
axs[1,2].grid(alpha=0.3)

plt.tight_layout(pad=0.5)
plt.savefig('/tmp/trace_at_nuhat_peak.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/trace_at_nuhat_peak.pdf')
print("\nwrote /tmp/trace_at_nuhat_peak.png + paper figs/trace_at_nuhat_peak.pdf")
