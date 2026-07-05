"""Detailed multi-panel BL diagnostic at one x station where the cavity and O-grid
differ most. We pick x/c = 0.07 (mid-divergence; both still pre-transition).
Plots: u, ω, d (=z-z_wall), Re_Ω, Γ, a, growth-rate kernel a·ω, and nuHat.
All as functions of d, on the SAME y-axis scale so the BL structure is comparable.
"""
import numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

NU_REF = 0.2 / 1e6   # Flow360 muRef = M/Re

# AFT activation constants (from Table 2 of the paper)
A_MAX = 0.05
S_PARAM = 35.0
RE0 = 1000.0
K_PARAM = 50.0
G_C = 1.04
RE_NA = 5.0      # no-amplification floor
M_EXP = 2.0

def activation(ReOmega, Gamma):
    """Return a(Re_Ω, Γ) using Eq.(rate) of the paper, with the no-amp floor."""
    a = np.zeros_like(ReOmega)
    mask = ReOmega > RE_NA
    if mask.any():
        # z = s·( (1/k)·log10(Re_Ω/Re_0) + Γ - g_c ) + m·ln(1 - Re_na/Re_Ω)
        ratio = np.maximum(1 - RE_NA/ReOmega[mask], 1e-30)
        z = S_PARAM * (np.log10(ReOmega[mask]/RE0)/K_PARAM + Gamma[mask] - G_C) \
            + M_EXP * np.log(ratio)
        a[mask] = A_MAX / (1.0 + np.exp(-z))
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

def extract(pts, arrs, xw_arr, d_arr, xt, dx=0.003, max_d=0.01):
    vel = arrs['velocity']; vort = arrs['vorticity']; nuHat = arrs.get('nuHat', None)
    u_mag = np.sqrt(vel[:,0]**2 + vel[:,2]**2); omega = np.abs(vort[:,1])
    m = (np.abs(xw_arr - xt) < dx) & (d_arr > 0) & (d_arr < max_d)
    if m.sum() < 5: return None
    d = d_arr[m]; un = u_mag[m]; on = omega[m]; nh = nuHat[m]
    ix = np.argsort(d)
    return d[ix], un[ix], on[ix], nh[ix]

def bin_median(d, val, bins):
    out = np.full(len(bins)-1, np.nan)
    for j in range(len(bins)-1):
        m = (d >= bins[j]) & (d < bins[j+1])
        if m.any(): out[j] = np.median(val[m])
    return out

DCAV = "proper_refineA4_cavity_xxfine"; DSTR = "refineA4_struct_sxxfine"
print("loading data + gradients ...", flush=True)
pc, ac = load_with_vorticity(DCAV); ps, as_ = load_with_vorticity(DSTR)
xc_cav, zc_cav = airfoil_polyline(DCAV); xc_str, zc_str = airfoil_polyline(DSTR)
print("computing wall distances ...", flush=True)
xw_cav, d_cav = project_signed(pc, xc_cav, zc_cav)
xw_str, d_str = project_signed(ps, xc_str, zc_str)

XT = 0.07   # divergence-zone center; both grids still laminar
print(f"\n=== detailed diagnostic at x/c = {XT} ===\n")

cav = extract(pc, ac, xw_cav, d_cav, XT)
sst = extract(ps, as_, xw_str, d_str, XT)
dc_raw, uc_raw, oc_raw, nhc_raw = cav
ds_raw, us_raw, os_raw, nhs_raw = sst

# bin by d to smooth scatter
bins = np.linspace(0, 0.01, 60); db = 0.5*(bins[1:]+bins[:-1])
uc = bin_median(dc_raw, uc_raw, bins); us_ = bin_median(ds_raw, us_raw, bins)
oc = bin_median(dc_raw, oc_raw, bins); os_ = bin_median(ds_raw, os_raw, bins)
nhc = bin_median(dc_raw, nhc_raw, bins); nhs = bin_median(ds_raw, nhs_raw, bins)
# derived
ReOc = db**2 * oc / NU_REF; ReOs = db**2 * os_ / NU_REF
odc = oc * db; ods = os_ * db
GAc = 2*odc*odc / (uc*uc + odc*odc + 1e-30); GAs = 2*ods*ods / (us_*us_ + ods*ods + 1e-30)
ac_a = activation(np.nan_to_num(ReOc), np.nan_to_num(GAc))
as_a = activation(np.nan_to_num(ReOs), np.nan_to_num(GAs))
growth_c = ac_a * oc; growth_s = as_a * os_

# u_edge for each grid
u_e_c = np.nanmax(uc); u_e_s = np.nanmax(us_)
d99c = db[np.where((uc/u_e_c >= 0.99) & (db > 1e-4))[0][0]] if ((uc/u_e_c >= 0.99) & (db > 1e-4)).any() else 0.005
d99s = db[np.where((us_/u_e_s >= 0.99) & (db > 1e-4))[0][0]] if ((us_/u_e_s >= 0.99) & (db > 1e-4)).any() else 0.005
print(f"  cavity: δ99 = {d99c:.4f},  u_edge = {u_e_c:.3f},  ω_wall (near) = {oc[0]:.1f}")
print(f"  O-grid: δ99 = {d99s:.4f},  u_edge = {u_e_s:.3f},  ω_wall (near) = {os_[0]:.1f}")

# build the 8-panel figure
fig, axs = plt.subplots(2, 4, figsize=(15, 9))
y_lim = 0.005   # consistent y limits
LBL_CAV = 'cavity (537k)'; LBL_STR = 'O-grid (255k)'
def style(ax, x_label, x_lim=None, log_x=False, vline=None):
    if log_x: ax.set_xscale('log')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylim(0, y_lim); ax.set_ylabel(r'wall-normal $d/c$', fontsize=11)
    ax.grid(alpha=0.3, which='both')
    if x_lim is not None: ax.set_xlim(*x_lim)
    if vline is not None: ax.axvline(vline, color='0.5', ls=':', lw=0.8)
    ax.tick_params(labelsize=9)

# 1) u/u_edge
ax = axs[0,0]
ax.plot(uc/u_e_c, db, 'k-', lw=1.4, label=LBL_CAV)
ax.plot(us_/u_e_s, db, 'r--', lw=1.4, label=LBL_STR)
ax.axhline(d99c, color='k', ls=':', lw=0.8); ax.axhline(d99s, color='r', ls=':', lw=0.8)
style(ax, r'$|u|/u_e$', x_lim=(0, 1.05))
ax.set_title(f'(a) velocity profile  $x/c={XT}$', fontsize=11)
ax.legend(fontsize=7.5, frameon=False, loc='lower right')

# 2) vorticity ω
ax = axs[0,1]; ax.plot(oc, db, 'k-', lw=1.4); ax.plot(os_, db, 'r--', lw=1.4)
style(ax, r'$|\omega|$', x_lim=(0, max(np.nanmax(oc), np.nanmax(os_))*1.1))
ax.set_title(r'(b) vorticity $\omega(d)$', fontsize=11)

# 3) Re_Omega
ax = axs[0,2]; ax.plot(ReOc, db, 'k-', lw=1.4); ax.plot(ReOs, db, 'r--', lw=1.4)
style(ax, r'$Re_\Omega = d^2\omega/\nu$', x_lim=(0, 1500))
ax.axvline(RE0, color='0.5', ls=':', lw=0.8)
ax.set_title(r'(c) $Re_\Omega$ ', fontsize=11)

# 4) Gamma
ax = axs[0,3]; ax.plot(GAc, db, 'k-', lw=1.4); ax.plot(GAs, db, 'r--', lw=1.4)
style(ax, r'$\Gamma = 2(\omega d)^2/(|u|^2+(\omega d)^2)$', x_lim=(0, 2))
ax.axvline(G_C, color='0.5', ls=':', lw=0.8)
ax.set_title(r'(d) $\Gamma$', fontsize=11)

# 5) activation a
ax = axs[1,0]; ax.plot(ac_a, db, 'k-', lw=1.4); ax.plot(as_a, db, 'r--', lw=1.4)
style(ax, r'$a(Re_\Omega, \Gamma)$', x_lim=(0, A_MAX*1.1))
ax.set_title(r'(e) activation $a$', fontsize=11)

# 6) growth rate a·ω
ax = axs[1,1]; ax.plot(growth_c, db, 'k-', lw=1.4); ax.plot(growth_s, db, 'r--', lw=1.4)
style(ax, r'$a\,\omega$', x_lim=(0, max(np.nanmax(growth_c), np.nanmax(growth_s))*1.2))
ax.set_title(r'(f) growth rate $a\cdot\omega$', fontsize=11)

# 7) nuHat
ax = axs[1,2]; ax.semilogx(np.abs(nhc)+1e-12, db, 'k-', lw=1.4); ax.semilogx(np.abs(nhs)+1e-12, db, 'r--', lw=1.4)
style(ax, r'$\hat\nu$', log_x=True, x_lim=(1e-10, 1e-5))
ax.set_title(r'(g) amplification factor $\hat\nu$', fontsize=11)

# 8) ω·d (this is the active integrand for amplification region)
ax = axs[1,3]; ax.plot(odc, db, 'k-', lw=1.4); ax.plot(ods, db, 'r--', lw=1.4)
style(ax, r'$\omega\cdot d$', x_lim=(0, max(np.nanmax(odc), np.nanmax(ods))*1.2))
ax.set_title(r'(h) $\omega\cdot d$ (numerator of $\Gamma$)', fontsize=11)

plt.tight_layout(pad=0.4)
plt.savefig('/tmp/divergence_zoom.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/divergence_zoom.pdf')
print("wrote /tmp/divergence_zoom.png + paper figs/divergence_zoom.pdf")

# integrate growth rate over BL
def integ_growth(d, g, d99):
    m = (d > 0) & (d < 1.5*d99) & np.isfinite(g)
    if m.sum() < 3: return np.nan
    return np.trapz(g[m], d[m])
ig_c = integ_growth(db, growth_c, d99c)
ig_s = integ_growth(db, growth_s, d99s)
print(f"\n  integrated growth rate ∫ a·ω dz over BL:")
print(f"    cavity = {ig_c:.4f}    O-grid = {ig_s:.4f}    ratio = {ig_c/ig_s:.3f}")
print(f"  peak (a·ω):  cavity = {np.nanmax(growth_c):.2f},  O-grid = {np.nanmax(growth_s):.2f}")
