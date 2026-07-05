"""Decompose the cavity-vs-O-grid amplification gap into its constituents:
- Re_Omega = d^2 * omega / nu
- Gamma    = 2 (omega*d)^2 / (|u|^2 + (omega*d)^2)

At each x station, compute omega via VTK gradient filter, then compare distributions
of Re_Omega(d), Gamma(d), and the activation a(Re_Omega, Gamma) on both grids.
"""
import numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

# Flow360 nondimensionalization: muRef = M/Re = 0.2/1e6 = 2e-7
# In nondim units, nu = muRef (assuming rho ~ 1)
NU_REF = 0.2 / 1e6  # = 2e-7

def compute_gradients(d):
    """Read slice, compute velocity gradients via VTK gradient filter, return
    points + arrays + (du/dz, du/dx, dw/dz, dw/dx)."""
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/slice_centerSpan.pvtu"); r.Update()
    g = r.GetOutput()
    # gradient filter
    gf = vtk.vtkGradientFilter()
    gf.SetInputData(g)
    gf.SetInputScalars(0, "velocity")  # vector field
    gf.SetResultArrayName("velgrad")
    gf.SetComputeVorticity(True)
    gf.SetVorticityArrayName("vorticity")
    gf.Update()
    out = gf.GetOutput()
    pd = out.GetPointData()
    arrs = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)) for i in range(pd.GetNumberOfArrays())}
    p = vtk_to_numpy(out.GetPoints().GetData())
    return p, arrs

def find_wall_z(pts, arrs, xt, dx=0.005):
    x = pts[:,0]; z = pts[:,2]
    vel = arrs['velocity']; u_mag = np.sqrt(vel[:,0]**2 + vel[:,2]**2)
    m = (np.abs(x - xt) < dx) & (z > 0) & (z < 0.15) & (u_mag < 0.02)
    if m.sum() == 0: return None
    return float(z[m].min())

def compute_quantities(pts, arrs, xt, dx=0.0008, max_dz=0.02):
    """At x_target, extract (z-z_wall, |u|, |omega|, Re_Omega, Gamma, nuHat) along the
    wall-normal direction (taking points in a tight x band)."""
    x = pts[:,0]; z = pts[:,2]
    z_wall = find_wall_z(pts, arrs, xt, dx=dx*3)
    if z_wall is None: return None
    vel = arrs['velocity']; nuHat = arrs.get('nuHat', None)
    vort = arrs.get('vorticity', None)  # 3-vector
    u_mag = np.sqrt(vel[:,0]**2 + vel[:,2]**2)
    # omega is the dominant component (y-vorticity for 2D in xz plane = dw/dx - du/dz)
    # vort = (curl_x, curl_y, curl_z). For our flow plane (x,z) the relevant component is curl_y.
    omega_mag = np.abs(vort[:,1])
    # mask: tight x band, z above wall
    m = (np.abs(x - xt) < dx) & (z >= z_wall - 1e-5) & (z < z_wall + max_dz)
    if m.sum() < 5: return None
    dzn = z[m] - z_wall  # wall-normal dist (positive above wall)
    u_n = u_mag[m]; o_n = omega_mag[m]; nh_n = nuHat[m] if nuHat is not None else None
    Re_Om = dzn**2 * o_n / NU_REF
    od = o_n * dzn
    Gamma = 2.0*od*od / (u_n*u_n + od*od + 1e-30)
    # sort by dzn
    ix = np.argsort(dzn)
    return dzn[ix], u_n[ix], o_n[ix], Re_Om[ix], Gamma[ix], (nh_n[ix] if nh_n is not None else None)

DCAV = "proper_refineA4_cavity_xxfine"; DSTR = "refineA4_struct_sxxfine"
print("computing gradients on cavity...", flush=True)
pc, ac = compute_gradients(DCAV)
print("computing gradients on struct...", flush=True)
ps, as_ = compute_gradients(DSTR)
print(f"cavity arrays: {list(ac.keys())}")
print(f"struct arrays: {list(as_.keys())}")

# pick 3 x stations spanning before/during/after divergence
X = [0.05, 0.07, 0.10]
fig, axs = plt.subplots(3, len(X), figsize=(3.0*len(X), 7.5), sharey='row')

print(f"\n{'x':>6s} {'grid':>8s} {'d99':>8s} {'ω_peak':>10s} {'Re_Ω_peak':>10s} {'Γ_peak':>8s} {'nu_peak':>10s}")
for i, xt in enumerate(X):
    cav = compute_quantities(pc, ac, xt); sst = compute_quantities(ps, as_, xt)
    if cav is None or sst is None: continue
    dzc, uc, oc, ReOc, GAc, nhc = cav
    dzs, us, os_, ReOs, GAs, nhs = sst
    u_edge_c = uc.max(); u_edge_s = us.max()
    # estimate δ99 (where u first reaches 99%)
    above_c = (uc >= 0.99*u_edge_c) & (dzc > 1e-4)
    above_s = (us >= 0.99*u_edge_s) & (dzs > 1e-4)
    d99c = dzc[above_c].min() if above_c.any() else 0.005
    d99s = dzs[above_s].min() if above_s.any() else 0.005
    # restrict to BL+slight buffer
    bl_c = dzc < 2*d99c
    bl_s = dzs < 2*d99s
    # peak quantities inside BL
    print(f"{xt:6.2f} {'cavity':>8s} {d99c:8.4f} {oc[bl_c].max():10.2e} {ReOc[bl_c].max():10.1f} {GAc[bl_c].max():8.3f} {(nhc[bl_c].max() if nhc is not None else 0):10.2e}")
    print(f"{xt:6.2f} {'O-grid':>8s} {d99s:8.4f} {os_[bl_s].max():10.2e} {ReOs[bl_s].max():10.1f} {GAs[bl_s].max():8.3f} {(nhs[bl_s].max() if nhs is not None else 0):10.2e}")

    # row 0: Re_Omega vs dz/δ
    norm_c = dzc / d99c; norm_s = dzs / d99s
    msk_c = (norm_c < 2.5) & (dzc > 1e-5); msk_s = (norm_s < 2.5) & (dzs > 1e-5)
    axs[0,i].plot(ReOc[msk_c], norm_c[msk_c], 'k.-', ms=2, lw=1.0, label='cavity')
    axs[0,i].plot(ReOs[msk_s], norm_s[msk_s], 'r.--', ms=2, lw=1.0, label='O-grid')
    axs[0,i].axvline(1000, color='0.5', ls=':', lw=0.8)
    axs[0,i].text(1050, 0.5, r'$Re_\Omega\!=\!10^3$', fontsize=7, color='0.5')
    axs[0,i].set_xlim(0, 6000); axs[0,i].set_ylim(0, 2.5)
    axs[0,i].set_xlabel(r'$Re_\Omega$'); axs[0,i].set_title(f"$x/c={xt:.2f}$", fontsize=10)
    if i == 0: axs[0,i].set_ylabel(r'$\Delta z/\delta_{99}$')
    axs[0,i].grid(alpha=0.3)
    # row 1: Gamma
    axs[1,i].plot(GAc[msk_c], norm_c[msk_c], 'k.-', ms=2, lw=1.0)
    axs[1,i].plot(GAs[msk_s], norm_s[msk_s], 'r.--', ms=2, lw=1.0)
    axs[1,i].axvline(1.04, color='0.5', ls=':', lw=0.8)
    axs[1,i].text(1.06, 0.5, r'$g_c\!=\!1.04$', fontsize=7, color='0.5')
    axs[1,i].set_xlim(0, 2); axs[1,i].set_ylim(0, 2.5)
    axs[1,i].set_xlabel(r'$\Gamma$')
    if i == 0: axs[1,i].set_ylabel(r'$\Delta z/\delta_{99}$')
    axs[1,i].grid(alpha=0.3)
    # row 2: vorticity magnitude
    axs[2,i].plot(oc[msk_c], norm_c[msk_c], 'k.-', ms=2, lw=1.0)
    axs[2,i].plot(os_[msk_s], norm_s[msk_s], 'r.--', ms=2, lw=1.0)
    axs[2,i].set_xlabel(r'$|\omega|$')
    if i == 0: axs[2,i].set_ylabel(r'$\Delta z/\delta_{99}$')
    axs[2,i].grid(alpha=0.3); axs[2,i].set_ylim(0, 2.5)

axs[0,0].legend(fontsize=7.5, frameon=False)
plt.tight_layout(pad=0.4)
plt.savefig('/tmp/ReOmega_Gamma_compare.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/ReOmega_Gamma_decomp.pdf')
print("\nwrote /tmp/ReOmega_Gamma_compare.png and paper figs/ReOmega_Gamma_decomp.pdf")
