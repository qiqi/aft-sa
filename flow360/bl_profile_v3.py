"""BL profile extraction v3: for each x station, take a narrow band, find wall by min-u,
then plot ALL points by z-z_wall (no thinning). The BL is narrow at LE so a tight band is
needed. We also show nuHat in absolute units (×1e6 for readability)."""
import numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def load_slice(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/slice_centerSpan.pvtu"); r.Update()
    g = r.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    arrs = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)) for i in range(pd.GetNumberOfArrays())}
    return p, arrs

def profile(pts, arrs, xt, dx=0.0005, max_dz_above=0.04):
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    vel = arrs['velocity']; nuHat = arrs['nuHat']
    u_mag = np.sqrt(vel[:,0]**2 + vel[:,2]**2)
    # broad band around xt to locate wall
    m_wide = (np.abs(x - xt) < dx*3) & (z > 0) & (z < 0.15)
    if m_wide.sum() < 10: return None
    # wall is where u is minimum (no-slip)
    # find points with u below half of (u_min + u_range)
    u_min_band = u_mag[m_wide].min()
    near_wall = m_wide & (u_mag < u_min_band + 0.05)
    if near_wall.sum() < 3: return None
    z_wall = z[near_wall].min()  # smallest z near wall
    # tight band around xt, take all points with z>=z_wall up to z_wall+max_dz
    m = (np.abs(x - xt) < dx) & (z >= z_wall - 1e-5) & (z < z_wall + max_dz_above)
    if m.sum() < 5: return None
    zr = z[m] - z_wall; ur = u_mag[m]; nr = nuHat[m]
    o = np.argsort(zr)
    return zr[o], ur[o], nr[o]

DCAV = "proper_refineA4_cavity_xxfine"; DSTR = "refineA4_struct_sxxfine"
p_cav, a_cav = load_slice(DCAV); p_str, a_str = load_slice(DSTR)

X = [0.02, 0.04, 0.07, 0.10, 0.13, 0.20]
fig, axs = plt.subplots(2, len(X), figsize=(2.0*len(X), 5.8), sharey='row')

print(f"{'x':>6s} {'cav δ99':>10s} {'cav nuPeak':>12s} {'str δ99':>10s} {'str nuPeak':>12s}  ratio")
for i, xt in enumerate(X):
    pc = profile(p_cav, a_cav, xt); ps = profile(p_str, a_str, xt)
    if pc is None or ps is None:
        axs[0,i].text(0.5, 0.5, "no data", transform=axs[0,i].transAxes, ha='center'); continue
    zc, uc, nc = pc; zs, us, ns = ps
    u_edge_c = uc.max(); u_edge_s = us.max()
    # δ99: smallest dz where u/u_edge >= 0.99
    above_c = (uc >= 0.99*u_edge_c) & (zc > 1e-4)
    above_s = (us >= 0.99*u_edge_s) & (zs > 1e-4)
    d99c = zc[above_c].min() if above_c.any() else 0.005
    d99s = zs[above_s].min() if above_s.any() else 0.005
    npc = float(nc.max()); nps_ = float(ns.max())
    print(f"{xt:6.2f} {d99c:10.5f} {npc:12.3e} {d99s:10.5f} {nps_:12.3e}  {npc/max(nps_,1e-30):.3f}")
    # plot u(z) — normalize by edge
    axs[0,i].plot(uc/u_edge_c, zc, 'k.-', ms=2, lw=1.0, label='cavity')
    axs[0,i].plot(us/u_edge_s, zs, 'r.--', ms=2, lw=1.0, label='O-grid')
    ylim = max(d99c, d99s) * 3
    if ylim == 0: ylim = 0.005
    axs[0,i].set_ylim(0, ylim); axs[0,i].set_xlim(-0.05, 1.1)
    axs[0,i].grid(alpha=0.3)
    axs[0,i].set_title(f"$x/c={xt:.2f}$", fontsize=10)
    if i == 0: axs[0,i].set_ylabel(r'$\Delta z/c$ (above wall)')
    axs[0,i].set_xlabel(r'$|u|/u_e$')
    # plot nuHat — log scale to span range
    axs[1,i].semilogx(np.abs(nc)+1e-12, zc, 'k.-', ms=2, lw=1.0, label='cavity')
    axs[1,i].semilogx(np.abs(ns)+1e-12, zs, 'r.--', ms=2, lw=1.0, label='O-grid')
    axs[1,i].set_xlim(1e-11, 1e-4); axs[1,i].set_ylim(0, ylim)
    axs[1,i].grid(alpha=0.3, which='both')
    if i == 0: axs[1,i].set_ylabel(r'$\Delta z/c$')
    axs[1,i].set_xlabel(r'$|\hat\nu|$')
axs[0,0].legend(fontsize=7.5, frameon=False, loc='upper right')
axs[1,0].legend(fontsize=7.5, frameon=False, loc='upper right')
plt.tight_layout(pad=0.4)
plt.savefig('/tmp/bl_profile_v3.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/bl_profile_compare.pdf')
print("\nwrote /tmp/bl_profile_v3.png and paper figs/bl_profile_compare.pdf")
