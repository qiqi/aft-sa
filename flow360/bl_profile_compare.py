"""Extract BL profiles (u, nuHat) at multiple x stations on the finest cavity and
finest O-grid meshes at alpha=4. Find where they diverge.

Approach: at each x station, locate the wall by finding the lowest-z point in the
slice with very small u. Then take a wall-normal cut and measure profile.
"""
import os, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

def load_slice(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/slice_centerSpan.pvtu"); r.Update()
    g = r.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData())
    pd = g.GetPointData()
    arrs = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)) for i in range(pd.GetNumberOfArrays())}
    return p, arrs

def find_wall_z(pts, arrs, x_target, dx=0.005, side='upper'):
    """Find the wall z-coordinate at x_target. The wall is where |u|=0 (no-slip).
    side='upper' picks z>0, side='lower' picks z<0."""
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    vel = arrs['velocity']
    u_mag = np.sqrt(vel[:,0]**2 + vel[:,2]**2)
    m = (np.abs(x - x_target) < dx)
    if side == 'upper': m &= (z > -0.05)
    else: m &= (z < 0.05)
    if m.sum() < 5: return None
    # candidate wall points have small u
    u_at = u_mag[m]; z_at = z[m]
    # the wall point is the one with smallest u (closest to no-slip)
    # but on a high-AR mesh, multiple points may have small u. pick the one with extremum z (closest to airfoil surface)
    threshold = u_at.min() + 0.01  # near-wall band
    near_wall = u_at < threshold
    if not near_wall.any(): return None
    if side == 'upper': return z_at[near_wall].min()
    else: return z_at[near_wall].max()

def extract_profile(pts, arrs, x_target, dx=0.002, side='upper', max_dz=0.02, U_inf=1.0):
    """Extract profile at x_target on the upper or lower surface.
    Returns z (wall-normal coord measured FROM the wall), |u|, nuHat.
    max_dz=0.02 captures BL up to ~2% chord above wall (BL at α=4 is <1% thick mostly)."""
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    vel = arrs['velocity']
    u_mag = np.sqrt(vel[:,0]**2 + vel[:,2]**2)
    nuHat = arrs.get('nuHat', None)
    z_wall = find_wall_z(pts, arrs, x_target, dx=dx*2, side=side)
    if z_wall is None: return None
    # mask: near x_target, on upper side, within max_dz of wall
    if side == 'upper': m = (np.abs(x - x_target) < dx) & (z >= z_wall) & (z < z_wall + max_dz)
    else: m = (np.abs(x - x_target) < dx) & (z <= z_wall) & (z > z_wall - max_dz)
    if m.sum() < 5: return None
    z_sel = z[m]; u_sel = u_mag[m]
    n_sel = nuHat[m] if nuHat is not None else None
    dz = np.abs(z_sel - z_wall)
    o = np.argsort(dz)
    return dz[o], u_sel[o], (n_sel[o] if n_sel is not None else None)

D_CAV = "proper_refineA4_cavity_xxfine"
D_STR = "refineA4_struct_sxxfine"
p_cav, a_cav = load_slice(D_CAV)
p_str, a_str = load_slice(D_STR)
print(f"cavity slice: N={len(p_cav)}, arrays={list(a_cav.keys())}")
print(f"struct slice: N={len(p_str)}, arrays={list(a_str.keys())}")

X_STATIONS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
fig, axs = plt.subplots(2, len(X_STATIONS), figsize=(2.2*len(X_STATIONS), 5.6))

# legend handles
from matplotlib.lines import Line2D
print(f"\n{'x/c':>6s} {'cav δ99':>9s} {'cav nuMax':>11s} {'str δ99':>9s} {'str nuMax':>11s} {'ratio cav/str':>12s}")
for i, xt in enumerate(X_STATIONS):
    cav = extract_profile(p_cav, a_cav, xt, dx=0.003, side='upper')
    sst = extract_profile(p_str, a_str, xt, dx=0.003, side='upper')
    if cav is None or sst is None:
        axs[0,i].text(0.5, 0.5, "no data", transform=axs[0,i].transAxes, ha='center'); continue
    dzc, uc, nc = cav
    dzs, us, ns = sst
    # find δ99 (where u reaches 99% of edge velocity)
    u_edge_c = uc.max(); u_edge_s = us.max()
    # find smallest dz where u/u_edge >= 0.99 (BL edge from below)
    above_99_c = (uc >= 0.99*u_edge_c) & (dzc > 0)
    above_99_s = (us >= 0.99*u_edge_s) & (dzs > 0)
    d99_c = dzc[above_99_c].min() if above_99_c.any() else 0.01
    d99_s = dzs[above_99_s].min() if above_99_s.any() else 0.01
    # plot u/u_edge
    axs[0,i].plot(uc/u_edge_c, dzc, 'k-', lw=1.2, label='cavity (537k)')
    axs[0,i].plot(us/u_edge_s, dzs, '--', color='C3', lw=1.2, label='O-grid (255k)')
    axs[0,i].set_xlim(0, 1.1)
    ylim = max(d99_c, d99_s) * 2.5
    axs[0,i].set_ylim(0, ylim)
    axs[0,i].set_title(f"$x/c={xt:.2f}$", fontsize=10)
    axs[0,i].grid(alpha=0.3)
    if i == 0: axs[0,i].set_ylabel('wall-normal $\\Delta z/c$')
    axs[0,i].set_xlabel('$|u|/u_e$')
    # plot nuHat (log scale)
    if nc is not None and ns is not None:
        # nuHat is normalized to muRef. The actual nuHat in physical units is muRef*nuHat
        # Just plot as-is
        axs[1,i].plot(nc * 1e6, dzc, 'k-', lw=1.2, label='cavity')
        axs[1,i].plot(ns * 1e6, dzs, '--', color='C3', lw=1.2, label='O-grid')
        # axs[1,i].set_xscale('symlog', linthresh=1)
        axs[1,i].set_ylim(0, ylim)
        axs[1,i].set_xlabel(r'$\hat\nu\times 10^{6}$')
        if i == 0: axs[1,i].set_ylabel('wall-normal $\\Delta z/c$')
        axs[1,i].grid(alpha=0.3)
    peak_nc = float(np.nanmax(nc)) if nc is not None else float('nan')
    peak_ns = float(np.nanmax(ns)) if ns is not None else float('nan')
    print(f"{xt:6.2f} {d99_c:9.4f} {peak_nc:11.3e} {d99_s:9.4f} {peak_ns:11.3e} {peak_nc/max(peak_ns,1e-30):12.3f}")

axs[0,0].legend(fontsize=7.5, frameon=False, loc='lower right')
axs[1,0].legend(fontsize=7.5, frameon=False, loc='upper right')
plt.tight_layout(pad=0.4)
plt.savefig('/tmp/bl_profile_compare.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/bl_profile_compare.pdf')
print("\nwrote /tmp/bl_profile_compare.png and paper figs/bl_profile_compare.pdf")
