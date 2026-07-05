"""Plot peak nuHat in the BL vs chord station x, for finest cavity and finest O-grid.
This directly addresses 'where do the two grids start to diverge'.

For each x, take a wide spatial band that captures the BL on either grid and
find the max of nuHat. The peak is robust to topology differences."""
import numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

def load(d):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/slice_centerSpan.pvtu"); r.Update()
    g = r.GetOutput(); p = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    arrs = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)) for i in range(pd.GetNumberOfArrays())}
    return p, arrs

def nu_peak_in_band(pts, arrs, xt, dx=0.002, z_band=0.05):
    """Return the maximum nuHat in a band around x_target, restricted to a narrow z-strip
    around the airfoil. Use 'narrow z' to avoid sweeping in O-grid far-field points."""
    x, z = pts[:,0], pts[:,2]
    nu = arrs['nuHat']
    # locate the airfoil upper surface z at this x: find the smallest u_mag near x
    vel = arrs['velocity']; u = np.sqrt(vel[:,0]**2 + vel[:,2]**2)
    m_loc = (np.abs(x - xt) < dx) & (z > 0) & (z < 0.15) & (u < 0.02)
    if m_loc.sum() < 1: return None, None
    z_wall = z[m_loc].min()  # take the smallest near-wall z (= airfoil surface)
    # extract band: same x, z from z_wall to z_wall + z_band
    m = (np.abs(x - xt) < dx) & (z >= z_wall - 1e-5) & (z < z_wall + z_band)
    if m.sum() < 3: return z_wall, None
    return z_wall, float(np.max(nu[m]))

DCAV = "proper_refineA4_cavity_xxfine"; DSTR = "refineA4_struct_sxxfine"
pc, ac = load(DCAV); ps, as_ = load(DSTR)

# fine x grid
xs = np.linspace(0.005, 0.6, 60)
nu_cav = []; nu_str = []; xs_used = []
for xt in xs:
    zwc, npc = nu_peak_in_band(pc, ac, xt)
    zws, nps = nu_peak_in_band(ps, as_, xt)
    if npc is None or nps is None: continue
    nu_cav.append(npc); nu_str.append(nps); xs_used.append(xt)
xs_used = np.array(xs_used); nu_cav = np.array(nu_cav); nu_str = np.array(nu_str)

fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2))
# panel 1: nuHat peak vs x (log y)
axs[0].semilogy(xs_used, nu_cav, 'k-', lw=1.4, label='cavity (537k cells)')
axs[0].semilogy(xs_used, nu_str, '--', color='C3', lw=1.4, label='O-grid (255k cells)')
axs[0].axhline(4e-9, color='0.7', ls=':', lw=1.0)
axs[0].text(0.4, 5e-9, r'freestream seed $\hat\nu_\infty$', fontsize=7.5, color='0.4')
axs[0].set_xlabel(r'$x/c$'); axs[0].set_ylabel(r'$\max_z \hat\nu$ in BL')
axs[0].set_title('peak amplification factor along chord', fontsize=10)
axs[0].legend(fontsize=8, frameon=False); axs[0].grid(alpha=0.3, which='both')
axs[0].set_xlim(0, 0.6); axs[0].set_ylim(1e-9, 1e-4)

# panel 2: ratio cavity / O-grid
ratio = nu_cav / np.maximum(nu_str, 1e-30)
axs[1].semilogy(xs_used, ratio, 'k-', lw=1.4)
axs[1].axhline(1.0, color='0.5', ls='--', lw=1.0)
axs[1].set_xlabel(r'$x/c$'); axs[1].set_ylabel(r'$\hat\nu^{\max}_{\rm cavity} / \hat\nu^{\max}_{\rm O-grid}$')
axs[1].set_title('cavity-vs-O-grid amplification ratio', fontsize=10)
axs[1].grid(alpha=0.3, which='both'); axs[1].set_xlim(0, 0.6); axs[1].set_ylim(0.01, 10)
plt.tight_layout(pad=0.4)
plt.savefig('/tmp/bl_nuhat_growth.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/bl_nuhat_growth.pdf')
print("wrote /tmp/bl_nuhat_growth.png + paper figs/bl_nuhat_growth.pdf")

# print key numbers
print("\n=== key values ===")
for xt in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]:
    i = int(np.argmin(np.abs(xs_used - xt)))
    if abs(xs_used[i] - xt) < 0.02:
        print(f"  x={xs_used[i]:.3f}: cavity={nu_cav[i]:.2e}  O-grid={nu_str[i]:.2e}  ratio={nu_cav[i]/nu_str[i]:.3f}")
