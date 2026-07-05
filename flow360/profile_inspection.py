"""Inspect what's actually happening: Cp(x), Cf(x), and velocity profiles u(d) at
sampled x/c on the structured-mesh NACA0012 variant-B case, compared with
Blasius profiles at the same Re_x. Plot everything; no conclusions until I see it.
"""
import sys, numpy as np
sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.integrate import odeint

NU = 2e-7
U_INF = 0.2

case_str = 'str_L1_varB_a0'
case_cav = 'cav_L1_varB_a0'
F = '/home/qiqi/flexcompute/aft-sa/flow360'

# --- Load surface fields ---
def load_surface(case):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f'{F}/{case}/surface_fluid_naca0012.pvtu'); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    p = vtk_to_numpy(g.GetPoints().GetData())
    cp = vtk_to_numpy(pd.GetArray('Cp'))
    cf_arr = vtk_to_numpy(pd.GetArray('Cf'))
    cf = np.linalg.norm(cf_arr, axis=1) if cf_arr.ndim==2 else np.abs(cf_arr)
    return p[:,0], p[:,2], cp, cf

# --- Load slice fields ---
def load_slice(case):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f'{F}/{case}/slice_centerSpan.pvtu'); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    p = vtk_to_numpy(g.GetPoints().GetData())
    wd = vtk_to_numpy(pd.GetArray('wallDistance'))
    om = vtk_to_numpy(pd.GetArray('vorticityMagnitude'))
    vel = vtk_to_numpy(pd.GetArray('velocity'))
    nh = vtk_to_numpy(pd.GetArray('nuHat'))
    return p, wd, om, vel, nh

# --- Blasius similarity ---
def blasius_rhs(F_, eta):
    return [F_[1], F_[2], -0.5*F_[0]*F_[2]]
eta_grid = np.linspace(0, 12, 800)
F_ = odeint(blasius_rhs, [0,0,0.332], eta_grid)
fp, fpp = F_[:,1], F_[:,2]

def blasius_profile_at_x(x_chord, U_e=U_INF, n_y=200):
    """Return (d, u, omega) for Blasius at given chord position, using local U_e."""
    Re_x = U_e * x_chord / NU
    sqx = np.sqrt(Re_x)
    # In Flow360 units (length=c, vel=c_sound): η = y·sqrt(U_e/(ν·x)) → y = η·sqrt(ν·x/U_e)
    y_grid = eta_grid * np.sqrt(NU * x_chord / U_e)
    u_pr = fp * U_e
    dudy = fpp * U_e / np.sqrt(NU * x_chord / U_e)  # in [vel/length] units
    omega = np.abs(dudy)
    return y_grid, u_pr, omega

# --- Extract BL profile from sim at given x ---
def bl_profile_sim(p, wd, om, vel, nh, x_t, dx=0.005, side='upper'):
    m = (np.abs(p[:,0]-x_t) < dx)
    if side == 'upper': m &= (p[:,2] > 0)
    else: m &= (p[:,2] < 0)
    m &= (wd > 0)
    pp = p[m]; wd_ = wd[m]; om_ = om[m]; vel_ = vel[m]; nh_ = nh[m]
    u_mag = np.linalg.norm(vel_, axis=1)
    order = np.argsort(wd_)
    return wd_[order], u_mag[order], om_[order], nh_[order]

# --- Load everything ---
xs_s, zs_s, cp_s, cf_s = load_surface(case_str)
xs_c, zs_c, cp_c, cf_c = load_surface(case_cav)
p_s, wd_s, om_s, vel_s, nh_s = load_slice(case_str)
p_c, wd_c, om_c, vel_c, nh_c = load_slice(case_cav)

# --- Plot ---
xs_sample = [0.05, 0.10, 0.18, 0.25, 0.40, 0.60]
fig = plt.figure(figsize=(16, 12))

# Top row: surface Cp + Cf
ax_cp = fig.add_subplot(3, 4, 1)
ax_cf = fig.add_subplot(3, 4, 2)
# Upper surface only, sorted by x
for color, xs, zs, cp, cf, lab in [('C0', xs_s, zs_s, cp_s, cf_s, 'str (O-grid)'),
                                    ('C2', xs_c, zs_c, cp_c, cf_c, 'cav (unstr)')]:
    up = zs > 0
    o = np.argsort(xs[up])
    ax_cp.plot(xs[up][o], cp[up][o], '-', color=color, lw=1.5, label=lab)
    ax_cf.plot(xs[up][o], cf[up][o], '-', color=color, lw=1.5, label=lab)
ax_cp.set_xlabel('x/c'); ax_cp.set_ylabel('Cp'); ax_cp.invert_yaxis()
ax_cp.set_title('Cp upper surface'); ax_cp.grid(alpha=0.3); ax_cp.legend(fontsize=8)
for x in xs_sample: ax_cp.axvline(x, color='gray', ls=':', lw=0.5, alpha=0.5)
ax_cf.set_xlabel('x/c'); ax_cf.set_ylabel('|Cf|'); ax_cf.set_ylim(0, 0.012)
ax_cf.set_title('|Cf| upper surface'); ax_cf.grid(alpha=0.3); ax_cf.legend(fontsize=8)
for x in xs_sample: ax_cf.axvline(x, color='gray', ls=':', lw=0.5, alpha=0.5)

# Compute U_e at each sample x from Cp (incompressible): U_e/U_inf = sqrt(1-Cp)
def Ue_from_Cp(cp, U_inf=U_INF):
    return U_inf * np.sqrt(np.maximum(1 - cp, 0))
# Re_x_local using local U_e and arc length ~ x (approx)
# For each x_t, find nearest surface point to read Cp, then U_e
def find_Ue(xs, zs, cp, x_t):
    up = zs > 0
    i = np.argmin(np.abs(xs[up] - x_t))
    return Ue_from_Cp(cp[up][i])

# Annotation panel: sampling table
ax_info = fig.add_subplot(3, 4, 3)
ax_info.axis('off')
info_lines = ['Sample points & local U_e:']
for x in xs_sample:
    Ue = find_Ue(xs_s, zs_s, cp_s, x)
    info_lines.append(f'  x={x:.2f}: U_e/U_∞={Ue/U_INF:.3f}, Re_x_local={Ue*x/NU:.2e}')
ax_info.text(0.05, 0.95, '\n'.join(info_lines), transform=ax_info.transAxes,
             fontsize=10, va='top', family='monospace')

# Empty slot for spacing
ax_blank = fig.add_subplot(3, 4, 4); ax_blank.axis('off')

# Middle + bottom rows: velocity profiles + Γ profile at each sample x
for k, x_t in enumerate(xs_sample):
    if k < 4:
        ax_u = fig.add_subplot(3, 4, 5+k)
    else:
        ax_u = fig.add_subplot(3, 4, 9+k-4)
    # str profile
    wd_str, u_str, om_str, nh_str = bl_profile_sim(p_s, wd_s, om_s, vel_s, nh_s, x_t)
    # cav profile
    wd_cav, u_cav, om_cav, nh_cav = bl_profile_sim(p_c, wd_c, om_c, vel_c, nh_c, x_t)
    # blasius using LOCAL U_e
    Ue = find_Ue(xs_s, zs_s, cp_s, x_t)
    d_bl, u_bl, om_bl = blasius_profile_at_x(x_t, U_e=Ue)

    ax_u.plot(u_str, wd_str, '-', color='C0', lw=1.3, label='str (sim)')
    ax_u.plot(u_cav, wd_cav, '-', color='C2', lw=1.3, label='cav (sim)', alpha=0.7)
    ax_u.plot(u_bl, d_bl, '--', color='k', lw=1.3, label=f'Blasius @ U_e={Ue/U_INF:.2f}U_∞')
    ax_u.set_xlabel('|V|')
    ax_u.set_ylabel('wall distance d')
    ax_u.set_title(f'x/c = {x_t}')
    delta_blasius = 5*np.sqrt(NU*x_t/Ue)
    ax_u.set_ylim(0, 3*delta_blasius)
    ax_u.set_xlim(0, max(u_bl[-1]*1.1, 0.25))
    ax_u.grid(alpha=0.3)
    if k==0: ax_u.legend(fontsize=8, frameon=False, loc='lower right')

plt.suptitle(f'NACA0012 var B BL profiles vs Blasius (local U_e) — str (clean) and cav (noisy unstr) overlaid',
             fontsize=12, y=0.995)
plt.tight_layout(rect=(0,0,1,0.97))
plt.savefig('/tmp/profile_inspect.png', dpi=130)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/profile_inspect.pdf')
plt.close()
print('wrote /tmp/profile_inspect.png')
