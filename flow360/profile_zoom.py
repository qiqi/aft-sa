"""Zoom into the BL profile at sampled x and look at:
  - u(d) compared to local Blasius profile (using local U_e from Cp)
  - ω(d), Γ(d), Re_Ω(d) profiles in the BL
  - Identify whether the Γ > 1 is real (small profile deviation) or noise
"""
import sys, numpy as np
sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.integrate import odeint

NU = 2e-7; U_INF = 0.2
F = '/home/qiqi/flexcompute/aft-sa/flow360'

# Blasius reference
eta_grid = np.linspace(0, 12, 800)
F_b = odeint(lambda Fb, e: [Fb[1], Fb[2], -0.5*Fb[0]*Fb[2]], [0, 0, 0.332], eta_grid)
fp, fpp = F_b[:,1], F_b[:,2]

def blasius_profile(x_chord, U_e, y_eval):
    """Blasius u(y), omega(y), evaluated at given y array."""
    eta_at_y = y_eval / np.sqrt(NU * x_chord / U_e)
    u = np.interp(eta_at_y, eta_grid, fp) * U_e
    om = np.interp(eta_at_y, eta_grid, fpp) * U_e / np.sqrt(NU * x_chord / U_e)
    return u, om

def load_case(case):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f'{F}/{case}/slice_centerSpan.pvtu'); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    p = vtk_to_numpy(g.GetPoints().GetData())
    wd = vtk_to_numpy(pd.GetArray('wallDistance'))
    om = vtk_to_numpy(pd.GetArray('vorticityMagnitude'))
    vel = vtk_to_numpy(pd.GetArray('velocity'))
    nh = vtk_to_numpy(pd.GetArray('nuHat'))
    # Cp from surface
    rs = vtk.vtkXMLPUnstructuredGridReader()
    rs.SetFileName(f'{F}/{case}/surface_fluid_naca0012.pvtu'); rs.Update()
    ps = vtk_to_numpy(rs.GetOutput().GetPoints().GetData())
    cps = vtk_to_numpy(rs.GetOutput().GetPointData().GetArray('Cp'))
    return p, wd, om, vel, nh, ps, cps

def col_at_x(p, wd, om, vel, nh, x_t, dx=0.004, side='upper'):
    m = (np.abs(p[:,0]-x_t) < dx)
    if side == 'upper': m &= (p[:,2] > 0)
    m &= (wd > 0)
    pp, w, o, v, n_ = p[m], wd[m], om[m], vel[m], nh[m]
    u_mag = np.linalg.norm(v, axis=1)
    order = np.argsort(w)
    return w[order], u_mag[order], o[order], n_[order]

def Ue_at(ps, cps, x_t):
    up = ps[:,2] > 0
    i = np.argmin(np.abs(ps[up,0] - x_t))
    Cp = cps[up][i]
    return U_INF * np.sqrt(max(1 - Cp, 0))

# Load both meshes
p_s, wd_s, om_s, vel_s, nh_s, ps_s, cps_s = load_case('str_L1_varB_a0')
p_c, wd_c, om_c, vel_c, nh_c, ps_c, cps_c = load_case('cav_L1_varB_a0')

xs_sample = [0.05, 0.10, 0.18, 0.25, 0.35, 0.50]
fig, axs = plt.subplots(len(xs_sample), 4, figsize=(16, 3.0*len(xs_sample)))

for row, x_t in enumerate(xs_sample):
    # str data
    w_s, u_s, o_s, n_s = col_at_x(p_s, wd_s, om_s, vel_s, nh_s, x_t)
    # cav data
    w_c, u_c, o_c, n_c = col_at_x(p_c, wd_c, om_c, vel_c, nh_c, x_t)
    # local U_e
    Ue = Ue_at(ps_s, cps_s, x_t)
    # Blasius reference at the SAME d points (use str d as reference)
    d_ref = np.linspace(1e-7, 0.01, 200)
    u_bl, om_bl = blasius_profile(x_t, Ue, d_ref)
    Gamma_bl = 2*(om_bl*d_ref)**2 / (u_bl**2 + (om_bl*d_ref)**2 + 1e-30)
    # Sim Γ at each point
    Gamma_s = 2*(o_s*w_s)**2 / (u_s**2 + (o_s*w_s)**2 + 1e-30)
    Gamma_c = 2*(o_c*w_c)**2 / (u_c**2 + (o_c*w_c)**2 + 1e-30)

    # Col 0: u(d) zoom
    ax = axs[row,0]
    delta_b = 5*np.sqrt(NU*x_t/Ue)
    mask = w_s < 2*delta_b
    mask_c = w_c < 2*delta_b
    mask_bl = d_ref < 2*delta_b
    ax.plot(u_s[mask]/Ue, w_s[mask]/delta_b, '-', color='C0', lw=1.5, label='str')
    ax.plot(u_c[mask_c]/Ue, w_c[mask_c]/delta_b, '-', color='C2', lw=1.0, alpha=0.6, label='cav')
    ax.plot(u_bl[mask_bl]/Ue, d_ref[mask_bl]/delta_b, '--', color='k', lw=1.2, label='Blasius')
    ax.set_xlabel('u/U_e'); ax.set_ylabel('d/δ_Bl')
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.5)
    ax.set_title(f'x={x_t}: u(d)/(U_e, δ_Bl)')
    ax.grid(alpha=0.3)
    if row==0: ax.legend(fontsize=8, loc='lower right')

    # Col 1: ω(d) zoom
    ax = axs[row,1]
    om_scale = Ue/delta_b   # natural scale Ue/δ
    ax.plot(o_s[mask]/om_scale, w_s[mask]/delta_b, '-', color='C0', lw=1.5)
    ax.plot(o_c[mask_c]/om_scale, w_c[mask_c]/delta_b, '-', color='C2', lw=1.0, alpha=0.6)
    ax.plot(om_bl[mask_bl]/om_scale, d_ref[mask_bl]/delta_b, '--', color='k', lw=1.2)
    ax.set_xlabel('ω / (U_e/δ_Bl)'); ax.set_ylabel('d/δ_Bl')
    ax.set_xlim(0, 6); ax.set_ylim(0, 1.5)
    ax.set_title(f'x={x_t}: ω(d)')
    ax.grid(alpha=0.3)

    # Col 2: ωd/|V| ratio (this is what saturates Γ)
    ax = axs[row,2]
    ratio_s = o_s[mask]*w_s[mask] / np.maximum(u_s[mask], 1e-30)
    ratio_c = o_c[mask_c]*w_c[mask_c] / np.maximum(u_c[mask_c], 1e-30)
    ratio_bl = om_bl[mask_bl]*d_ref[mask_bl] / np.maximum(u_bl[mask_bl], 1e-30)
    ax.plot(ratio_s, w_s[mask]/delta_b, '-', color='C0', lw=1.5, label='str')
    ax.plot(ratio_c, w_c[mask_c]/delta_b, '-', color='C2', lw=1.0, alpha=0.6, label='cav')
    ax.plot(ratio_bl, d_ref[mask_bl]/delta_b, '--', color='k', lw=1.2, label='Blasius')
    ax.axvline(1.0, color='gray', ls=':', lw=1)
    ax.set_xlabel('ω·d / |V|'); ax.set_ylabel('d/δ_Bl')
    ax.set_xlim(0, 1.5); ax.set_ylim(0, 1.5)
    ax.set_title(f'x={x_t}: ωd/|V| — drives Γ')
    ax.grid(alpha=0.3)

    # Col 3: Γ(d) profile
    ax = axs[row,3]
    ax.plot(Gamma_s[mask], w_s[mask]/delta_b, '-', color='C0', lw=1.5, label='str')
    ax.plot(Gamma_c[mask_c], w_c[mask_c]/delta_b, '-', color='C2', lw=1.0, alpha=0.6, label='cav')
    ax.plot(Gamma_bl[mask_bl], d_ref[mask_bl]/delta_b, '--', color='k', lw=1.2, label='Blasius')
    ax.axvline(1.04, color='red', ls=':', lw=1, label='sigmoid center')
    ax.set_xlabel('Γ'); ax.set_ylabel('d/δ_Bl')
    ax.set_xlim(0, 1.2); ax.set_ylim(0, 1.5)
    ax.set_title(f'x={x_t}: Γ profile')
    ax.grid(alpha=0.3)
    if row==0: ax.legend(fontsize=8, loc='lower left')

plt.suptitle('NACA0012 var B BL profiles at sampled x/c: str (O-grid), cav (unstr), Blasius reference\n'
             'd normalized by δ_Blasius=5√(νx/U_e); u/U_e on x-axis',
             fontsize=11, y=0.998)
plt.tight_layout(rect=(0,0,1,0.99))
plt.savefig('/tmp/profile_zoom.png', dpi=120)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/profile_zoom.pdf')
plt.close()
print('wrote /tmp/profile_zoom.png')

# Print numerical Γ_max in the BL
print(f"\n{'x':>5} {'Ue/Uin':>8} {'Γmax_str':>10} {'Γmax_cav':>10} {'Γmax_Bl':>10} {'d_Γmax_s/δBl':>12}")
for x_t in xs_sample:
    w_s, u_s, o_s, _ = col_at_x(p_s, wd_s, om_s, vel_s, nh_s, x_t)
    w_c, u_c, o_c, _ = col_at_x(p_c, wd_c, om_c, vel_c, nh_c, x_t)
    Ue = Ue_at(ps_s, cps_s, x_t)
    delta_b = 5*np.sqrt(NU*x_t/Ue)
    Gs = 2*(o_s*w_s)**2 / (u_s**2 + (o_s*w_s)**2 + 1e-30)
    Gc = 2*(o_c*w_c)**2 / (u_c**2 + (o_c*w_c)**2 + 1e-30)
    bl_mask_s = w_s < 1.5*delta_b
    bl_mask_c = w_c < 1.5*delta_b
    Gs_max = Gs[bl_mask_s].max() if bl_mask_s.any() else np.nan
    Gc_max = Gc[bl_mask_c].max() if bl_mask_c.any() else np.nan
    j = int(np.argmax(Gs[bl_mask_s])) if bl_mask_s.any() else -1
    d_Gmax_s = w_s[bl_mask_s][j]/delta_b if bl_mask_s.any() else float('nan')
    # Blasius max Γ
    d_ref = np.linspace(1e-7, 1.5*delta_b, 500)
    u_bl, om_bl = blasius_profile(x_t, Ue, d_ref)
    G_bl = 2*(om_bl*d_ref)**2/(u_bl**2 + (om_bl*d_ref)**2 + 1e-30)
    print(f'{x_t:>5.2f} {Ue/U_INF:>8.4f} {Gs_max:>10.4f} {Gc_max:>10.4f} {G_bl.max():>10.4f} {d_Gmax_s:>12.4f}')
