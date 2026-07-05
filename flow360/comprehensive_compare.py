"""Comprehensive amplification-rate comparison.

For each model, plot N(Re_theta) and dN/dRe_theta(Re_theta) on common axes:
   1. mfoil Ncrit=9    (Drela e^N envelope on NACA0012)
   2. mfoil Ncrit=15.65
   3. Blasius transport solver (our model at a_max=0.2)
   4. NACA0012 cav_L1, baseline (a_max=0.05)
   5. NACA0012 cav_L1, variant B (a_max=0.20, chi_inf=1.6e-7)
   6. Drela analytical m(H_k=2.59) for Blasius reference

Goal: pinpoint exactly where and by how much our rate exceeds Drela's.
"""
import sys, os, pickle, numpy as np
sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.ndimage import maximum_filter1d, gaussian_filter1d

F = "/home/qiqi/flexcompute/aft-sa/flow360"
PD = "/home/qiqi/flexcompute/aft-sa/paper"
NU = 2e-7
U_INF = 0.2

# -------- 1. mfoil data --------
mfoil = pickle.load(open(f"{F}/mfoil_naca0012_compare.pkl", "rb"))

# -------- 2. Drela's m(H_k=2.59) analytical (Blasius) --------
def drela_m(H_k):
    # Drela 1987 fit:
    #  m(H_k) = 0.028*(H_k-1) - 0.0345*exp(-(3.87/(H_k-1)-2.52)**2)
    return 0.028*(H_k-1) - 0.0345*np.exp(-(3.87/(H_k-1)-2.52)**2)

H_k_Blasius = 2.59
m_Blasius = drela_m(H_k_Blasius)   # ≈ 0.01

# Re_theta range
Re_theta_arr = np.linspace(1, 1500, 200)
# Drela "envelope" curve: dN/dRe_theta = m_Blasius for Re_theta > Re_theta_crit, else 0
# Re_theta_crit at H=2.59 from Drela: roughly 200
Re_theta_crit_Drela = 200.0
dN_dRet_Drela = np.where(Re_theta_arr > Re_theta_crit_Drela, m_Blasius, 0.0)
N_Drela = np.cumsum(dN_dRet_Drela) * (Re_theta_arr[1]-Re_theta_arr[0])

# -------- 3. Run Blasius transport solver and extract N(Re_x) --------
import matplotlib; matplotlib.use('Agg')
from src.solvers.boundary_layer_solvers import NuHatBlasiusSolver
s = NuHatBlasiusSolver()
nuHat = np.array(s(1.0))
Re_x_Bl = np.arange(s.nx+1) * s.dx
N_Bl = np.log(np.maximum(nuHat, 1e-30)).max(axis=1)
# Smooth and compute dN/dRe_x
N_Bl_sm = gaussian_filter1d(N_Bl, sigma=2)
dN_dRex_Bl = np.gradient(N_Bl_sm, s.dx)
# Convert to Re_theta = 0.664*sqrt(Re_x)
Re_th_Bl = 0.664*np.sqrt(np.maximum(Re_x_Bl, 1))
# dN/dRe_theta = dN/dRe_x / (dRe_theta/dRe_x)
dRet_dRex = 0.5 * 0.664 / np.sqrt(np.maximum(Re_x_Bl, 1))
dN_dRet_Bl = dN_dRex_Bl / np.maximum(dRet_dRex, 1e-30)

# -------- 4. NACA0012 data: pull max_y N (from binned slice) and convert x->Re_theta --------
def naca_N_from_slice(case_dir, nu_inf):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f'{case_dir}/slice_centerSpan.pvtu'); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    p = vtk_to_numpy(g.GetPoints().GetData())
    nh = vtk_to_numpy(pd.GetArray('nuHat'))
    x_ = p[:,0]; z_ = p[:,2]
    mask = (np.abs(z_) < 0.05) & (x_ > 0.0) & (x_ < 1.0)
    x_, z_, nh = x_[mask], z_[mask], nh[mask]
    edges = np.linspace(0, 1, 60)
    centers = 0.5*(edges[:-1]+edges[1:])
    max_nh = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        m = (x_ >= edges[i]) & (x_ < edges[i+1])
        if m.any(): max_nh[i] = nh[m].max()
    max_nh = maximum_filter1d(np.where(np.isnan(max_nh), -np.inf, max_nh), size=3)
    max_nh = np.where(max_nh == -np.inf, np.nan, max_nh)
    N = np.log(np.maximum(max_nh, 1e-35) / nu_inf)
    # Convert x to Re_theta_Blasius: Re_theta = 0.664*sqrt(x/nu)
    Re_th = 0.664 * np.sqrt(centers / NU * U_INF)   # treating ν=NU, U=U_INF=0.2
    # Actually want Re_theta = U_e·theta/ν.  Use Blasius approximation: theta = 0.664*sqrt(nu·x/U_e), so Re_theta = 0.664*sqrt(U_e·x/nu). For U_e≈U_inf=0.2: Re_theta = 0.664*sqrt(0.2·x/NU) = 0.664·sqrt(x/1e-6) = 664·sqrt(x)
    return centers, Re_th, N

x_b, Reth_b, N_base = naca_N_from_slice(f'{F}/cav_L1_base_a0', 4e-9)
x_v, Reth_v, N_var = naca_N_from_slice(f'{F}/cav_L1_varB_a0', 3.2e-14)

# Smooth and differentiate
N_base_sm = gaussian_filter1d(np.where(np.isnan(N_base), 0, N_base), sigma=2)
N_var_sm  = gaussian_filter1d(np.where(np.isnan(N_var),  0, N_var),  sigma=2)
dx_b = (x_b[1]-x_b[0])
# dN/dx_chord -> dN/dRe_theta:  Re_theta = 664·sqrt(x); dRe_theta/dx = 332/sqrt(x); dN/dRe_theta = (dN/dx)·sqrt(x)/332
sqx = np.sqrt(np.maximum(x_b, 1e-6))
dN_dx_base = np.gradient(N_base_sm, dx_b)
dN_dx_var  = np.gradient(N_var_sm,  dx_b)
dN_dRet_base = dN_dx_base * sqx / 332.0
dN_dRet_var  = dN_dx_var  * sqx / 332.0

# -------- 5. mfoil dN/dRe_theta --------
def mfoil_dN_dRet(md):
    x_m = md['x']; n_m = md['n_amp']
    # Re_theta = 664·sqrt(x) (NACA at α=0, U_e≈U_inf approx)
    Re_th = 664*np.sqrt(np.maximum(x_m, 1e-6))
    valid = ~np.isnan(n_m) & (np.arange(len(x_m)) > 0)
    # sort by Re_th
    order = np.argsort(Re_th[valid])
    Re_th_s = Re_th[valid][order]
    n_s = n_m[valid][order]
    n_s_sm = gaussian_filter1d(n_s, sigma=2)
    dRet = np.gradient(Re_th_s)
    dN_dRet = np.gradient(n_s_sm) / np.maximum(dRet, 1e-30)
    return Re_th_s, n_s, dN_dRet

Reth_m9, N_m9, dN_m9 = mfoil_dN_dRet(mfoil[9.0])
Reth_m15, N_m15, dN_m15 = mfoil_dN_dRet(mfoil[15.65])

# -------- 6. Plot --------
matplotlib.rcParams.update({'font.size':11, 'axes.titlesize':11, 'axes.labelsize':11,
                            'legend.fontsize':9, 'xtick.labelsize':10, 'ytick.labelsize':10})

fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
ax_N, ax_rate = axs

# Left: N(Re_theta)
ax_N.plot(Re_th_Bl, N_Bl, '-', color='C0', lw=1.8, label='Our model on Blasius BL ($a_{\\max}\\!=\\!0.2$)')
ax_N.plot(Reth_b,    N_base_sm, '-', color='C5', lw=1.8, label='Our model on NACA0012, $a_{\\max}\\!=\\!0.05$ (baseline)')
ax_N.plot(Reth_v,    N_var_sm,  '-', color='C2', lw=1.8, label='Our model on NACA0012, $a_{\\max}\\!=\\!0.20$ (var B)')
ax_N.plot(Reth_m9,   N_m9,      '--', color='C1', lw=1.8, label='mfoil on NACA0012, $N_{\\rm crit}\\!=\\!9$')
ax_N.plot(Reth_m15,  N_m15,     '--', color='C3', lw=1.8, label='mfoil on NACA0012, $N_{\\rm crit}\\!=\\!15.65$')
ax_N.plot(Re_theta_arr, N_Drela, ':', color='k', lw=1.5, label='Drela $m(H_k\\!=\\!2.59)$ envelope (analytic Blasius)')

ax_N.set_xlim(0, 1500)
ax_N.set_ylim(0, 20)
ax_N.set_xlabel('$Re_\\theta$ (Blasius approx from $x$)')
ax_N.set_ylabel('$N\\!=\\!\\ln(\\tilde\\nu/\\tilde\\nu_\\infty)$')
ax_N.set_title('Amplification factor $N$ vs $Re_\\theta$')
ax_N.grid(alpha=0.4)
ax_N.legend(fontsize=8.5, frameon=False, loc='upper left')

# Right: dN/dRe_theta(Re_theta)
ax_rate.plot(Re_th_Bl[5:], np.maximum(dN_dRet_Bl[5:], 1e-5), '-', color='C0', lw=1.8, label='Our model, Blasius BL ($a_{\\max}\\!=\\!0.2$)')
ax_rate.plot(Reth_b[2:],   np.maximum(dN_dRet_base[2:], 1e-5), '-', color='C5', lw=1.8, label='Our model on NACA0012, $a_{\\max}\\!=\\!0.05$')
ax_rate.plot(Reth_v[2:],   np.maximum(dN_dRet_var[2:], 1e-5),  '-', color='C2', lw=1.8, label='Our model on NACA0012, $a_{\\max}\\!=\\!0.20$')
ax_rate.plot(Reth_m9[3:],   np.maximum(dN_m9[3:], 1e-5),    '--', color='C1', lw=1.8, label='mfoil $N_{\\rm crit}\\!=\\!9$')
ax_rate.plot(Reth_m15[3:],  np.maximum(dN_m15[3:], 1e-5),   '--', color='C3', lw=1.8, label='mfoil $N_{\\rm crit}\\!=\\!15.65$')
ax_rate.axhline(m_Blasius, color='k', ls=':', lw=1.5, label=f'Drela $m(H_k\\!=\\!2.59)\\!=\\!{m_Blasius:.3f}$')

ax_rate.set_yscale('log')
ax_rate.set_xlim(0, 1500)
ax_rate.set_ylim(1e-4, 3)
ax_rate.set_xlabel('$Re_\\theta$')
ax_rate.set_ylabel('$dN/dRe_\\theta$ (log)')
ax_rate.set_title('Amplification rate $dN/dRe_\\theta$ — what scales as what')
ax_rate.grid(alpha=0.4, which='both')
ax_rate.legend(fontsize=8.5, frameon=False, loc='upper right')

plt.suptitle('Comprehensive amplification comparison: our model (Blasius solver + NACA0012 RANS) vs mfoil vs Drela', fontsize=11.5)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig('/tmp/comprehensive_compare.png', dpi=130)
plt.savefig(f'{PD}/figs/comprehensive_amp_compare.pdf')
plt.close()
print('wrote /tmp/comprehensive_compare.png')

# Print numerical summary at a few Re_theta points
print()
print(f"{'Re_θ':>8} {'Bl(0.2)':>9} {'Bl_dN':>9} {'NACA b(0.05)':>13} {'NACAb dN':>10} {'NACA v(0.2)':>12} {'NACAv dN':>10} {'mfoil9':>8} {'mfoil9_dN':>10} {'Drela_m':>9}")
for Rt in [50, 100, 200, 300, 500, 800, 1100, 1300]:
    i_Bl = np.searchsorted(Re_th_Bl, Rt)
    i_b = np.searchsorted(Reth_b, Rt) if Rt < Reth_b.max() else len(Reth_b)-1
    i_v = np.searchsorted(Reth_v, Rt) if Rt < Reth_v.max() else len(Reth_v)-1
    i_m9 = np.searchsorted(Reth_m9, Rt) if Rt < Reth_m9.max() else -1
    def safe(arr, i): return arr[i] if 0 <= i < len(arr) else float('nan')
    print(f'{Rt:>8} {safe(N_Bl,i_Bl):>9.3f} {safe(dN_dRet_Bl,i_Bl):>9.4f} '
          f'{safe(N_base_sm,i_b):>13.3f} {safe(dN_dRet_base,i_b):>10.4f} '
          f'{safe(N_var_sm,i_v):>12.3f} {safe(dN_dRet_var,i_v):>10.4f} '
          f'{safe(N_m9,i_m9):>8.3f} {safe(dN_m9,i_m9):>10.4f} '
          f'{m_Blasius if Rt > Re_theta_crit_Drela else 0:>9.4f}')
