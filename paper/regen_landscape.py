"""Visualize the 2D (Re_Omega, Gamma) landscape of the amplification rate
a(Re_Omega, Gamma) with the recalibrated constants, and trace the upper/lower
surface trajectories on top.

Output: paper/figs/landscape_trajectories.pdf

Layout: 2 panels (alpha=0 left, alpha=4 right).
  - Black contours of a(Re_Omega, Gamma) with the current committed constants
    (s=5.263, g_c=1.572, a_max=0.25, sharp Γ-dependent cliff with p=4).
  - Upper surface trajectory: blue line, lower surface: red line.
  - Markers along each trajectory at:
      * begin: chi crosses chi_inf (LE region)
      * mid:   chi crosses sqrt(chi_inf*c_v1) (log mid-point)
      * end:   chi crosses c_v1 (f_v1 = 1/2 transition criterion)
"""
import numpy as np, vtk, pickle
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = "/home/qiqi/flexcompute/sa-ai/flow360"
PD = "/home/qiqi/flexcompute/sa-ai/paper"
NU = 2e-7

# Current committed kernel constants (match ModelConstants.h / aft_sources.py).
# Option A kernel (Γ-only sigmoid) + sharp Γ-dependent cliff barrier.
G_C, SLOPE, A_MAX = 1.572, 5.263, 0.15
RE_OMEGA_FLOOR = 100.0
TILT_SLOPE = 1.0e6  # effectively disabled (cliff is pure floor)
BARRIER_POWER = 4.0

C_V1 = 7.1
CHI_INF = C_V1 * np.exp(-9.0)         # match the cases (~8.76e-4)
# Marker χ values (linear-amplification regime: chi_inf < chi < 1, before BL
# starts squashing into a turbulent profile):
CHI_BEGIN = 2.0 * CHI_INF             # ≈ 1.75e-3, just barely above freestream
CHI_MID   = np.sqrt(CHI_INF)          # ≈ 2.96e-2, geometric mid of [chi_inf, 1]
CHI_END   = 1.0                        # χ reaches 1: end of the clean amplification line
CHI_TAIL  = C_V1                       # carry a thin tail out to f_v1 = 1/2

def rate(Re_O, Gamma):
    """Current Option A kernel with Γ-dependent sharp cliff. Matches Flow360
    SAAftTransition.h::__aftRate and JAX src/numerics/aft_sources.py."""
    Re_O = np.asarray(Re_O); Gamma = np.asarray(Gamma)
    log_floor = np.log10(RE_OMEGA_FLOOR)
    log_extra = np.where(Gamma < 1.0, (1.0 - Gamma) / TILT_SLOPE, 0.0)
    Re_cliff = 10.0 ** (log_floor + log_extra)
    safe = np.maximum(Re_O, Re_cliff + 1e-12)
    ratio = Re_cliff / safe
    bar_inside = np.maximum(1.0 - np.power(ratio, BARRIER_POWER), 1e-20)
    barr = np.where(Re_O > Re_cliff, np.log(bar_inside), -np.inf)
    z = SLOPE * (Gamma - G_C) + barr
    a = A_MAX / (1.0 + np.exp(-z))
    return np.where(Re_O > Re_cliff, a, 0.0)

# ---- Landscape grid ----
ReO_grid = np.logspace(-0.5, 5, 240)
G_grid = np.linspace(0, 2.0, 200)
RG, GG = np.meshgrid(ReO_grid, G_grid)
A_grid = rate(RG, GG)

def trajectory(case_dir, side='upper', nbins=80, d_max=0.05):
    """At each x bin on the specified side, find the wall-normal location where the
    kernel a*omega is maximum (i.e., where amplification is being deposited), and
    record (Re_Omega, Gamma) at that point.  Also track max chi in the BL band so
    we can mark thresholds along the trajectory."""
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{case_dir}/slice_centerSpan.pvtu"); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    arrs = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    if not all(n in arrs for n in ['nuHat','wallDistance','vorticityMagnitude']):
        return None
    p = vtk_to_numpy(g.GetPoints().GetData())
    nh = vtk_to_numpy(pd.GetArray('nuHat'))
    wd_arr = vtk_to_numpy(pd.GetArray('wallDistance'))
    om = vtk_to_numpy(pd.GetArray('vorticityMagnitude'))
    vel = vtk_to_numpy(pd.GetArray('velocity'))
    u_mag = np.linalg.norm(vel, axis=1)
    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5*(edges[:-1] + edges[1:])
    Re_O = np.full(nbins, np.nan); Gamma = np.full(nbins, np.nan)
    chi = np.full(nbins, np.nan)
    side_mask = (p[:,2] > 0) if side == 'upper' else (p[:,2] < 0)
    band = (wd_arr > 0) & (wd_arr <= d_max)
    for i in range(nbins):
        m = (p[:,0] >= edges[i]) & (p[:,0] < edges[i+1]) & band & side_mask
        if m.sum() < 3: continue
        wdi = wd_arr[m]; omi = om[m]; ui = u_mag[m]; nhi = nh[m]
        omd = omi * wdi
        Gi = 2 * omd**2 / (ui**2 + omd**2 + 1e-30)
        Re_Oi = wdi**2 * omi / NU
        # Choose the point where the kernel a*omega is largest (rate-deposit peak).
        ai = rate(Re_Oi, Gi)
        aw = ai * omi
        # If kernel is zero everywhere (very early BL), fall back to argmax Gamma so
        # we still get a sensible (Re_Omega, Gamma) trajectory point.
        if aw.max() <= 0:
            j = int(np.argmax(Gi))
        else:
            j = int(np.argmax(aw))
        Re_O[i] = Re_Oi[j]
        Gamma[i] = Gi[j]
        chi[i] = nhi.max() / NU
    return centers, Re_O, Gamma, chi

def find_x_at_chi(x, chi, chi_target):
    """First x at which chi(x) crosses chi_target (going up)."""
    chi = np.where(np.isnan(chi), 0, chi)
    above = chi >= chi_target
    if not above.any(): return None
    i = int(np.argmax(above))
    if i == 0: return x[0]
    # Linear interpolation
    x0, x1 = x[i-1], x[i]; c0, c1 = chi[i-1], chi[i]
    if c1 == c0: return x1
    return x0 + (chi_target - c0) * (x1 - x0) / (c1 - c0)

def values_at_x(x_arr, ReO_arr, G_arr, chi_arr, x_target):
    if x_target is None: return None
    i = int(np.clip(np.searchsorted(x_arr, x_target), 1, len(x_arr)-1))
    # interpolate
    frac = (x_target - x_arr[i-1]) / (x_arr[i] - x_arr[i-1] + 1e-30)
    ro = (1-frac)*ReO_arr[i-1] + frac*ReO_arr[i]
    g  = (1-frac)*G_arr[i-1]   + frac*G_arr[i]
    return ro, g

# ---- Plot ----
matplotlib.rcParams.update({'font.size': 10.5, 'axes.titlesize': 11, 'axes.labelsize': 10.5,
                            'legend.fontsize': 9, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5})

fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)

# Color convention (matches Fig 5): upper=blue, lower=red
UP_COLOR, LO_COLOR = 'C0', 'C3'
LEVELS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.18, 0.195]

for col, a in enumerate([0, 4]):
    ax = axs[col]
    # Landscape contours
    cs = ax.contour(RG, GG, A_grid, levels=LEVELS,
                    colors='k', linewidths=0.8, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.3f')
    # Mark no-amp floor
    ax.axvline(RE_NA, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.text(RE_NA*1.05, 0.02, '  $Re_{\\Omega,na}\\!=\\!5$', fontsize=8, color='gray', va='bottom')
    # Mark sigmoid center
    ax.axhline(G_C, color='gray', ls=':', lw=0.5, alpha=0.4)
    ax.text(0.5, G_C+0.02, '$g_c\\!=\\!1.571$', fontsize=8, color='gray', alpha=0.7)

    # Trajectory and markers for each surface (structured O-grid only)
    case_str = f"{B}/strprop_naca0012_a{a}"
    for side, color in [('upper', UP_COLOR), ('lower', LO_COLOR)]:
        traj = trajectory(case_str, side=side)
        if traj is None: continue
        xs, ReO, Gamma, chi = traj
        valid = ~np.isnan(ReO) & ~np.isnan(Gamma) & (ReO > 0)
        # Smooth out cell-to-cell binning noise in (Re_O, Gamma)
        from scipy.ndimage import gaussian_filter1d
        ReO_s = ReO.copy(); G_s = Gamma.copy()
        ReO_s[~valid] = np.nan; G_s[~valid] = np.nan
        # Fill nans by linear interpolation to allow smoothing through gaps
        idx = np.arange(len(xs))
        ReO_filled = np.interp(idx, idx[valid], np.log(np.maximum(ReO_s[valid], 1e-30)))
        G_filled   = np.interp(idx, idx[valid], G_s[valid])
        ReO_smooth = np.exp(gaussian_filter1d(ReO_filled, sigma=2.0))
        G_smooth   = gaussian_filter1d(G_filled, sigma=2.0)
        # Two segments:
        #   main (chi <= 1):  thick solid — linear amplification regime
        #   tail (1 < chi <= c_v1):  thin dashed — BL squashing toward turbulent
        chi_filled = np.interp(idx, idx[valid], np.maximum(chi[valid], 1e-30))
        first_cross_end  = np.argmax(chi_filled >= CHI_END)  if (chi_filled >= CHI_END).any()  else None
        first_cross_tail = np.argmax(chi_filled >= CHI_TAIL) if (chi_filled >= CHI_TAIL).any() else None
        n = len(xs)
        main_end = (first_cross_end + 1) if first_cross_end is not None else n
        tail_end = (first_cross_tail + 1) if first_cross_tail is not None else n
        main_keep = valid & (np.arange(n) < main_end)
        tail_keep = valid & (np.arange(n) >= max(main_end-1, 0)) & (np.arange(n) < tail_end)
        ax.plot(ReO_smooth[main_keep], G_smooth[main_keep], '-',  color=color, lw=1.8, label=f'{side}')
        ax.plot(ReO_smooth[tail_keep], G_smooth[tail_keep], '--', color=color, lw=1.0, alpha=0.7)
        # Markers
        x_begin = find_x_at_chi(xs, chi, CHI_BEGIN)
        x_mid   = find_x_at_chi(xs, chi, CHI_MID)
        x_end   = find_x_at_chi(xs, chi, CHI_END)
        for xt, mk in [(x_begin, 'o'), (x_mid, 's'), (x_end, '^')]:
            if xt is None: continue
            i = int(np.clip(np.searchsorted(xs, xt), 1, len(xs)-1))
            frac = (xt - xs[i-1]) / (xs[i] - xs[i-1] + 1e-30)
            ro = np.exp((1-frac)*np.log(max(ReO_smooth[i-1], 1e-30)) + frac*np.log(max(ReO_smooth[i], 1e-30)))
            g  = (1-frac)*G_smooth[i-1] + frac*G_smooth[i]
            ax.plot(ro, g, mk, ms=10, color=color, mfc='white', mec=color, mew=1.6, zorder=5)
            ax.annotate(f'  $x\\!=\\!{xt:.2f}$', (ro, g),
                        fontsize=8, color=color, alpha=0.9,
                        xytext=(8, 2), textcoords='offset points')

    ax.set_xscale('log')
    ax.set_xlim(0.5, 1e5)
    ax.set_ylim(0, 2.05)
    ax.set_xlabel('$Re_\\Omega = d^2|\\omega|/\\nu$')
    ax.set_title(rf'$\alpha={a}^\circ$')
    ax.grid(alpha=0.3, which='major')
    if col == 0: ax.set_ylabel('$\\Gamma = 2(\\omega d)^2 / (|V|^2 + (\\omega d)^2)$')

# Composite legend
mh = [Line2D([],[],marker='o',color='k',mfc='white',mec='k',mew=1.6,ls='',label=r'begin: $\chi\!=\!2\chi_\infty$'),
      Line2D([],[],marker='s',color='k',mfc='white',mec='k',mew=1.6,ls='',label=r'mid:   $\chi\!=\!\sqrt{\chi_\infty}$'),
      Line2D([],[],marker='^',color='k',mfc='white',mec='k',mew=1.6,ls='',label=r'end:   $\chi\!=\!1$'),
      Line2D([],[],color='k',ls='--',lw=1.0,alpha=0.7,label=r'tail: $1\!<\!\chi\!\leq\!c_{v1}$'),
      Line2D([],[],color=UP_COLOR,lw=2,label='upper surface'),
      Line2D([],[],color=LO_COLOR,lw=2,label='lower surface')]
axs[1].legend(handles=mh, fontsize=8.5, frameon=True, loc='lower right')

plt.suptitle('Amplification-rate landscape and BL trajectories — NACA 0012, $Re\\!=\\!10^6$, structured O-grid', fontsize=11.5, y=0.995)
plt.tight_layout(rect=(0,0,1,0.97))
plt.savefig(f'{PD}/figs/landscape_trajectories.pdf')
plt.savefig('/tmp/landscape_trajectories.png', dpi=140)
plt.close()
print(f"wrote landscape_trajectories.pdf")
print(f"chi_inf={CHI_INF:.3e}, chi_mid={CHI_MID:.3e}, c_v1={C_V1}")
