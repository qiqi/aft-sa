"""Stage 2: is the Gamma_g excess operator noise or solution roughness?

Three curvature estimates at matched (mesh-scale) filtering, per station:
  c1 = |lap u|            probed field (VTK double-LSQ of u)      [what Q sees]
  c2 = |d(omega_s)/dn|    solver's own omega, smoothed, FD once   [1 solver LSQ]
  c3 = |d2(u_t,s)/dn2|    raw sampled u_t, smoothed, FD twice     [no LSQ at all]
plus the raw staggering amplitude of the solution itself:
  wiggle(u_t) = rms(u_t - smooth(u_t)) / U_e     per station, band-averaged
  wiggle(om)  = rms(omega - smooth(omega)) / max(omega)
If c3 (pure resampled velocity) already shows the cavL2 excess -> the SOLUTION
carries mesh-scale oscillations whose curvature is real; any consistent second
derivative must report it. If c3 is clean while c1 spikes -> operator noise.
"""
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_gate_noise_term import D, dists, med, p90   # reuses the probed data

h = dists[1] - dists[0]
SIG = 25e-6/h          # Gaussian filter sigma = 25e-6 (>= local mesh spacing)
band = (dists >= 80e-6) & (dists <= 300e-6)

print(f"filter sigma = 25e-6 ({SIG:.1f} samples), band d in [80,300]e-6")
print(f"\n{'grid':>7} {'c1=|lap u|':>12} {'c2=|dw_s/dn|':>13} {'c3=|d2u_s/dn2|':>15}  (band medians)")
for n in ('strL2', 'cavL1', 'cavL2'):
    o = D[n]
    om_s = gaussian_filter1d(o['omg'], SIG, axis=0, mode='nearest')
    ut_s = gaussian_filter1d(o['ut'],  SIG, axis=0, mode='nearest')
    c2 = np.abs(np.gradient(om_s, h, axis=0))
    c3 = np.abs(np.gradient(np.gradient(ut_s, h, axis=0), h, axis=0))
    print(f"{n:>7} {med(o['lap'][band]):12.3g} {med(c2[band]):13.3g} {med(c3[band]):15.3g}")

print(f"\n=== raw solution roughness (sub-filter residual), band-averaged ===")
print(f"{'grid':>7} {'rms(u_t wiggle)/U_e':>20} {'rms(omega wiggle)/max':>22}")
for n in ('strL2', 'cavL1', 'cavL2'):
    o = D[n]
    ut_s = gaussian_filter1d(o['ut'], SIG, axis=0, mode='nearest')
    om_s = gaussian_filter1d(o['omg'], SIG, axis=0, mode='nearest')
    wu = o['ut'] - ut_s
    wo = o['omg'] - om_s
    Ue = np.nanmax(o['U'], axis=0)
    ru = np.sqrt(np.nanmean(wu[band]**2, axis=0))/Ue
    ro = np.sqrt(np.nanmean(wo[band]**2, axis=0))/np.nanmax(o['omg'], axis=0)
    print(f"{n:>7} {med(ru):20.2e} {med(ro):22.2e}")

# one raw trace to SEE it: x~0.12, fine d grid
print("\n=== raw u_t(d) and omega(d), station nearest x=0.12, d=100..300e-6 ===")
for n in ('strL2', 'cavL2'):
    o = D[n]
    i0 = int(np.argmin(np.abs(o['x'] - 0.12)))
    jj = np.where(band)[0][::6]
    ut_s = gaussian_filter1d(o['ut'][:, i0], SIG, mode='nearest')
    print(f"--- {n} (x={o['x'][i0]:.3f}) ---")
    print("  d(e-6): " + " ".join(f"{dists[j]*1e6:7.0f}" for j in jj))
    print("  u_t   : " + " ".join(f"{o['ut'][j, i0]:7.4f}" for j in jj))
    print("  u_t-s : " + " ".join(f"{(o['ut'][j, i0]-ut_s[j])*1e4:+7.2f}" for j in jj) + "  (x1e-4)")
    print("  omega : " + " ".join(f"{o['omg'][j, i0]:7.1f}" for j in jj))
    print("  lap   : " + " ".join(f"{o['lap'][j, i0]:7.3g}" for j in jj))

# spectral character: where does the wiggle live along x for cavL2?
print("\n=== per-station wiggle amplitude along x (cavL2), vs Gg spike ===")
o = D['cavL2']
ut_s = gaussian_filter1d(o['ut'], SIG, axis=0, mode='nearest')
wu = o['ut'] - ut_s
ru = np.sqrt(np.nanmean(wu[band]**2, axis=0))/np.nanmax(o['U'], axis=0)
d2 = dists[:, None]**2
den = o['U']**2 + (o['omg']*dists[:, None])**2
Gg = (d2*o['lap'])**2/np.maximum(den, 1e-30)
Ggb = np.nanmax(Gg[band], axis=0)
for k in range(0, len(o['x']), 4):
    print(f"  x={o['x'][k]:.3f}  wiggle={ru[k]:.2e}  max_band_Gg={Ggb[k]:.3f}")
