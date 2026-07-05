"""Numerical FD verification of the chi^{1/4} σ_t form and its analytic Jacobian.

This mirrors the new __aftIsTurb in SAAftTransition.h:

    chi_eff = chi^{1/4}
    isTurb  = 0,                                      chi <= switchCenter
              1 - exp(-(chi_eff - switchCenter)/τ),   chi >  switchCenter
    d(isTurb)/d(nuHat) = (1/(4τν))·χ^{-3/4}·exp(...)

Run: python3 verify_chi_quartic_jacobian.py
"""
import numpy as np

nu = 1e-5
switchCenter = 1.0
switchWidth  = 4.0

def isTurb(nuHat):
    chi = nuHat / nu
    if chi <= switchCenter:
        return 0.0
    chi_eff = chi ** 0.25
    return 1.0 - np.exp(-(chi_eff - switchCenter) / switchWidth)

def dIsTurb_dNuHat_analytic(nuHat):
    chi = nuHat / nu
    if chi <= switchCenter:
        return 0.0
    chi_eff = chi ** 0.25
    e = np.exp(-(chi_eff - switchCenter) / switchWidth)
    # d(chi_eff)/d(nuHat) = (1/(4 nu)) * chi^{-3/4}
    dChiEff_dNuHat = 0.25 / (nu * chi_eff**3)
    return (e / switchWidth) * dChiEff_dNuHat

def fd_central(nuHat, h_rel=1e-5):
    h = h_rel * nuHat
    return (isTurb(nuHat + h) - isTurb(nuHat - h)) / (2 * h)

print(f"{'chi':>8s} {'isTurb':>10s} {'dIsTurb/dnu (analytic)':>26s} {'FD':>16s} {'rel err':>10s}")
for chi_test in [0.5, 1.0, 1.05, 1.5, 2.0, 5.0, 20.0, 100.0, 1000.0]:
    nuHat = chi_test * nu
    val   = isTurb(nuHat)
    da    = dIsTurb_dNuHat_analytic(nuHat)
    dfd   = fd_central(nuHat) if chi_test > switchCenter else 0.0
    rel   = abs(dfd - da) / (abs(da) + 1e-30)
    print(f"{chi_test:>8.2f} {val:>10.4e} {da:>26.6e} {dfd:>16.6e} {rel:>10.2e}")

# Compare with old chi-linear form
print("\nReference (old chi-linear σ_t for comparison):")
def isTurb_old(nuHat):
    chi = nuHat / nu
    if chi <= switchCenter: return 0.0
    return 1.0 - np.exp(-(chi - switchCenter) / switchWidth)
print(f"{'chi':>8s} {'σ_t,old(chi)':>14s} {'σ_t,new(chi)':>14s} {'physical match?':>18s}")
print("  (σ_t,new should follow σ_t,old(chi^{1/4}); at chi=1 both 0; at chi^{1/4}=3.77 old=0.5)")
for chi_test in [1.0, 1.0**4, 3.77, 3.77**4, 5**4]:
    nuHat = chi_test * nu
    print(f"{chi_test:>8.2f} {isTurb_old(nuHat):>14.4f} {isTurb(nuHat):>14.4f}")

# Verify the physical-band-width preservation:
# old σ_t = 0.5 at χ_old ≈ 3.77
# new σ_t = 0.5 at χ_new = 3.77^4 ≈ 201.5
# in physical space these correspond to the same Δs because χ_new = χ_old^4 along streamlines
chi_50_old = 1 + switchWidth * np.log(2)
chi_50_new = chi_50_old ** 4
print(f"\nσ_t = 0.5 at:  chi_old = {chi_50_old:.3f}   chi_new = chi_old^4 = {chi_50_new:.3f}")

# χ_∞ rescaling: 0.02 → 0.02^4 = 1.6e-7
chi_inf_old = 0.02
chi_inf_new = chi_inf_old ** 4
print(f"χ_∞ rescaling:  0.02 (legacy) → {chi_inf_new:.2e} (internal BC/IC)")
print(f"               N_crit = ln(1/0.02) = {np.log(1/0.02):.2f} unchanged")
