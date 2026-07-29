"""Handover eddy-viscosity ratio xi = nu_t/nu at/after transition, from the
XFOIL/Drela integral closure.

Purpose
-------
Support the Drela handover feedback (expert_feedback.md, items 3 and 5): estimate
what viscosity ratio chi = nu_t/nu = nu_tilde/nu the transitional layer carries,
so we can put a number on "hand over at chi ~ 30, bubble-size-scaled" instead of
at chi ~ 1.

It also serves as the reproducibility check for
`expert_feedback/handover_viscosity_ratio.tex`: it recomputes the equilibrium
shear-stress coefficient directly and CROSS-CHECKS it against the actual codebase
closure `src/validation/mfoil.py::get_cteq` / `get_cttr` (XFOIL/Fidkowski mfoil),
so the algebra in the note is tied to the code we ship.

Two viscosity ratios are reported, and only the first answers the handover
question:

  xi_init  : AT the handover station -- the quantity of interest. XFOIL
             initializes Ctau = c^2 * Ctau_eq with c = CtauC*exp(-CtauE/(Hk-1))
             (get_cttr). c depends on Hk ALONE (no N dependence); it is applied
             once, at the station where N first reaches N_crit, and c = 1 only at
             Hk = 1 + CtauE/ln(CtauC) = 6.6. So the layer is handed over holding
             only c^2 of its equilibrium stress.
  xi_eq    : the local equilibrium the lag equation relaxes toward (Ctau_eq).
             Reported only because xi_init = c^2 * xi_eq; it is not the
             handover target.

Closure (verbatim constants from mfoil Param defaults, incompressible H = Hk):
  CC = 0.5/(GA^2 GB),  GA=6.7, GB=0.75          -> 0.01485   (the "0.015")
  Hs = get_Hs (turbulent KE shape parameter correlation)
  Us = 0.5 Hs (1 - (1/GB)(Hk-1)/H)              (get_Us)
  Ctau_eq = CC Hs (Hk-1) Hkc^2 / [(1-Us) H Hk^2],  Hkc = Hk-1-GC/Ret  (get_cteq)
  c  = CtauC exp(-CtauE/(Hk-1)), CtauC=1.8, CtauE=3.3               (get_cttr)

Eddy-viscosity mapping (turns the integral Ctau into a nu_t):
  nu_t,max = u_e^2 Ctau / (du/dy)_max,   (du/dy)_max = G * u_e/theta
  => xi = Re_theta * Ctau / G
The peak-shear coefficient G is NOT assumed: it is calibrated here on the
canonical Falkner-Skan family (`calibrate_G`), which is the right family for an
LSB because the profile at the transition station is still the laminar
(near-/post-separation) shear layer. The result is that G is essentially
INDEPENDENT of Hk, G ~ 0.207 +/- 2% over the whole inflectional range
Hk = 2.7 -> 4.9 (mild APG, through separation, into reversed flow). In
particular G does NOT scale like (Hk-1)/Hk.

Run:  python3 handover_viscosity_ratio.py
"""
import os
import sys

import numpy as np

# --- constants, verbatim from src/validation/mfoil.py Param defaults ----------
GA, GB, GC = 6.7, 0.75, 18.0
CC = 0.5 / (GA**2 * GB)          # = 0.014851, the derivation's "0.015"
CTAUC, CTAUE = 1.8, 3.3          # transition initialization factor c
GPEAK = 0.207                    # (theta/u_e)(du/dy)_max, calibrated below


def calibrate_G():
    """Peak shear G = (theta/u_e)(du/dy)_max on the Falkner-Skan family.

    In similarity variables (eta = y/L, u = u_e f'), theta = L*Ith with
    Ith = int f'(1-f') deta, so (theta/u_e)(du/dy) = Ith*f''  and
    G = Ith * max(f'').  Returns (list of (Hk, G, eta_peak), Gmean, spread).
    """
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..")))
    from lib.boundary_layer import FalknerSkanWedge

    cases = [(-0.05, None), (-0.10, None), (-0.14, None), (-0.17, None),
             (-0.19, None), (-0.1988, None), (-0.19, -0.03)]
    out = []
    for beta, guess in cases:
        fs = FalknerSkanWedge(beta, guess=guess)
        eta, fp, fpp = fs.eta, fs.u, fs.dudeta
        Ith = np.trapezoid(fp*(1 - fp), eta)
        Hk = np.trapezoid(1 - fp, eta)/Ith
        j = int(np.argmax(fpp))
        out.append((Hk, Ith*fpp[j], eta[j]))
    Gs = np.array([o[1] for o in out])
    return out, Gs.mean(), (Gs.max() - Gs.min())/2/Gs.mean()


def Hs_turb(Hk, Ret):
    """Turbulent KE shape parameter H* (mfoil get_Hs, incompressible)."""
    Hsmin, dHsinf = 1.5, 0.015
    Ho = 3.0 + 400.0 / Ret if Ret > 400 else 4.0
    Reb = max(Ret, 200.0)
    if Hk < Ho:
        Hr = (Ho - Hk) / (Ho - 1.0)
        aa = (2.0 - Hsmin - 4.0 / Reb) * Hr**2
        return Hsmin + 4.0 / Reb + aa * 1.5 / (Hk + 0.5)
    lrb = np.log(Reb)
    aa = Hk - Ho + 4.0 / lrb
    bb = 0.007 * lrb / aa**2 + dHsinf / Hk
    return Hsmin + 4.0 / Reb + (Hk - Ho)**2 * bb


def Us_(Hk, Hs, H):
    """Normalized slip velocity Us (mfoil get_Us, non-wake clip)."""
    Us = 0.5 * Hs * (1.0 - (1.0 / GB) * (Hk - 1.0) / H)
    return 0.98 if Us > 0.95 else Us


def ctau_eq(Hk, Ret):
    """Equilibrium shear-stress coefficient Ctau_eq (mfoil get_cteq^2)."""
    H = Hk                                       # incompressible
    Hs = Hs_turb(Hk, Ret)
    Us = Us_(Hk, Hs, H)
    Hkc = max(Hk - 1.0 - GC / Ret, 0.01)
    num = CC * Hs * (Hk - 1.0) * Hkc**2
    den = (1.0 - Us) * H * Hk**2
    return num / den, Hs, Us


def c_factor(Hk):
    return CTAUC * np.exp(-CTAUE / (Hk - 1.0))


def xi_from_ctau(Ctau, Hk, Ret):
    """nu_t/nu from integral Ctau via the calibrated peak shear G."""
    return Ret * Ctau / GPEAK


# ---------------------------------------------------------------------------
# Cross-check the reimplemented closure against the actual mfoil code.
# ---------------------------------------------------------------------------
def mfoil_crosscheck():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))  # sa-ai/
    for p in (os.path.join(root, "src", "validation"), os.path.join(root, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import mfoil as mf
    except Exception as e:  # noqa: BLE001
        return f"  (mfoil import failed: {e}; skipping direct code cross-check)"

    param = mf.Param()
    param.Minf = 0.0
    param.rho0 = 1.0
    param.turb = True
    param.wake = False

    lines = ["  mfoil get_cteq / get_cttr direct cross-check (incompressible):",
             f"  {'Hk':>5} {'Ret':>6} {'Ctau_eq(mine)':>14} {'Ctau_eq(mfoil)':>15}"
             f" {'Ctau_init(mine)':>16} {'Ctau_init(mfoil)':>17}"]
    ok = True
    for Hk in (1.5, 2.5, 3.5):
        for Ret in (200.0, 800.0):
            mu0 = 1.0 / Ret                      # th=ue=1 => Ret = th*ue/mu0
            param.mu0 = mu0
            U = np.array([1.0, Hk, 0.0, 1.0])    # [th, ds=Hk*th, sa, ue]
            cteq_m, _ = mf.get_cteq(U, param)
            cttr_m, _ = mf.get_cttr(U, param)
            Ceq_mf = cteq_m**2
            Cinit_mf = cttr_m**2
            Ceq_me, _, _ = ctau_eq(Hk, Ret)
            Cinit_me = c_factor(Hk)**2 * Ceq_me
            ok = ok and np.isclose(Ceq_mf, Ceq_me, rtol=1e-9) \
                and np.isclose(Cinit_mf, Cinit_me, rtol=1e-9)
            lines.append(f"  {Hk:5.1f} {Ret:6.0f} {Ceq_me:14.6f} {Ceq_mf:15.6f}"
                         f" {Cinit_me:16.6f} {Cinit_mf:17.6f}")
    lines.append(f"  => reimplementation matches mfoil closure: {ok}")
    return "\n".join(lines)


def report(label, Hk, Ret):
    Ceq, Hs, Us = ctau_eq(Hk, Ret)
    c = c_factor(Hk)
    Cinit = c**2 * Ceq
    print(f"{label:26s} Hk={Hk:.2f} Re_th={Ret:5.0f} | Hs={Hs:.3f} "
          f"Us={Us:.3f} c={c:.4g}")
    print(f"{'':26s}   xi_eq={xi_from_ctau(Ceq, Hk, Ret):6.2f}  "
          f"-> xi_init (HANDOVER) = c^2 xi_eq = "
          f"{xi_from_ctau(Cinit, Hk, Ret):.3g}")


if __name__ == "__main__":
    print(f"CC = 0.5/(GA^2 GB) = {CC:.5f}  (this is the derivation's 0.015)\n")
    print(mfoil_crosscheck())
    print()
    cal, Gmean, spread = calibrate_G()
    print("  Peak-shear calibration on Falkner-Skan, G = (theta/u_e)(du/dy)_max:")
    print(f"  {'Hk':>7} {'G':>8} {'eta_peak':>9}")
    for Hk, G, ep in cal:
        print(f"  {Hk:7.3f} {G:8.4f} {ep:9.3f}")
    print(f"  => G = {Gmean:.3f} +/- {100*spread:.1f}% over Hk = "
          f"{cal[0][0]:.1f}-{cal[-1][0]:.1f}: essentially independent of Hk")
    print(f"     (using GPEAK = {GPEAK}; note G is NOT ~ (Hk-1)/Hk)")
    print()
    print("chi = nu_t/nu (see module docstring):  xi_eq = local equilibrium,")
    print("xi_init = c^2 xi_eq = the level actually handed over at N=N_crit.")
    print(f"c = 1 only at Hk = 1 + {CTAUE}/ln({CTAUC}) = "
          f"{1 + CTAUE/np.log(CTAUC):.2f}\n")

    print("=== A. Attached / flat-plate transition ===")
    report("flat plate (turb Hk=1.5)", 1.5, 500)
    report("flat plate (turb Hk=1.5)", 1.5, 1000)
    report("flat-plate laminar Hk~2.6", 2.6, 700)
    print()
    print("=== B. Laminar separation bubble transition ===")
    for Hk in (2.5, 3.0, 3.5):
        for Ret in (200.0, 800.0):
            report("bubble", Hk, Ret)
    print()
    print("Takeaways:")
    print("  * 0.015 is legitimate (= CC of the G-beta locus).")
    print("  * G = (theta/u_e)(du/dy)_max ~ 0.207 is Hk-independent on the")
    print("    Falkner-Skan family, so xi = Re_theta*Ctau/G ~ 4.8 Re_theta Ctau.")
    print("  * c has NO N dependence: it is a function of Hk alone, applied at")
    print("    the N=N_crit station. The layer is handed over holding only c^2")
    print("    of equilibrium stress (4% at Hk=2.5, 23% at Hk=3.5).")
    print("  * Handover level xi_init = O(0.2-7), rising ~8x from Hk=2.5 to 3.5:")
    print("    'bigger bubble -> later handover' is quantitative, but XFOIL's")
    print("    own initialization is chi of order a few, NOT ~30.")
