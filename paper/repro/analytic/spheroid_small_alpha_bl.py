"""What the viscous flow does on the two symmetry meridians at infinitesimal
incidence: the O(alpha) response of the spheroid boundary layer, built on the
exact potential edge field of `spheroid_potential.py`.

THE QUESTION.  At alpha = 0 the spheroid boundary layer is known at every
station.  Put on an infinitesimal alpha.  What happens in the circumferential
direction, and does it explain why Stock's and the measured transition fronts
are near-symmetric between phi_w = 0 and 180 while ours sweeps by 0.147 x/L?

WHAT THE POTENTIAL FIELD GIVES, exactly.  Writing phi_w from the windward
generator and eta = 2x/L - 1,

    u_s   = U [ f0(eta) + alpha f1(eta) cos(phi_w) ],
    u_phi = U   alpha g sin(phi_w),

    f1(eta) = T eta / (c sqrt(xi0^2 - eta^2)),     g = T / b = const,

with T = c sqrt(xi0^2-1) + B Q11(xi0).  Two exact facts follow, and both matter:

  (1) **g is CONSTANT along the body.**  The azimuthal edge velocity is
      U alpha (T/b) sin(phi_w) at every station.  T/b = 1.9171 for 6:1, and
      T/b -> 2 as b/a -> 0, which is the two-dimensional cylinder crossflow
      factor -- an independent check on the transverse mode.

  (2) **f1 is ANTISYMMETRIC about mid-body**, f1 propto eta, vanishing exactly
      at x/L = 0.5.  So at O(alpha) the windward meridian is SLOWER than the
      axisymmetric one over the front half and FASTER over the rear half, and
      the perturbation passes through zero at mid-body.  The alpha = 0 front
      sits at x/L = 0.425, i.e. almost exactly where f1 -> 0.

Fact (2) is the first half of the answer: the edge-velocity asymmetry is
concentrated near the nose, where Re_theta is far below critical and N
accumulates nothing, and it dies out just where transition actually happens.

WHAT THIS SCRIPT ADDS.  Fact (2) alone is not enough -- N is an integral, so
what matters is the perturbation weighted over the whole amplifying run.  This
marches the generalized Thwaites momentum integral on each symmetry meridian,

    theta^2 u_e^6 W^2 = 0.45 nu integral u_e^5 W^2 ds,
    W = r0 exp( integral k ds ),   k = (1/(u_e r0)) d(u_phi)/d(phi_w),

so W is the physical width of a streamtube straddling the symmetry plane: r0
for the axisymmetric part and exp(int k) for the extra azimuthal straining,
k > 0 windward (diverging) and k < 0 leeward (converging) by fact (1).  Then a
Drela-Giles envelope on the marched Re_theta gives the front, and differencing
in alpha gives the sensitivity dx_tr/d(alpha).

Three marches are compared, to separate the two mechanisms:
    'ue'      edge-velocity perturbation only, W = r0     (what a 2-D envelope
                                                           on each meridian sees)
    'div'     divergence only, u_e = U f0
    'both'    the full O(alpha) response

The march start x0 is swept, because int k ds picks up a logarithm at the nose.
It turns out not to matter: the 'div' answer is 0.219 at x0 = 0.004, 0.010 and
0.025 alike, because the integral is dominated by mid-body, not the nose.

WHAT COMES OUT, and it refutes the hypothesis this script was written to test.
At alpha = 2.5, Re_L = 7.2e6, windward minus leeward front:

    edge-velocity perturbation only          +0.068   (dx/dalpha = 0.78)
    lateral divergence only                 +0.219   (dx/dalpha = 2.51)
    both                                    +0.287   (dx/dalpha = 3.28)
    ---
    SA-AI, chi = 1, calibrated seed          +0.147
    measured, DFVLR hot films                +0.018
    Stock, two-N-factor e^N on a 3-D BL      -0.030

  1. The edge-velocity channel is CONFIRMED and is not contaminated.  Marching
     on the exact potential u_e gives +0.068, against +0.067 from marching on
     our own computed c_p (diag_spheroid_symmetry_planes.py).  The two agree to
     0.001, and the potential and computed windward/leeward u_e ratio at
     x/L = 0.05 agree to three digits at alpha = 5 (0.895 vs 0.896).  So the
     earlier worry that our own transition was corrupting that march is closed:
     it was not.

  2. The lateral divergence does NOT rescue the symmetry -- it makes it three
     times WORSE, and in the same direction.  This kills the hypothesis that
     Stock's full 3-D boundary layer gets the symmetry right because it carries
     a divergence term our axisymmetric march omits.  It carries the term, and
     the term has the wrong sign to help.

  3. A textbook 3-D momentum integral plus a Drela-Giles envelope therefore
     predicts an asymmetry of +0.287 -- TWICE our own kernel's +0.147 and
     sixteen times the measurement.  Our over-sweep is not an SA-AI defect
     against a competent classical baseline; the classical baseline is worse.

  4. So the deficiency is not in the amplification rate and not in the
     divergence bookkeeping.  It is in how a stability estimate built on local
     Re_theta and a local pressure-gradient parameter responds to the O(alpha)
     perturbation at all.  The streamtube-width model assumes lateral straining
     stretches the whole profile uniformly; in a real 3-D layer the fluid
     converging onto the leeward plane arrives with its own profile, the
     near-wall and edge crossflow differ, and the SHAPE -- which is what governs
     stability -- responds far more weakly than the width does.  Stock resolves
     the profile (121 wall-normal points, finite-difference 3-D BL, local LST)
     and gets the near-symmetry; every integral/local method here, ours
     included, does not.

  This also explains why the two-source kernel could not help
  (diag_spheroid_vg_kernel.py): it modifies the rate, and the defect is in the
  profile-shape response.

Run from paper/:  python3 repro/analytic/spheroid_small_alpha_bl.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'cfd')))

from spheroid_potential import (A_AX, B_AX, C_FOC, XI0,          # noqa: E402
                                surface_fields, metrics)
from spheroid_a0_physics import (Re_theta0, dN_dRe_theta,        # noqa: E402
                                front_crossing)

RE_L = 7.20e6                    # the alpha = 0 / 2.5 condition
N_TARGET = 8.0                   # Stock's Goettingen limiting N_TS
NPT = 6000


def grid(x0, x1):
    eta = np.linspace(2 * x0 - 1, 2 * x1 - 1, NPT)
    _, h_eta, _ = metrics(XI0, eta)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (h_eta[1:] + h_eta[:-1])
                                         * np.diff(eta))])
    return eta, s, B_AX * np.sqrt(1.0 - eta**2)


def march(alpha, side, mode, x0, nu):
    """side = +1 windward (phi_w=0), -1 leeward (phi_w=180)."""
    eta, s, r0 = grid(x0, 0.97)
    f0, f1, g = surface_fields(eta)
    ue = f0 + (alpha * f1 * side if mode in ('ue', 'both') else 0.0)
    ue = np.maximum(ue, 1e-6)
    if mode in ('div', 'both'):
        k = alpha * side * g / (ue * r0)
        lnW = np.log(r0) + np.concatenate(
            [[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(s))])
    else:
        lnW = np.log(r0)
    W = np.exp(lnW - lnW.max())                     # scale-free; W^2 cancels
    w = ue**5 * W**2
    I = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(s))])
    th2 = 0.45 * nu * I / np.maximum(ue**6 * W**2, 1e-300)
    lam = np.clip(th2 / nu * np.gradient(ue, s), -0.12, 0.25)
    H = np.where(lam >= 0, 2.61 - 3.75 * lam + 5.24 * lam**2,
                 2.088 + 0.0731 / (lam + 0.14))
    Rt = ue * np.sqrt(th2) / nu
    Rt0 = np.asarray(Re_theta0(H))
    dN = np.where(Rt > Rt0,
                  np.maximum(np.asarray(dN_dRe_theta(H)) * np.gradient(Rt, s),
                             0.0), 0.0)
    N = np.concatenate([[0.0], np.cumsum(0.5 * (dN[1:] + dN[:-1]) * np.diff(s))])
    xl = 0.5 * (1.0 + eta)
    return front_crossing(xl, N, N_TARGET)


def main():
    nu = 1.0 / RE_L
    _, f1_mid, g = surface_fields(np.array([0.0]))
    print(f'exact O(alpha) edge field:  g = T/b = {g[0]:.4f} (constant along '
          f'the body; -> 2 in the slender limit, the 2-D cylinder crossflow)')
    print(f'                            f1(x/L=0.5) = {f1_mid[0]:.2e}  '
          f'(f1 propto eta, so it vanishes at mid-body)')
    print(f'{"x/L":>6} {"f0":>8} {"f1":>8}   f1 is the O(alpha) meridional '
          f'perturbation, xcos(phi_w)')
    for xl in (0.05, 0.15, 0.30, 0.425, 0.55, 0.80):
        f0, f1, _ = surface_fields(np.array([2 * xl - 1]))
        print(f'{xl:6.3f} {f0[0]:8.4f} {f1[0]:8.4f}')

    print(f'\nfront at N = {N_TARGET:g}, Re_L = {RE_L:.3g}, marched from '
          f'x/L = x0 on each symmetry meridian:')
    print(f'{"x0":>6} {"mode":>5} {"alpha":>7} {"windward":>9} {"leeward":>9} '
          f'{"wind-lee":>9} {"d/dalpha":>9}')
    base = {}
    for x0 in (0.004, 0.010, 0.025):
        for mode in ('ue', 'div', 'both'):
            row = {}
            for al_deg in (0.0, 2.5):
                al = np.radians(al_deg)
                w = march(al, +1, mode, x0, nu)
                l = march(al, -1, mode, x0, nu)
                row[al_deg] = (w, l)
                if al_deg == 0.0:
                    base[(x0, mode)] = w
            w, l = row[2.5]
            al = np.radians(2.5)
            print(f'{x0:6.3f} {mode:>5} {2.5:7.2f} {w:9.3f} {l:9.3f} '
                  f'{w-l:9.3f} {(w-l)/(2*al):9.2f}')
        print(f'{x0:6.3f} {"--":>5} {0.0:7.2f} {base[(x0,"ue")]:9.3f} '
              f'{base[(x0,"ue")]:9.3f} {0.0:9.3f} {"":>9}  '
              f'(alpha = 0 reference)')

    print('\nfor comparison, windward - leeward at alpha = 2.5, Re_L = 7.2e6:')
    print('  measured (DFVLR hot films, 3 stations)     +0.018')
    print('  Stock, two-N-factor e^N on a 3-D BL        -0.030')
    print('  SA-AI, chi=1 front, calibrated seed        +0.147')


if __name__ == '__main__':
    main()
