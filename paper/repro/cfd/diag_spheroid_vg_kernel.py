"""Does the two-source ("vg") kernel fix the alpha = 2.5 azimuthal over-sweep?

No.  It moves the whole front upstream by a nearly uniform 0.034 x/L and leaves
the azimuthal structure essentially untouched.

The test.  `spheroid/run_vg_case.sh` clones the converged canon case and reruns
it with the two-branch rate+gate switched on by environment variable only --
AI_A_VISC = 0.0276, AI_REOMC_BC = 130 -- so Flow360.json is byte-identical and
the comparison isolates the kernel.  Both runs went the full 45000 pseudo-steps
to a momentum residual of 1.2e-11.  Both surfaces were then probed with the SAME
`spheroid/surface_map.py` call, so the two fronts are like-for-like; the canon
front also matches the independent harvest path (sweep 0.147, dx/dbeta 4.25 here
against 4.30 there), which is what rules out a method offset.

Result at alpha = 2.5, Re_L = 7.2e6, calibrated seed:

    azimuthal sweep of the chi = 1 front   0.147 -> 0.137   (-7%)
    dx_front/dbeta                          4.25 ->  3.93   (-7%)
    corr(beta, front)                      +0.992 -> +0.991 (unchanged)
    mean shift                              -0.034, spread over azimuth 0.011

Stock's alpha = 2.5 front sweeps 0.030 over the whole body, and his Fig. 18
shows it dead vertical, so the target is ~0, not "somewhat less than 0.147".

Why this is the expected answer in hindsight.  The over-sweep lives in the GATE:
Re_Omega^c = k * softmin2(C, A + B/P^2) turns a 24% drop in P = max_y OmegaHat*Ihat
into a 47-59% threshold rise (see diag_spheroid_fpg_sensitivity.py).  The vg
branch adds a RATE floor a_visc, which contributes wherever OmegaHat*Ihat is
small -- that is a roughly uniform additive amplification, hence the uniform
upstream shift -- plus a second gate branch A + B_c/P_curv^2 with B_c = 130 that
evidently never wins on this geometry.  Neither touches the B/P^2 sensitivity
that produces the sweep.

It is NOT a pure failure, and should not be reported as one.  Against the
measured hot films at the two symmetry planes (windward 0.450, leeward 0.432)
the two-plane RMS error falls from 0.112 to 0.083, about 26%, because the
uniform shift is in the right direction.  The vg leeward front lands at 0.429
against a measured 0.432.  The windward front is still 0.117 late.

Run from paper/:  python3 repro/cfd/diag_spheroid_vg_kernel.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, 'cache_spheroid_surface')

from spheroid_a0_physics import front_crossing            # noqa: E402

STEM = 'surface_map_case_ogrid_L1_tun_a2p5_gcal'
PHIS = np.arange(5.0, 180.0, 15.0)
# measured DFVLR hot films at the two symmetry planes, alpha = 2.5 (fig14b)
MEAS_WIND, MEAS_LEE = 0.450, 0.432
STOCK_SWEEP = 0.030


def curve(d):
    """-> (phi, chi=1 front, mean Falkner-Skan beta ahead of it)."""
    xl, phd = d['xl'], d['phi_deg']
    ue = np.sqrt(np.clip(1.0 - d['cp'], 1e-9, None))
    out = []
    for ph in PHIS:
        j = int(np.argmin(abs(phd - ph)))
        f = front_crossing(xl, d['chimax'][j] - 1.0, 0.0)
        m = (xl > 0.08) & (xl < min(f if f == f else 0.6, 0.6))
        ml = (xl / np.maximum(ue[j], 1e-9)) * np.gradient(ue[j], xl)
        out.append((ph, f, float(np.mean(2 * ml[m] / (ml[m] + 1)))))
    return np.array(out)


def main():
    try:
        can = np.load(os.path.join(CACHE, f'{STEM}.npz'))
        vg = np.load(os.path.join(CACHE, f'{STEM}_vg.npz'))
    except FileNotFoundError as e:
        print(f'missing probe cache: {e}')
        return
    A, B = curve(can), curve(vg)

    print(f'{"phi":>5} {"canon":>7} {"vg":>7} {"vg-canon":>9} {"beta":>8}')
    for (p, fa, ba), (_, fb, _) in zip(A, B):
        print(f'{p:5.0f} {fa:7.3f} {fb:7.3f} {fb-fa:9.3f} {ba:8.4f}')

    for nm, X in (('canon', A), ('vg', B)):
        r = float(np.corrcoef(X[:, 2], X[:, 1])[0, 1])
        s = float(np.polyfit(X[:, 2], X[:, 1], 1)[0])
        print(f'\n{nm:>5}: front {X[:,1].min():.3f}..{X[:,1].max():.3f}  '
              f'sweep {np.ptp(X[:,1]):.3f}  dx/dbeta {s:5.2f}  r {r:+.3f}')

    d = B[:, 1] - A[:, 1]
    print(f'\nsweep {np.ptp(A[:,1]):.3f} -> {np.ptp(B[:,1]):.3f} '
          f'({100*(np.ptp(B[:,1])/np.ptp(A[:,1])-1):+.0f}%);  '
          f'Stock sweeps {STOCK_SWEEP:.3f}, and his Fig. 18 is dead vertical')
    print(f'mean shift {d.mean():+.4f}, spread over azimuth {np.ptp(d):.4f} '
          f'-- i.e. nearly uniform, a rate effect, not a gate effect')

    for nm, X in (('canon', A), ('vg', B)):
        w = float(X[0, 1]) - MEAS_WIND
        l = float(X[-1, 1]) - MEAS_LEE
        print(f'{nm:>5}: vs measured planes  windward {w:+.3f}  leeward {l:+.3f}'
              f'   RMS {np.sqrt(0.5*(w*w+l*l)):.3f}')


if __name__ == '__main__':
    main()
