"""Does the FROZEN instrument reproduce the ~1-4% Q2 effect the coupled CFD
shows? Q2-on vs Q2-off on the same separated Falkner-Skan profiles and the
same composite-gate constants, sweeping the advection floor ufrac -- and
reporting WHERE the spatial eigenmode sits (u at its peak), which is the
suspected source of the frozen-vs-coupled disagreement.

Q2-off = Q1s alone (shape 'off'); Q2-on = Q1s * parabola-loop Q2, c2=8.
"""
import sys
import numpy as np
from scipy.linalg import eigh_tridiagonal
import scipy.sparse as sp  # noqa

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa
from _saai import C_NU_AI, SIGMA_SA
from explore_q2_shape import gate, build_profile, drela_spatial, PROFILES


def sigma_x_vec(pr, Re_th, shape, c2, ufrac):
    """Leading spatial growth rate AND the u at the eigenmode's energy peak."""
    Th = pr['Theta']
    yh = pr['eta'][1:-1]/Th
    u = pr['u'][1:-1]
    w = Th*pr['up'][1:-1]
    dwdy = Th*Th*pr['upp'][1:-1]
    Gam = 2*(w*yh)**2/((w*yh)**2 + u**2 + 1e-300)
    from lib.aft_sources import compute_aft_amplification_rate
    K = dict(gc=0.9874, s=10.68, floor=243.7)
    rate = np.asarray(compute_aft_amplification_rate(
        np.abs(w)*yh**2*Re_th, Gam, lambda_p=0.0, sigmoid_center=K['gc'],
        sigmoid_slope=K['s'], re_omega_floor=K['floor'], barrier_power=4.0))
    b = rate*gate(w, dwdy, u, yh, shape, 2.0, c2)*np.abs(w)
    h = yh[1] - yh[0]
    kd = (C_NU_AI/SIGMA_SA)/Re_th
    uf = np.maximum(u, ufrac)
    d = (b - 2*kd/h**2)/uf
    e = (kd/h**2)/np.sqrt(uf[:-1]*uf[1:])
    lam, vec = eigh_tridiagonal(d, e, select='i',
                                select_range=(len(yh)-1, len(yh)-1))
    v = np.abs(vec[:, 0])/np.sqrt(uf)
    jp = int(np.argmax(v))
    return float(lam[0]), float(u[jp])


def main():
    print("Frozen separated FS: Q2 on/off, composite constants, "
          "R_x = dN/dx vs Drela.\n")
    for uf in (0.03, 0.10, 0.30):
        print(f"=== advection floor ufrac = {uf} ===")
        print(f"{'H':>7} {'Re':>5} | {'off(Q1)':>8} {'on(Q1Q2)':>9} "
              f"{'on/off':>7} | {'u@mode off/on':>14}")
        for b, g in PROFILES:
            pr = build_profile(b, g)
            dS = drela_spatial(pr['H'])
            for Re in (200.0, 400.0):
                soff, uoff = sigma_x_vec(pr, Re, 'off', 8.0, uf)
                son, uon = sigma_x_vec(pr, Re, 'band', 8.0, uf)
                roff, ron = soff/dS, son/dS
                ratio = son/soff if soff else float('nan')
                print(f"{pr['H']:7.2f} {Re:5.0f} | {roff:8.2f} {ron:9.2f} "
                      f"{ratio:7.2f} | {uoff:+.3f}/{uon:+.3f}", flush=True)
        print()


if __name__ == "__main__":
    main()
