"""Is the frozen dead-air spatial growth limited by the advection floor ufrac
(numerical) or by molecular diffusion (physical, hence Re-dependent)?

Drive ufrac -> 0 and sweep Re_theta on a deep separated FS profile, Q2-off
(bare dead-air singularity) and Q2-on. Also report the u at the eigenmode
peak, so we see whether the mode stays pinned to the u=0 line.

If dN/dx keeps rising as ufrac -> 0 with Re fixed  -> ufrac-limited.
If dN/dx saturates as ufrac -> 0 but rises with Re -> diffusion-limited.
"""
import sys
import numpy as np
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa
from _saai import C_NU_AI, SIGMA_SA
from explore_q2_shape import gate, build_profile, drela_spatial
from lib.aft_sources import compute_aft_amplification_rate

K = dict(gc=0.9874, s=10.68, floor=243.7)


def sigma_x(pr, Re_th, shape, c2, ufrac):
    Th = pr['Theta']
    yh = pr['eta'][1:-1]/Th
    u = pr['u'][1:-1]
    w = Th*pr['up'][1:-1]
    dwdy = Th*Th*pr['upp'][1:-1]
    Gam = 2*(w*yh)**2/((w*yh)**2 + u**2 + 1e-300)
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
    return float(lam[0]), float(u[int(np.argmax(v))])


def main():
    pr = build_profile(-0.15, -0.08)     # H ~ 7.5, deep bubble
    dS = drela_spatial(pr['H'])
    print(f"deep FS profile H={pr['H']:.2f}, Drela dN/dx = {dS:.4f}\n")
    for shape, tag in (('off', 'Q2 OFF (bare)'), ('band', 'Q2 ON  (c2=8)')):
        print(f"=== {tag} :  R_x = dN/dx / Drela ===")
        print(f"{'ufrac':>7} |" +
              "".join(f" Re={r:<5.0f}(u@pk)" for r in (100, 200, 400, 800)))
        for uf in (0.10, 0.03, 0.01, 0.003, 0.001):
            cells = []
            for Re in (100, 200, 400, 800):
                s, upk = sigma_x(pr, float(Re), shape, 8.0, uf)
                cells.append(f" {s/dS:5.2f}({upk:+.2f})")
            print(f"{uf:7.3f} |" + "".join(cells), flush=True)
        print()


if __name__ == "__main__":
    main()
