"""Correcting the earlier error: for a free shear layer at varying wall
distance h/delta, WHERE does the wavepacket PRODUCTION peak (critical layer)
sit in Lambda_v -- not where |Lambda_v| is maximal.

At the inflection u''=0 so Lambda_v=0 exactly, regardless of wall distance;
the critical layer sits at/near the inflection. So the production peak
should be near Lambda_v=0 for ALL h/delta -- meaning a thin band near
Lambda_v~0 (slightly +) would generalize to free shear AND exclude the
dead-air crossing (Lambda_v=-2). This tests that.

Rayleigh (inviscid) most-unstable mode of u=0.5(1+tanh((y-h)/delta)) on a
wall at y=0; production p(y) = (alpha/2)|Im(phi' phi*)| |u'|.
"""
import numpy as np
from scipy.linalg import eig
from scipy.optimize import minimize_scalar

DELTA = 1.0


def profile(h, ymax, N):
    y = np.linspace(0, ymax, N)
    u = 0.5*(1 + np.tanh((y - h)/DELTA))
    up = np.gradient(u, y)
    upp = np.gradient(up, y)
    return y, u, up, upp


def rayleigh_prodpeak(h):
    ymax = h + 10*DELTA
    N = 900
    y, u, up, upp = profile(h, ymax, N)
    hgy = y[1] - y[0]
    n = N - 2
    D2 = (np.diag(np.ones(n-1), -1) - 2*np.eye(n) + np.diag(np.ones(n-1), 1))/hgy**2

    def wi(al, vec=False):
        B = D2 - al**2*np.eye(n)
        A = np.diag(u[1:-1]) @ B - np.diag(upp[1:-1])
        c, V = eig(A, B, right=True)
        ci = np.imag(c)
        j = int(np.argmax(ci))
        return (complex(c[j]), V[:, j]) if vec else al*float(ci[j])

    r = minimize_scalar(lambda a: -wi(a), bounds=(0.05/DELTA, 1.2/DELTA),
                        method='bounded', options={'xatol': 2e-3})
    al = float(r.x)
    c, v = wi(al, vec=True)
    phi = np.zeros(N, complex); phi[1:-1] = v
    dphi = np.gradient(phi, hgy)
    p = 0.5*al*np.abs(np.imag(dphi*np.conj(phi)))*np.abs(up)
    jp = int(np.argmax(p))
    # Lambda_v at the production peak (d = wall distance = y)
    d = y[jp]
    Lv = -up[jp]*upp[jp]*d**3/((up[jp]*d)**2 + u[jp]**2 + 1e-30)
    Gam = 2*(up[jp]*d)**2/((up[jp]*d)**2 + u[jp]**2 + 1e-30)
    # core extent in Lambda_v (p>0.5)
    core = p > 0.5*p.max()
    dd = y[core]
    Lc = -up[core]*upp[core]*dd**3/((up[core]*dd)**2 + u[core]**2 + 1e-30)
    return al, c.real, y[jp]-h, Lv, Gam, Lc.min(), Lc.max()


if __name__ == "__main__":
    print("free tanh shear layer (delta=1) at wall distance h, "
          "Rayleigh most-unstable mode:")
    print(f"{'h/delta':>8} {'c':>6} {'(y-h)@peak':>11} {'Lv@peak':>9} "
          f"{'Gam@peak':>9} {'Lv core[min,max]':>20}")
    for h in (1.5, 3, 6, 12, 25):
        al, c, dyp, Lv, Gam, Lmin, Lmax = rayleigh_prodpeak(h)
        print(f"{h:8.1f} {c:6.2f} {dyp:+11.2f} {Lv:+9.2f} {Gam:9.2f} "
              f"[{Lmin:+.2f},{Lmax:+.2f}]", flush=True)
