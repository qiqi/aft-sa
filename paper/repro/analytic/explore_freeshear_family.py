"""Does the production peak sit near Lambda_v=0 for DIFFERENT free-shear
profiles (not just one tanh)? Symmetric mixing layer, asymmetric mixing
ratios, a jet edge, and a wake -- each placed far from the wall (h/delta
large, so it is a genuine free shear layer). Report (Gamma, Lambda_v) at
the Rayleigh production peak, and how far the peak Lambda_v strays from 0.

Also print the wall (attached-linear) and the dead-air crossing for
reference, to show what Lambda_v alone can and cannot separate.
"""
import numpy as np
from scipy.linalg import eig
from scipy.optimize import minimize_scalar

H = 12.0            # wall distance of the layer center (far from wall)


def prodpeak(uf, upf, uppf, ymax, N=1000):
    y = np.linspace(0, ymax, N)
    u, up, upp = uf(y), upf(y), uppf(y)
    hgy = y[1]-y[0]
    n = N-2
    D2 = (np.diag(np.ones(n-1), -1)-2*np.eye(n)+np.diag(np.ones(n-1), 1))/hgy**2

    def wi(al, vec=False):
        B = D2-al**2*np.eye(n)
        A = np.diag(u[1:-1])@B - np.diag(upp[1:-1])
        c, V = eig(A, B, right=True)
        j = int(np.argmax(np.imag(c)))
        return (complex(c[j]), V[:, j]) if vec else al*float(np.imag(c[j]))
    r = minimize_scalar(lambda a: -wi(a), bounds=(0.05, 1.4),
                        method='bounded', options={'xatol': 2e-3})
    al = float(r.x)
    c, v = wi(al, vec=True)
    phi = np.zeros(N, complex); phi[1:-1] = v
    dphi = np.gradient(phi, hgy)
    p = 0.5*al*np.abs(np.imag(dphi*np.conj(phi)))*np.abs(up)
    jp = int(np.argmax(p))
    d = y[jp]
    den = (up[jp]*d)**2+u[jp]**2+1e-30
    return (2*(up[jp]*d)**2/den, -up[jp]*upp[jp]*d**3/den, c.real, y[jp]-H)


def tanh_prof(Um):
    # u = Um + tanh((y-H)); Um shifts the mean (mixing ratio)
    return (lambda y: Um+np.tanh(y-H),
            lambda y: 1/np.cosh(y-H)**2,
            lambda y: -2*np.tanh(y-H)/np.cosh(y-H)**2, H+9)


def jet_edge():
    # single edge of a jet: u = sech^2 half -- use u = 1/cosh^2((y-H)) rising edge
    return (lambda y: 1/np.cosh(y-H)**2,
            lambda y: -2*np.sinh(y-H)/np.cosh(y-H)**3,
            lambda y: (-2+4*np.sinh(y-H)**2/np.cosh(y-H)**2)/np.cosh(y-H)**2,
            H+9)


def wake():
    # wake deficit: u = 1 - 0.7 sech^2((y-H))
    a = 0.7
    return (lambda y: 1-a/np.cosh(y-H)**2,
            lambda y: 2*a*np.sinh(y-H)/np.cosh(y-H)**3,
            lambda y: -a*(-2+4*np.sinh(y-H)**2/np.cosh(y-H)**2)/np.cosh(y-H)**2,
            H+9)


if __name__ == "__main__":
    print(f"free-shear profiles centered at wall distance H={H}; "
          "Rayleigh production peak:")
    print(f"{'profile':>22} {'Gamma':>7} {'Lambda_v':>9} {'c':>6} "
          f"{'(y-H)@pk':>9}")
    cases = [("mixing layer Um=0", tanh_prof(0.0)),
             ("mixing ratio Um=0.3", tanh_prof(0.3)),
             ("mixing ratio Um=0.6", tanh_prof(0.6)),
             ("jet edge (sech^2)", jet_edge()),
             ("wake (1-0.7sech^2)", wake())]
    for name, (uf, upf, uppf, ym) in cases:
        try:
            G, Lv, c, dyp = prodpeak(uf, upf, uppf, ym)
            print(f"{name:>22} {G:7.2f} {Lv:+9.2f} {c:6.2f} {dyp:+9.2f}",
                  flush=True)
        except Exception as e:
            print(f"{name:>22}  failed: {e}")
    print("\nreference loci (not production peaks):")
    print(f"{'attached wall':>22} {'~1.0':>7} {'~0.00':>9}   (Gamma=1, Lv=0)")
    print(f"{'dead-air bubble xing':>22} {'2.00':>7} {'-2.00':>9}   "
          "(Gamma=2, Lv=-2)")
