"""Linear ramp on the frozen Hiemenz field: eigenvalue, Airy asymptote, march.

Below chi = 1 the SA equation linearizes exactly (destruction is O(chi^3),
the f_v2 term in S~ is O(chi^2), nu^ -> 1).  Dropping v-advection and
x-diffusion, both measured small on the converged field, leaves

    x f'(y) chi_x  =  P(y) x chi  +  D chi_yy ,      D = c_nu/sigma

with P = cb1 f'' for standard SA and P = a_ai f'' for SA-AI.  x cancels
between advection and production, so chi = G(y) exp(int lambda dx) gives the
eigenvalue problem

    (D/x) G'' + P(y) G = lambda f'(y) G ,     G(0) = G(inf) = 0 .

Near the wall f' = a y, f'' = a with a = f''(0), and standard SA's version is
Airy's equation in zeta = (lambda a x / D)^{1/3} (y - cb1/lambda).  The wall
condition G(0) = 0 selects the first Airy zero a1 = -2.33811, closing it:

    lambda(x) = (cb1/|a1|)^{3/2} sqrt(a x / D)  ~  sqrt(x)
    ln chi    = (2/3) (cb1/|a1|)^{3/2} sqrt(a/D) x^{3/2}

Entry points:
    lam_airy, yp_airy   near-wall Airy asymptote (standard SA)
    lam_exact           exact eigenvalue on the full Hiemenz profile
    eig_lambda          same, selecting standard SA or SA-AI production
    march               independent check: fine-grid backward-Euler march
                        of the linear PDE, no WKB and no eigenvalue
"""
import os
import sys

import numpy as np
from scipy.linalg import eigh, solve_banded
from scipy.special import ai_zeros

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stagnation_bistability import CB1, SIGMA, PAPER, C_NU_AI, hiemenz

sys.path.insert(0, os.path.join(PAPER, 'repro'))
from lib.sphere_kernel import sphere_rate

A0 = 1.2325876568                      # f''(0), Hiemenz
A1 = ai_zeros(1)[0][0]                 # first zero of Ai
AIP = -1.0187929716                    # first zero of Ai' (peak of Ai)


def lam_airy(x, D=1.0/SIGMA):
    """Near-wall Airy asymptote for the standard-SA growth rate."""
    return (CB1/abs(A1))**1.5*np.sqrt(A0*x/D)


def yp_airy(x, D=1.0/SIGMA):
    """Height of the eigenfunction peak, same asymptote."""
    lam = lam_airy(x, D)
    return (abs(A1)-abs(AIP))*(D/(x*A0*lam))**(1.0/3.0)


def _solve(x, P, D, y):
    """Largest eigenvalue of (D/x) G'' + P G = lambda f' G, G(0)=G(inf)=0."""
    h = y[1]-y[0]
    F = hiemenz()
    fp = F(np.minimum(y, 20.0))[1]
    m = len(y)
    L = (np.diag(-2.0*np.ones(m)) + np.diag(np.ones(m-1), 1)
         + np.diag(np.ones(m-1), -1))/h**2
    A = (D/x)*L + np.diag(P)                 # symmetric
    s = 1.0/np.sqrt(fp)                      # B = diag(f') > 0 for y > 0
    w, V = eigh((A*s).T*s)                   # B^{-1/2} A B^{-1/2}, symmetric
    k = int(np.argmax(w))
    g = s*V[:, k]
    g = g/np.max(np.abs(g))
    return w[k], y[int(np.argmax(np.abs(g)))], y, g


def lam_exact(x, D=1.0/SIGMA, ymax=12.0, n=2000):
    """Exact standard-SA eigenvalue on the full Hiemenz profile."""
    y = np.linspace(0.0, ymax, n+1)[1:]      # G(0)=0 enforced by omission
    F = hiemenz()
    fpp = F(np.minimum(y, 20.0))[2]
    return _solve(x, CB1*fpp, D, y)


def eig_lambda(x, mode='sa', ymax=12.0, n=1500):
    """Eigenvalue and peak height for mode in {'sa', 'ai'}."""
    y = np.linspace(0.0, ymax, n+1)[1:]
    F = hiemenz()
    fy = F(np.minimum(y, 20.0))
    f, fp, fpp = fy[0], fy[1], fy[2]
    if mode == 'sa':
        P, D = CB1*fpp, 1.0/SIGMA
    else:
        fppp = -(f*fpp + 1.0 - fp**2)
        P, D = sphere_rate(x*fp, x*fpp, x*fppp, y, nu=1.0)*fpp, C_NU_AI/SIGMA
    w, yp = _solve(x, P, D, y)[:2]
    return w, yp


def march(x0, x1, D=1.0/SIGMA, chi0=None, ymax=12.0, ny=1200, dx=0.05):
    """Backward-Euler march of the linear ramp equation. Returns ln growth."""
    y = np.linspace(0.0, ymax, ny+1)
    dy = y[1]-y[0]
    F = hiemenz()
    fy = F(np.minimum(y, 20.0))
    fp, fpp = fy[1], fy[2]
    yi, fpi, fppi = y[1:-1], fp[1:-1], fpp[1:-1]
    P = CB1*fppi/fpi
    c = chi0(yi) if chi0 else np.exp(-((yi-0.6)/0.4)**2)
    c = c/np.max(c)
    n = len(yi)
    lg = 0.0
    for k in range(int(round((x1-x0)/dx))):
        xx = x0 + (k+1)*dx
        Q = D/(xx*fpi)
        a = -dx*Q/dy**2
        b = 1.0 - dx*P + 2.0*dx*Q/dy**2
        ab = np.zeros((3, n))
        ab[0, 1:] = a[:-1]
        ab[1, :] = b
        ab[2, :-1] = a[1:]
        c = solve_banded((1, 1), ab, c)
        m = np.max(c)
        lg += np.log(m)
        c = c/m
    return lg, yi, c


def wkb(x0, x1, npts=41, D=1.0/SIGMA):
    """int lambda dx between two stations, exact eigenvalue."""
    xs = np.linspace(x0, x1, npts)
    return np.trapz([lam_exact(x, D)[0] for x in xs], xs)


if __name__ == '__main__':
    D = 1.0/SIGMA
    print(f'D = c_nu/sigma = {D:.4f}   (standard SA)')
    print(f'{"x":>7}{"lam_exact":>12}{"lam_Airy":>11}{"ratio":>8}'
          f'{"yp_exact":>11}{"yp_Airy":>10}')
    for x in (25.0, 50.0, 100.0, 150.0, 200.0, 400.0):
        le, ype = lam_exact(x, D)[:2]
        print(f'{x:7.1f}{le:12.5f}{lam_airy(x, D):11.5f}'
              f'{le/lam_airy(x, D):8.3f}{ype:11.4f}{yp_airy(x, D):10.4f}')

    print('\ngrowth rate of the near-wall linear ramp, standard SA vs SA-AI')
    print(f'{"x":>7}{"lam_SA":>11}{"yp_SA":>9}{"lam_SAAI":>12}{"yp_SAAI":>10}')
    for x in (30., 50., 100., 200., 400., 1000., 2000., 3000.):
        ls, ys = eig_lambda(x, 'sa')
        la, ya = eig_lambda(x, 'ai')
        print(f'{x:7.0f}{ls:11.5f}{ys:9.3f}{la:12.5f}{ya:10.3f}')

    print('\nindependent march vs the eigenvalue integral')
    print(f'{"x0":>8}{"x1":>8}{"march":>10}{"int lam dx":>12}{"peak y":>9}')
    for (x0, x1) in [(102., 148.8), (200., 400.)]:
        lg, yi, c = march(x0, x1, D)
        print(f'{x0:8.1f}{x1:8.1f}{lg:10.3f}{wkb(x0, x1, D=D):12.3f}'
              f'{yi[int(np.argmax(c))]:9.4f}')
