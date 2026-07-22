"""Verification (paper Sec. III.E): wall-layer exactness of the sigma_D tie.

Constant-stress SA vs SA-AI wall layer (chi=kappa*y+) under
sigma_D = 1 - R_TIE*(1 - sigma_P): both solutions coincide with the linear
nuHat = kappa*y+ (ASSERTED: relative deviation < 1e-10, |dB| < 1e-10 --
the residual is roundoff). No figure: the paper states the identity and
this script certifies it. SA constants imported from src.spalart_allmaras;
c_nu,ai / tau / R_TIE imported canonical (_saai)."""
import _saai
from _saai import C_NU_AI as CNU, TAU, R_TIE
from lib.spalart_allmaras import CB1 as cb1, SIGMA as sig, CB2 as cb2, \
    KAPPA as kap, CW1 as cw1, CW2 as cw2, CW3 as cw3, CV1 as cv1, CV2 as cv2
import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
cv3 = 0.9  # standard SA-2012 S-tilde limiter constant (not an SA-AI kernel constant)

fv1 = lambda c: c**3 / (c**3 + cv1**3)
fv2 = lambda c: 1.0 - c / (1.0 + c*fv1(c))
sigt = lambda N, tau: np.maximum(1.0 - np.exp(-(np.maximum(N - 1.0, 0.0))/tau), 0.0)

def Stilde(N, eta):
    N = np.maximum(N, 1e-9)
    nut = N*fv1(N); Sp = 1.0/(1.0 + nut)
    Sbar = N*fv2(N)/(kap**2*eta**2)
    lim = Sp + Sp*(cv2**2*Sp + cv3*Sbar)/((cv3 - 2*cv2)*Sp - Sbar)
    return np.maximum(np.where(Sbar >= -cv2*Sp, Sp + Sbar, lim), 1e-8), Sp

def fw(N, eta, St):
    r = np.minimum(N/(St*kap**2*eta**2), 10.0); g = r + cw2*(r**6 - r)
    return g*((1 + cw3**6)/(g**6 + cw3**6))**(1.0/6.0)

E0, EMAX = 0.3, 400.0
eta = np.linspace(E0, EMAX, 700)
bc = lambda ya, yb: np.array([ya[0] - kap*E0, yb[0] - kap*EMAX])

def rhs(e, y, lam):
    N, Np = y; N = np.maximum(N, 1e-9)
    St, Sp = Stilde(N, e)
    P = cb1*St*N; D = cw1*fw(N, e, St)*(N/e)**2
    sP = 1.0 - lam*(1.0 - sigt(N, TAU))
    sD = 1.0 - lam*R_TIE*(1.0 - sigt(N, TAU))   # the linearity tie
    dc = (1.0 + N) - lam*(1.0 - CNU)
    return np.vstack([Np, (-sig*(sP*P - sD*D) - (1 + cb2)*Np**2)/dc])

def solve(lam_target):
    sol = solve_bvp(lambda e, y: rhs(e, y, 0.0), bc, eta,
                    np.vstack([kap*eta, kap*np.ones_like(eta)]), max_nodes=300000, tol=1e-9)
    for lam in [0.15, 0.3, 0.45, 0.6, 0.75, 0.85, 0.92, 0.97, lam_target]:
        if lam > lam_target: break
        sol = solve_bvp(lambda e, y: rhs(e, y, lam), bc, eta,
                        np.vstack([sol.sol(eta)[0], sol.sol(eta)[1]]), max_nodes=500000, tol=1e-9)
    return sol


def intercept(sol, yp):
    N = sol.sol(yp)[0]
    nut = N*fv1(N)
    up = cumulative_trapezoid(1.0/(1.0 + nut), yp, initial=0.0) + E0
    m = (yp > 80) & (yp < 250)
    return float(np.mean(up[m] - np.log(yp[m])/kap)), nut, up


def main():
    sol0, sol1 = solve(0.0), solve(1.0)
    yp = np.logspace(np.log10(E0), np.log10(EMAX), 900)
    B0, nut0, up0 = intercept(sol0, yp)
    B1, nut1, up1 = intercept(sol1, yp)
    # EXACTNESS of the tie (paper Sec. III.E): the linear profile solves the
    # gated equation at every height, so both solutions ARE kappa*y+ and the
    # footprint is identically zero -- asserted, not calibrated.
    lin = kap*yp
    dev0 = float(np.max(np.abs(sol0.sol(yp)[0] - lin)/lin))
    dev1 = float(np.max(np.abs(sol1.sol(yp)[0] - lin)/lin))
    print(f"linearity: max|N/kap*y+ - 1| = {dev0:.1e} (SA), {dev1:.1e} (SA-AI tie)")
    assert dev1 < 1e-10 and abs(B1 - B0) < 1e-10, \
        f"tie exactness violated: dev={dev1:.1e}, dB={B1-B0:.1e}"
    NUP = 1.0
    diff_sum = (nut1 - nut0)/(nut0 + NUP)*100.0
    k = int(np.nanargmax(np.abs(diff_sum))); peak, ypk = float(diff_sum[k]), float(yp[k])
    print(f"B0={B0:.3f}  B1(tie)={B1:.3f}  dB={B1-B0:+.1e}")
    print(f"max |d nu_t+/(nu_t+ + nu+)| = {abs(peak):.2f}% at y+={ypk:.1f}")

if __name__ == '__main__':
    main()
