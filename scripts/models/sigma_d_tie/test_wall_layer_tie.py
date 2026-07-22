"""sigma-d-tie Test 1: wall-layer exactness of the pointwise linearity tie.

sigma_D = 1 - R*(1 - sigma_P), R = c_b1/(kappa^2 c_w1)  (~ 0.751 + 0.249 sigma_P).
On the constant-stress wall layer the SA-AI solution must equal standard SA's
linear nuHat = kappa*y+ to machine precision, with NO tau_D.
Result (2026-07-15): max relative deviation 3e-16 for both; dB = 0.0 exactly.
"""
import sys
sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai/paper/repro/analytic')
import _saai
from _saai import C_NU_AI as CNU, TAU
from lib.spalart_allmaras import CB1 as cb1, SIGMA as sig, CB2 as cb2, \
    KAPPA as kap, CW1 as cw1, CW2 as cw2, CW3 as cw3, CV1 as cv1, CV2 as cv2
import numpy as np
from scipy.integrate import solve_bvp, cumulative_trapezoid
cv3 = 0.9
R_TIE = cb1/(kap**2*cw1)

fv1 = lambda c: c**3/(c**3 + cv1**3)
fv2 = lambda c: 1.0 - c/(1.0 + c*fv1(c))
sigt = lambda N, tau: np.maximum(1.0 - np.exp(-(np.maximum(N - 1.0, 0.0))/tau), 0.0)

def Stilde(N, eta):
    N = np.maximum(N, 1e-9)
    nut = N*fv1(N); Sp = 1.0/(1.0 + nut)
    Sbar = N*fv2(N)/(kap**2*eta**2)
    lim = Sp + Sp*(cv2**2*Sp + cv3*Sbar)/((cv3 - 2*cv2)*Sp - Sbar)
    return np.maximum(np.where(Sbar >= -cv2*Sp, Sp + Sbar, lim), 1e-8)

def fw(N, eta, St):
    r = np.minimum(N/(St*kap**2*eta**2), 10.0); g = r + cw2*(r**6 - r)
    return g*((1 + cw3**6)/(g**6 + cw3**6))**(1.0/6.0)

E0, EMAX = 0.3, 400.0
eta = np.linspace(E0, EMAX, 700)
bc = lambda ya, yb: np.array([ya[0] - kap*E0, yb[0] - kap*EMAX])

def rhs(e, y, lam):
    N, Np = y; N = np.maximum(N, 1e-9)
    St = Stilde(N, e)
    P = cb1*St*N; D = cw1*fw(N, e, St)*(N/e)**2
    sP = 1.0 - lam*(1.0 - sigt(N, TAU))
    sD = 1.0 - lam*R_TIE*(1.0 - sigt(N, TAU))
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
    return float(np.mean(up[m] - np.log(yp[m])/kap)), N, nut

if __name__ == '__main__':
    print(f"tie: sigma_D = 1 - {R_TIE:.4f}*(1 - sigma_P); floor = {1-R_TIE:.4f}")
    yp = np.logspace(np.log10(E0), np.log10(EMAX), 900)
    B0, N0, nut0 = intercept(solve(0.0), yp)
    B1, N1, nut1 = intercept(solve(1.0), yp)
    lin = kap*yp
    print(f"standard SA : B = {B0:.5f}, max|N/kap*y+ - 1| = {np.max(np.abs(N0-lin)/lin):.2e}")
    print(f"SA-AI tied  : B = {B1:.5f}, max|N/kap*y+ - 1| = {np.max(np.abs(N1-lin)/lin):.2e}")
    print(f"dB = {B1-B0:+.2e}   max|dnut/(nut+1)| = {np.max(np.abs(nut1-nut0)/(nut0+1)):.2e}")
    assert np.max(np.abs(N1-lin)/lin) < 1e-12 and abs(B1-B0) < 1e-12
    print("EXACT: linear nuHat recovered, tau_D eliminated")
