"""Coarse-mesh (y+) robustness check for the tau_D-calibrated SA-AI.

Same finite-difference constant-stress solve as paper/regen_yplus_table.py,
extended with a separate destruction-gate width TAU_D. Prints the Table-1-style
rows for standard SA, SA-AI (tau_D = 4, current), and SA-AI (tau_D = TAU_D*):
recovered log-law intercept B and grid-induced |dCf/Cf| vs the finest grid.
Expected: B(SA-AI, tau_D*) == B(SA) to ~1e-3 at every resolution, grid
sensitivity unchanged.
"""
import sys
import numpy as np
from scipy.optimize import root
from scipy.integrate import cumulative_trapezoid

TAU_D_STAR = float(sys.argv[1]) if len(sys.argv) > 1 else 1.36

cb1, sig, cb2, kap = 0.1355, 2.0/3.0, 0.622, 0.41
cw1 = cb1/kap**2 + (1+cb2)/sig; cw2, cw3 = 0.3, 2.0; cv1, cv2, cv3 = 7.1, 0.7, 0.9
CNU, TAU_P = 1.0/12.0, 4.0
fv1 = lambda c: c**3/(c**3+cv1**3)
fv2 = lambda c: 1.0 - c/(1.0+c*fv1(c))
sigt = lambda N, tau: np.maximum(1.0-np.exp(-(np.maximum(N-1.0, 0.0))/tau), 0.0)
def St(N, y):
    N = np.maximum(N, 1e-9); nut = N*fv1(N); Sp = 1.0/(1.0+nut); Sb = N*fv2(N)/(kap**2*y**2)
    lim = Sp + Sp*(cv2**2*Sp+cv3*Sb)/((cv3-2*cv2)*Sp-Sb)
    return np.maximum(np.where(Sb >= -cv2*Sp, Sp+Sb, lim), 1e-8)
def fw(N, y, S):
    r = np.minimum(N/(S*kap**2*y**2), 10.0); g = r+cw2*(r**6-r)
    return g*((1+cw3**6)/(g**6+cw3**6))**(1.0/6.0)
def grid(Y, ye=1000.0, r=1.2):
    ys = [0.0, Y]
    while ys[-1] < ye: ys.append(ys[-1] + (ys[-1]-ys[-2])*r)
    ys = np.array(ys); ys[-1] = ye; return ys
def deriv(y):
    M = len(y); D1 = np.zeros((M, 3)); D2 = np.zeros((M, 3))
    for j in range(1, M-1):
        hm = y[j]-y[j-1]; hp = y[j+1]-y[j]
        D1[j] = [-hp/(hm*(hm+hp)), (hp-hm)/(hm*hp), hm/(hp*(hm+hp))]
        D2[j] = [2/(hm*(hm+hp)), -2/(hm*hp), 2/(hp*(hm+hp))]
    return D1, D2
def solve(Y, aftsa, tauD, ye=1000.0, r=1.2):
    y = grid(Y, ye, r); M = len(y); D1, D2 = deriv(y); Ne = kap*ye
    def resid(Nin, lam):
        N = np.empty(M); N[0] = 0.0; N[-1] = Ne; N[1:-1] = Nin; N = np.maximum(N, 1e-9)
        R = np.zeros(M-2)
        for jj, j in enumerate(range(1, M-1)):
            S = St(N[j], y[j])
            Np = D1[j, 0]*N[j-1]+D1[j, 1]*N[j]+D1[j, 2]*N[j+1]
            Npp = D2[j, 0]*N[j-1]+D2[j, 1]*N[j]+D2[j, 2]*N[j+1]
            P = cb1*S*N[j]; Dd = cw1*fw(N[j], y[j], S)*(N[j]/y[j])**2
            sP = 1.0 - lam*(1.0-sigt(N[j], TAU_P))
            sD = 1.0 - lam*(1.0-sigt(N[j], tauD))
            dc = (1.0+N[j]) - lam*(1.0-CNU)
            R[jj] = dc*Npp + sig*(sP*P-sD*Dd) + (1+cb2)*Np**2
        return R
    sol = kap*y[1:-1]
    for lam in ([0.0] if not aftsa else [0.0, 0.3, 0.6, 0.85, 1.0]):
        sol = root(lambda z: resid(z, lam), sol, method='hybr', tol=1e-11).x
    N = np.empty(M); N[0] = 0; N[-1] = Ne; N[1:-1] = sol
    nut = N*fv1(N); up = cumulative_trapezoid(1.0/(1.0+nut), y, initial=0.0)
    m = (y > 50) & (y < 300); return float(np.mean(up[m]-np.log(y[m])/kap)), M

Ys = [0.25, 0.5, 1, 2, 4, 8]
BSA, B4, BS, NN = [], [], [], []
for Y in Ys:
    b, M = solve(Y, False, 4.0); BSA.append(b); NN.append(M)
    B4.append(solve(Y, True, 4.0)[0])
    BS.append(solve(Y, True, TAU_D_STAR)[0])
BSA, B4, BS = map(np.array, (BSA, B4, BS)); ue = np.log(1000)/kap + BSA[0]
print(f"tau_D* = {TAU_D_STAR}")
print(f"{'y+_1':>6}{'nodes':>7}{'B_SA':>8}{'dCf':>7}{'B_tauD4':>9}{'dCf':>7}{'B_tauD*':>9}{'dCf':>7}{'B*-B_SA':>9}")
for i, Y in enumerate(Ys):
    f = lambda B: 'ref' if i == 0 else f'{abs(-2*(B[i]-B[0])/ue*100):.1f}%'
    print(f"{Y:>6}{NN[i]:>7}{BSA[i]:>8.2f}{f(BSA):>7}{B4[i]:>9.2f}{f(B4):>7}"
          f"{BS[i]:>9.2f}{f(BS):>7}{BS[i]-BSA[i]:>+9.3f}")
