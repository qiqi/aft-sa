"""tab:yplus -> printed table. Coarse-mesh (y+) robustness of the SA-AI wall layer.

Solves the constant-stress SA wall-layer ODE by 2nd-order FD on grids whose first
node sits at y+ = 0.25..8, for standard SA and SA-AI, reporting recovered log-law
intercept B and grid-induced |dCf/Cf|. The velocity is reconstructed on the SAME
coarse grid (trapezoidal integration of du+/dy+ = 1/(1+nut+) over the coarse
nodes, second order), so B and dCf include the discretization error of both the
nuHat transport and the momentum integration. Same constants/machinery as fig07."""
import _saai
from _saai import C_NU_AI as CNU, TAU, R_TIE
from lib.spalart_allmaras import CB1 as cb1, SIGMA as sig, CB2 as cb2, \
    KAPPA as kap, CW1 as cw1, CW2 as cw2, CW3 as cw3, CV1 as cv1, CV2 as cv2
import numpy as np
from scipy.optimize import root
from scipy.integrate import cumulative_trapezoid
cv3 = 0.9  # standard SA-2012 S-tilde limiter constant

fv1 = lambda c: c**3/(c**3 + cv1**3)
fv2 = lambda c: 1.0 - c/(1.0 + c*fv1(c))
sigt = lambda N, tau: np.maximum(1.0 - np.exp(-(np.maximum(N - 1.0, 0.0))/tau), 0.0)

def St(N, y):
    N = np.maximum(N, 1e-9); nut = N*fv1(N); Sp = 1.0/(1.0 + nut); Sb = N*fv2(N)/(kap**2*y**2)
    lim = Sp + Sp*(cv2**2*Sp + cv3*Sb)/((cv3 - 2*cv2)*Sp - Sb)
    return np.maximum(np.where(Sb >= -cv2*Sp, Sp + Sb, lim), 1e-8)

def fw(N, y, S):
    r = np.minimum(N/(S*kap**2*y**2), 10.0); g = r + cw2*(r**6 - r)
    return g*((1 + cw3**6)/(g**6 + cw3**6))**(1.0/6.0)

def grid(Y, ye=1000.0, r=1.2):
    ys = [0.0, Y]
    while ys[-1] < ye: ys.append(ys[-1] + (ys[-1] - ys[-2])*r)
    ys = np.array(ys); ys[-1] = ye; return ys

def deriv(y):
    M = len(y); D1 = np.zeros((M, 3)); D2 = np.zeros((M, 3))
    for j in range(1, M-1):
        hm = y[j]-y[j-1]; hp = y[j+1]-y[j]
        D1[j] = [-hp/(hm*(hm+hp)), (hp-hm)/(hm*hp), hm/(hp*(hm+hp))]
        D2[j] = [2/(hm*(hm+hp)), -2/(hm*hp), 2/(hp*(hm+hp))]
    return D1, D2

def solve(Y, aftsa, ye=1000.0, r=1.2):
    y = grid(Y, ye, r); M = len(y); D1, D2 = deriv(y); Ne = kap*ye
    def resid(Nin, lam):
        N = np.empty(M); N[0] = 0.0; N[-1] = Ne; N[1:-1] = Nin; N = np.maximum(N, 1e-9)
        R = np.zeros(M-2)
        for jj, j in enumerate(range(1, M-1)):
            S = St(N[j], y[j])
            Np = D1[j, 0]*N[j-1] + D1[j, 1]*N[j] + D1[j, 2]*N[j+1]
            Npp = D2[j, 0]*N[j-1] + D2[j, 1]*N[j] + D2[j, 2]*N[j+1]
            P = cb1*S*N[j]; Dd = cw1*fw(N[j], y[j], S)*(N[j]/y[j])**2
            sP = 1.0 - lam*(1.0 - sigt(N[j], TAU))
            sD = 1.0 - lam*R_TIE*(1.0 - sigt(N[j], TAU))   # the linearity tie
            dc = (1.0 + N[j]) - lam*(1.0 - CNU)
            R[jj] = dc*Npp + sig*(sP*P - sD*Dd) + (1 + cb2)*Np**2
        return R
    sol = kap*y[1:-1]
    for lam in ([0.0] if not aftsa else [0.0, 0.3, 0.6, 0.85, 1.0]):
        sol = root(lambda z: resid(z, lam), sol, method='hybr', tol=1e-11).x
    N = np.empty(M); N[0] = 0; N[-1] = Ne; N[1:-1] = sol
    nut = N*fv1(N); up = cumulative_trapezoid(1.0/(1.0 + nut), y, initial=0.0)
    m = (y > 50) & (y < 300); return float(np.mean(up[m] - np.log(y[m])/kap)), M


def main():
    Ys = [0.25, 0.5, 1, 2, 4, 8]; BSA = []; BAF = []; NN = []
    for Y in Ys:
        b, M = solve(Y, False); BSA.append(b); NN.append(M); BAF.append(solve(Y, True)[0])
    BSA = np.array(BSA); BAF = np.array(BAF); ue = np.log(1000)/kap + BSA[0]
    print(f"{'y+_1':>6}{'nodes':>7}{'B_SA':>8}{'dCf_SA':>8}{'B_SAAI':>9}{'dCf_SAAI':>10}")
    for i, Y in enumerate(Ys):
        dsa = abs(-2*(BSA[i]-BSA[0])/ue*100); daf = abs(-2*(BAF[i]-BAF[0])/ue*100)
        r_sa = 'ref' if i == 0 else f'{dsa:.1f}%'; r_af = 'ref' if i == 0 else f'{daf:.1f}%'
        print(f"{Y:>6}{NN[i]:>7}{BSA[i]:>8.2f}{r_sa:>8}{BAF[i]:>9.2f}{r_af:>10}")


if __name__ == '__main__':
    main()
