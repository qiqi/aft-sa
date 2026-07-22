"""fig:walllayer -> paper/figs/wall_layer.pdf.

Constant-stress SA vs SA-AI wall layer (chi=kappa*y+): eddy-viscosity, its
normalized difference, and the log-law shift dB. SA constants imported from
src.spalart_allmaras; c_nu,ai / tau / tau_D imported canonical (_saai)."""
import _saai
from _saai import C_NU_AI as CNU, TAU, TAU_D
from src.physics.spalart_allmaras import CB1 as cb1, SIGMA as sig, CB2 as cb2, \
    KAPPA as kap, CW1 as cw1, CW2 as cw2, CW3 as cw3, CV1 as cv1, CV2 as cv2
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
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

def rhs(e, y, lam, tauD):
    N, Np = y; N = np.maximum(N, 1e-9)
    St, Sp = Stilde(N, e)
    P = cb1*St*N; D = cw1*fw(N, e, St)*(N/e)**2
    sP = 1.0 - lam*(1.0 - sigt(N, TAU)); sD = 1.0 - lam*(1.0 - sigt(N, tauD))
    dc = (1.0 + N) - lam*(1.0 - CNU)
    return np.vstack([Np, (-sig*(sP*P - sD*D) - (1 + cb2)*Np**2)/dc])

def solve(lam_target, tauD):
    sol = solve_bvp(lambda e, y: rhs(e, y, 0.0, tauD), bc, eta,
                    np.vstack([kap*eta, kap*np.ones_like(eta)]), max_nodes=300000, tol=1e-9)
    for lam in [0.15, 0.3, 0.45, 0.6, 0.75, 0.85, 0.92, 0.97, lam_target]:
        if lam > lam_target: break
        sol = solve_bvp(lambda e, y: rhs(e, y, lam, tauD), bc, eta,
                        np.vstack([sol.sol(eta)[0], sol.sol(eta)[1]]), max_nodes=500000, tol=1e-7)
    return sol


def main():
    sol0, sol1 = solve(0.0, TAU_D), solve(1.0, TAU_D)
    yp = np.logspace(np.log10(E0), np.log10(EMAX), 900)
    N0, N1 = sol0.sol(yp)[0], sol1.sol(yp)[0]
    nut0, nut1 = N0*fv1(N0), N1*fv1(N1)
    up0 = cumulative_trapezoid(1.0/(1.0 + nut0), yp, initial=0.0) + E0
    up1 = cumulative_trapezoid(1.0/(1.0 + nut1), yp, initial=0.0) + E0
    m = (yp > 80) & (yp < 250)
    B0 = float(np.mean(up0[m] - np.log(yp[m])/kap)); B1 = float(np.mean(up1[m] - np.log(yp[m])/kap))
    NUP = 1.0
    diff_sum = (nut1 - nut0)/(nut0 + NUP)*100.0
    k = int(np.nanargmax(np.abs(diff_sum))); peak, ypk = float(diff_sum[k]), float(yp[k])
    print(f"B0={B0:.3f}  B1(tauD={TAU_D})={B1:.3f}  dB={B1-B0:+.4f}")
    print(f"max |d nu_t+/(nu_t+ + nu+)| = {abs(peak):.2f}% at y+={ypk:.1f}")

    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5})
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(9.2, 3.1))
    PL = dict(fontsize=15, va='top', fontweight='bold')
    axA.loglog(yp, nut0, 'k-', lw=1.7); axA.loglog(yp, nut1, 'k--', lw=1.7)
    axA.loglog(yp, kap*yp, ':', color='0.5', lw=1.4)
    axA.axhline(1.0, color='0.7', ls=(0, (1, 2)), lw=0.9)
    axA.set_xlim(E0, EMAX); axA.set_ylim(1e-6, 5e2)
    axA.set_xlabel(r'$y^+$'); axA.set_ylabel(r'$\nu_t^+$')
    axA.text(0.05, 0.95, '(a)', transform=axA.transAxes, **PL); axA.grid(alpha=0.25, which='both')
    axB.axvspan(3, 30, color='0.85', lw=0); axB.semilogx(yp, diff_sum, 'k-', lw=1.9)
    axB.axhline(0, color='0.6', lw=0.8); axB.set_xlim(E0, EMAX)
    axB.set_xlabel(r'$y^+$'); axB.set_ylabel(r'$\Delta\nu_t^+/(\nu_t^+ + \nu^+)$  (%)')
    axB.text(0.05, 0.95, '(b)', transform=axB.transAxes, **PL); axB.grid(alpha=0.25)
    axC.semilogx(yp, up0, 'k-', lw=1.7); axC.semilogx(yp, up1, 'k--', lw=1.7)
    axC.semilogx(yp, np.log(yp)/kap + B0, ':', color='0.5', lw=1.4)
    axC.set_xlim(E0, EMAX); axC.set_xlabel(r'$y^+$'); axC.set_ylabel(r'$u^+$')
    axC.text(0.05, 0.95, '(c)', transform=axC.transAxes, **PL); axC.grid(alpha=0.25)
    axD = axC.twinx(); axD.semilogx(yp, up1 - up0, color='0.4', ls='-.', lw=1.8)
    axD.set_ylim(-0.05, 0.05); axD.set_ylabel(r'$\Delta u^+$')
    plt.tight_layout(); plt.savefig('figs/wall_layer.pdf'); plt.close()
    print("wrote figs/wall_layer.pdf")


if __name__ == '__main__':
    main()
