"""Does baseline SA sustain turbulence at the bubble-reattachment
Re_theta, or relax toward chi ~ c_v1? (author question, 2026-07-24)

Pure-SA parallel march (sigma_P=1, no amplification; full term set incl.
c_b2) on a frozen Reichardt law-of-the-wall profile, nuHat initialized
at the SA log-layer solution (kappa*u_tau*y, outer-capped at
0.08*u_tau*delta), fetch 2000 delta. Result: SUSTAINED at every
Re_tau in {100,200,400,1000} (Re_theta 176-1887) with equilibrium
chi_max ~ 0.09*Re_tau -- i.e., at reattachment Re_theta ~ 180-350 the
attractor exists but lives at chi ~ 9-18, just above c_v1. The
Delta-x(chi:1->30) handover metric is therefore mis-scaled at low Re.

Run: python3 sa_sustain.py
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

CB1, CV1, KAP, CB2, SIG = 0.1355, 7.1, 0.41, 0.622, 2.0/3.0
CW1 = CB1/KAP**2 + (1 + CB2)/SIG
CW2, CW3 = 0.3, 2.0


def sustain(Re_tau, nx=4000, ny=600, fetch_delta=2000.0):
    yp = np.linspace(0.2, 1.2*Re_tau, ny)
    dy = yp[1] - yp[0]
    k, C = KAP, 7.8
    up = (1/k)*np.log(1 + k*yp) + C*(1 - np.exp(-yp/11) - (yp/11)*np.exp(-yp/3))
    ue = np.interp(Re_tau, yp, up)
    u = np.minimum(up, ue)
    dudy = np.gradient(u, yp)
    om = np.abs(dudy) + 1e-30
    Re_theta = np.trapezoid((u/ue)*(1 - u/ue), yp)*ue
    nu = np.minimum(KAP*yp, 0.08*Re_tau)
    dx = fetch_delta*Re_tau/nx
    hist = [nu.max()]
    for i in range(nx):
        chi = nu
        fv1 = chi**3/(chi**3 + CV1**3)
        fv2 = 1 - chi/(1 + chi*fv1)
        St = np.maximum(om + chi*fv2/(KAP**2*yp**2), 0.3*om)
        r = np.clip(chi/(St*KAP**2*yp**2 + 1e-30), 0, 10)
        gw = r + CW2*(r**6 - r)
        fw = gw*((1 + CW3**6)/(gw**6 + CW3**6))**(1/6.)
        Dcoef = CW1*fw*chi/yp**2
        Pcoef = CB1*St
        knode = (1.0 + chi)/SIG/dy**2
        kface = 0.5*(knode[1:] + knode[:-1])
        di = np.zeros(ny)
        di[:-1] += kface
        di[1:] += kface
        main = u/dx + di + Dcoef
        dnudy = np.gradient(nu, dy)
        rhs = u/dx*nu + Pcoef*nu + (CB2/SIG)*dnudy**2
        A = sp.diags([-kface, main, -kface], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        if i % (nx//8) == 0:
            hist.append(nu.max())
    verdict = 'SUSTAINED' if nu.max() > CV1 else 'COLLAPSED toward/below c_v1'
    print(f"Re_tau={Re_tau:.0f} (Re_theta~{Re_theta:.0f}): "
          f"trajectory {['%.1f' % h for h in hist]}  final {nu.max():.2f}  ({verdict})")


if __name__ == '__main__':
    for Rt in (100.0, 200.0, 400.0, 1000.0):
        sustain(Rt)
