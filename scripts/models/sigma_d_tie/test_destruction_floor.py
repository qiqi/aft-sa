"""sigma-d-tie Test 2: laminar-instrument envelope shift under the destruction floor.

Adds D = sigma_D(chi) * c_w1 * f_w(r) * (chi/y)^2 to the eq:transport march,
with sigma_D = 1 - R*(1 - sigma_P(chi)), R = c_b1/(kappa^2 c_w1), and the
physical amplitude chi = chi_inf * nuhat (seed chi_inf = c_v1 e^-9, the Mack
N_crit = 9 level used by every airfoil case). Compares the envelope measures
(Rt1, Rt9, mean and late secants) with and without the destruction floor for
the two anchor profiles and the K_r fit point.
"""
import sys
sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai/paper/repro/analytic')
import _saai
from _saai import C_NU_AI, SIGMA_SA, K_R
from lib.spalart_allmaras import CB1 as cb1, SIGMA as sig_sa, CB2 as cb2, \
    KAPPA as kap, CW1 as cw1, CW2 as cw2, CW3 as cw3, CV1 as cv1, CV2 as cv2
cv3 = 0.9  # 2012 S-tilde limiter
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0
from lib.aft_sources import compute_aft_amplification_rate, compute_q4_gate

R_TIE = cb1/(kap**2*cw1)
CHI_INF = cv1*np.exp(-9.0)
TAU_P = 4.0

fv1 = lambda c: c**3/(c**3 + cv1**3)
fv2 = lambda c: 1.0 - c/(1.0 + c*fv1(c))

def fw(r):
    r = np.minimum(r, 10.0)
    g = r + cw2*(r**6 - r)
    return g*((1 + cw3**6)/(g**6 + cw3**6))**(1.0/6.0)

def march(fs, x_max, nx=800, ny=600, y_top=None, seed=1.0, beta=None,
          k_r=K_R, destruction=False):
    m = None if beta is None else beta/(2.0 - beta)
    if y_top is None:
        eta99 = np.interp(0.99, fs.u, fs.eta)
        y_top = 8.0*eta99*np.sqrt(x_max/fs.inviscid_at(x_max))
    dy = y_top/ny; yc = (np.arange(ny) + 0.5)*dy; dx = x_max/nx
    nu = np.ones(ny)*seed; N = [0.0]; xs = [0.0]
    k = (C_NU_AI/SIGMA_SA)/dy**2
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, np.arange(ny + 1)*dy, cellCentered=True)
        u = np.maximum(u, 1e-12)
        vp = np.clip(v, 0, None)/dy; vm = np.clip(-v, 0, None)/dy
        di = vp + vm + 2*k; lo = -(vp[1:] + k); up = -(vm[:-1] + k)
        di[0] += k; di[-1] -= k
        lam = 0.0 if m is None else m*yc**2*fs.inviscid_at(x)**2/(x*u)
        rate = np.asarray(compute_aft_amplification_rate(
            yc**2*np.abs(dudy), 2*(dudy*yc)**2/(u**2 + (dudy*yc)**2),
            lambda_p=lam, fpg_rate_slope=k_r))
        q4 = compute_q4_gate(np.gradient(dudy, yc), np.abs(dudy), u, yc)
        b = rate*q4*np.abs(dudy)
        rhs = u/dx*nu + b*nu
        main = u/dx + di
        if destruction:
            chi = CHI_INF*np.clip(nu, 0.0, 1e100)
            # 2012-limited S-tilde (as in the solver): Sp = |omega|, Sbar = chi*fv2/(k^2 y^2)
            Sp = np.abs(dudy)
            Sbar = chi*fv2(chi)/(kap**2*yc**2)
            lim = Sp + Sp*(cv2**2*Sp + cv3*Sbar)/((cv3 - 2*cv2)*Sp - Sbar)
            St = np.maximum(np.where(Sbar >= -cv2*Sp, Sp + Sbar, lim), 1e-12)
            r = chi/(St*kap**2*yc**2)
            sigP = np.where(chi > 1.0, 1.0 - np.exp(-(chi - 1.0)/TAU_P), 0.0)
            sigD = 1.0 - R_TIE*(1.0 - sigP)
            # implicit (linearized) quadratic sink: D = coeff*nu, coeff = sigD*cw1*fw*chi_inf*nu_old/y^2
            Dcoef = sigD*cw1*fw(r)*CHI_INF*np.clip(nu, 0.0, 1e100)/yc**2
            main = main + Dcoef
            if False:
                P = b*np.clip(nu, 1e-300, None)
                j = int(np.argmax(np.clip(nu, 0, None)))
                dp = Dcoef[j]*nu[j]/max(P[j], 1e-300)
                print(f"    x={x:.3g} N={np.log(max(nu.max(),1e-300)):.1f} chi_pk={chi[j]:.2e} "
                      f"D/P@peak={dp:.2e} max(D/P)={np.max(Dcoef*nu/np.maximum(P,1e-300)):.2e}", flush=True)
        rhs[-1] += vm[-1]*seed
        A = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = np.clip(spla.spsolve(A, rhs), -1e100, 1e100)
        xs.append((i + 1)*dx); N.append(float(np.log(max(nu.max()/seed, 1e-300))))
    return np.array(xs), np.array(N)

def profile_ints(fs):
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    I_ds = np.trapezoid(1 - fs.u, fs.eta)
    return I_th, I_ds/I_th

def measures(beta, destruction):
    fs = FalknerSkanWedge(beta); I_th, H = profile_ints(fs)
    bb = beta
    x_max = 4e6 if beta == 0.0 else (3e5 if beta > 0 else 1.2e6)
    for _ in range(12):
        xs, N = march(fs, x_max, beta=bb, destruction=destruction)
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.15; continue
        if N[-1] > 10.5:
            x_max = 1.1*float(np.interp(10.5, N, xs))
            xs, N = march(fs, x_max, beta=bb, destruction=destruction); break
        x_max *= 3.0
    Ue = fs.inviscid_at(np.maximum(xs, 1e-12)); Rt = I_th*np.sqrt(xs*Ue)
    Rt1 = float(np.interp(1.0, N, Rt)); Rt5 = float(np.interp(5.0, N, Rt))
    Rt9 = float(np.interp(9.0, N, Rt))
    return H, Rt1, Rt9, 8.0/(Rt9 - Rt1), 4.0/(Rt9 - Rt5)

print(f"R_TIE = {R_TIE:.4f}  floor = {1-R_TIE:.4f}  chi_inf = {CHI_INF:.3e}", flush=True)
for beta in [0.0, -0.1988, 0.35]:
    H, Rt1a, Rt9a, ma, la = measures(beta, destruction=False)
    _, Rt1b, Rt9b, mb, lb = measures(beta, destruction=True)
    print(f"beta={beta:+.4f} H={H:.3f}", flush=True)
    print(f"  off : Rt1={Rt1a:7.1f} Rt9={Rt9a:7.1f} mean={ma:.5e} late={la:.5e}")
    print(f"  on  : Rt1={Rt1b:7.1f} Rt9={Rt9b:7.1f} mean={mb:.5e} late={lb:.5e}")
    print(f"  shift: Rt1 {100*(Rt1b/Rt1a-1):+.3f}%  Rt9 {100*(Rt9b/Rt9a-1):+.3f}%  "
          f"mean {100*(mb/ma-1):+.3f}%  late {100*(lb/la-1):+.3f}%", flush=True)
