"""CPU favorable-PG test for the sphere kernel (P-recast).

Marches the disturbance transport (eq:transport) on Falkner-Skan wedges with the
NEW sphere kernel written on the definite product P = Shat*g:
    X,Y,Z = (u, d u', 1/2 d^2 u'')/R           (unit-sphere indicators, d=y)
    Shat  = Y/sqrt(X^2+Y^2)                     (shear fraction = sin longitude; odd)
    g     = Yn - Xn - Zn                         (Rayleigh coord; odd)
    P     = Shat*g                               (EVEN => RP^2-definite)
    a     = a_max * min(1, <P>+ + c_sqrt*sqrt(<P>+))
    onset = ramp(Re_Omega / Re_Omega_crit(P)),  Re_Omega = d^2|u'|/nu  (=yc^2|u'| in BL units)
    source b = a*onset*|u'|

Question: does P<0 on favorable wedges (beta>0) hold them laminar by itself, so
the old lambda_p cliff/f_lambda favorable treatment is unnecessary?
"""
import _saai
from _saai import C_NU_AI, SIGMA_SA
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0

M_POW = 1.0
G_POW = 0.5
A_MAX   = 0.19          # Michalke free-shear eigenvalue (paper's "at face value")
C_SQRT  = 0.03
REOM_C  = 88.0
REOM_Q  = 0.60
REOM_FLOOR = 65.0
RAMP_W  = 0.35          # onset ramp half-width in units of the ratio

def reom_crit(P):
    Pp = np.maximum(P, 1e-6)
    return np.maximum(REOM_FLOOR, REOM_C*Pp**(-REOM_Q))

def sphere_source(u, dudy, yc):
    """return (b_over_nu coefficient) = a*onset*|u'|, plus diagnostics P,g."""
    d2u = np.gradient(dudy, yc)
    X = u; Y = yc*dudy; Z = 0.5*yc**2*d2u
    R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
    Xn, Yn, Zn = X/R, Y/R, Z/R
    Shat = Y/np.sqrt(X*X + Y*Y + 1e-30)
    g = Yn - Xn - Zn
    P = Shat*g
    # rate = a_max * Shat * G(g), G(g)=<g>+^p an ODD function of g (product with the
    # odd Shat is EVEN => RP^2-definite).  p shapes the rise with inflection content;
    # ceiling a_max reached at the shear pole (Shat=1,g=1).
    gp = np.clip(g, 0.0, None)
    a = A_MAX*np.minimum(1.0, Shat*gp**G_POW)
    ReOm = yc**2*np.abs(dudy)
    ratio = ReOm/reom_crit(P)
    onset = 0.5*(1.0 + np.tanh((ratio - 1.0)/RAMP_W))
    return a*onset*np.abs(dudy), P, g

def march(fs, x_max, nx=500, ny=500, y_top=None, seed=1.0):
    if y_top is None:
        eta99 = np.interp(0.99, np.maximum.accumulate(fs.u), fs.eta)
        y_top = 8.0*eta99*np.sqrt(x_max/fs.inviscid_at(x_max))
    dy = y_top/ny; yc = (np.arange(ny) + 0.5)*dy; dx = x_max/nx
    nu = np.ones(ny)*seed; N = [0.0]; xs = [0.0]; Pmax = -9.0
    k = (C_NU_AI/SIGMA_SA)/dy**2
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, np.arange(ny + 1)*dy, cellCentered=True)
        u = np.maximum(u, 1e-12)
        vp = np.clip(v, 0, None)/dy; vm = np.clip(-v, 0, None)/dy
        di = vp + vm + 2*k; lo = -(vp[1:] + k); up = -(vm[:-1] + k)
        di[0] += k; di[-1] -= k
        b, P, g = sphere_source(u, dudy, yc)
        Pmax = max(Pmax, float(np.nanmax(P)))
        main = u/dx + di; rhs = u/dx*nu + b*nu; rhs[-1] += vm[-1]*seed
        A = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        xs.append((i + 1)*dx); N.append(float(np.log(max(nu.max()/seed, 1e-300))))
    return np.array(xs), np.array(N), Pmax

def run(beta):
    fs = FalknerSkanWedge(beta)
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    I_ds = np.trapezoid(1 - fs.u, fs.eta); H = I_ds/I_th
    x_max = 4e6 if beta == 0.0 else (3e5 if beta > 0 else 1.2e6)
    xs, N, Pmax = march(fs, x_max)
    Ue = fs.inviscid_at(np.maximum(xs, 1e-12)); Rt = I_th*np.sqrt(xs*Ue)
    def at_N(n): return float(np.interp(n, N, Rt)) if N[-1] >= n else float('nan')
    Rt1, Rt5, Rt9 = at_N(1.0), at_N(5.0), at_N(9.0)
    drela = float(dN_dRe_theta(H))
    s_mean = 8.0/(Rt9 - Rt1) if Rt9 > Rt1 else float('nan')
    s_late = 4.0/(Rt9 - Rt5) if Rt9 > Rt5 else float('nan')
    tag = 'FAV' if beta > 0 else ('ADV' if beta < 0 else 'BLAS')
    print(f"beta={beta:+.4f} [{tag}] H={H:.3f}  Pmax={Pmax:+.4f}  N_end={N[-1]:5.1f}  "
          f"Re_th(N1)={Rt1:7.0f} Re_th(N9)={Rt9:8.0f}  "
          f"s_mean={s_mean:.3e}({s_mean/drela:4.2f}x) s_late={s_late:.3e}({s_late/drela:4.2f}x) "
          f"Drela={drela:.2e}", flush=True)
    return H, Pmax, N[-1], Rt1, Rt5, Rt9, s_mean, s_late, drela

if __name__ == '__main__':
    print(f"a_max={A_MAX} c_sqrt={C_SQRT}  reom_crit={REOM_C}*P^-{REOM_Q} floor {REOM_FLOOR}")
    print("--- adverse / Blasius ---")
    for b in [-0.1988, -0.10, 0.0]:
        run(b)
    print("--- favorable (test: should stay laminar if P<0 holds) ---")
    for b in [0.05, 0.10, 0.20, 0.35, 0.55, 1.0]:
        run(b)
