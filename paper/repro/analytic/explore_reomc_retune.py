"""Sec. II.D stage (c): re-anchor the onset threshold Re_Omega_crit for a
given laminar-diffusion reduction c_nu,ai.

With c_nu,ai chosen (stage b, explore_cnuai_sweep.py), the marched N=1
crossings must land on the Drela-Giles N=1 stations
    Re_theta(N=1) = Re_theta_c(H) + 1/(dN/dRe_theta)(H)
across the attached Falkner-Skan family. The current solver canon
softmin_2(1670, 112, 1.28) was anchored at Blasius WITH c=1/12 and carries a
family tilt (adverse ~+13%, favorable -15/-20% at c=1/12; see the sweep JSON).
Here we fit all three constants (CEIL, A, B) in log space, n=2 and ramp
w=0.35 held, minimizing sum of squared log errors of the marched N=1
crossing over beta in [-0.18, +0.10] (9 wedges).

Run:  python3 explore_reomc_retune.py [c_nu ...]   (default: 1/12 1/24)
Out:  figs_explore/results_reomc_retune.json + stdout tables.
"""
import json
import sys
import _saai  # noqa: F401
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0

SIGMA_SA = 2.0/3.0
A_MAX = 0.19
RAMP_W = 0.35
REOM_N = 2.0
BETAS = [-0.18, -0.15, -0.12, -0.09, -0.06, -0.03, 0.0, 0.05, 0.10]


def sphere_rate(u, dudy, yc, ceil, a_, b_):
    d2u = np.gradient(dudy, yc)
    X = u; Y = yc*dudy; Z = 0.5*yc**2*d2u
    R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
    Shat = Y/np.sqrt(X*X + Y*Y + 1e-30)
    g = (Y - X - Z)/R
    P = Shat*g
    a = A_MAX*np.minimum(1.0, np.clip(P, 0.0, None))
    ReOm = yc**2*np.abs(dudy)
    _pw = a_ + b_*np.maximum(P, 1e-6)**(-2.0)
    reomc = (ceil**(-REOM_N) + _pw**(-REOM_N))**(-1.0/REOM_N)
    onset = 0.5*(1.0 + np.tanh((ReOm/reomc - 1.0)/RAMP_W))
    return a*onset


def march_to_N(fs, x_max, c_nu, consts, nx=600, ny=500, seed=1.0):
    eta99 = np.interp(0.99, np.maximum.accumulate(fs.u), fs.eta)
    y_top = 8.0*eta99*np.sqrt(x_max/fs.inviscid_at(x_max))
    dy = y_top/ny; yc = (np.arange(ny) + 0.5)*dy; dx = x_max/nx
    nu = np.ones(ny)*seed; N = [0.0]; xs = [0.0]
    k = (c_nu/SIGMA_SA)/dy**2
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, np.arange(ny + 1)*dy, cellCentered=True)
        u = np.maximum(u, 1e-12)
        vp = np.clip(v, 0, None)/dy; vm = np.clip(-v, 0, None)/dy
        di = vp + vm + 2*k; lo = -(vp[1:] + k); up = -(vm[:-1] + k)
        di[0] += k; di[-1] -= k
        b = sphere_rate(u, dudy, yc, *consts)*np.abs(dudy)
        main = u/dx + di; rhs = u/dx*nu + b*nu; rhs[-1] += vm[-1]*seed
        A = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        xs.append((i + 1)*dx); N.append(float(np.log(max(nu.max()/seed, 1e-300))))
    return np.array(xs), np.array(N)


class Wedge:
    def __init__(self, beta, c_nu):
        self.fs = FalknerSkanWedge(beta)
        self.I_th = float(np.trapezoid(self.fs.u*(1 - self.fs.u), self.fs.eta))
        H = float(np.trapezoid(1 - self.fs.u, self.fs.eta))/self.I_th
        self.H = H
        d = float(np.asarray(dN_dRe_theta(H)))
        self.N1_drela = float(np.asarray(Re_theta0(H))) + 1.0/d
        # FIXED march domain per wedge (independent of the trial constants),
        # so the objective is smooth in the fit parameters: generous headroom
        # to Re_theta = 4x the Drela N=1 station covers thresholds up to ~3x
        # late even at full molecular diffusion.
        Ue1 = self.fs.inviscid_at(1.0)
        self.x_max = (4.0*self.N1_drela/self.I_th)**2/Ue1
        self.c_nu = c_nu

    def Rt1(self, consts):
        """N=1 crossing Re_theta on the fixed domain (deterministic)."""
        xs, N = march_to_N(self.fs, self.x_max, self.c_nu, consts,
                           nx=800, ny=500)
        if not np.all(np.isfinite(N)) or N[-1] < 1.2:
            return float('nan')
        Ue = self.fs.inviscid_at(np.maximum(xs, 1e-12))
        Rt = self.I_th*np.sqrt(xs*Ue)
        return float(np.interp(1.0, N, Rt))


def fit_for_c(c_nu, x0=(1670.0, 112.0, 1.28)):
    wedges = [Wedge(b, c_nu) for b in BETAS]

    def errs_for(consts):
        errs = []
        for w in wedges:
            r = w.Rt1(consts)
            if not np.isfinite(r):
                return None
            errs.append(np.log(r/w.N1_drela))
        return np.array(errs)

    def objective(logp):
        consts = tuple(np.exp(logp))
        errs = errs_for(consts)
        if errs is None:
            return 1e3
        e = float(np.sum(errs**2))
        print(f"  eval ({consts[0]:7.0f},{consts[1]:6.1f},{consts[2]:6.3f}) "
              f"rms={np.sqrt(e/len(errs)):.4f}", flush=True)
        return e

    # Stage 1: 1-D scale fit (robust) -- multiply the whole start point by s.
    x0 = np.asarray(x0, float)
    best_s, best_e = 1.0, np.inf
    for s in np.geomspace(0.4, 3.0, 13):
        errs = errs_for(tuple(s*x0))
        if errs is None:
            continue
        e = float(np.sum(errs**2))
        print(f"  scale s={s:5.2f}  rms={np.sqrt(e/len(BETAS)):.4f}", flush=True)
        if e < best_e:
            best_s, best_e = s, e
    # Stage 2: Nelder-Mead on log(C, A, B) from the scaled start.
    res = minimize(objective, np.log(best_s*x0), method='Nelder-Mead',
                   options=dict(xatol=2e-3, fatol=1e-6, maxfev=150))
    consts = tuple(np.exp(res.x))
    table = []
    for b, w in zip(BETAS, wedges):
        r = w.Rt1(consts)
        table.append(dict(beta=b, H=w.H, N1_drela=w.N1_drela, Rt1=r,
                          ratio=r/w.N1_drela))
    return consts, float(np.sqrt(res.fun/len(wedges))), table


def main():
    cvals = [float(eval(a)) for a in sys.argv[1:]] or [1.0/12.0, 1.0/24.0]
    out = {}
    for c in cvals:
        print(f"=== fitting Re_Omega_crit constants for c_nu,ai = {c:.5f} ===",
              flush=True)
        consts, rms, table = fit_for_c(c)
        print(f"c={c:.5f}: (CEIL, A, B) = ({consts[0]:.0f}, {consts[1]:.1f}, "
              f"{consts[2]:.3f}), rms log-error {rms:.4f}")
        for t in table:
            print(f"  beta={t['beta']:+.2f} H={t['H']:.3f} "
                  f"N1_drela={t['N1_drela']:6.0f} Rt1={t['Rt1']:6.0f} "
                  f"ratio={t['ratio']:.3f}")
        out[f"{c:.6f}"] = dict(consts=list(consts), rms=rms, table=table)
    jpath = 'repro/analytic/figs_explore/results_reomc_retune.json'
    json.dump(out, open(jpath, 'w'), indent=1)
    print(f'wrote {jpath}')


if __name__ == '__main__':
    main()
