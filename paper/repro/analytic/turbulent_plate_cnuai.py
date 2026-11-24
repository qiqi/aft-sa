"""Effect of c_nu,ai on a fully turbulent flat-plate boundary layer (Sec. II.D).

Spalart and Allmaras calibrated c_w2 on the flat-plate skin friction at
Re_theta = 1e4 (C_f = 0.00262). The factor c_nu,ai multiplies the molecular
viscosity in the diffusion term of nu~ only,

    (1/sigma) [ d/dy((c nu + nu~) dnu~/dy) + c_b2 (dnu~/dy)^2 ],

which vanishes on the linear wall-layer solution but not at the outer edge of
the boundary layer, where chi passes through one. This script marches the
boundary-layer equations with baseline SA for c = 1 and c = 1/6 and compares
the two at Re_theta = 1e4.

Units: U_e = 1, nu = 1, so x = Re_x and the momentum thickness is Re_theta.
Unknowns (u, v, nu~) on a uniform y grid, central differences, backward Euler
in x. Each station is solved by a fully coupled Newton iteration with an
exact (automatic-differentiation) banded Jacobian. The layer starts laminar
with nu~ seeded inside it and is turbulent by Re_theta of about 500.

Result (h = 40 nu/U_e, and the same to the digits quoted at h = 20):
c = 1 gives C_f = 0.00262 at Re_theta = 1e4. With c = 1/6, C_f is 0.07% higher,
H is 0.02% lower, and the eddy viscosity differs by at most 0.25% of its peak.
The differences scale as 1/Re_theta (C_f +0.38% at 2000, +0.15% at 5000).

Run from paper/:  python3 repro/analytic/turbulent_plate_cnuai.py   (about 5 minutes)
"""
import sys, json, time
import numpy as np
from scipy.linalg import solve_banded
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

CB1, SIG, CB2, KAP, CW2, CW3, CV1 = 0.1355, 2.0/3.0, 0.622, 0.41, 0.3, 2.0, 7.1
CW1 = CB1/KAP**2 + (1.0 + CB2)/SIG


def residual(q, qn, dx, c, h, yy, chi_inf):
    u, v, nt = q[:, 0], q[:, 1], q[:, 2]
    un, ntn = qn[:, 0], qn[:, 2]
    chi = jnp.maximum(nt, 0.0)
    fv1 = chi**3/(chi**3 + CV1**3)
    nue = 1.0 + nt*fv1
    nf = 0.5*(nue[1:] + nue[:-1])
    # momentum, interior
    mom = (u[1:-1]*(u[1:-1] - un[1:-1])/dx + v[1:-1]*(u[2:] - u[:-2])/(2*h)
           - (nf[1:]*(u[2:] - u[1:-1]) - nf[:-1]*(u[1:-1] - u[:-2]))/h**2)
    mom = jnp.concatenate([u[:1], mom, u[-1:] - 1.0])
    # continuity (row j couples j-1 and j), v_0 = 0
    con = v[1:] - v[:-1] + 0.5*h*((u[1:] - un[1:]) + (u[:-1] - un[:-1]))/dx
    con = jnp.concatenate([v[:1], con])
    # SA
    uy = (u[2:] - u[:-2])/(2*h); om = jnp.abs(uy)
    nti, yi, chii, fv1i = chi[1:-1], yy[1:-1], chi[1:-1], fv1[1:-1]   # sources use max(nu~, 0)
    fv2 = 1.0 - chii/(1.0 + chii*fv1i)
    St = jnp.maximum(om + nti*fv2/(KAP**2*yi**2), 0.3*om)
    r = jnp.minimum(nti/jnp.maximum(St*KAP**2*yi**2, 1e-30), 10.0)
    g = r + CW2*(r**6 - r)
    fw = g*((1.0 + CW3**6)/(g**6 + CW3**6))**(1.0/6.0)
    src = CB1*St*nti - CW1*fw*(nti/yi)**2
    Dn = c + chi; Df = 0.5*(Dn[1:] + Dn[:-1])
    dif = ((1.0 + CB2)*(Df[1:]*(nt[2:] - nt[1:-1]) - Df[:-1]*(nt[1:-1] - nt[:-2]))
           - CB2*Dn[1:-1]*(nt[2:] - 2*nt[1:-1] + nt[:-2]))/(SIG*h**2)
    sa = (u[1:-1]*(nt[1:-1] - ntn[1:-1])/dx + v[1:-1]*(nt[2:] - nt[:-2])/(2*h) - src - dif)
    sa = jnp.concatenate([nt[:1], sa, nt[-1:] - chi_inf])
    return jnp.stack([mom, con, sa], axis=1)


@jax.jit
def res_and_jvps(q, qn, dx, c, h, yy, chi_inf, seeds):
    f = lambda z: residual(z, qn, dx, c, h, yy, chi_inf)
    R = f(q)
    J = jax.vmap(lambda s: jax.jvp(f, (q,), (s,))[1])(seeds)     # (9, n, 3)
    return R, J


class Banded:
    """Banded Jacobian (l = u = 5) from 9 colored JVPs; unknown index 3 j + k."""
    def __init__(self, n):
        self.n = n; N = 3*n
        seeds = np.zeros((9, n, 3))
        for jm in range(3):
            for k in range(3):
                seeds[3*jm + k, jm::3, k] = 1.0
        self.seeds = jnp.asarray(seeds)
        jr = np.repeat(np.arange(n), 3)                 # row node
        rows, cols, colr = [], [], []
        I = np.arange(N)
        for col in range(9):
            jm, k = divmod(col, 3)
            jc = jr + ((jm - jr) % 3 + 1) % 3 - 1       # the node in {jr-1, jr, jr+1} with jc % 3 == jm
            ok = (jc >= 0) & (jc < n)
            rows.append(I[ok]); cols.append(3*jc[ok] + k); colr.append(np.full(ok.sum(), col))
        self.rows = np.concatenate(rows); self.cols = np.concatenate(cols); self.colr = np.concatenate(colr)
        self.ab_r = 5 + self.rows - self.cols

    def solve(self, Jv, R):
        J = np.asarray(Jv).reshape(9, -1)
        ab = np.zeros((11, 3*self.n))
        ab[self.ab_r, self.cols] = J[self.colr, self.rows]
        return solve_banded((5, 5), ab, -np.asarray(R).reshape(-1)).reshape(self.n, 3)


def integrals(u, h):
    f = u*(1.0 - u); g = 1.0 - u
    th = h*(f.sum() - 0.5*(f[0] + f[-1])); ds = h*(g.sum() - 0.5*(g[0] + g[-1]))
    cf = 2.0*(-3.0*u[0] + 4.0*u[1] - u[2])/(2.0*h)
    return th, ds, cf


def march(c, h, dx_max, y_max=2.0e5, x0=1.0e5, rth_stop=1.05e4, chi_inf=8.76e-4,
          targets=(2.0e3, 5.0e3, 1.0e4), log=None, growth=1.01):
    y = np.arange(0.0, y_max + 0.5*h, h); n = y.size
    yy = jnp.asarray(np.maximum(y, h))
    d0 = 5.0*np.sqrt(x0)
    u = np.where(y < d0, np.sin(0.5*np.pi*np.minimum(y/d0, 1.0)), 1.0)
    nt = chi_inf + 30.0*np.exp(-(y/(0.7*d0))**4)*np.minimum(y/(0.05*d0), 1.0)
    nt[0] = 0.0
    q = np.stack([u, np.zeros(n), nt], axis=1)
    B = Banded(n)
    x, dx, step = x0, 20.0, 0
    hist, prof, prev = [], {}, None
    t0 = time.time()
    while True:
        qn = q.copy()
        ok = False
        for it in range(30):
            R, J = res_and_jvps(jnp.asarray(q), jnp.asarray(qn), dx, c, h, yy, chi_inf, B.seeds)
            dq = B.solve(J, R)
            err = max(np.abs(dq[:, 0]).max(), (np.abs(dq[:, 2])/(1.0 + np.abs(q[:, 2]))).max())
            lam = 1.0
            if it < 25 and err > 0.5:
                lam = 0.5/err
            q = q + lam*dq
            if err < 1e-10:
                ok = True
                break
        if not ok:
            raise RuntimeError('Newton did not converge at x=%g (err %.2e)' % (x, err))
        x += dx; step += 1
        u, nt = q[:, 0], q[:, 2]
        th, ds, cf = integrals(u, h)
        row = (x, th, ds, cf, it + 1)
        hist.append(row)
        for T in targets:
            if prev is not None and prev[0][1] < T <= th and T not in prof:
                w = (T - prev[0][1])/(th - prev[0][1])
                prof[T] = dict(x=prev[0][0] + w*(x - prev[0][0]), u=prev[1] + w*(u - prev[1]),
                               nt=prev[2] + w*(nt - prev[2]), cf=prev[0][3] + w*(cf - prev[0][3]),
                               ds=prev[0][2] + w*(ds - prev[0][2]))
        prev = (row, u.copy(), nt.copy())
        if log and step % 500 == 0:
            print('%s step %6d x=%.4e Rth=%8.1f H=%.4f Cf=%.6f newton=%d min(nt)=%.2e  %.0fs'
                  % (log, step, x, th, ds/th, cf, it + 1, nt.min(), time.time() - t0), flush=True)
        if th >= rth_stop or x > 3e7:
            break
        dx = min(dx*growth, dx_max)
    return y, np.array(hist), prof


def _run(args):
    c, h, dxm = args
    y, hist, prof = march(c, h, dxm)
    return y, prof


def compare(h=40.0, dx_max=2000.0):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with ProcessPoolExecutor(2, mp_context=mp.get_context('spawn')) as ex:
        (y, pa), (_, pb) = ex.map(_run, [(1.0, h, dx_max), (1.0/6.0, h, dx_max)])
    out = {}
    for T in sorted(pa):
        a, b = pa[T], pb[T]
        fv = lambda n: n*np.maximum(n, 0.0)**3/(np.maximum(n, 0.0)**3 + CV1**3)
        fa, fb = fv(a['nt']), fv(b['nt'])
        out[T] = dict(cf=a['cf'], dcf=100*(b['cf']/a['cf'] - 1), dH=100*(b['ds']/a['ds'] - 1),
                      dnut=100*np.abs(fb - fa).max()/fa.max(), nut=fa.max(),
                      du=np.abs(b['u'] - a['u']).max())
        print('Re_theta = %5d:  C_f(c=1) = %.5f   dC_f = %+.3f%%   dH = %+.3f%%   '
              'max |d nu_t| / max nu_t = %.2f%%   max |du|/U_e = %.1e'
              % (T, a['cf'], out[T]['dcf'], out[T]['dH'], out[T]['dnut'], out[T]['du']))
    return out


if __name__ == '__main__':
    r = compare()[1.0e4]
    assert abs(r['cf']/0.00262 - 1) < 0.005, r['cf']          # SA's calibration point
    assert 0.06 < r['dcf'] < 0.08, r['dcf']
    assert -0.03 < r['dH'] < -0.015, r['dH']
    assert 0.2 < r['dnut'] < 0.3, r['dnut']
    print('Quoted numbers reproduced: C_f +0.07%, eddy viscosity within 0.25% of its peak.')
