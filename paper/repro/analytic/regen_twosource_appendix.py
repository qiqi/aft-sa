"""Two-source (inviscid + viscous) appendix figures -- ANALYTIC ONLY, no CFD.

Regenerates the four Drela-meeting figures under the UPDATED kernel: the two
branches are now DECOUPLED -- each carries its own rate AND its own onset gate,
and the two complete SOURCES are soft-maxed.  The retired 'vg' form soft-maxed
the rates but soft-MINned the two thresholds into one shared gate, which lets
the inviscid branch's low threshold switch on the viscous rate before the
viscous mechanism is critical (+7.4% on the Blasius anchor).

    P_I      = Om_hat <I_hat>_+                 inflectional (inviscid)
    P_curv   = Om_hat <-Z/R>_+                  curvature (viscous)
    thr_inv  = softmin_n( C , A + B  /P_I^2 )   canon gate, ceiling RETAINED
    thr_visc =             A + B_c/P_curv^2     A_c = A shared
    a*S      = softmax_2( a_max  P_I    S(Re_Om/thr_inv ),
                          a_visc P_curv S(Re_Om/thr_visc) )

Retaining C is what keeps the calibrated family unchanged (beta=+0.20 would
otherwise drop to 0.63x canon).

Outputs (paper/figs/):
    twosource_calibrate.pdf    rate + onset vs H, canon / two-source / Drela
    twosource_onsetgraze.pdf   each branch grazing its OWN family in its OWN coord
    twosource_transport.pdf    marched disturbance transport, 4 wedges
    twosource_sphere.pdf       low-H Falkner-Skan on RP^2

Run from paper/:  python3 repro/analytic/regen_twosource_appendix.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _saai  # noqa: F401  (puts repro/ on the path)
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0
from lib.sphere_kernel import (A_MAX, RAMP_W, REOM_A, REOM_B, REOM_CEIL,
                               REOM_N)

A_VISC = 0.0276          # viscous-branch rate  (a_visc = eps_r a_max)
B_C = 130.0              # curvature-branch onset coefficient (k-carrying)
SQ2 = np.sqrt(2.0)
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figs')


# ----------------------------------------------------------------- kernel --
def coords(u, dudy, d2u, y):
    X, Y, Z = u, y*dudy, 0.5*y*y*d2u
    R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-300
    Ohat = Y/(np.sqrt(X*X + Y*Y) + 1e-300)
    Ihat = (Y - X - Z)/R
    return Ohat*np.clip(Ihat, 0, None), Ohat*np.clip(-Z/R, 0, None)


def thr_inv(P_I):
    b = REOM_A + REOM_B/np.maximum(P_I, 1e-300)**2
    return (b**(-REOM_N) + REOM_CEIL**(-REOM_N))**(-1.0/REOM_N)


def thr_visc(P_c):
    return REOM_A + B_C/np.maximum(P_c, 1e-300)**2


def _S(ReOm, thr):
    return 0.5*(1.0 + np.tanh((ReOm/thr - 1.0)/RAMP_W))


def rate_twosource(u, dudy, yc, nu=1.0):
    """Dimensionless a*S in the figures' normalisation (nu = 1)."""
    d2u = np.gradient(dudy, yc)
    P_I, P_c = coords(u, dudy, d2u, yc)
    ReOm = yc*yc*np.abs(dudy)/nu
    si = A_MAX*P_I*_S(ReOm, thr_inv(P_I))
    sv = A_VISC*P_c*_S(ReOm, thr_visc(P_c))
    return np.sqrt(si*si + sv*sv)


# patch the shared machinery so march/measures use the two-source kernel
import fig04_shapefactor as f4                                    # noqa: E402
_CANON_RATE = f4.sphere_rate
f4.sphere_rate = rate_twosource


def with_kernel(fn, two_source=True):
    """Run fn() with either kernel active."""
    f4.sphere_rate = rate_twosource if two_source else _CANON_RATE
    try:
        return fn()
    finally:
        f4.sphere_rate = rate_twosource


# ------------------------------------------------------------------ data --
BETAS = [1.0, 0.55, 0.35, 0.20, 0.10, 0.05, 0.0,
         -0.05, -0.10, -0.15, -0.19, -0.1988]
LOWH = [(1.0, 'H=2.22'), (0.55, '2.30'), (0.35, '2.34'),
        (0.20, '2.41'), (0.0, '2.59 (Blasius)')]


def fs_at(beta, Re_th, ny=3000, ymax=14.0):
    """(u, dudy, d2u, y) at the station whose Re_theta matches, edge-normalised."""
    w = FalknerSkanWedge(beta)

    def sample(Rex):
        sc = w.inviscid_at(Rex)
        yg = np.linspace(0.0, ymax*np.sqrt(Rex/sc), ny + 1)
        y, u, dudy, _ = w.at(Rex, yg)
        ue = u[-1]
        return np.trapezoid(u/ue*(1 - u/ue), y)*ue, y, u, dudy
    lo, hi = 1e1, 1e13
    for _ in range(80):
        m = np.sqrt(lo*hi)
        if sample(m)[0] > Re_th:
            hi = m
        else:
            lo = m
    _, y, u, dudy = sample(hi)
    return u, dudy, np.gradient(dudy, y), y


# --------------------------------------------------------------- FIGURE A --
def fig_calibrate():
    rows = []
    for b in BETAS:
        try:
            H, s_e, s_l, Rt1 = f4.measures_for_beta(b, verbose=False,
                                                    wedge_lambda=False)
            Hc, sc_e, sc_l, Rc1 = with_kernel(
                lambda b=b: f4.measures_for_beta(b, verbose=False,
                                                 wedge_lambda=False),
                two_source=False)
            rows.append((b, H, s_e, s_l, Rt1, sc_e, sc_l, Rc1))
            print('  beta %+.4f  H %.3f  two-source Rt1 %8.1f  canon Rt1 %8.1f'
                  % (b, H, Rt1, Rc1), flush=True)
        except Exception as e:                                  # noqa: BLE001
            print('  beta %+.4f skipped (%s)' % (b, e), flush=True)
    rows = np.array([r for r in rows if np.isfinite(r[1])])
    H = rows[:, 1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    dg = np.array([float(dN_dRe_theta(h)) for h in H])
    ax[0].semilogy(H, dg, 'r--', lw=2, label='Drela--Giles')
    ax[0].semilogy(H, rows[:, 2], 'k-', lw=1.8, label='two-source, early $N\\in[1,5]$')
    ax[0].semilogy(H, rows[:, 3], 'k:', lw=1.8, label='two-source, late $N\\in[5,9]$')
    ax[0].semilogy(H, rows[:, 5], color='0.55', ls='-', lw=1.4,
                   label='single-source, early')
    ax[0].set_xlabel('$H$'); ax[0].set_ylabel(r'$dN/dRe_\theta$')
    ax[0].set_title('(a) amplification rate'); ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3)
    dgs = np.array([float(Re_theta0(h)) + 1.0/float(dN_dRe_theta(h)) for h in H])
    ax[1].loglog(dgs, rows[:, 4], 'ko', ms=5, label='two-source')
    ax[1].loglog(dgs, rows[:, 7], 'o', ms=5, mfc='none', color='0.55',
                 label='single-source')
    lim = [min(dgs.min(), 1e2), max(dgs.max()*1.2, 1e4)]
    ax[1].loglog(lim, lim, 'r--', lw=1.5, label='Drela--Giles $N\\!=\\!1$')
    ax[1].set_xlabel(r'Drela--Giles $N\!=\!1$ station $Re_\theta$')
    ax[1].set_ylabel(r'model $N\!=\!1$ station $Re_\theta$')
    ax[1].set_title('(b) onset'); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'twosource_calibrate.pdf'))
    plt.close(fig)
    return rows


# --------------------------------------------------------------- FIGURE B --
def fig_onsetgraze():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    adverse = [0.0, -0.05, -0.10, -0.15, -0.19, -0.1988]
    cm = plt.cm.coolwarm
    for k, b in enumerate(adverse):
        Rt0 = float(Re_theta0(_H_of(b)))
        u, dudy, d2u, y = fs_at(b, Rt0)
        P_I, _ = coords(u, dudy, d2u, y)
        ReOm = y*y*np.abs(dudy)
        m = P_I > 1e-6
        ax[0].loglog(P_I[m], ReOm[m], color=cm(k/max(len(adverse)-1, 1)), lw=1.2,
                     label=r'$\beta=%+.3f$' % b)
    p = np.logspace(-3, 0, 400)
    ax[0].loglog(p, thr_inv(p), 'k--', lw=2, label='inflectional branch')
    ax[0].set_xlabel(r'$P_I=\hat\Omega\langle\hat I\rangle_+$')
    ax[0].set_ylabel(r'$Re_\Omega$'); ax[0].set_ylim(1e1, 1e5)
    ax[0].set_title('(a) inviscid branch grazes the adverse family')
    ax[0].legend(fontsize=6.5); ax[0].grid(alpha=.3)
    for k, (b, lab) in enumerate(LOWH):
        Rt0 = float(Re_theta0(_H_of(b)))
        u, dudy, d2u, y = fs_at(b, Rt0)
        _, P_c = coords(u, dudy, d2u, y)
        ReOm = y*y*np.abs(dudy)
        m = P_c > 1e-6
        ax[1].loglog(P_c[m], ReOm[m], color=cm(1 - k/max(len(LOWH)-1, 1)),
                     lw=1.2, label=lab)
    ax[1].loglog(p, thr_visc(p), 'k--', lw=2, label='viscous branch')
    ax[1].set_xlabel(r'$P_\mathrm{curv}=\hat\Omega\langle-\hat Z\rangle_+$')
    ax[1].set_ylabel(r'$Re_\Omega$'); ax[1].set_ylim(1e1, 1e5)
    ax[1].set_title('(b) viscous branch grazes the low-$H$ family')
    ax[1].legend(fontsize=6.5); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'twosource_onsetgraze.pdf'))
    plt.close(fig)


def _H_of(beta):
    fs = FalknerSkanWedge(beta)
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    return np.trapezoid(1 - fs.u, fs.eta)/I_th


# --------------------------------------------------------------- FIGURE C --
def fig_transport():
    rows = [(-0.10, 'adverse'), (0.0, 'Blasius'), (0.10, 'favorable'),
            (1.0, 'stagnation')]
    fig, ax = plt.subplots(len(rows), 1, figsize=(7.2, 3.0*len(rows)))
    for i, (b, tag) in enumerate(rows):
        try:
            fs = FalknerSkanWedge(b)
            H = _H_of(b)
            # adaptive: extend until the envelope reaches N ~ 10, as
            # measures_for_beta does; favorable wedges need a far longer run
            x_max = 4e6 if b <= 0.0 else 1e8
            for _ in range(14):
                xs, N = f4.march(fs, x_max)
                if not np.all(np.isfinite(N)) or N[-1] > 60.0:
                    x_max *= 0.15
                    continue
                if N[-1] > 11.0:
                    x_max = 1.05*float(np.interp(11.0, N, xs))
                    xs, N = f4.march(fs, x_max)
                    break
                x_max *= 4.0
            Ue = fs.inviscid_at(np.maximum(xs, 1e-12))
            I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
            Rt = I_th*np.sqrt(xs*Ue)
            ax[i].plot(Rt, N, 'k-', lw=1.8, label='two-source, marched')
            dg = float(dN_dRe_theta(H)); r0 = float(Re_theta0(H))
            ax[i].plot(Rt, np.clip((Rt - r0)*dg, 0, None), 'r--', lw=1.6,
                       label='Drela--Giles envelope')
            ax[i].set_xlim(0, min(Rt.max(), 4*r0 + 4.0/dg))
            ax[i].set_ylim(0, 10)
            ax[i].set_ylabel('$N$')
            ax[i].set_title(r'$\beta=%+.2f$, $H=%.2f$ (%s)' % (b, H, tag),
                            fontsize=9)
            ax[i].legend(fontsize=7); ax[i].grid(alpha=.3)
            print('  transport beta %+.2f  H %.3f  N_end %.2f' % (b, H, N[-1]),
                  flush=True)
        except Exception as e:                                  # noqa: BLE001
            print('  transport beta %+.2f failed (%s)' % (b, e), flush=True)
    ax[-1].set_xlabel(r'$Re_\theta$')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'twosource_transport.pdf'))
    plt.close(fig)


# --------------------------------------------------------------- FIGURE D --
def fig_sphere():
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.0))
    th = np.linspace(0, 2*np.pi, 400)
    for a in ax:
        a.plot(np.cos(th), np.sin(th), color='0.8', lw=1)
        a.set_aspect('equal'); a.axis('off')
    cm = plt.cm.viridis
    hs, vs = [], []
    for k, (b, lab) in enumerate(LOWH):
        u, dudy, d2u, y = fs_at(b, max(float(Re_theta0(_H_of(b))), 50.0))
        X0, Y0, Z0 = u, y*dudy, 0.5*y*y*d2u
        R = np.sqrt(X0**2 + Y0**2 + Z0**2) + 1e-30
        X, Y, Z = X0/R, Y0/R, Z0/R
        h, v = (Y - X)/SQ2, Z
        m = np.isfinite(h) & np.isfinite(v)
        c = cm(k/max(len(LOWH)-1, 1))
        ax[0].plot(h[m], v[m], color=c, lw=1.3, label=lab)
        ax[1].plot(h[m], v[m], color=c, lw=1.6)
        hs.append(h[m]); vs.append(v[m])
    # neutral locus Ihat = 0, i.e. Y - X - Z = 0, projected: v = -SQ2*h
    tt = np.linspace(-1, 1, 200)
    for a in ax:
        a.plot(tt, -SQ2*tt, color='0.4', ls=':', lw=1.2)
    ax[0].plot(0, 0, 'k+', ms=8)                       # wall point
    hs = np.concatenate(hs); vs = np.concatenate(vs)
    pad = 0.06
    ax[0].set_xlim(-1.05, 1.05); ax[0].set_ylim(-1.05, 1.05)
    ax[1].set_xlim(hs.min()-pad, hs.max()+pad)
    ax[1].set_ylim(vs.min()-pad, vs.max()+pad)
    ax[0].set_title(r'(a) low-$H$ family on $\mathbb{RP}^2$', fontsize=10)
    ax[1].set_title('(b) detail; dotted: neutral locus $\\hat I=0$', fontsize=10)
    ax[1].set_aspect('auto')
    ax[0].legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'twosource_sphere.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(FIGS, exist_ok=True)
    print('a_inv=%.3f  a_visc=%.4f  A=%.1f  B=%.3f  B_c=%.1f  C=%.1f'
          % (A_MAX, A_VISC, REOM_A, REOM_B, B_C, REOM_CEIL))
    print('[1/4] onset graze', flush=True); fig_onsetgraze()
    print('[2/4] sphere', flush=True); fig_sphere()
    print('[3/4] transport', flush=True); fig_transport()
    print('[4/4] calibrate', flush=True); fig_calibrate()
    print('done ->', os.path.normpath(FIGS))
