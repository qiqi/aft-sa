"""STEP 7 -- a point SOURCE at the fore trailing edge.

User's formulation (2026-08-04): rather than deforming the surface, put a
potential-flow source at the trailing edge and ask how strong it must be to
bring Cp there to a target (-0.4). The source strength then converts directly
into the displacement thickness the boundary layer would have to supply.

Conversion. A 2D source of volume flux Q in a stream of speed U_e opens a
half-body of asymptotic TOTAL width Q/U_e. That width is exactly the wake's
total displacement thickness, i.e. the sum over the two surfaces:

    Q / U_e  =  delta*_upper + delta*_lower      ->   per surface  Q/(2 U_e)

So a required Q maps to a required delta*, and with theta from Blasius, to a
required H.

The source enters the panel solve as a known singularity: it contributes to
the tangency right-hand side and to the Kutta rows, and to the surface
tangential velocity afterwards.

Run:  python3 step7_te_source.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel2e as M
import step4_coupled as S
from step1_flap_and_camber import flap_nodes, FORE_CHORD, ALPHA
from step2_questions import refit_inc

T_FORE, NPAN = 0.08, 400
H_BLASIUS, H_SEP = 2.59, 3.98
RE = 1.0e6


WAKE_LEN, N_SRC = 0.05, 40          # distribute over the near wake


def src_vel(px, pz, xs, zs, Q, tdir=(1.0, 0.0), L=WAKE_LEN, n=N_SRC):
    """Velocity from a LINE source of total strength Q laid along the wake.

    A point source sitting exactly at the trailing edge is singular on the
    adjacent panel's collocation point (cosine spacing puts it a few 1e-5 away),
    which makes Cp_TE meaningless. The boundary layer's displacement is in any
    case not a point: it is a source distribution that develops along the wake.
    Spreading Q over the first L of wake removes the singularity and is the
    more faithful model.
    """
    t = np.asarray(tdir, float); t /= np.linalg.norm(t)
    ds = L/n
    ux = np.zeros_like(np.asarray(px, float))
    uz = np.zeros_like(ux)
    k = Q/n/(2.0*np.pi)
    for i in range(n):
        cx, cz = xs + t[0]*(i + 0.5)*ds, zs + t[1]*(i + 0.5)*ds
        dx, dz = px - cx, pz - cz
        r2 = np.maximum(dx*dx + dz*dz, 1e-12)
        ux += k*dx/r2
        uz += k*dz/r2
    return ux, uz


def solve_with_source(els, alpha, xsrc, zsrc, Q):
    """Hess-Smith with an extra fixed point source (known RHS contribution)."""
    P = M.Panels(els)
    a = np.radians(alpha)
    Vx0, Vz0 = np.cos(a), np.sin(a)
    usx, usz = src_vel(P.xc, P.zc, xsrc, zsrc, Q)
    Vfx, Vfz = Vx0 + usx, Vz0 + usz          # freestream + source

    dim = P.n + P.n_elem
    A = np.zeros((dim, dim)); b = np.zeros(dim)
    us, ws, uv, wv = M.influence(P.xc, P.zc, P, self_idx=np.arange(P.n))
    A[:P.n, :P.n] = us*P.nx[:, None] + ws*P.nz[:, None]
    vn_v = uv*P.nx[:, None] + wv*P.nz[:, None]
    for k in range(P.n_elem):
        A[:P.n, P.n + k] = vn_v[:, P.eid == k].sum(axis=1)
    b[:P.n] = -(Vfx*P.nx + Vfz*P.nz)

    for k in range(P.n_elem):
        idx = np.where(P.eid == k)[0]
        f, l = idx[0], idx[-1]
        tgt = np.array([f, l])
        us2, ws2, uv2, wv2 = M.influence(P.xc[tgt], P.zc[tgt], P, self_idx=tgt)
        tx, tz = P.tx[tgt][:, None], P.tz[tgt][:, None]
        A[P.n + k, :P.n] = (us2*tx + ws2*tz).sum(axis=0)
        vt_v = (uv2*tx + wv2*tz).sum(axis=0)
        for j in range(P.n_elem):
            A[P.n + k, P.n + j] = vt_v[P.eid == j].sum()
        b[P.n + k] = -((Vfx[f]*P.tx[f] + Vfz[f]*P.tz[f])
                       + (Vfx[l]*P.tx[l] + Vfz[l]*P.tz[l]))

    sol = np.linalg.solve(A, b)
    q, gam = sol[:P.n], sol[P.n:]
    Vt = Vfx*P.tx + Vfz*P.tz + (us*P.tx[:, None] + ws*P.tz[:, None]) @ q
    vt_v = uv*P.tx[:, None] + wv*P.tz[:, None]
    for k in range(P.n_elem):
        Vt += gam[k]*vt_v[:, P.eid == k].sum(axis=1)
    P.Vt, P.Cp = Vt, 1.0 - Vt**2
    return P


if __name__ == '__main__':
    n2 = flap_nodes()
    Pf, _ = M.solve_elements([n2], ALPHA)
    xs_, zs_ = M.streamline_camber(Pf, 0.0, 0.0, FORE_CHORD, n=400)
    inc = refit_inc(S.M_CAM, S.P_CAM, xs_, zs_)
    els = S.build(T_FORE, inc)
    te = els[0][0]                          # first node = fore lower TE
    print('fore TE at (%.4f, %.4f)' % (te[0], te[1]))

    # baseline (no source) reference levels
    P0 = solve_with_source(els, ALPHA, te[0], te[1], 0.0)
    _, up0 = M.surfaces(P0, 0)
    xf0 = P0.xc[up0]/FORE_CHORD
    i95 = int(np.argmin(np.abs(xf0 - 0.95)))
    print('baseline: Cp at x/c=0.95 is %.4f, at the last panel %.4f'
          % (P0.Cp[up0][i95], P0.Cp[up0][-1]))

    # local edge speed just ahead of the recovery -> the U_e for the conversion
    Ue = float(np.sqrt(max(1.0 - P0.Cp[up0][i95], 1e-9)))
    s_te = FORE_CHORD
    theta = 0.664*np.sqrt(s_te/RE)
    print('U_e ahead of the recovery = %.4f ;  Blasius theta = %.5f c\n' % (Ue, theta))

    print('%9s %10s %10s %12s %12s %9s' %
          ('Q/(U c)', 'Cp_TE', 'Cp@0.95', 'd_delta*/c', 'per side', 'H needed'))
    rows = []
    for Q in (0.0, 0.0005, 0.001, 0.002, 0.004, 0.007, 0.010, 0.015, 0.025):
        P = solve_with_source(els, ALPHA, te[0], te[1], Q)
        _, up = M.surfaces(P, 0)
        cp_te, cp95 = float(P.Cp[up][-1]), float(P.Cp[up][i95])
        dd = Q/Ue                          # total (both surfaces)
        per = 0.5*dd
        Hneed = (H_BLASIUS*theta + per)/theta
        rows.append((Q, cp_te, cp95, dd, per, Hneed, P))
        print('%9.4f %10.4f %10.4f %12.5f %12.5f %9.2f'
              % (Q, cp_te, cp95, dd, per, Hneed))

    # interpolate to the target
    Qs = np.array([r[0] for r in rows]); cps = np.array([r[1] for r in rows])
    for tgt in (-0.4, -0.25, 0.0):
        if cps.min() <= tgt <= cps.max():
            Qn = float(np.interp(tgt, cps[::-1], Qs[::-1])) if cps[0] > cps[-1] \
                else float(np.interp(tgt, cps, Qs))
            dd = Qn/Ue
            Hn = (H_BLASIUS*theta + 0.5*dd)/theta
            print('\n  Cp_TE = %+.2f  needs Q/(U c) = %.4f'
                  '  ->  delta* +%.5f c per side  ->  H = %.2f  (%s H_sep=%.2f)'
                  % (tgt, Qn, 0.5*dd, Hn,
                     'PAST' if Hn > H_SEP else 'below', H_SEP))
        else:
            print('\n  Cp_TE = %+.2f not reached within the swept Q' % tgt)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for Q, cp_te, cp95, dd, per, Hn, P in rows:
        if Q in (0.0, 0.005, 0.02, 0.05, 0.12):
            _, up = M.surfaces(P, 0)
            ax[0].plot(P.xc[up]/FORE_CHORD, P.Cp[up], '-', lw=1.4,
                       label='Q=%.3f (H=%.1f)' % (Q, Hn))
    ax[0].set_xlim(0.80, 1.01); ax[0].invert_yaxis(); ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=7.5); ax[0].set_xlabel('$x/c_{fore}$')
    ax[0].set_ylabel('$C_p$'); ax[0].set_title('TE source unloading', fontsize=10)

    ax[1].plot([r[5] for r in rows], [r[1] for r in rows], 'o-',
               color='#1f4e9c', lw=1.6)
    ax[1].axvline(H_SEP, color='#b91c1c', ls='--', lw=1.4, label='$H_{sep}$=3.98')
    ax[1].axhline(-0.4, color='#059669', ls=':', lw=1.4, label='target $C_p$=-0.4')
    ax[1].set_xlabel('required $H$ at the TE'); ax[1].set_ylabel('$C_p$ at the TE')
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    ax[1].set_title('what the source costs in $H$', fontsize=10)
    fig.tight_layout()
    fig.savefig('step7_te_source.pdf'); fig.savefig('step7_te_source.png', dpi=125)
    print('\nwrote step7_te_source.pdf / .png')
