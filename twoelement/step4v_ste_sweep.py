"""Is the camber TE slope the optimizer chose (+18.8 deg) actually the right
one, or is it just where Nelder-Mead stopped?

Sweep the trailing-edge camber slope, re-optimising the thickness coefficients
at EACH value so the comparison is between converged designs rather than
between one converged design and a set of detuned ones. Also re-states the
step2v baseline properly: the earlier comparison padded step2v's 6-coefficient
thickness vector to 7 by duplicating the last entry, which changes the shape
(max t/c 0.1145 rather than its real 0.0917), so that row was not step2v.

Run:  python3 step4v_ste_sweep.py
"""
import numpy as np
from scipy.optimize import minimize

import panel2e as M
import step1v_camber as S
import step2v_build as B
import step4v_te_tune as T
from flap_displacement import flap_body

OPER = -1.0


def diag(nd, flap_nd, x_end, alphas=T.ALPHAS):
    """Flatness, slot-side suction and flap peak, averaged and at operating."""
    tot, rows = 0.0, {}
    for al in alphas:
        P, res = M.solve_elements([nd, flap_nd], al)
        c, d = M.fore_flatness(P)
        lo, up = M.surfaces(P, 0)
        xn = P.xc[lo]/x_end
        m = xn >= 0.80
        sp, lvl = M.flap_suction(P)
        rows[al] = dict(flat=c, rms_up=d['rms_up'], rms_lo=d['rms_lo'],
                        cp_aft=float(P.Cp[lo][m].min()), CL=res['Cl'],
                        flap_pk=lvl)
        tot += c
    return tot/len(alphas), rows


def tune_thickness(s_te, xs_s, zs_s, x_end, flap_nd, a0, maxfev=1400):
    def f(a):
        return T.cost(np.concatenate([[s_te], a]), xs_s, zs_s, x_end, flap_nd)
    r = minimize(f, a0, method='Nelder-Mead',
                 options=dict(maxiter=maxfev, maxfev=maxfev, xatol=1e-5,
                              fatol=1e-7, adaptive=True))
    return r.x, r.fun


if __name__ == '__main__':
    d4 = np.load('step4v_te_tune.npz')
    d2 = np.load('step2v_build.npz')
    x_end = float(d4['x_end'])
    flap_nd, _ = flap_body(with_wake=False)
    rf = S.RansField('case_A%+05.1f' % T.ANCHOR)
    xs_s, zs_s = S.trace(rf, 0.0, 0.0, x_end, n=900)
    te_half = 0.5*B.TE_BASE/x_end

    # ---- the REAL step2v baseline, 6 thickness coefficients, unpadded
    nd2, _, _, th2 = B.fore_nodes(d2['thk'], d2['cam_par'], x_end)
    s2 = -d2['cam_par'][B.N_CAM] + d2['cam_par'][B.N_CAM+1]/x_end
    ang2 = 2*np.degrees(np.arctan(abs(te_half - d2['thk'][-1])))
    f2, r2 = diag(nd2, flap_nd, x_end)
    print('\nSTEP2v baseline (true, 6 thickness coefficients)')
    print('  camber TE slope %+.2f deg, TE included angle %.2f deg, '
          'max t/c %.4f' % (np.degrees(np.arctan(s2)), ang2,
                            2*th2.max()/x_end))
    print('  alpha=-1: flat %.4f  Cp_lo_aft %.3f  flap peak %.3f'
          % (r2[OPER]['flat'], r2[OPER]['cp_aft'], r2[OPER]['flap_pk']))

    a_start = d4['thk']
    print('\nSweeping the camber TE slope, thickness re-optimised at each value')
    print('%-9s %-8s %9s %10s %10s %9s %9s'
          % ('s_te', 'deg', 'TE ang', 'flat(mean)', 'Cp_lo_aft', 'flap pk',
             'max t/c'))
    best = None
    for s_te in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.3402, 0.40):
        a, fv = tune_thickness(s_te, xs_s, zs_s, x_end, flap_nd, a_start)
        nd, cam, rms, th, xf, zc = T.build(np.concatenate([[s_te], a]),
                                           xs_s, zs_s, x_end)
        fm, rows = diag(nd, flap_nd, x_end)
        ang = 2*np.degrees(np.arctan(abs(te_half - a[-1])))
        gap = float(np.min(np.hypot(flap_nd[:, 0] - nd[0, 0],
                                    flap_nd[:, 1] - nd[0, 1])))
        print('%-9.4f %-8.2f %9.2f %10.4f %10.3f %9.3f %9.4f'
              % (s_te, np.degrees(np.arctan(s_te)), ang, fm,
                 rows[OPER]['cp_aft'], rows[OPER]['flap_pk'],
                 2*th.max()/x_end))
        if best is None or fm < best[0]:
            best = (fm, s_te, a, nd, cam, ang, gap)
    fm, s_te, a, nd, cam, ang, gap = best
    print('\nBEST: s_te %+.4f (%.2f deg), TE included angle %.2f deg, '
          'gap %.4f, flat %.4f'
          % (s_te, np.degrees(np.arctan(s_te)), ang, gap, fm))
    np.savez('step4v_ste_sweep.npz', s_te=s_te, thk=a, cam_par=cam,
             x_end=x_end, fore=nd, flap=flap_nd, te_angle=ang, gap=gap)
    np.savetxt('step4v_fore_best.dat', nd, fmt='%12.8f')
    print('wrote step4v_ste_sweep.npz, step4v_fore_best.dat')
