"""How well SA's wall-layer solution satisfies the SA-AI equation (Sec. II.E).

On the constant-stress wall layer SA's solution is nu~ = kappa u_tau d, i.e.
chi = kappa y+, with the velocity profile (1 + chi f_v1) du+/dy+ = 1. With the
destruction tie of Eq. (w:sigma), SA's own production, destruction and
diffusion balance on that profile for every value of the handover weight. The
only term left over is the amplification source, which is the whole production
where sigma_P = 0 (chi < 1, y+ < 2.44). This script evaluates the residual of
the full balance, P - D + diffusion with P the gated maximum of Eq. (w:blend),
as a fraction of SA's production c_b1 u_tau^2.

Result: at most 8e-6, at y+ = 2.44 (chi = 1). Once sigma_P exceeds about
1e-5, which is by chi = 1.0001, SA's weighted production is the larger of the
two and the residual is zero to rounding error.

Run from paper/:  python3 repro/analytic/wall_layer_tie_residual.py
"""
import _saai  # noqa: F401
import numpy as np
from lib import sphere_kernel as SK

KAPPA, CB1, CB2, SIGMA, CV1, TAU = 0.41, 0.1355, 0.622, 2.0/3.0, 7.1, 4.0
CW1 = CB1/KAPPA**2 + (1.0 + CB2)/SIGMA


def main():
    yp = np.geomspace(1e-3, 2e3, 400001)
    chi = KAPPA*yp
    fv1 = chi**3/(chi**3 + CV1**3)
    up = 1.0/(1.0 + chi*fv1)
    u = np.concatenate([[0.0], np.cumsum(0.5*(up[1:] + up[:-1])*np.diff(yp))]) + yp[0]
    upp = np.gradient(up, yp)
    prod, _, _, _ = SK.sphere_indicators(u, up, upp, yp)
    a = SK.A_MAX*np.clip(prod, 0.0, 1.0)
    s_re = 0.5*(1.0 + np.tanh((yp**2*np.abs(up)/SK.reom_crit(prod) - 1.0)/SK.RAMP_W))
    s_p = np.maximum(1.0 - np.exp(-(chi - 1.0)/TAU), 0.0)
    s_d = 1.0 - (CB1/(KAPPA**2*CW1))*(1.0 - s_p)
    p_ai = a*s_re*up*chi                                  # in units of u_tau^2
    p = np.maximum((1.0 - s_p)*p_ai, s_p*CB1)
    sa_only = s_p*CB1 - s_d*CW1*KAPPA**2 + (1.0 + CB2)*KAPPA**2/SIGMA
    full = (p - s_d*CW1*KAPPA**2 + (1.0 + CB2)*KAPPA**2/SIGMA)/CB1
    j = int(np.argmax(np.abs(full)))
    above = chi > 1.0001
    j_sp = int(np.argmax((s_p*CB1 >= (1.0 - s_p)*p_ai) & (chi > 1.0)))
    print('SA terms alone:  max |residual| = %.1e' % np.abs(sa_only).max())
    print('full equation:   max residual / (c_b1 u_tau^2) = %.2e at y+ = %.2f' % (full[j], yp[j]))
    print('SA production becomes the larger at chi = %.6f, sigma_P = %.1e' % (chi[j_sp], s_p[j_sp]))
    print('above chi = 1.0001:  max |residual| / (c_b1 u_tau^2) = %.1e' % np.abs(full[above]).max())
    assert np.abs(sa_only).max() < 1e-12
    assert 5e-6 < full[j] < 1e-5 and yp[j] < 2.45
    assert np.abs(full[above]).max() < 1e-12
    print('Quoted number reproduced: 8e-6.')


if __name__ == '__main__':
    main()
