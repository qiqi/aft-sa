"""The analytic margin behind the fv1-bypass q gate (Sec. II.F): how far can
an ATTACHED equilibrium wall layer stray from q = 1?

q = Re_u / (y+ U+_Reichardt(y+)) with y+ := chi/kappa. In an SA-consistent
attached layer nuHat = kappa y u_tau exactly (that is fv1's design premise),
so y+ = chi/kappa recovers the true y+ and q = U+_actual(y+)/U+_Reichardt(y+).
The distortion of U+_actual away from Reichardt under pressure gradient is
modeled by the Coles composite,

    U+(y+; Pi, delta+) = U+_Reichardt(y+) + (2 Pi / kappa) sin^2(pi/2 * y/delta),

and the bypass gate only acts in the chi band (1, ~30), i.e. y+ in
(2.4, ~73), where fv1 < 0.99. This script sweeps the wake parameter Pi and
delta+ and reports max q over the gated band -- the analytic envelope the
paper's "(2,4)" gate margin is scoped to.

  python3 fv1_qmargin_composite.py
"""
import numpy as np

KAPPA = 0.41


def u_reichardt(yp):
    return (1.0 / KAPPA) * np.log(1.0 + KAPPA * yp) + 7.8 * (
        1.0 - np.exp(-yp / 11.0) - (yp / 11.0) * np.exp(-yp / 3.0))


def q_max_in_band(Pi, dplus, band=(2.4, 73.0)):
    yp = np.linspace(band[0], min(band[1], dplus), 400)
    wake = (2.0 * Pi / KAPPA) * np.sin(0.5 * np.pi * yp / dplus) ** 2
    q = (u_reichardt(yp) + wake) / u_reichardt(yp)
    return float(q.max())


def q_sa_wall_layer(yp_max=1.0e5, n=400001):
    """q in SA's own solution of the wall layer: nuHat = kappa u_tau d, so that
    chi = kappa y+, and dU+/dy+ = 1/(1 + chi f_v1) in the constant-stress layer.
    Returns (chi, q) with q = U+_SA / U+_Reichardt."""
    yp = np.concatenate([[0.0], np.geomspace(1e-3, yp_max, n)])
    chi = KAPPA * yp
    slope = 1.0 / (1.0 + chi * chi ** 3 / (chi ** 3 + 7.1 ** 3))
    U = np.concatenate([[0.0], np.cumsum(0.5 * (slope[1:] + slope[:-1]) * np.diff(yp))])
    return chi[1:], U[1:] / u_reichardt(yp[1:])


def shape(Pi, dplus):
    """Shape factor, Re_theta and C_f of the composite profile."""
    yp = np.linspace(0.0, dplus, 200001)
    U = u_reichardt(yp) + (2.0 * Pi / KAPPA) * np.sin(0.5 * np.pi * yp / dplus) ** 2
    f = U / U[-1]
    ds = np.trapezoid(1.0 - f, yp); th = np.trapezoid(f * (1.0 - f), yp)
    return ds / th, th * U[-1], 2.0 / U[-1] ** 2


if __name__ == '__main__':
    print(f"{'Pi':>5} " + " ".join(f"d+={d:>6}" for d in (100, 300, 1000, 3000)))
    for Pi in (0.0, 0.55, 1.0, 2.0, 3.5, 5.0, 7.0):
        row = [q_max_in_band(Pi, d) for d in (100, 300, 1000, 3000)]
        print(f"{Pi:5.2f} " + " ".join(f"{q:8.3f}" for q in row))
    # the three profiles quoted in Sec. II.F: (Pi, delta+) -> (Re_theta, H, max q)
    quoted = {(0.55, 300): (780, 1.54, 1.02), (3.5, 300): (1700, 1.95, 1.15), (3.5, 100): (520, 2.27, 1.88)}
    for (Pi, d), (rt, hh, qq) in quoted.items():
        H, Rt, Cf = shape(Pi, d)
        print('Pi=%.2f delta+=%d: Re_theta=%.0f H=%.3f Cf=%.5f max q=%.3f' % (Pi, d, Rt, H, Cf, q_max_in_band(Pi, d)))
        assert abs(Rt/rt - 1) < 0.03 and abs(H - hh) < 0.006 and abs(q_max_in_band(Pi, d) - qq) < 0.006
    # q with the velocity of SA's own wall-layer solution, at every chi
    chi, q = q_sa_wall_layer()
    print('SA wall layer: q min %.3f at chi=%.1f, max %.3f at chi=%.1f, q=%.3f at chi=%.0f'
          % (q.min(), chi[q.argmin()], q.max(), chi[q.argmax()], q[-1], chi[-1]))
    assert abs(q.min() - 0.96) < 0.005 and abs(q.max() - 1.07) < 0.005
    yp = 0.5/KAPPA
    assert abs(yp*u_reichardt(yp) - 1.5) < 0.01            # denominator of q at chi = 0.5
    assert 1.0/(1.0 + 7.1**3) < 0.003                       # f_v1 at chi = 1
    # Clauser's parameter for Pi = 3.5 from Das's correlation, beta = 0.42 Pi^2 + 0.76 Pi - 0.4
    beta = 0.42*3.5**2 + 0.76*3.5 - 0.4
    print('Pi = 3.5: Clauser beta = %.2f (Das 1987); Pi = 0.55: beta = %.2f' % (beta, 0.42*0.55**2 + 0.76*0.55 - 0.4))
    assert abs(beta - 7.4) < 0.05
    # q - 1 = (2 Pi / kappa) sin^2(pi y+ / 2 delta+) / U+(y+): increasing in Pi, decreasing in delta+
    grid = [[q_max_in_band(P, d) for d in (100, 200, 300, 1000, 3000)] for P in (0.55, 1.0, 2.0, 3.5, 5.0)]
    g = np.array(grid)
    assert (np.diff(g, axis=0) > 0).all() and (np.diff(g, axis=1) < 0).all()
    print("Quoted numbers reproduced.")
    print()
    print("Reading: q stays below the gate's lower edge (2) throughout the")
    print("gated band for Pi <= 3.5 at any delta+ >= 100 -- i.e. every")
    print("attached equilibrium layer up to strong adverse gradient. Only")
    print("near-separation wakes (Pi ~ 5-7) at low delta+ push q past 2,")
    print("opening G partially; the flat-plate and NLF batteries bound that")
    print("consequence below 1.5 drag counts (data/fv1bypass_battery_results.json).")
