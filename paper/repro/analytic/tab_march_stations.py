"""Sec. II.D: the table of the Blasius growth rate against c_nu,ai, its
insensitivity to the onset factor c_on, the N = 1 stations that set c_on, and
the stations and rates the text quotes for the Falkner-Skan family. No figure.

Every case is the disturbance transport of Eq. (1) with laminar diffusion, marched
on the similarity profile (fig04_shapefactor.march, grid NX x NY). The N = 1 and
N = 9 stations are compared with the Drela-Giles envelope: critical Re_theta plus
N/(dN/dRe_theta). The early rate is the N = 1 to 5 secant.

The numbers quoted in the paper are asserted at the end."""
import _saai
import numpy as np
import fig04_shapefactor as F4
from fig04_shapefactor import measures_many, drela, Re_theta0

SWEEP = [(1.0, '1'), (0.5, '1/2'), (0.25, '1/4'), (1/6, '1/6'), (1/12, '1/12'),
         (1/24, '1/24'), (1/48, '1/48'), (0.0, '0')]
FAMILY = [0.35, 0.15, 0.10, 0.05, 0.0, -0.05, -0.10, -0.15, -0.19, -0.1988]
CONS = [1.0, F4.K_ANCHOR, 0.5]     # onset factors: none, the model's, one half
# quoted in Sec. II.D: early rate at the model's c_on (percent of Drela-Giles)
QUOTED_RATE = {'1': 61, '1/2': 70, '1/4': 79, '1/6': 83, '1/12': 90, '1/24': 96, '1/48': 100, '0': 107}
# quoted in Sec. II.D: N = 1 station at c_nu,ai = 1/6 for each onset factor
QUOTED_STATION = {1.0: 398, F4.K_ANCHOR: 331, 0.5: 281}


def main():
    n = len(SWEEP)
    res = measures_many([(0.0, {})]*(n*len(CONS)) + [(b, {}) for b in FAMILY],
                        c=[c for _ in CONS for c, _ in SWEEP] + [1/6]*len(FAMILY),
                        con=[k for k in CONS for _ in SWEEP] + [F4.K_ANCHOR]*len(FAMILY))
    sweep = {k: res[i*n:(i + 1)*n] for i, k in enumerate(CONS)}
    family = res[n*len(CONS):]
    d0 = float(drela(sweep[CONS[0]][0][0]))

    print(f'grid: {F4.NX} streamwise steps, {F4.NY} wall-normal cells')
    print('\nBlasius march: early rate / Drela-Giles, and N = 1 station, against c_nu,ai and c_on')
    print('  c_nu,ai  ' + ''.join(f'   c_on = {k:.3f}     ' for k in CONS))
    spread = 0.0
    for i, (c, name) in enumerate(SWEEP):
        pct = {k: 100*sweep[k][i][1]/d0 for k in CONS}
        st = {k: sweep[k][i][3] for k in CONS}
        print(f'  {name:>6}  ' + ''.join(f'   {pct[k]:5.1f}%  {st[k]:6.1f}  ' for k in CONS))
        spread = max(spread, max(pct.values()) - min(pct.values()))
        assert abs(pct[F4.K_ANCHOR] - QUOTED_RATE[name]) <= 0.6, (name, pct)
        if name == '1/6':
            for k in CONS:
                assert abs(st[k] - QUOTED_STATION[k]) <= 0.6, (k, st[k])
            late = st[1.0]/(float(Re_theta0(sweep[1.0][i][0])) + 1/d0) - 1
            print(f'           at c_on = 1 the N = 1 station is {100*late:+.1f}% from Drela-Giles')
            assert 0.17 < late < 0.19
    print(f'  largest change of the rate between c_on = 1 and 1/2: {spread:.1f}% of the Drela-Giles slope')
    assert spread < 2.0

    print('\nFalkner-Skan family at c_nu,ai = 1/6')
    print('  beta      H     N=1 model  N=1 D-G   ratio   N=9 model  N=9 D-G   ratio   early  late')
    dev = {}
    for beta, (H, se, sl, Rt1) in zip(FAMILY, family):
        d = float(drela(H)); r0 = float(Re_theta0(H))
        Rt9 = Rt1 + 4/se + 4/sl
        dev[beta] = (Rt1/(r0 + 1/d) - 1, Rt9/(r0 + 9/d) - 1, se/d, Rt1/r0, Rt1, Rt9)
        print(f'  {beta:+.4f}  {H:.2f}  {Rt1:9.1f}  {r0 + 1/d:8.1f}  {Rt1/(r0 + 1/d):6.3f}  '
              f'{Rt9:9.1f}  {r0 + 9/d:8.1f}  {Rt9/(r0 + 9/d):6.3f}   {se/d:.2f}   {sl/d:.2f}')
    attached = [b for b in FAMILY if b <= 0.15]
    rms = float(np.sqrt(np.mean([dev[b][0]**2 for b in attached])))
    adverse = [dev[b][0] for b in attached if b < 0]
    favorable = [dev[b][0] for b in attached if b > 0]
    print(f'\nN = 1 station against Drela-Giles, beta = +0.15 ... -0.1988: rms {100*rms:.1f}%')
    print(f'  adverse wedges {100*min(adverse):+.1f}% ... {100*max(adverse):+.1f}%')
    print(f'  favorable wedges {100*min(favorable):+.1f}% ... {100*max(favorable):+.1f}%')
    print(f'  Blasius: N = 1 at {dev[0.0][4]:.1f} ({100*dev[0.0][0]:+.1f}%), '
          f'N = 9 at {dev[0.0][5]:.1f} ({100*dev[0.0][1]:+.1f}%)')
    print(f'  beta = +0.35: rate {dev[0.35][2]:.2f} x Drela-Giles, N = 1 station '
          f'{dev[0.35][3]:.2f} x its critical Re_theta')

    # numbers quoted in Sec. II.D and Table 1
    assert abs(dev[0.0][4] - 331) < 0.6 and abs(dev[0.0][0]) < 0.025
    assert abs(dev[0.0][5] - 1175) < 1.5 and 0.055 < dev[0.0][1] < 0.065
    assert 0.065 < rms < 0.075
    assert max(abs(x) for x in adverse) < 0.055
    assert -0.135 < min(favorable) and max(favorable) < -0.09
    assert 0.08 < dev[0.35][2] < 0.12 and 2.6 < dev[0.35][3] < 2.8
    print('\nAll quoted numbers reproduced.')


if __name__ == '__main__':
    main()
