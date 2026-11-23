"""Sensitivity of the N=1 stations to the onset ramp width w (Sec. II.D).

Marches the attached Falkner-Skan family as tab_march_stations.py does, with
the ramp width of Eq. (w:rate) at its value 0.35, halved and doubled, and
asserts the two numbers Section II.D quotes: the N=1 stations move by at most
1.6% with w halved and by at most 3.3% with w doubled.

Run from paper/:  python3 repro/analytic/ramp_width_sensitivity.py
(about 20 minutes on 32 cores)
"""
import _saai  # noqa: F401
import lib.sphere_kernel as SK
import fig04_shapefactor as F4

FAMILY = [0.15, 0.10, 0.05, 0.0, -0.05, -0.10, -0.15, -0.19, -0.1988]
W0 = 0.35
QUOTED = {0.5*W0: 1.6, 2.0*W0: 3.3}      # max |change| of the N=1 station, percent


def stations(w):
    SK.RAMP_W = w
    F4.RAMP_W = w
    res = F4.measures_many([(b, {'verbose': False}) for b in FAMILY],
                           c=[1/6]*len(FAMILY), con=[F4.K_ANCHOR]*len(FAMILY))
    return [r[3] for r in res]


def main():
    assert abs(SK.RAMP_W - W0) < 1e-12
    base = stations(W0)
    print('w = %.3f:' % W0, ' '.join('%.1f' % x for x in base), flush=True)
    for w, quoted in QUOTED.items():
        st = stations(w)
        d = [100*(a/b - 1) for a, b in zip(st, base)]
        worst = max(abs(x) for x in d)
        print('w = %.3f:' % w, ' '.join('%+.2f' % x for x in d), ' max = %.2f%%' % worst, flush=True)
        assert abs(worst - quoted) < 0.06, (w, worst, quoted)
    SK.RAMP_W = W0
    print('All quoted numbers reproduced.')


if __name__ == '__main__':
    main()
