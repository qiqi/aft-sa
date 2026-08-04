"""Why the spheroid fronts are too swept in azimuth: the onset gate's
sensitivity to weak favorable pressure gradient.

The question this answers: at alpha = 2.5 deg and Re_L = 7.2e6 our front is
far more swept in azimuth than either the measurement or Stock's e^N -- and
at that incidence Stock's front is TS-coded, so a missing crossflow channel
cannot be the explanation.

Method (all from committed data -- the surface probe caches and the harvest
JSON; no CFD, no case tree):

  1. Per azimuth, get the inviscid edge speed from the surface pressure,
     u_e/U_inf = sqrt(1 - c_p), and average the Falkner-Skan-like gradient
     parameter beta = 2m/(m+1), m = (x/u_e) du_e/dx, over the laminar run
     ahead of the computed front.
  2. Regress the computed chi = 1 front on that beta.
  3. Compare the resulting sensitivity with what the kernel's own structure
     predicts, using the whitepaper's Falkner-Skan values of the rate
     coordinate P = max_y OmegaHat*Ihat and the onset threshold
     Re_Omega^c = k * softmin2(C, A + B/P^2), (C,A,B) = (2600,175,2).

Result: the front is a near-linear function of beta ALONE -- r = 0.995 at
alpha = 2.5 -- with a sensitivity dx_front/dbeta ~ 4.2 that is the same at
2.5, 5 and 10 degrees, against ~1.4 for the measurement.  The amplifier is
the SQUARE in B/P^2: P falls steeply with beta (d ln P/d beta = -7.4 on the
Falkner-Skan family), so a 25% drop in P raises the onset threshold ~60%,
and since Re_Omega grows roughly linearly along a laminar layer that is
~60% of the onset station.

Run from paper/:  python3 repro/cfd/diag_spheroid_fpg_sensitivity.py
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHE = os.path.join(HERE, 'cache_spheroid_surface')
HARV = os.path.join(HERE, 'figs_explore', 'spheroid_tunnel_harvest.json')

# whitepaper Fig. 2 / Fig. 3 captions: max_y OmegaHat*Ihat on the FS family
FS_BETA = np.array([-0.10, 0.0, 0.10])
FS_P = np.array([0.162, 0.078, 0.037])
A_GATE, B_GATE = 175.0, 2.0          # whitepaper (C, A, B) = (2600, 175, 2)

CONDS = [('a2p5', 'a2p5_gcal', 2.5), ('a5hi', 'a5hi_gcal', 5.0),
         ('a10hi', 'a10hi_gcal', 10.0)]


def beta_and_front(tag, seedtag, H):
    d = np.load(glob.glob(os.path.join(CACHE, f'*{tag}_*'))[0])
    xl, phd, cp = d['xl'], d['phi_deg'], d['cp']
    ue = np.sqrt(np.clip(1.0 - cp, 1e-9, None))
    front = {p['phi']: p['chi1'] for p in H[seedtag]['curve']
             if p['chi1'] is not None}
    B, X = [], []
    for j, ph in enumerate(phd):
        k = min(front, key=lambda q: abs(q - ph))
        if abs(k - ph) > 4:
            continue
        xf = front[k]
        m = (xl > 0.08) & (xl < min(xf, 0.6))
        if m.sum() < 10:
            continue
        ml = (xl / np.maximum(ue[j], 1e-9)) * np.gradient(ue[j], xl)
        B.append(float(np.mean(2 * ml[m] / (ml[m] + 1))))
        X.append(float(xf))
    return np.array(B), np.array(X)


def measured_sweep():
    """Measured front sweep at alpha=10, 6.56e6, over its own azimuths."""
    d = json.load(open(os.path.join(PAPER, 'data',
                                    'stock2006_fig15a_digitized.json')))
    xs = [p['xL'] for p in d['re_6p56e6_alpha10_circles']]
    return max(xs) - min(xs)


def main():
    H = json.load(open(HARV))
    print('computed chi=1 front regressed on the local FS beta '
          '(calibrated seed):')
    print(f'{"case":>7} {"alpha":>6} {"beta range":>19} {"front range":>18}'
          f' {"r":>7} {"dx/dbeta":>9}')
    for tag, seed, al in CONDS:
        B, X = beta_and_front(tag, seed, H)
        r = float(np.corrcoef(B, X)[0, 1])
        s = float(np.polyfit(B, X, 1)[0])
        print(f'{tag:>7} {al:6.1f} {B.min():8.4f}..{B.max():-8.4f}'
              f' {X.min():8.3f}..{X.max():-8.3f} {r:7.3f} {s:9.2f}')

    slope = float(np.polyfit(FS_BETA, np.log(FS_P), 1)[0])
    print(f'\nkernel structure: d ln(OmegaHat*Ihat)/d beta = {slope:.2f} '
          '(whitepaper FS family)')
    B, X = beta_and_front('a2p5', 'a2p5_gcal', H)
    dbeta = B.max() - B.min()
    ratio = float(np.exp(slope * dbeta))
    print(f'alpha=2.5: d beta = {dbeta:.4f} over azimuth -> P falls to '
          f'{ratio:.3f} of its leeward value')
    for p0 in (0.05, 0.06, 0.078):
        r0 = A_GATE + B_GATE / p0 ** 2
        r1 = A_GATE + B_GATE / (p0 * ratio) ** 2
        print(f'   P_lee={p0:.3f}: Re_Omega^c {r0:6.0f} -> {r1:6.0f} '
              f'(+{100*(r1/r0-1):3.0f}%)  => front shift ~{(r1/r0-1)*0.46:.2f} L')
    print(f'   observed alpha=2.5 front sweep {X.max()-X.min():.3f} L')
    print(f'\nfor scale, the MEASURED front sweep at alpha=10 / 6.56e6 is '
          f'{measured_sweep():.3f} L; ours is '
          f'{np.ptp(beta_and_front("a10hi","a10hi_gcal",H)[1]):.3f} L')


if __name__ == '__main__':
    main()
