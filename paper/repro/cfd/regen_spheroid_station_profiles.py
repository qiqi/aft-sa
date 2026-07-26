"""Station-waterfall profiles for the 6:1 spheroid in the presentation of
Stock 2006 Figs. 2-5 (the DFVLR measurement stations): C_p(phi) at the 13
pressure stations (each curve displaced +0.14 in -C_p) and total skin
friction C_ft(x1e3) / wall-shear direction gamma(phi) at the 7 hot-film
stations (displaced -1.5 and -15 deg per station). Computed curves from the
surface-map .npz dumps; filled squares mark where the model's chi=c_v1
front crosses each station (Stock's transition-location marks). Measured
symbol overlays follow with the per-condition digitization.

Run from paper/: python3 repro/cfd/regen_spheroid_station_profiles.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = '/home/qiqi/flexcompute/sa-ai/paper'
CV1 = 7.1
CP_STATIONS = [-0.99880, -0.98520, -0.96560, -0.92840, -0.86520, -0.74320,
               -0.53700, -0.28780, -0.03760, 0.21240, 0.46160, 0.66920,
               0.79240]
HF_STATIONS = [-0.894, -0.722, -0.382, -0.040, 0.304, 0.650, 0.766]
DCP, DCF, DGAM = 0.14, -1.5, -15.0


def load(npz):
    d = np.load(npz)
    return d['xl'], d['phi_deg'], d


def station_curve(xl, field2d, Xa):
    xs = (Xa + 1.0) / 2.0
    j = np.searchsorted(xl, xs)
    j = np.clip(j, 1, len(xl) - 1)
    f = (xs - xl[j-1]) / (xl[j] - xl[j-1])
    return field2d[:, j-1] * (1 - f) + field2d[:, j] * f


def front_phi(xl, ph, chimax, Xa):
    """phi where the chi=c_v1 front crosses this station (if any):
    the smallest phi at which the front x(phi) <= x_station boundary."""
    xs = (Xa + 1.0) / 2.0
    fx = np.full(len(ph), np.nan)
    for i in range(len(ph)):
        c = chimax[i]
        hits = np.where(np.isfinite(c) & (c > CV1))[0]
        if len(hits) and hits[0] > 0:
            fx[i] = xl[hits[0]]
    # crossing of fx(phi) with xs
    out = []
    for i in range(1, len(ph)):
        a, b = fx[i-1] - xs, fx[i] - xs
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            out.append(ph[i-1] + (0 - a) / (b - a) * (ph[i] - ph[i-1]))
    return out


def cp_waterfall(npz, out, title):
    xl, ph, d = load(npz)
    if 'cp' not in d:
        raise SystemExit(f'{npz} lacks cp -- re-probe with the updated '
                         'surface_map.py first')
    fig, ax = plt.subplots(figsize=(6.4, 8.4))
    for j, Xa in enumerate(CP_STATIONS):
        c = station_curve(xl, d['cp'], Xa)
        ax.plot(ph, -c + DCP * j, '-', color='k', lw=0.8)
        ax.annotate(f'$X/a={Xa:g}$', (181, (-c + DCP * j)[-1]), fontsize=6.5,
                    va='center')
    ax.set_xlim(0, 215)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_xlabel(r'$\phi$ [deg]')
    ax.set_ylabel(r'$-C_p$ (curves displaced by $+0.14$ per station)')
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{PAPER}/figs/{out}.pdf')
    plt.close(fig)
    print('wrote', out)


def cf_gamma_waterfall(npz, out, title):
    xl, ph, d = load(npz)
    fig, axs = plt.subplots(2, 1, figsize=(6.8, 9.2), sharex=True)
    for j, Xa in enumerate(HF_STATIONS):
        cf = station_curve(xl, d['cf'], Xa) * 1e3
        gm = station_curve(xl, d['gamma_w'], Xa)
        axs[0].plot(ph, cf + DCF * j, '-', color='k', lw=0.8)
        axs[1].plot(ph, gm + DGAM * j, '-', color='k', lw=0.8)
        axs[0].annotate(f'$X/a={Xa:g}$', (181, (cf + DCF * j)[-1]),
                        fontsize=6.5, va='center')
        axs[1].annotate(f'$X/a={Xa:g}$', (181, (gm + DGAM * j)[-1]),
                        fontsize=6.5, va='center')
        for pstar in front_phi(xl, ph, d['chimax'], Xa):
            ci = np.interp(pstar, ph, cf)
            gi = np.interp(pstar, ph, gm)
            axs[0].plot(pstar, ci + DCF * j, 's', color='k', ms=5)
            axs[1].plot(pstar, gi + DGAM * j, 's', color='k', ms=5)
    axs[0].set_ylabel(r'$C_{ft}\times10^3$ (displaced $-1.5$ per station)')
    axs[1].set_ylabel(r'$\gamma_w$ [deg] (displaced $-15^\circ$ per station)')
    axs[1].set_xlabel(r'$\phi$ [deg]')
    axs[1].set_xlim(0, 215)
    axs[1].set_xticks([0, 30, 60, 90, 120, 150, 180])
    for a in axs:
        a.grid(alpha=0.3)
    axs[0].set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{PAPER}/figs/{out}.pdf')
    plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    for tag, ttl in (('L2', r'$\alpha=10^\circ$'),
                     ('a29p7_L2', r'$\alpha=29.7^\circ$')):
        npz = f'{PAPER}/figs/spheroid_maps_{tag}.npz'
        base = ('a10' if tag == 'L2' else 'a29p7')
        cp_waterfall(npz, f'spheroid_stations_cp_{base}',
                     f'{ttl}, $Re_L=1.5\\times10^6$ (L2): '
                     '$C_p$ at the DFVLR pressure stations')
        cf_gamma_waterfall(npz, f'spheroid_stations_cfgamma_{base}',
                           f'{ttl}, $Re_L=1.5\\times10^6$ (L2): wall shear '
                           'at the DFVLR hot-film stations')
