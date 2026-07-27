"""Station-waterfall profiles for the 6:1 spheroid in the presentation of
Stock 2006 Figs. 2-5 (the DFVLR measurement stations): C_p(phi) at the 13
pressure stations (each curve displaced +0.42 in -C_p) and total skin
friction C_ft(x1e3) / wall-shear direction gamma(phi) at the 7 hot-film
stations (displaced -1.5 and -15 deg per station). Computed curves from the
surface-map .npz dumps; filled squares mark where the model's chi=c_v1
front crosses each station (Stock's transition-location marks). Open red
circles are the DFVLR measured chains digitized from Stock's printed
figures (data/stock2006_fig{2,3,4,5}_digitized.json, emitted by
digitize_stock_waterfalls.py; chains truncated where the printed symbol
tangle makes identity unrecoverable), subsampled to ~2 deg for plotting
and drawn at the same per-station displacement as the computed curve.

Run from paper/: python3 repro/cfd/regen_spheroid_station_profiles.py
"""
import json
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
# per-incidence conventions, READ FROM STOCK'S PRINTED FIGURES (pass 43):
# - Fig 5 (29.7 deg) displaces by -3.0/-20 deg and its hot-film station 5
#   is X/a = 0.130 (the array moved between runs), unlike Fig 4's 0.304;
# - the Cp displacement STEP is 0.42 per station in BOTH printed
#   figures (pass 43, printed axes); the printed families additionally
#   carry a +0.14 base displacement (the value Stock's text states),
#   which the digitizer removes -- see the offset derivation against
#   exact potential theory in digitize_stock_waterfalls.py.  Our own
#   waterfalls use the plain 0.42*j displacement with no base.
DCP = 0.42
CASES = {
    'a10':   dict(hf=[-0.894, -0.722, -0.382, -0.040, 0.304, 0.650, 0.766],
                  dcf=-1.5, dgam=-15.0,
                  meas_cp='stock2006_fig2_digitized.json',
                  meas_cfg='stock2006_fig4_digitized.json'),
    'a29p7': dict(hf=[-0.894, -0.722, -0.382, -0.040, 0.130, 0.650, 0.766],
                  dcf=-3.0, dgam=-20.0,
                  meas_cp='stock2006_fig3_digitized.json',
                  meas_cfg='stock2006_fig5_digitized.json'),
}
MEAS_KW = dict(marker='o', ls='none', ms=2.6, mew=0.7, mfc='none',
               color='red')


def load_meas(fname):
    return json.load(open(f'{PAPER}/data/{fname}'))['stations']


def meas_chain(stations, panel, Xa, dphi=2.0, tol=0.006):
    """Digitized (phi, value) for the station matching Xa, subsampled
    so plotted symbols sit ~dphi apart (the tracker emits per-column
    points, far denser than the printed symbols)."""
    for k, s in stations.items():
        if k.startswith(panel + '_') and abs(s['station'] - Xa) < tol:
            phi, val, last = [], [], -1e9
            for p, v in zip(s['phi'], s['value']):
                if p - last >= dphi:
                    phi.append(p); val.append(v); last = p
            return phi, np.array(val)
    return [], np.array([])


def load(npz):
    d = np.load(npz)
    return d['xl'], d['phi_deg'], d


def station_curve(xl, field2d, Xa):
    xs = (Xa + 1.0) / 2.0
    j = np.searchsorted(xl, xs)
    j = np.clip(j, 1, len(xl) - 1)
    # no extrapolation: the first pressure station (X/a=-0.9988 ->
    # x/L=0.0006) sits below the probe grid start (0.004) and is read at
    # the grid edge (stated in the caption)
    f = np.clip((xs - xl[j-1]) / (xl[j] - xl[j-1]), 0.0, 1.0)
    return field2d[:, j-1] * (1 - f) + field2d[:, j] * f


def front_phi(xl, ph, chimax, Xa):
    """ALL phi crossings of the chi=c_v1 front x(phi) with this station
    (right for wavy fronts); fronts at the first x-index and crossings
    adjacent to NaN spans are skipped -- none occur on the current maps."""
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


def cp_waterfall(npz, out, meas):
    xl, ph, d = load(npz)
    if 'cp' not in d:
        raise SystemExit(f'{npz} lacks cp -- re-probe with the updated '
                         'surface_map.py first')
    fig, ax = plt.subplots(figsize=(6.4, 8.4))
    for j, Xa in enumerate(CP_STATIONS):
        c = station_curve(xl, d['cp'], Xa)
        ax.plot(ph, -c + DCP * j, '-', color='k', lw=0.8)
        pm, vm = meas_chain(meas, 'cp', Xa)
        ax.plot(pm, vm + DCP * j, **MEAS_KW)   # digitized value is -Cp
        ax.annotate(f'$X/a={Xa:g}$', (181, (-c + DCP * j)[-1]), fontsize=6.5,
                    va='center')
    ax.plot([], [], '-', color='k', lw=0.8, label='SA-AI (L2)')
    ax.plot([], [], label='measured (DFVLR)', **MEAS_KW)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.set_xlim(0, 215)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_xlabel(r'$\phi$ [deg]')
    ax.set_ylabel(rf'$-C_p$ (curves displaced by $+{DCP:g}$ per station)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{PAPER}/figs/{out}.pdf')
    plt.close(fig)
    print('wrote', out)


def cf_gamma_waterfall(npz, out, hf, dcf, dgam, meas):
    xl, ph, d = load(npz)
    fig, axs = plt.subplots(2, 1, figsize=(6.8, 9.2), sharex=True)
    for j, Xa in enumerate(hf):
        cf = station_curve(xl, d['cf'], Xa) * 1e3
        gm = station_curve(xl, d['gamma_w'], Xa)
        axs[0].plot(ph, cf + dcf * j, '-', color='k', lw=0.8)
        axs[1].plot(ph, gm + dgam * j, '-', color='k', lw=0.8)
        pm, vm = meas_chain(meas, 'cft', Xa)
        axs[0].plot(pm, vm + dcf * j, **MEAS_KW)
        pm, vm = meas_chain(meas, 'gam', Xa)
        axs[1].plot(pm, vm + dgam * j, **MEAS_KW)
        axs[0].annotate(f'$X/a={Xa:g}$', (181, (cf + dcf * j)[-1]),
                        fontsize=6.5, va='center')
        axs[1].annotate(f'$X/a={Xa:g}$', (181, (gm + dgam * j)[-1]),
                        fontsize=6.5, va='center')
        for pstar in front_phi(xl, ph, d['chimax'], Xa):
            ci = np.interp(pstar, ph, cf)
            gi = np.interp(pstar, ph, gm)
            axs[0].plot(pstar, ci + dcf * j, 's', color='k', ms=5)
            axs[1].plot(pstar, gi + dgam * j, 's', color='k', ms=5)
    axs[0].plot([], [], '-', color='k', lw=0.8, label='SA-AI (L2)')
    axs[0].plot([], [], label='measured (DFVLR)', **MEAS_KW)
    axs[0].legend(fontsize=8, loc='upper right', framealpha=0.9)
    axs[0].set_ylabel(rf'$C_{{ft}}\times10^3$ (displaced ${dcf:g}$ per station)')
    axs[1].set_ylabel(rf'$\gamma_w$ [deg] (displaced ${dgam:g}^\circ$ per station)')
    axs[1].set_xlabel(r'$\phi$ [deg]')
    axs[1].set_xlim(0, 215)
    axs[1].set_xticks([0, 30, 60, 90, 120, 150, 180])
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{PAPER}/figs/{out}.pdf')
    plt.close(fig)
    print('wrote', out)


def re_label(npz):
    d = np.load(npz)
    re_l = float(d['mach']) / float(d['muref'])
    return f'{re_l/10**int(np.log10(re_l)):.3g}\\times10^{int(np.log10(re_l))}'


if __name__ == '__main__':
    for tag, base in (('L2', 'a10'), ('a29p7_L2', 'a29p7')):
        npz = f'{PAPER}/figs/spheroid_maps_{tag}.npz'
        c = CASES[base]
        cp_waterfall(npz, f'spheroid_stations_cp_{base}',
                     load_meas(c['meas_cp']))
        cf_gamma_waterfall(npz, f'spheroid_stations_cfgamma_{base}',
                           c['hf'], c['dcf'], c['dgam'],
                           load_meas(c['meas_cfg']))
