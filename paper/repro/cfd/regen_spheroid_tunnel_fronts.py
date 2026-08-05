"""Three-way spheroid front comparison: SA-AI vs Stock's e^N vs measurement.

-> paper/figs/spheroid_tunnel_fronts.pdf

One panel per condition, in the order of tab:sphtunnel (incidence, then
Reynolds number, Goettingen before ONERA).  Each panel carries all three legs:

  measurement      DFVLR hot films, open red squares
  Stock's e^N      grey line, LINESTYLE CODED BY MECHANISM as printed
                   (solid TS / dashed TS+CF / long-dash CF / dotted ambiguous)
  SA-AI            blue, solid at the facility-calibrated seed and dashed at
                   the facility-measured hot-wire seed

Identity is never carried by colour alone: the measurement is the only marked
series, Stock is the only grey one, and the two SA-AI seeds differ in dash as
well.  The three hues were checked for CVD separation before use -- red
#D62728 / blue #1F4E9C / grey #7F7F7F give OKLab dE (x100) of 33.7 red-blue,
21.1 red-grey and 21.0 blue-grey under normal vision, and no pair falls below
16 under simulated deuteranopia or protanopia.  A near-black for Stock, which
is the obvious choice, FAILS: it collides with the red at dE 4.8 under
protanopia.

Run from paper/:  python3 repro/cfd/regen_spheroid_tunnel_fronts.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.lines import Line2D                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, '..', '..'))
HARV = os.path.join(HERE, 'figs_explore', 'spheroid_tunnel_harvest.json')
DATA = os.path.join(PAPER, 'data')
OUT = os.path.join(PAPER, 'figs', 'spheroid_tunnel_fronts.pdf')

C_MEAS, C_OURS, C_STOCK = '#D62728', '#1F4E9C', '#7F7F7F'
MECH_LS = {'TS': '-', 'TS+CF': (0, (5, 2)), 'CF': (0, (7, 3, 1, 3)),
           'ambiguous:CF?': (0, (1, 2)), 'ambiguous': (0, (1, 2))}

# condition -> (label, Stock file, Stock key)
PANELS = [
    # Reynolds-major: the low-Re band first (layer laminar to the open
    # free-vortex separation; Stock's "front" there IS the separation line),
    # then the high-Re band where transition is real.  Within each band the
    # walk is by incidence.
    ('a5lo',  r'$\alpha=5^\circ$, $Re_L=1.52\times10^6$',
     'stock2006_fig14c_digitized.json', 'front_re1p52e6'),
    ('a10lo', r'$\alpha=10^\circ$, $Re_L=1.52\times10^6$',
     'stock2006_fig15a_digitized.json', 'separation_line_short_dashed'),
    # Stock digitized no computed curve for Fig. 16c, so this panel carries the
    # measurement and SA-AI only.
    ('a29p7', r'$\alpha=29.7^\circ$, $Re_L=1.53\times10^6$',
     'stock2006_fig16c_digitized.json', None),
    ('a0',    r'$\alpha=0^\circ$, $Re_L=7.2\times10^6$',
     'stock2006_fig14a_digitized.json', 'computed_ts_front'),
    ('a2p5',  r'$\alpha=2.5^\circ$, $Re_L=7.2\times10^6$',
     'stock2006_fig14b_digitized.json', 'computed_ts_front'),
    ('a5hi',  r'$\alpha=5^\circ$, $Re_L=6.49\times10^6$',
     'stock2006_fig14c_digitized.json', 'front_re6p49e6'),
    ('a10hi', r'$\alpha=10^\circ$, $Re_L=6.56\times10^6$, Goettingen',
     'stock2006_fig15a_digitized.json', 'front_re6p56e6'),
    ('a10f1', r'$\alpha=10^\circ$, $Re_L=6.56\times10^6$, ONERA F1',
     'stock2006_fig17a_digitized.json', 'front_re6p56e6'),
]


def stock_segments(fn, key):
    """-> list of (mechanism, phi[], xL[]) so each style is drawn separately."""
    d = json.load(open(os.path.join(DATA, fn)))
    if key is None:
        return []
    if key == 'computed_ts_front':
        c = d[key]
        if not isinstance(c.get('phi_deg'), list):
            return []
        a = np.array(sorted(zip(c['phi_deg'], c['xL'])))
        return [('TS', a[:, 0], a[:, 1])]
    blk = d.get('stock_computed_lines', {}).get(key)
    if not blk:
        return []
    pts = sorted(((p['phi_deg'], p['xL']) for p in blk['points']))
    phi = np.array([p[0] for p in pts]); xl = np.array([p[1] for p in pts])
    segs = blk.get('mechanism_segments') or [{'mechanism': 'TS',
                                              'phi_from': phi.min(),
                                              'phi_to': phi.max()}]
    out = []
    for s in segs:
        lo, hi = sorted((s['phi_from'], s['phi_to']))
        m = (phi >= lo - 0.6) & (phi <= hi + 0.6)
        if m.sum() > 1:
            out.append((s['mechanism'], phi[m], xl[m]))
    return out


def main():
    H = json.load(open(HARV))
    fig, axes = plt.subplots(5, 2, figsize=(7.2, 11.4), sharex=True,
                             sharey=True)
    axes = axes.ravel()

    for ax, (cond, label, sfn, skey) in zip(axes, PANELS):
        cal = H.get(f'{cond}_gcal') or H.get(f'{cond}_fcal')
        mea = H.get(f'{cond}_gmeas') or H.get(f'{cond}_fmeas')

        for mech, phi, xl in stock_segments(sfn, skey):
            ax.plot(xl, phi, color=C_STOCK, lw=1.6,
                    ls=MECH_LS.get(mech, (0, (1, 2))), zorder=2)

        for rec, ls in ((cal, '-'), (mea, (0, (4, 2)))):
            if not rec or not rec.get('curve'):
                continue
            c = [(p['phi'], p['chi1']) for p in rec['curve']
                 if p['chi1'] is not None]
            if len(c) > 1:
                ax.plot([p[1] for p in c], [p[0] for p in c],
                        color=C_OURS, lw=1.8, ls=ls, zorder=3)

        ref = cal or mea
        if ref:
            ax.plot([r['x_meas'] for r in ref['rows']],
                    [r['phi'] for r in ref['rows']], 's', mfc='none',
                    mec=C_MEAS, mew=1.4, ms=6.0, zorder=4)

        ax.set_title(label, fontsize=8.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 180)
        ax.set_yticks([0, 45, 90, 135, 180])
        ax.grid(True, lw=0.4, color='0.88', zorder=0)
        ax.tick_params(labelsize=8)

    for ax in axes[len(PANELS):]:
        ax.axis('off')
    handles = [
        Line2D([], [], ls='none', marker='s', mfc='none', mec=C_MEAS, mew=1.4,
               ms=6, label='measured (DFVLR hot films)'),
        Line2D([], [], color=C_OURS, lw=1.8, label='SA-AI, calibrated seed'),
        Line2D([], [], color=C_OURS, lw=1.8, ls=(0, (4, 2)),
               label='SA-AI, measured seed'),
        Line2D([], [], color=C_STOCK, lw=1.6, label=r"Stock $e^N$: TS"),
        Line2D([], [], color=C_STOCK, lw=1.6, ls=(0, (5, 2)),
               label=r'Stock $e^N$: TS$+$CF'),
        Line2D([], [], color=C_STOCK, lw=1.6, ls=(0, (7, 3, 1, 3)),
               label=r'Stock $e^N$: CF'),
        Line2D([], [], color=C_STOCK, lw=1.6, ls=(0, (1, 2)),
               label=r'Stock $e^N$: style ambiguous'),
    ]
    axes[-1].legend(handles=handles, loc='center', fontsize=8, frameon=False,
                handlelength=3.4, labelspacing=0.9)

    for k in (len(PANELS) - 2, len(PANELS) - 1):
        axes[k].set_xlabel('$x/L$', fontsize=9)
        axes[k].tick_params(labelbottom=True, labelsize=8)
    for r in range(4):
        axes[2 * r].set_ylabel(r'$\phi$ [deg]', fontsize=9)
    fig.tight_layout(pad=0.6)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
