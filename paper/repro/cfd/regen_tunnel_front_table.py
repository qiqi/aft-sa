"""Emit the spheroid tunnel-condition front table from the harvest JSON.

-> paper/tables/tab_spheroid_tunnel.tex  (\\input by sa-ai.tex and whitepaper.tex)

Rows are ordered by incidence, then Reynolds number, then seed.  Columns carry
the comparison that matters: the mean and rms of (computed chi=1 front minus
measured front) over the azimuths where a measurement exists, and the signed
residual at the most leeward and most windward measured azimuth separately --
the leeward/windward split is the finding.

Azimuth convention throughout: phi = 0 windward, 180 leeward (Stock's and
spheroid/surface_map.py's `phi_lit`).  NOTE: the older
spheroid_reseed_harvest.offset_meridian_sweep docstring labels its argument as
MESH phi (0 leeward) while passing it to surface_frame as phi_lit (0 windward),
so its printed windward/leeward labels are interchanged; this table does not
use that helper.

Run from paper/:  python3 repro/cfd/regen_tunnel_front_table.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC = os.path.join(HERE, 'figs_explore', 'spheroid_tunnel_harvest.json')
OUT = os.path.join(PAPER, 'tables', 'tab_spheroid_tunnel.tex')
OUT_WP = os.path.join(PAPER, 'tables', 'tab_spheroid_tunnel_wp.tex')

FAC = {'NWG': r'G\"ottingen', 'F1': 'ONERA F1'}

# tag -> (digitized file, key of Stock's COMPUTED front for that condition).
# At Re 1.52e6 Stock's computed transition line IS the separation line -- the
# layer is laminar up to separation there -- which is why a10lo maps to it.
STOCK = {
    'a0':    ('stock2006_fig14a_digitized.json', 'computed_ts_front'),
    'a2p5':  ('stock2006_fig14b_digitized.json', 'computed_ts_front'),
    'a5lo':  ('stock2006_fig14c_digitized.json', 'front_re1p52e6'),
    'a5hi':  ('stock2006_fig14c_digitized.json', 'front_re6p49e6'),
    'a10lo': ('stock2006_fig15a_digitized.json', 'separation_line_short_dashed'),
    'a10hi': ('stock2006_fig15a_digitized.json', 'front_re6p56e6'),
    'a10f1': ('stock2006_fig17a_digitized.json', 'front_re6p56e6'),
}
DATA = os.path.join(PAPER, 'data')


def stock_front(tag):
    """-> (phi[], xL[]) of Stock's computed front, or None."""
    cond = tag.rsplit('_', 1)[0]
    if cond not in STOCK:
        return None
    fn, key = STOCK[cond]
    d = json.load(open(os.path.join(DATA, fn)))
    if key == 'computed_ts_front':                 # alpha = 0 / 2.5: TS only
        c = d[key]
        phi, xl = c['phi_deg'], c['xL']
        if not isinstance(phi, list):
            return None
        a = np.array(sorted(zip(phi, xl)))
        return a[:, 0], a[:, 1]
    blk = d.get('stock_computed_lines', {}).get(key)
    if not blk:
        return None
    a = np.array(sorted((p['phi_deg'], p['xL']) for p in blk['points']))
    return a[:, 0], a[:, 1]


def stock_resid(tag, rows):
    f = stock_front(tag)
    if f is None:
        return None, None
    phi_s, xl_s = f
    dd = []
    for r in rows:
        if not (phi_s.min() - 2 <= r['phi'] <= phi_s.max() + 2):
            continue
        dd.append(float(np.interp(r['phi'], phi_s, xl_s)) - r['x_meas'])
    if not dd:
        return None, None
    dd = np.array(dd)
    return float(dd.mean()), float(np.sqrt((dd ** 2).mean()))

SEED_LABEL = {'gcal': 'calibrated', 'gmeas': 'measured',
              'fcal': 'calibrated', 'fmeas': 'measured'}


def tu_of_n(n):
    import math
    return 100.0 * math.exp(-(n + 8.43) / 2.4)


def main():
    d = json.load(open(SRC))
    recs = sorted(d.values(), key=lambda r: (r['alpha'], r['Re'],
                                            r['facility'] != 'NWG', -r['chi_inf']))
    rows = []
    for r in recs:
        rw = [x for x in r['rows'] if x['chi1'] == x['chi1']]     # drop NaN
        if not rw:
            continue
        lee = max(rw, key=lambda x: x['phi'])
        wind = min(rw, key=lambda x: x['phi'])
        seed = r['tag'].rsplit('_', 1)[1]
        sm, sr = stock_resid(r['tag'], rw)
        stock_cols = (f"{sm:+.3f} & {sr:.3f}" if sm is not None else "-- & --")
        rows.append(
            f"${r['alpha']:g}^\\circ$ & ${r['Re']/1e6:g}$ & {FAC[r['facility']]} & "
            f"{SEED_LABEL[seed]} & {tu_of_n(r['n_crit']):.3f} & "
            f"{r['stat']['n']}/{len(r['rows'])} & "
            f"{r['stat']['mean']:+.3f} & {r['stat']['rms']:.3f} & "
            f"{lee['d_chi1']:+.3f} & {wind['d_chi1']:+.3f} & {stock_cols}")

    hdr = (r'$\alpha$ & $Re_L$ & tunnel & seed & $Tu$ & $n$ &'
           r' \multicolumn{4}{c}{SA-AI $-$ measured} &'
           r' \multicolumn{2}{c}{Stock $-$ measured} \\'
           '\n    & $[10^6]$ & & & [\\%] & & mean & rms & lee & wind & mean & rms')
    cap = (r'6:1 prolate spheroid at the tunnel conditions: computed '
           r'near-wall $\chi\!=\!1$ front minus the measured front, at every '
           r'azimuth carrying a measurement, on the L1 O-grid at the '
           r'measured Mach number. Each condition is run at the '
           r'facility-calibrated seed (the tunnel-calibrated limiting '
           r'$N_{TS}$) and at the '
           r'facility-measured hot-wire level. $\Delta$ lee / $\Delta$ wind '
           r'are the residuals at the most leeward and most windward measured '
           r'azimuth ($\phi\!=\!0$ windward). The $n$ column is the number of '
           r'measured azimuths at which the computed front was found, over '
           r'the number measured: where they differ the near-wall $\chi$ '
           r'never reaches $1$ within $x/L\!\le\!0.97$, i.e. the computed '
           r'layer is still laminar at the tail and the residual there is a '
           r'bound, not a value. The last two columns are the same residual '
           r'for Stock''s own two-N-factor $e^N$ computation, digitized from '
           r'the printed panels at the same azimuths; at '
           r'$Re_L\!=\!1.52\!\times\!10^6$ his computed transition line is '
           r'the laminar separation line.')
    for path, place in ((OUT, '[tp]'), (OUT_WP, '[H]')):
      with open(path, 'w') as f:
        f.write('\\begin{table}' + place + '\n  \\centering\\small\n'
                f'  \\caption{{{cap}}}\n'
                '  \\label{tab:sphtunnel}\n'
                '  \\begin{tabular}{cc l l c c cccc cc}\n    \\toprule\n'
                f'    {hdr} \\\\\n    \\midrule\n')
        for r in rows:
            f.write(f'    {r} \\\\\n')
        f.write('    \\bottomrule\n  \\end{tabular}\n\\end{table}\n')
      print('wrote', path, f'({len(rows)} rows)')


if __name__ == '__main__':
    main()
