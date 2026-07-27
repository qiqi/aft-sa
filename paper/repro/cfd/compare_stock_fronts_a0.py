"""alpha=0 pure-TS control comparison: computed spheroid transition
fronts (Re_L=7.2e6, alpha=0, ladder L0-L2) against Stock 2006 Fig. 14a
-- BOTH the measured front (DFVLR hot films, Re=7.20e6, an exact
condition match) and Stock's computed pure-TS e^N front (N_TS=8.0, his
DFVLR-tunnel limit). Digitized inputs:
paper/data/stock2006_fig14a_digitized.json (digitize_stock_fig14a.py).

This is the control for fig:spheroidfront's mechanism attribution: at
alpha=0 the flow is axisymmetric -- no crossflow, no free-vortex
separation -- so the front miss here is attributable to the TS
amplification channel alone.

Front definitions per compare_stock_fronts.py (same code): chi=1 and
chi=c_v1 crossings of the wall-normal max chi; cf-rise front (first
x/L past the nose where cf exceeds k x its running minimum, k=1.5,
sensitivity band k=1.25-2). At alpha=0 every front is
azimuth-independent; the table reports the median over phi and the
max-min spread.

SEED NOTE (verified from the run artifacts, 2026-07-27): the spheroid
campaign runs carry Flow360.json modifiedTurbulentViscosityRatio =
8.76e-6 and ai_laminarSlowdown = 1.0 (per the campaign ai_constants
convention, cf. daedalus/ai_constants/*.log = same launcher), so the
PHYSICAL delivered seed is chi_inf = 8.76e-6 (N_crit ~ 13.6, the
flight-quiet Daedalus level; ambient chi probed in all three re72a0
volume fields = 8.76e-6 to three digits) -- NOT the airfoil-study
8.76e-4. The wind tunnel itself calibrates to N_TS = 8.0
(chi_inf ~ 2.4e-3). To separate seed from amplification, the script
also reports LINEAR-REMAP front estimates: below the blend the
amplification is linear in nuHat and the laminar field upstream of the
as-run front is seed-independent, so the chi=1 front at a physical
seed s sits where the as-run envelope crosses chi = 8.76e-6/s
(estimates, not re-runs).

Outputs: paper/figs/spheroid_front_compare_a0.pdf/png (overlay in the
spheroid_front_compare style, no in-figure title) and
paper/data/spheroid_front_summary_a0.json.

Run from paper/: python3 repro/cfd/compare_stock_fronts_a0.py
(inputs: figs/spheroid_maps_re72a0_{L0,L1,L2}.npz, built by
spheroid/surface_map.py from spheroid_fv1/case_ogrid_L*_saai_re72a0)
"""
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = '/home/qiqi/flexcompute/sa-ai/paper'
CV1 = 7.1
CHI_RUN = 8.76e-6                 # physical delivered seed, as run
SEEDS = {'as_run_N13.6': 8.76e-6, 'airfoil_N9.0': 8.76e-4,
         'tunnel_N8.0': 2.38e-3}  # c_v1 e^-N


def level_front(xl, row, level):
    """First sub-cell x/L where `row` crosses `level` from below."""
    hits = np.where(np.isfinite(row) & (row > level))[0]
    if len(hits) and hits[0] > 0:
        j = hits[0]
        f = (level - row[j-1]) / (row[j] - row[j-1])
        return xl[j-1] + f * (xl[j] - xl[j-1])
    return np.nan


def cf_front(xl, cf_row, k):
    """First sub-cell-interpolated x/L (x>0.2) where cf exceeds k x its
    running minimum (identical to compare_stock_fronts.py)."""
    runmin = np.inf
    prev_ex = None
    for j in range(len(xl)):
        v = cf_row[j]
        if not np.isfinite(v) or xl[j] < 0.05:
            continue
        runmin = min(runmin, v)
        ex = v - k * runmin
        if xl[j] > 0.2 and ex > 0 and prev_ex is not None and prev_ex[1] <= 0:
            xj_1, e0 = prev_ex
            f = -e0 / (ex - e0)
            return xj_1 + f * (xl[j] - xj_1)
        prev_ex = (xl[j], ex)
    return np.nan


D = json.load(open(f'{PAPER}/data/stock2006_fig14a_digitized.json'))
SQ = D['measured_squares']
TS = D['computed_ts_front']

res, summary = {}, {'levels': {}}
for lev in ('L0', 'L1', 'L2'):
    d = np.load(f'{PAPER}/figs/spheroid_maps_re72a0_{lev}.npz')
    xl, ph, chi, cf = d['xl'], d['phi_deg'], d['chimax'], d['cf']
    n = len(ph)
    f_chi1 = np.array([level_front(xl, chi[i], 1.0) for i in range(n)])
    f_cv1 = np.array([level_front(xl, chi[i], CV1) for i in range(n)])
    fcf = {k: np.array([cf_front(xl, cf[i], k) for i in range(n)])
           for k in (1.25, 1.5, 2.0)}
    remap = {}
    for name, s in SEEDS.items():
        lvl = CHI_RUN / s
        remap[name] = np.array([level_front(xl, chi[i], lvl)
                                for i in range(n)])
    res[lev] = (xl, ph, f_chi1, f_cv1, fcf, cf)
    ent = {}
    for name, arr in [('cf_k1.25', fcf[1.25]), ('cf_k1.5', fcf[1.5]),
                      ('cf_k2.0', fcf[2.0]), ('chi1', f_chi1),
                      ('cv1', f_cv1)] + [(f'chi1_remap_{k}', v)
                                         for k, v in remap.items()]:
        ent[name] = dict(median=round(float(np.nanmedian(arr)), 4),
                         spread=round(float(np.nanmax(arr)
                                            - np.nanmin(arr)), 4))
    summary['levels'][lev] = ent
    print(f"{lev}: cf-rise k=1.5 {ent['cf_k1.5']['median']:.3f} "
          f"(band {ent['cf_k1.25']['median']:.3f}-"
          f"{ent['cf_k2.0']['median']:.3f}, phi-spread "
          f"{ent['cf_k1.5']['spread']:.3f}) | chi1 "
          f"{ent['chi1']['median']:.3f} cv1 {ent['cv1']['median']:.3f}"
          f" | remap N9 {ent['chi1_remap_airfoil_N9.0']['median']:.3f}"
          f" N8 {ent['chi1_remap_tunnel_N8.0']['median']:.3f}")

meas = [q['xL'] for q in SQ]
summary.update(
    condition='Re_L=7.2e6, alpha=0 (measured 7.20e6 -- exact match)',
    seed_note='physical delivered seed chi_inf=8.76e-6 (N~13.6); '
              'chi1_remap_* = linear-amplification front estimates at '
              'the named physical seeds (see script docstring)',
    measured_xL=dict(mean=round(float(np.mean(meas)), 4),
                     std=round(float(np.std(meas)), 4), n=len(meas)),
    stock_eN_xL=dict(mean=TS['mean_xL'], std=TS['std_xL'],
                     N_TS=8.0),
    criterion='cf-rise k=1.5 x running min, band k=1.25-2; '
              'chi=1 / chi=c_v1 crossings of wall-normal max chi')
json.dump(summary, open(f'{PAPER}/data/spheroid_front_summary_a0.json',
                        'w'), indent=1)

# ---- overlay figure (spheroid_front_compare style + Cf panel) ---------------
# Top: the (x/L, phi) front plane of fig:spheroidfront (proves the
# computed azimuth-independence and the grid trend). Bottom: the
# meridional c_f itself -- the hot-film detection quantity -- at
# phi=90 per level: the measured front sits where the experiment's
# shear rises; the model's c_f keeps decaying past it.
xl, ph, f_chi1, f_cv1, fcf, cf = res['L2']
fig, (ax, ax2) = plt.subplots(
    2, 1, figsize=(9.6, 7.4), sharex=True,
    gridspec_kw=dict(height_ratios=[1.35, 1.0]), constrained_layout=True)
cs = ax.contour(xl, ph, cf * 1e3, levels=np.arange(0, 6.51, 0.5),
                colors='0.65', linewidths=0.5)
ax.clabel(cs, levels=np.arange(0, 6.51, 1.0), fmt='%g', fontsize=6.5)
ax.fill_betweenx(ph, fcf[1.25], fcf[2.0], color='C0', alpha=0.18, lw=0,
                 label=r'L2 criterion band ($k=1.25$--$2$)')
for lev, ls in (('L0', ':'), ('L1', '--'), ('L2', '-')):
    _, phL, _, _, fcfL, _ = res[lev]
    ax.plot(fcfL[1.5], phL, ls, color='C0', lw=1.4,
            label=f'{lev} $C_f$-rise front ($k=1.5$)')
ax.plot(f_chi1, ph, ':', color='C4', lw=1.3, label=r'L2 $\chi=1$ front')
ax.plot(f_cv1, ph, '-', color='C4', lw=1.3,
        label=r'L2 $\chi=c_{v1}$ front')
ax.plot(TS['xL'], TS['phi_deg'], '-.', color='k', lw=1.3,
        label=r'pure-TS $e^N$ front (Stock, computed, $N_{TS}=8$)')
ax.plot([q['xL'] for q in SQ], [q['phi_deg'] for q in SQ], 's',
        color='red', mfc='none', ms=9, mew=2,
        label='measured (Stock Fig. 14a, DFVLR)')
ax.set_ylabel(r'$\phi$ [deg]  (0 = windward)')
ax.set_ylim(0, 180)
ax.set_xlim(0, 1)
ax.legend(fontsize=8, loc='center left', framealpha=0.9)

# ---- bottom panel: meridional c_f (the hot-film quantity), phi=90 ----------
for lev, ls in (('L0', ':'), ('L1', '--'), ('L2', '-')):
    xlL, phL, _, _, _, cfL = res[lev]
    i90 = np.argmin(np.abs(phL - 90.0))
    ax2.plot(xlL, cfL[i90] * 1e3, ls, color='C0', lw=1.4,
             label=fr'{lev} $c_f$ ($\phi=90^\circ$)')
ax2.axvline(float(np.mean(meas)), color='red', lw=1.6)
ax2.annotate('measured front', xy=(float(np.mean(meas)), 2.55),
             xytext=(6, 0), textcoords='offset points',
             color='red', fontsize=8)
ax2.axvline(TS['mean_xL'], color='k', lw=1.2, ls='-.')
ax2.annotate(r'$e^N$ ($N_{TS}=8$)', xy=(TS['mean_xL'], 3.05),
             xytext=(-6, 0), textcoords='offset points', ha='right',
             color='k', fontsize=8)
ax2.set_xlabel('$x/L$')
ax2.set_ylabel(r'$c_f \times 10^3$')
ax2.set_ylim(0, 3.6)
ax2.legend(fontsize=8, loc='upper right', framealpha=0.9)
# no in-figure title (paper-wide rule); the caption carries the condition
fig.savefig(f'{PAPER}/figs/spheroid_front_compare_a0.pdf')
fig.savefig(f'{PAPER}/figs/spheroid_front_compare_a0.png', dpi=140)
print('wrote spheroid_front_compare_a0.pdf/png + summary json')
