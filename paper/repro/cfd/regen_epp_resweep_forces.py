"""fig:eppresweepforces -> paper/figs/eppler_resweep_forces.pdf.

Eppler-387 alpha=5 Reynolds-sweep section forces across the full
L0/L1/L2 suite (24 solutions): c_l (top) and c_d (bottom,
log) versus Re (log), with the paper's established conventions --
structured O-grid solid blue / unstructured cavity dashed orange, L0/L1/L2
by geometric line thickness (POLAR_LW), the e^9 panel reference dotted
gray (mfoil squares; XFOIL diamonds at 6e4 and 4.6e5 per the table
footnotes), and the LTPT measurement in black with its documented repeat
scatter (+-0.006 c_l / +-0.0003 c_d at Re>=1e5; an order of magnitude
larger at the bistable 6e4).

The grid-convergence fan at each Re is the point: it closes with
refinement at 2-4.6e5 and OPENS at 1e5, where the computation sits past
the model's bursting boundary and refinement drives both families away
from the measurement.

Run from paper/: python3 repro/cfd/regen_epp_resweep_forces.py
"""
import os, json, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fr")
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(_HERE, '..', '..', 'figs'))
PREV = os.path.abspath(os.path.join(_HERE, '..', 'analytic', 'figs_explore'))
POLAR_LW = {'L0': 0.8, 'L1': 1.6, 'L2': 3.2}
FAM = {'str': dict(color='C0', ls='-'), 'cav': dict(color='C1', ls='--')}
RES = [60, 100, 200, 300, 460]

EXP = {60: (0.838, 0.0439), 100: (0.873, 0.0237), 200: (0.891, 0.0138),
       300: (0.901, 0.0114), 460: (0.914, 0.0093)}
# repeat scatter: +-0.006/0.0003 at >=1e5 (tab:eppresweep caption); at the
# bistable 60k, the half-spread of the three consecutive alpha=4.00 repeats
# (cl 0.643/0.697/0.721, cd 0.0431/0.0386/0.0400)
EXP_ERR = {Rk: (0.006, 0.0003) if Rk >= 100 else (0.039, 0.0023) for Rk in RES}

bench = json.load(open(f'{B}/sphere_campaign_eppler_results.json'))
swp = json.load(open(f'{B}/sphere_campaign_epp_sweep_results.json'))
lvl = json.load(open(f'{B}/sphere_campaign_epp_sweep_levels_results.json'))

def case(fam, L, Rk):
    if Rk == 200:
        return bench[f'{fam}{L}prop_eppler387_Re200k_a5']
    if L == 'L1':
        return swp[f'sweep_Re{Rk}k_a5' if fam == 'cav' else f'sweep_str_Re{Rk}k_a5']
    return lvl[f'sweep_{fam}{L}_Re{Rk}k_a5']

mf = pickle.load(open(f'{B}/mfoil_eppler387_sweep_a5.pkl', 'rb'))
xf = pickle.load(open(f'{B}/xfoil_eppler387_sweep_a5.pkl', 'rb'))

fig, (axl, axd) = plt.subplots(2, 1, figsize=(6.6, 7.4), sharex=True)
Re = np.array(RES, float)*1e3

for fam in ('str', 'cav'):
    for L in ('L0', 'L1', 'L2'):
        cl = [case(fam, L, Rk)['CL'] for Rk in RES]
        cd = [case(fam, L, Rk)['CD'] for Rk in RES]
        kw = dict(color=FAM[fam]['color'], ls=FAM[fam]['ls'], lw=POLAR_LW[L],
                  marker='o' if fam == 'str' else '^', ms=3.5)
        axl.semilogx(Re, cl, **kw)
        axd.loglog(Re, cd, **kw)

# e^9 reference: mfoil where converged/reliable (100-300k), XFOIL at 60k/460k
# mfoil quoted at 100-300k (300k near-converged, as the table footnote
# documents); XFOIL at the 60k/460k ends
mre = [100, 200, 300]
axl.semilogx([r*1e3 for r in mre], [mf[r]['cl'] for r in mre], ':', color='0.45',
             marker='s', mfc='none', ms=6, lw=1.3)
axd.loglog([r*1e3 for r in mre], [mf[r]['cd'] for r in mre], ':', color='0.45',
           marker='s', mfc='none', ms=6, lw=1.3)
xre = [Rk for Rk in (60, 460) if Rk in xf]
axl.semilogx([r*1e3 for r in xre], [xf[r]['cl'] for r in xre], ls='none',
             color='0.45', marker='D', mfc='none', ms=6, mew=1.3)
axd.loglog([r*1e3 for r in xre], [xf[r]['cd'] for r in xre], ls='none',
           color='0.45', marker='D', mfc='none', ms=6, mew=1.3)

# experiment with repeat scatter
axl.errorbar(Re, [EXP[r][0] for r in RES], yerr=[EXP_ERR[r][0] for r in RES],
             fmt='o', color='k', ms=5, capsize=3, zorder=6)
axd.errorbar(Re, [EXP[r][1] for r in RES], yerr=[EXP_ERR[r][1] for r in RES],
             fmt='o', color='k', ms=5, capsize=3, zorder=6)

axl.set_ylabel('$c_l$'); axd.set_ylabel('$c_d$'); axd.set_xlabel('$Re$')
axl.grid(alpha=0.3, which='both'); axd.grid(alpha=0.3, which='both')
axl.set_xlim(4.5e4, 5.6e5); axd.set_xlim(4.5e4, 5.6e5)
axl.set_ylim(0.55, 1.02)
import matplotlib.ticker as mticker
for ax in (axl, axd):
    ax.xaxis.set_major_locator(mticker.FixedLocator([6e4, 1e5, 2e5, 3e5, 4.6e5]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(
        ['$0.6$', '$1$', '$2$', '$3$', '$4.6$']))
axd.set_xlabel(r'$Re\ (\times 10^5)$')
axd.annotate('bursting boundary', xy=(1.0e5, 0.034), xytext=(1.6e5, 0.042),
             fontsize=8.5, color='0.3',
             arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8))
handles = [Line2D([], [], color='k', marker='o', ls='none', ms=5,
                  label='Experiment (LTPT, repeat scatter)'),
           Line2D([], [], color='0.45', ls=':', marker='s', mfc='none', ms=6,
                  label='mfoil ($e^9$)'),
           Line2D([], [], color='0.45', ls='none', marker='D', mfc='none',
                  ms=6, mew=1.3, label='XFOIL ($e^9$, $6{\\times}10^4$, $4.6{\\times}10^5$)'),
           Line2D([], [], color='C0', ls='-', marker='o', ms=3.5,
                  label='SA-AI, structured (O-grid)'),
           Line2D([], [], color='C1', ls='--', marker='^', ms=3.5,
                  label='SA-AI, unstructured'),
           Line2D([], [], color='0.4', lw=POLAR_LW['L0'], label='L0'),
           Line2D([], [], color='0.4', lw=POLAR_LW['L1'], label='L1'),
           Line2D([], [], color='0.4', lw=POLAR_LW['L2'], label='L2')]
fig.legend(handles=handles, fontsize=8, ncol=3, frameon=False,
           loc='lower center', bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(rect=(0, 0.08, 1, 1))
os.makedirs(PREV, exist_ok=True)
plt.savefig(f'{OUT}/eppler_resweep_forces.pdf')
plt.savefig(f'{PREV}/epp_resweep_forces.png', dpi=140)
print('wrote', f'{OUT}/eppler_resweep_forces.pdf')
