"""fig:nlfaft -> paper/figs/nlf_aft_transition.pdf.

NLF(1)-0416 at Re=4e6: transition location x_t/c versus c_l, upper (left)
and lower (right) surfaces -- the LTPT microphone measurements, the AFT
model's OVERFLOW sweeps from Coder's dissertation at its nominal tunnel
calibration (Ncrit=10.07, Tu=0.045%) and at the recalibrated Ncrit=7.18
(Tu=0.15%) chosen there to close the transition-location discrepancy, the
XFOIL e^9 reference from the same source, and SA-AI's four incidences on
both L2 mesh families (untuned N=9-class seed chi_inf = c_v1 e^-9).

AFT/XFOIL/experiment curves are vector-exact extractions from the
dissertation PDF (repro: data/aft_nlf0416_digitized.json).
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_H = os.path.dirname(os.path.abspath(__file__))
PD = os.path.abspath(os.path.join(_H, '..', '..'))
B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fr")

D = json.load(open(f'{PD}/data/aft_nlf0416_digitized.json'))['transition']
camp = json.load(open(f'{B}/sphere_campaign_nlf_results.json'))

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
for ax, side, lab in ((axs[0], 'upper', 'upper surface'),
                      (axs[1], 'lower', 'lower surface')):
    e = D[f'exp_{side}']
    ax.plot(e['xt'], e['cl'], 'o', mfc='none', mec='k', ms=6, mew=1.2,
            label='LTPT experiment', zorder=6)
    a10 = D[f'aft_ncrit10_{side}']
    ax.plot(a10['xt'], a10['cl'], '--', color='0.45', lw=1.4,
            label='AFT, $N_\\mathrm{crit}=10.07$ (nominal)')
    a7 = D[f'aft_ncrit718_{side}']
    ax.plot(a7['xt'], a7['cl'], '-', color='0.45', lw=1.4,
            label='AFT, $N_\\mathrm{crit}=7.18$ (recalibrated)')
    xf = D[f'xfoil_{side}']
    ax.plot(xf['xt'], xf['cl'], ':', color='0.6', lw=1.2, label='XFOIL ($e^9$)')
    for fam, c, mk in (('str', 'C0', 's'), ('cav', 'C1', '^')):
        cls, xts = [], []
        for a in (0, 4, 9, 15):
            r = camp[f'{fam}L2prop_nlf0416_Re4M_a{a}']
            cls.append(r['CL']); xts.append(r['xtr_up' if side == 'upper' else 'xtr_lo'])
        ax.plot(xts, cls, mk, color=c, ms=7, mfc='none', mew=1.6,
                label=f"SA-AI L2, {'O-grid' if fam=='str' else 'unstructured'}",
                zorder=5)
    ax.set_xlabel('$x_t/c$'); ax.set_title(lab, fontsize=10)
    ax.grid(alpha=0.3); ax.set_xlim(0, 0.75)
axs[0].set_ylabel('$c_l$'); axs[0].set_ylim(-0.6, 2.1)
axs[0].legend(fontsize=7.5, loc='upper right')
plt.tight_layout()
out = f'{PD}/figs/nlf_aft_transition.pdf'
plt.savefig(out); plt.savefig('/tmp/nlf_aft.png', dpi=130)
print('wrote', out)
