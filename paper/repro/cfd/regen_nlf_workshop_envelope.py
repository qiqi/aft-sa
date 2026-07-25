"""fig:nlfworkshop -> paper/figs/nlf_workshop_envelope.pdf.

The 1st AIAA Transition Modeling & Prediction Workshop context figure:
transition location vs incidence for the NLF(1)-0416 (Case 2B alpha
sweep), upper and lower surfaces, showing
  - the min-max ENVELOPE of all ~14 workshop submittals (digitized from
    Coder's NAS-2021 workshop-summary deck p. 22; raster, lossless --
    data/workshop_nlf_envelope.json),
  - the experiment circles recoverable from under the letter clusters,
  - Piotrowski & Zingg's SA-LM2015 (fine grid; their alpha sweep is
    -6..6 step 2, vector-digitized from their Fig 6(f)),
  - SA-AI on all six grids at alpha = 0, 4 (and 9 upper) plus the
    finest-grid negative pair at -4 (markers as fig:nlfaft).
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_H = os.path.dirname(os.path.abspath(__file__))
PD = os.path.abspath(os.path.join(_H, '..', '..'))
B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fr")

W = json.load(open(f'{PD}/data/workshop_nlf_envelope.json'))
camp = json.load(open(f'{B}/sphere_campaign_nlf_results.json'))

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, side, key in ((axs[0], 'upper', 'xtr_up'), (axs[1], 'lower', 'xtr_lo')):
    env = W['envelope'][side]
    a = np.asarray(env['alpha'], float)
    ax.fill_between(a, env['xtr_min'], env['xtr_max'], color='0.5', alpha=0.22,
                    lw=0)
    ax.plot(a, env['xtr_min'], color='0.4', lw=0.7, alpha=0.6)
    ax.plot(a, env['xtr_max'], color='0.4', lw=0.7, alpha=0.6)
    ex = W['experiment'][side]
    ax.plot(ex['alpha'], ex['xtr'], 'o', mfc='none', mec='k', ms=6, mew=1.2,
            zorder=6)
    # Piotrowski SA-LM2015 fine: their sweep is alpha=-6..6 step 2
    pz = W['piotrowski'][f'sa_lm2015_{side}']
    apz = np.arange(-6, 6.1, 2)[:len(pz['xtr'])]
    ax.plot(apz, pz['xtr'], '-', color='C2', lw=1.2, marker='x', ms=4)
    # SA-AI: six grids at 0/4(/9 upper), finest grids at -4
    for fam, c, mk in (('str', 'C0', 's'), ('cav', 'C1', '^')):
        for lv, ms, mew, al in ((0, 3.5, 0.8, 0.5), (1, 5.2, 1.1, 0.75),
                                (2, 7.0, 1.6, 1.0)):
            A, X = [], []
            alphas = [0, 4]   # 9 deg lies beyond the envelope's alpha domain (-4..8)
            keys = [(f'{fam}L{lv}prop_nlf0416_Re4M_a{q}', q) for q in alphas]
            if lv == 2:
                keys.append((f'{fam}L2prop_nlf0416_Re4M_am4', -4))
            for k, q in keys:
                r = camp.get(k)
                if r is None or r.get(key) is None:
                    continue
                A.append(q); X.append(r[key])
            if A:
                ax.plot(A, X, mk, color=c, ms=ms, mfc='none', mew=mew,
                        alpha=al, zorder=5)
    ax.set_xlabel(r'$\alpha$ (deg)'); ax.set_xlim(-6.5, 8.5)
    ax.set_ylim(0, 1.0 if side == 'upper' else 0.8)
    ax.set_ylabel(f'$x_t/c$ ({side} surface)')
    ax.grid(alpha=0.3)
handles = [plt.Rectangle((0, 0), 1, 1, fc='0.5', alpha=0.22,
                         label='workshop submittals (min–max of ~14)'),
           Line2D([], [], color='k', ls='none', marker='o', mfc='none', ms=6,
                  label='LTPT experiment'),
           Line2D([], [], color='C2', ls='-', lw=1.2, marker='x', ms=4,
                  label='SA-LM2015 (Piotrowski–Zingg, fine)'),
           Line2D([], [], color='C0', ls='none', marker='s', mfc='none', ms=5.5,
                  label='SA-AI, O-grid (L0$\\to$L2 by size)'),
           Line2D([], [], color='C1', ls='none', marker='^', mfc='none', ms=5.5,
                  label='SA-AI, unstructured (L0$\\to$L2 by size)')]
axs[0].legend(handles=handles, fontsize=7.5, loc='lower left')
plt.tight_layout()
out = f'{PD}/figs/nlf_workshop_envelope.pdf'
plt.savefig(out); plt.savefig('/tmp/nlf_workshop.png', dpi=130)
print('wrote', out)
