"""fig:nlfaft -> paper/figs/nlf_aft_transition.pdf.

NLF(1)-0416 at Re=4e6: transition location x_t/c versus c_l, upper (left)
and lower (right) surfaces -- the LTPT microphone measurements, the AFT
model's OVERFLOW sweeps from Coder's dissertation at its nominal tunnel
calibration (Ncrit=10.07, Tu=0.045%) and at the recalibrated Ncrit=7.18
(Tu=0.15%) chosen there to close the transition-location discrepancy, the
XFOIL e^9 reference from the same source, and SA-AI's four incidences on
ALL SIX grids (both mesh families, L0-L2; marker size grows with
refinement) at the untuned N=9-class seed chi_inf = c_v1 e^-9.

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
# Langtry-Menter (OVERFLOW gamma-Re_theta, fine grid) at the same condition:
# raster-digitized from Denison et al.'s transition-workshop Fig 11
# (repro: data/lm_nlf0416_transition_digitized.json)
LM = json.load(open(f'{PD}/data/lm_nlf0416_transition_digitized.json'))['transition']
camp = json.load(open(f'{B}/sphere_campaign_nlf_results.json'))

# The full workshop-submittal underlay (all 14 participants, digitized from
# the summary deck -- data/workshop_nlf_submittals.json). The deck's sweep
# is xtr vs alpha; it is placed on this figure's lift axis via the computed
# structured-L2 lift curve (smooth and near-linear over alpha in [-4,8];
# a +-0.05 c_l placement error is imperceptible at underlay weight).
WS = json.load(open(f'{PD}/data/workshop_nlf_submittals.json'))
PZ = json.load(open(f'{PD}/data/workshop_nlf_envelope.json'))['piotrowski']
_amap = [(-8, 'am8'), (-4, 'am4'), (0, 'a0'), (4, 'a4'), (9, 'a9'), (15, 'a15')]
_aa = [a for a, t in _amap if f'strL2prop_nlf0416_Re4M_{t}' in camp]
_cc = [camp[f'strL2prop_nlf0416_Re4M_{t}']['CL'] for a, t in _amap
       if f'strL2prop_nlf0416_Re4M_{t}' in camp]


def cl_of_alpha(a):
    return np.interp(a, _aa, _cc)


def ws_style(fam):
    if 'AFT' in fam:
        return dict(color='mediumpurple', lw=0.9, alpha=0.75, zorder=1.6)
    if 'B-C' in fam:
        return dict(color='seagreen', lw=1.1, alpha=0.9, zorder=1.6)
    return dict(color='0.82', lw=0.55, alpha=1.0, zorder=1.2)


fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
for ax, side, lab in ((axs[0], 'upper', 'upper surface'),
                      (axs[1], 'lower', 'lower surface')):
    for name, ser in WS['transition_vs_alpha'].items():
        sd = ser.get(side)
        if not sd or not sd.get('alpha'):
            continue
        a = np.asarray(sd['alpha'], float)
        xt = np.asarray(sd['xtr'], float)
        o = np.argsort(a)
        ax.plot(xt[o], cl_of_alpha(a[o]), '-', **ws_style(ser['family']))
    pz = PZ[f'sa_lm2015_{side}']
    ax.plot(pz['xtr'], pz['cl'], '-.', color='olive', lw=1.0, alpha=0.9,
            zorder=1.8, label='SA-LM2015 (Piotrowski–Zingg)')
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
    lm = LM[f'lm_fine_grid_{side}']
    ax.plot(lm['xtr'], lm['cl'], '-', color='C2', lw=1.2, marker='x', ms=4,
            label='$\\gamma$\u2013$Re_\\theta$ (Denison)')
    for fam, c, mk in (('str', 'C0', 's'), ('cav', 'C1', '^')):
        for lv, ms, mew, al in ((0, 3.5, 0.8, 0.5), (1, 5.2, 1.1, 0.75),
                                (2, 7.0, 1.6, 1.0)):
            cls, xts = [], []
            keys = [f'{fam}L{lv}prop_nlf0416_Re4M_a{a}' for a in (0, 4, 9, 15)]
            if lv == 2:   # negative-incidence pair exists on the finest grids only
                keys += [f'{fam}L2prop_nlf0416_Re4M_am4',
                         f'{fam}L2prop_nlf0416_Re4M_am8']
            for k in keys:
                r = camp.get(k)
                if r is None:
                    continue
                xt = r['xtr_up' if side == 'upper' else 'xtr_lo']
                if xt is None:
                    continue                       # front lost on the coarse grid
                cls.append(r['CL']); xts.append(xt)
            ax.plot(xts, cls, mk, color=c, ms=ms, mfc='none', mew=mew,
                    alpha=al,
                    label=(f"SA-AI, {'O-grid' if fam=='str' else 'unstructured'}"
                           " (L0$\\to$L2 by size)") if lv == 2 else None,
                    zorder=5)
    ax.set_xlabel('$x_t/c$'); ax.set_title(lab, fontsize=10)
    ax.grid(alpha=0.3); ax.set_xlim(0, 0.95)
axs[0].set_ylabel('$c_l$'); axs[0].set_ylim(-0.6, 2.1)
_h, _l = axs[0].get_legend_handles_labels()
from matplotlib.lines import Line2D as _L2
_h += [_L2([], [], color='0.82', lw=0.55, label='workshop submittals (14, all models)'),
       _L2([], [], color='mediumpurple', lw=0.9, label='workshop AFT submittals'),
       _L2([], [], color='seagreen', lw=1.1, label='workshop algebraic (B–C)')]
axs[0].legend(handles=_h, fontsize=7.5, loc='upper right')
plt.tight_layout()
out = f'{PD}/figs/nlf_aft_transition.pdf'
plt.savefig(out); plt.savefig('/tmp/nlf_aft.png', dpi=130)
print('wrote', out)
