"""fig:eppbubble -> paper/figs/eppler_bubble_stations.pdf.

The promotion of the reattachment table (old tab:eppxtr) to a figure in the
style of the NLF transition figure: laminar-separation location (left) and
turbulent-reattachment location (right) against incidence, Eppler 387 at
Re=2e5.  Series:
  - oil-flow visualization, TM-4062 Table III, R=200k (nine incidences;
    alpha=8 shows natural transition, no bubble, and is omitted);
  - mfoil e^9 dense alpha sweep (data/mfoil_eppler_bubble_sweep.json);
  - gamma-Re_theta (LM) and SA-BC digitized from Shahjahan et al. Fig 10a
    (data/lmbcm_eppler387_digitized.json, key bubble_re200k), if present;
  - SA-AI on all six grids (both families, L0-L2 by marker size), stations
    from the signed-Cf crossings (same definitions as the old table).
"""
import os, sys, json, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H)
import regen_eppler_v2 as R
from regen_epp_reattach import reattach

PD = os.path.abspath(os.path.join(_H, '..', '..'))
B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fr")

# TM-4062 Table III, R=200,000: alpha -> (LS, TR).  alpha=8 is "(NT at .32)"
# (natural transition, no bubble): excluded.  8.5 is the post-stall LE bubble.
OIL = {-2: (0.53, 0.80), 0: (0.48, 0.74), 2: (0.43, 0.67), 4: (0.40, 0.62),
       5: (0.38, 0.59), 6: (0.37, 0.55), 7: (0.33, 0.48), 8.5: (0.03, 0.18)}


def separation(x, cf, thr=0.0, xlo=0.05, xhi=0.85):
    """First downward zero-crossing of signed C_f in the LSB window."""
    x = np.asarray(x, float); cf = np.asarray(cf, float)
    m = (x > xlo) & (x < xhi); x, cf = x[m], cf[m]
    o = np.argsort(x); x, cf = x[o], cf[o]
    if cf.min() >= thr:
        return None
    for i in range(1, len(x)):
        if cf[i] < thr and cf[i - 1] >= thr:
            f = (thr - cf[i - 1]) / (cf[i] - cf[i - 1])
            return float(x[i - 1] + f * (x[i] - x[i - 1]))
    return None


fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
ax_ls, ax_tr = axs

# oil flow
oa = sorted(OIL)
ax_ls.plot([OIL[a][0] for a in oa], oa, 'o-', mfc='none', mec='k', color='k',
           ms=6, mew=1.2, lw=0.8, zorder=6)
ax_tr.plot([OIL[a][1] for a in oa], oa, 'o-', mfc='none', mec='k', color='k',
           ms=6, mew=1.2, lw=0.8, zorder=6)

# mfoil dense sweep
XF_DRAWN = []
swp = f'{PD}/data/mfoil_eppler_bubble_sweep.json'
if os.path.exists(swp):
    s = json.load(open(swp))
    xf = s.pop('xfoil', None)
    aa = sorted(float(k) for k in s)
    ls = [s[f'{a:.1f}']['ls'] for a in aa]
    tr = [s[f'{a:.1f}']['tr'] for a in aa]
    ok = [i for i, (l, t) in enumerate(zip(ls, tr))
          if s[f'{aa[i]:.1f}']['conv'] and l is not None and t is not None
          and aa[i] <= 6.5]     # mfoil unreliable at the stall edge (see text)
    ax_ls.plot([ls[i] for i in ok], [aa[i] for i in ok], ':', color='0.5', lw=1.4)
    ax_tr.plot([tr[i] for i in ok], [aa[i] for i in ok], ':', color='0.5', lw=1.4)
    if xf:   # XFOIL would carry the e^9 reference past mfoil's edge, but
        # xfoil 6.99 SIGFPEs at these conditions and its pickled alpha=7 Cf
        # never crosses zero (min +2e-4) -- branch draws only if real data
        # ever lands; the legend entry is gated on XF_DRAWN.
        ax2 = sorted(float(k) for k in xf)
        for ax, q in ((ax_ls, 'ls'), (ax_tr, 'tr')):
            pts = [(xf[f'{a:.1f}'][q], a) for a in ax2
                   if xf[f'{a:.1f}']['conv'] and xf[f'{a:.1f}'][q] is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], '-.',
                        color='0.5', lw=1.1)
                XF_DRAWN.append(q)

# literature transition models (digitized), if the bubble extraction landed
lit = json.load(open(f'{PD}/data/lmbcm_eppler387_digitized.json'))
bub = lit.get('bubble_re200k', {})
# the Selig-experiment markers from the same source figure (a second,
# independent tunnel): small gray diamonds
sel = bub.get('experiment_selig')
if sel:
    ax_ls.plot(sel['ls'], sel['alpha'], 'D', color='0.35', ms=3.5, mfc='none',
               zorder=4)
    ax_tr.plot(sel['tr'], sel.get('alpha_tr', sel['alpha']), 'D', color='0.35',
               ms=3.5, mfc='none', zorder=4)
for key, col, mk in (('langtry_menter', 'C2', 'x'), ('sa_bc', 'C4', '+')):
    d = bub.get(key)
    if not d:
        continue
    al = np.asarray(d['alpha'], float)
    for ax, q in ((ax_ls, 'ls'), (ax_tr, 'tr')):
        v = np.asarray([np.nan if x is None else x for x in d[q]], float)
        m = np.isfinite(v)
        ax.plot(v[m], al[m], '-', color=col, lw=1.0, marker=mk, ms=4.5)

# SA-AI, six grids
for mesh, col, mk in (('str', 'C0', 's'), ('cav', 'C1', '^')):
    for lev, ms, mew, al_ in (('L0', 3.5, 0.8, 0.5), ('L1', 5.2, 1.1, 0.75),
                              ('L2', 7.0, 1.6, 1.0)):
        A, LS, TR = [], [], []
        for a in (0, 2, 5, 7):
            d = f"{B}/{mesh}{lev}prop_eppler387_Re200k_a{a}"
            if not os.path.isdir(d):
                continue
            try:
                (xu, cfu, _), _ = R.airfoil_walk_contour(d)
            except Exception:
                continue
            A.append(a); LS.append(separation(xu, cfu)); TR.append(reattach(xu, cfu))
        for ax, V in ((ax_ls, LS), (ax_tr, TR)):
            pts = [(v, a) for v, a in zip(V, A) if v is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], mk, color=col,
                        ms=ms, mfc='none', mew=mew, alpha=al_, zorder=5)

for ax, lab in ((ax_ls, 'laminar separation $x_{LS}/c$'),
                (ax_tr, 'turbulent reattachment $x_R/c$')):
    ax.set_xlabel(lab); ax.grid(alpha=0.3); ax.set_xlim(0, 0.9)
ax_ls.set_ylabel(r'$\alpha$ (deg)'); ax_ls.set_ylim(-3, 9.2)
handles = [Line2D([], [], color='k', ls='-', lw=0.8, marker='o', mfc='none', ms=6,
                  label='Oil flow (LTPT, Table III)'),
           Line2D([], [], color='0.35', ls='none', marker='D', mfc='none', ms=3.5,
                  label='Experiment (Selig et al.)'),
           Line2D([], [], color='0.5', ls=':', lw=1.4, label='mfoil ($e^9$)'),
           Line2D([], [], color='C2', ls='-', lw=1.0, marker='x', ms=4.5,
                  label='$\\gamma$\u2013$Re_\\theta$ (Shahjahan et al.)'),
           Line2D([], [], color='C4', ls='-', lw=1.0, marker='+', ms=5,
                  label='SA-BC (Shahjahan et al.)'),
           Line2D([], [], color='C0', ls='none', marker='s', mfc='none', ms=5.5,
                  label='SA-AI, O-grid (L0$\\to$L2 by size)'),
           Line2D([], [], color='C1', ls='none', marker='^', mfc='none', ms=5.5,
                  label='SA-AI, unstructured (L0$\\to$L2 by size)')]
if XF_DRAWN:
    handles.insert(3, Line2D([], [], color='0.5', ls='-.', lw=1.1,
                             label='XFOIL ($e^9$, $\\alpha\\geq6.5^\\circ$)'))
ax_tr.legend(handles=handles, fontsize=7.5, loc='lower left')
plt.tight_layout()
out = f'{PD}/figs/eppler_bubble_stations.pdf'
plt.savefig(out); plt.savefig('/tmp/eppler_bubble.png', dpi=130)
print('wrote', out)
