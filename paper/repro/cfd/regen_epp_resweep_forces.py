"""fig:eppresweepforces -> paper/figs/eppler_resweep_forces.pdf.

Two-by-two Eppler-387 alpha=5 Reynolds-sweep summary (2026-07-28 rework):
  [0,0] c_l | [0,1] c_d (log)   -- section forces
  [1,0] c_m(c/4) | [1,1] laminar-separation & turbulent-reattachment
                                  stations combined on one x/c axis
  vs Re for all 24 SA-AI solutions (structured solid blue / unstructured
  dashed orange, L0-L2 by line thickness), the e^9 panel reference
  (mfoil squares 0.6-3e5 where its solve returns forces; XFOIL diamonds
  at all five Re, c_m from both), the LTPT measurement (large filled
  black circles -- the visually dominant experiment; individual TM-4062
  Table B1 run readings near alpha=5 plotted as separate circles where
  more than one was hand-read from the scan, see EXP_* below), and small
  gray published-calculation symbols read at alpha=5:
    6e4: Frere+ 2016 ILES (alpha=4) & coupled RANS-e^N, Carreno Ruiz &
         D'Ambrosio 2022 gamma-Retheta, IJSRP 2019 k-kL-omega, and
         Frere's compiled Delft/Stuttgart experimental lift (small dots);
    1e5 & 3e5: IJSRP 2019 transition-SST.
  Station panel: SA-AI from the wall C_f walk (x_R pinned near 1.0 = no
  closure ahead of the TE); the TM-4062 Table III oil flow (large filled
  black circles, Re=1/2/3e5 only -- Table III tabulates oil-flow closure
  at alpha=5 for those three Re only); and the published alpha=4-6deg
  bracket pairs straddling the computed alpha=5 (no exact alpha=5 station
  datum exists in those sources) drawn as capped range bars: Cole &
  Mueller 1990 at 1e5, Ghimire+ 2025's gamma variants at 3e5.

The experiment/other-calculation distinction is carried by SIZE and FILL:
LTPT = large filled black; published calcs = small open gray.

Run from paper/: python3 repro/cfd/regen_epp_resweep_forces.py
"""
import os, sys, csv, json, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fv1")
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(_HERE, '..', '..', 'figs'))
DATA = os.path.abspath(os.path.join(_HERE, '..', '..', 'data'))
PREV = os.path.abspath(os.path.join(_HERE, '..', 'analytic', 'figs_explore'))
sys.path.insert(0, _HERE)
import regen_eppler_v2 as R

POLAR_LW = {'L0': 0.8, 'L1': 1.6, 'L2': 3.2}
FAM = {'str': dict(color='C0', ls='-'), 'cav': dict(color='C1', ls='--')}
RES = [60, 100, 200, 300, 460]
ALPHA = 5.0

# TM-4062 Table B1 individual run readings near alpha=5, hand-read from the
# scanned listing (provenance: regen_resweep_table.py docstring). Each list is
# EVERY individual value that was hand-read at the tabulated angle nearest 5 deg
# -- plotted as a separate black circle so repeat measurements read as multiple
# points, not a mean+bar. Coverage is uneven because only these readings were
# transcribed: 200k c_l has the four repeat runs (9/10/13 a=5.00/5.03/5.06/5.06
# = .891/.895/.894/.897) and c_d three (.0137-.0139); 100k c_d has the run
# 15/16 repeat (.0237/.0240); the other Re have a single transcribed reading.
# Full per-run digitization (all runs x all near-5 alphas, all columns) would
# require re-reading TM-4062 Table B1 from the PDF -- see the rework record.
EXP_CL = {60: [0.838], 100: [0.873], 200: [0.891, 0.895, 0.894, 0.897],
          300: [0.901], 460: [0.914]}
EXP_CD = {60: [0.0439], 100: [0.0237, 0.0240],
          200: [0.0137, 0.0138, 0.0139], 300: [0.0114], 460: [0.0093]}
EXP_CM = {60: [-0.1139], 100: [-0.0889], 200: [-0.0809],
          300: [-0.0799], 460: [-0.0807]}
# TM-4062 Table III oil flow at alpha=5 (x_sep, x_reattach). Tabulated for
# Re=1/2/3e5 only (no 6e4 or 4.6e5 oil-flow closure at alpha=5 in Table III).
OIL = {100: (0.34, 0.67), 200: (0.38, 0.59), 300: (0.39, 0.55)}

bench = json.load(open(f'{B}/sphere_campaign_eppler_results.json'))
_swp_p = f'{B}/sphere_campaign_epp_sweep_l1_results.json'
if not os.path.exists(_swp_p):
    _swp_p = f'{B}/sphere_campaign_epp_sweep_results.json'   # legacy tree
swp = json.load(open(_swp_p))
lvl = json.load(open(f'{B}/sphere_campaign_epp_sweep_levels_results.json'))


def case_dir(fam, L, Rk):
    if Rk == 200:
        return f'{B}/{fam}{L}prop_eppler387_Re200k_a5'
    if L == 'L1':
        uni = f'{B}/sweep_{fam}L1_Re{Rk}k_a5'
        if os.path.isdir(uni):
            return uni
        return f'{B}/sweep_Re{Rk}k_a5' if fam == 'cav' else f'{B}/sweep_str_Re{Rk}k_a5'
    return f'{B}/sweep_{fam}{L}_Re{Rk}k_a5'


def forces(d):
    rows = [r for r in list(csv.reader(open(f"{d}/total_forces_v2.csv")))[1:]
            if len(r) > 10]
    t = rows[int(0.8 * len(rows)):]
    cl = float(np.median([float(r[2]) for r in t]))
    cd = float(np.median([float(r[3]) for r in t]))
    cmy = float(np.median([float(r[8]) for r in t]))
    a = np.radians(ALPHA)
    return cl, cd, cmy + 0.25 * (cl * np.cos(a) + cd * np.sin(a))


def stations(d):
    (xu, cfu, _), _ = R.airfoil_walk_contour(d)
    o = np.argsort(xu)
    x, cf = xu[o], cfu[o]
    m = (x > 0.05) & (x < 0.995)
    xs = None
    xw, cw = x[m], cf[m]
    for i in range(1, len(xw)):
        if cw[i] < 0 and cw[i - 1] >= 0:
            f = -cw[i - 1] / (cw[i] - cw[i - 1])
            xs = float(xw[i - 1] + f * (xw[i] - xw[i - 1]))
            break
    neg = np.where(cf < 0)[0]
    xr = None
    if len(neg):
        i = neg[-1]
        xr = float(x[i] + (0 - cf[i]) * (x[i + 1] - x[i]) / (cf[i + 1] - cf[i])) \
            if i + 1 < len(x) else float(x[i])
    return xs, xr


mf = pickle.load(open(f'{B}/mfoil_eppler387_sweep_a5.pkl', 'rb'))
xf = pickle.load(open(f'{B}/xfoil_eppler387_sweep_a5.pkl', 'rb'))


def lit(fname):
    p = f'{DATA}/{fname}'
    return json.load(open(p)) if os.path.exists(p) else None


carreno = lit('carreno2022_e387_re60k_polar.json')
ijsrp = lit('ijsrp2019_e387_polars.json')
frere = lit('frere2016_e387_re60k_polar.json')
ghim = lit('ghimire2025_e387_bubble_stations.json')
cm90 = lit('colemueller1990_e387_bubble_stations.json')


def at5(series):
    """cl, cd at alpha=5 by linear interpolation."""
    a = np.asarray(series['alpha'], float)
    cl = np.asarray(series['cl'], float)
    cd = np.asarray(series['cd'], float)
    return float(np.interp(5.0, a, cl)), float(np.interp(5.0, a, cd))


fig = plt.figure(figsize=(9.6, 8.4))
gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28)
axl = fig.add_subplot(gs[0, 0])   # c_l
axd = fig.add_subplot(gs[0, 1])   # c_d (log)
axm = fig.add_subplot(gs[1, 0])   # c_m (c/4)
axst = fig.add_subplot(gs[1, 1])  # combined separation + reattachment
axs = axst   # laminar-separation stations share the combined panel
axr = axst   # turbulent-reattachment stations share the combined panel
Re = np.array(RES, float) * 1e3

sa_st = {}   # (fam, L) -> (xs list, xr list)
_dump = {}   # appendix data tables: forces + stations per family/level/Re
for fam in ('str', 'cav'):
    for L in ('L0', 'L1', 'L2'):
        F = [forces(case_dir(fam, L, Rk)) for Rk in RES]
        S = [stations(case_dir(fam, L, Rk)) for Rk in RES]
        sa_st[(fam, L)] = S
        _dump[f'{fam}_{L}'] = {str(Rk): {'cl': F[i][0], 'cd': F[i][1],
                                         'cm': F[i][2], 'xsep': S[i][0],
                                         'xr': S[i][1]}
                               for i, Rk in enumerate(RES)}
        kw = dict(color=FAM[fam]['color'], ls=FAM[fam]['ls'], lw=POLAR_LW[L],
                  marker='o' if fam == 'str' else '^', ms=3.0)
        axl.semilogx(Re, [f[0] for f in F], **kw)
        axd.loglog(Re, [f[1] for f in F], **kw)
        axm.semilogx(Re, [f[2] for f in F], **kw)
        axs.semilogx(Re, [s[0] for s in S], **kw)
        axr.semilogx(Re, [s[1] for s in S], **kw)

# e^9 reference: mfoil dotted wherever its solve returns finite forces
# (460k fails outright); xfoil diamonds at every computed Reynolds number,
# so the reference spans the full data range.
mre = [r for r in sorted(mf) if np.isfinite(mf[r].get('cl') or np.nan)]
axl.semilogx([r*1e3 for r in mre], [mf[r]['cl'] for r in mre], ':',
             color='0.45', marker='s', mfc='none', ms=5.5, lw=1.2)
axd.loglog([r*1e3 for r in mre], [mf[r]['cd'] for r in mre], ':',
           color='0.45', marker='s', mfc='none', ms=5.5, lw=1.2)
axm.semilogx([r*1e3 for r in mre], [mf[r]['cm'] for r in mre], ':',
             color='0.45', marker='s', mfc='none', ms=5.5, lw=1.2)
xre = sorted(xf)
axl.semilogx([r*1e3 for r in xre], [xf[r]['cl'] for r in xre], ls='none',
             color='0.45', marker='D', mfc='none', ms=5.5, mew=1.2)
axd.loglog([r*1e3 for r in xre], [xf[r]['cd'] for r in xre], ls='none',
           color='0.45', marker='D', mfc='none', ms=5.5, mew=1.2)
xrm = [r for r in xre if xf[r].get('cm') is not None]
axm.semilogx([r*1e3 for r in xrm], [xf[r]['cm'] for r in xrm], ls='none',
             color='0.45', marker='D', mfc='none', ms=5.5, mew=1.2)

# experiment: LTPT measurement, made visually dominant -- LARGE FILLED black
# circles, one per individual Table B1 run reading (repeat runs -> multiple
# circles), heavy edge, top zorder.
EXP_KW = dict(ls='none', marker='o', color='k', mfc='k', mec='k',
              ms=8.5, mew=1.0, zorder=12)
for ax, tab in ((axl, EXP_CL), (axd, EXP_CD), (axm, EXP_CM)):
    for Rk in RES:
        for v in tab[Rk]:
            ax.plot([Rk*1e3], [v], **EXP_KW)
# oil-flow separation & reattachment: same dominant filled black circle
for Rk, (xs_, xr_) in OIL.items():
    axst.plot([Rk*1e3], [xs_], **EXP_KW)
    axst.plot([Rk*1e3], [xr_], **EXP_KW)

# ---- literature: forces at 6e4 / 1e5 / 3e5 ----
LIT_MS = 6.5


def lit_pt(ax_c, ax_d, Rk, cl, cd, marker, color, label=None):
    for ax, v in ((ax_c, cl), (ax_d, cd)):
        if ax is not None and v is not None:
            ax.plot([Rk*1e3], [v], ls='none', marker=marker, color=color,
                    ms=LIT_MS, mfc='none', mew=1.4, zorder=7, label=label)


if carreno:
    cl5, cd5 = at5(carreno['series']['gamma-Re_theta (STAR-CCM+)'])
    lit_pt(axl, axd, 60, cl5, cd5, 'v', '0.25')
if ijsrp:
    s = ijsrp['re_60k'].get('kklw')
    if s:
        cl5, cd5 = at5(s)
        lit_pt(axl, axd, 60, cl5, cd5, 'P', '0.25')
    for key, Rk in (('re_100k', 100), ('re_300k', 300)):
        s = (ijsrp.get(key) or {}).get('transition_sst')
        if s:
            cl5, cd5 = at5(s)
            lit_pt(axl, axd, Rk, cl5, cd5, 'X', '0.25')
if frere:
    S = frere['series']
    # coupled URANS-e^N: exact alpha=5 points
    for name, mk in (('rans_eN_N7', '<'), ('rans_eN_N9', '>')):
        if name in S and S[name].get('alpha'):
            cl5, cd5 = at5(S[name])
            lit_pt(axl, axd, 60, cl5, cd5, mk, '0.25')
    # ILES: only alpha=4 and 8, straddling the reattachment jump -- an
    # alpha=5 interpolation is meaningless; plot the alpha=4 (pre-jump) state
    if 'iles' in S and 4 in [int(a) for a in S['iles']['alpha']]:
        i4 = [int(a) for a in S['iles']['alpha']].index(4)
        lit_pt(axl, axd, 60, S['iles']['cl'][i4], S['iles']['cd'][i4],
               '*', '0.25')
    # facility spread at the bistable condition: exact alpha=5 lift points
    # (Delft, Stuttgart; CD hidden by overlaps in the source at alpha=5)
    for name in ('exp_delft', 'exp_stuttgart'):
        pan = (S.get(name) or {}).get('cl_panel') or {}
        aa = pan.get('alpha') or []
        if 5.0 in [round(float(a), 1) for a in aa]:
            i = [round(float(a), 1) for a in aa].index(5.0)
            lit_pt(axl, None, 60, pan['cl'][i], None, '.', '0.55')

# ---- literature: stations, alpha=4/6 bracket pairs straddling alpha=5 ----
# No source tabulates a station at exactly alpha=5, so each is drawn as a
# CAPPED RANGE BAR spanning the alpha=4 and alpha=6 values (the small end
# markers keep the per-model identity); the caps make it read as a range, not
# a bare vertical line. One shared legend entry explains the convention.
_bracket_handle = [None]


def bracket(ax, Rk, v4, v6, marker, color='0.4'):
    vv = [v for v in (v4, v6) if v is not None]
    if not vv:
        return
    lo, hi = min(vv), max(vv)
    mid = 0.5 * (lo + hi)
    x = Rk * 1e3
    eb = ax.errorbar([x], [mid], yerr=[[mid - lo], [hi - mid]], fmt='none',
                     ecolor=color, elinewidth=1.1, capsize=5, capthick=1.1,
                     zorder=5)
    if _bracket_handle[0] is None:
        _bracket_handle[0] = eb
    for v in vv:
        ax.plot([x], [v], ls='none', marker=marker, color=color, ms=4.5,
                mfc='none', mew=1.1, zorder=6)


if cm90 and 're_100k' in cm90:
    rows = {r['alpha_deg']: r['x_over_c'] for r in cm90['re_100k']}
    g = lambda a, k: (rows.get(a) or {}).get(k)
    bracket(axst, 100, g(4, 'x_sep'), g(6, 'x_sep'), 's')
    bracket(axst, 100, g(4, 'x_reattach'), g(6, 'x_reattach'), 's')
if ghim:
    for model, mk in (('gamma_retheta_sst', 'v'), ('gamma_sst', '<'),
                      ('kgamma_sst', '>')):
        s4 = (ghim.get('alpha_4') or {}).get(model) or {}
        s6 = (ghim.get('alpha_6') or {}).get(model) or {}
        bracket(axst, 300, s4.get('x_sep'), s6.get('x_sep'), mk)
        bracket(axst, 300, s4.get('x_reattach'), s6.get('x_reattach'), mk)
    e4 = (ghim.get('alpha_4') or {}).get('experiment') or {}
    e6 = (ghim.get('alpha_6') or {}).get('experiment') or {}
    bracket(axst, 300, e4.get('x_sep'), e6.get('x_sep'), '.', '0.55')
    bracket(axst, 300, e4.get('x_reattach'), e6.get('x_reattach'), '.', '0.55')

for ax, lab in ((axl, '$c_l$'), (axd, '$c_d$'), (axm, '$c_m$ ($c/4$)'),
                (axst, r'station $x/c$')):
    ax.set_ylabel(lab)
    ax.grid(alpha=0.3, which='both')
    ax.set_xlim(4.5e4, 5.6e5)
    ax.xaxis.set_major_locator(
        mticker.FixedLocator([6e4, 1e5, 2e5, 3e5, 4.6e5]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(
        mticker.FixedFormatter(['$0.6$', '$1$', '$2$', '$3$', '$4.6$']))
# x labels only on the bottom row (2x2 shares the Re axis by column)
for ax in (axm, axst):
    ax.set_xlabel(r'$Re\ (\times 10^5)$')
for ax in (axl, axd):
    ax.tick_params(labelbottom=False)
axl.set_ylim(0.55, 1.02)
axst.set_ylim(0.28, 1.05)
axst.axhline(1.0, color='0.7', lw=0.6)
# name the two curve bands directly so one panel reads unambiguously
axst.text(5.0e5, 0.34, 'laminar\nseparation', fontsize=8, color='0.35',
          ha='right', va='center')
axst.text(1.15e5, 0.86, 'turbulent\nreattachment', fontsize=8, color='0.35',
          ha='left', va='center')
axd.annotate('bursting boundary', xy=(1.0e5, 0.034), xytext=(1.55e5, 0.044),
             fontsize=8, color='0.3',
             arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8))
axst.annotate('no closure', xy=(0.85e5, 0.995), xytext=(0.62e5, 0.905),
              fontsize=8, color='0.3',
              arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8))

handles = [
    Line2D([], [], color='C0', ls='-', marker='o', ms=3.0,
           label='SA-AI, structured (L0--L2 by weight)'),
    Line2D([], [], color='C1', ls='--', marker='^', ms=3.0,
           label='SA-AI, unstructured'),
    Line2D([], [], color='k', marker='o', ls='none', ms=8.5, mfc='k',
           label='LTPT experiment (individual runs; oil flow)'),
    Line2D([], [], color='0.45', ls=':', marker='s', mfc='none', ms=5.5,
           label='mfoil ($e^9$)'),
    Line2D([], [], color='0.45', ls='none', marker='D', mfc='none', ms=5.5,
           label='XFOIL ($e^9$)'),
    Line2D([], [], color='0.25', ls='none', marker='*', mfc='none', ms=7,
           label='ILES (Frère+, $\\alpha{=}4$)'),
    Line2D([], [], color='0.55', ls='none', marker='.', ms=6,
           label='other facilities / source exp.'),
    Line2D([], [], color='0.25', ls='none', marker='<', mfc='none', ms=6,
           label='RANS-$e^N$ $N{=}7$ / $\\gamma$-$SST$ (Ghimire+)'),
    Line2D([], [], color='0.25', ls='none', marker='>', mfc='none', ms=6,
           label='RANS-$e^N$ $N{=}9$ / K$\\gamma$ (Ghimire+)'),
    Line2D([], [], color='0.25', ls='none', marker='v', mfc='none', ms=6,
           label='$\\gamma$-$Re_\\theta$ (Carreño Ruiz; Ghimire+)'),
    Line2D([], [], color='0.25', ls='none', marker='P', mfc='none', ms=6,
           label='k-k$_L$-$\\omega$ (IJSRP)'),
    Line2D([], [], color='0.25', ls='none', marker='X', mfc='none', ms=6,
           label='transition-SST (IJSRP)'),
    Line2D([], [], color='0.25', ls='none', marker='s', mfc='none', ms=5,
           label='Cole--Mueller LDV/$C_p$ ($\\alpha{=}4/6$)'),
]
if _bracket_handle[0] is not None:
    _bracket_handle[0].set_label(
        r'$\alpha{=}4$--$6^\circ$ bracket (no $\alpha{=}5$ station datum)')
    handles.append(_bracket_handle[0])
# figure-level legend BELOW the axes; bbox_inches='tight' at save expands the
# canvas to include it, so it never overlaps the bottom-row x-axis labels.
fig.subplots_adjust(left=0.075, right=0.985, top=0.975, bottom=0.135)
leg = fig.legend(handles=handles, fontsize=7.8, ncol=4, frameon=False,
                 loc='upper center', bbox_to_anchor=(0.5, 0.085))
os.makedirs(PREV, exist_ok=True)
plt.savefig(f'{OUT}/eppler_resweep_forces.pdf',
            bbox_inches='tight', bbox_extra_artists=(leg,))
import json as _json
_json.dump(_dump, open(f'{DATA}/eppresweep_forces_computed.json', 'w'), indent=1)
plt.savefig(f'{PREV}/epp_resweep_forces.png', dpi=140,
            bbox_inches='tight', bbox_extra_artists=(leg,))
missing = [n for n, v in (('frere', frere), ('carreno', carreno),
                          ('ijsrp', ijsrp), ('ghimire', ghim),
                          ('cole-mueller', cm90)) if v is None]
print('wrote', f'{OUT}/eppler_resweep_forces.pdf',
      f'(missing lit: {missing})' if missing else '(all lit present)')
