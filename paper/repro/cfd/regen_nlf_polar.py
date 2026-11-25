"""NLF(1)-0416 drag polar (Re=4e6): SA-AI vs digitized experiment, Eppler-style.
Experiment digitized from Somers, NASA TP-1861, Fig 11(d) (pixel-calibrated).
-> figs/nlf_polar_compare.pdf
"""
import csv, os, pickle, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Canvas is 7.4in but the figure is included at \textwidth = 6.5in, so the PDF
# is scaled down by 6.5/7.4 on the page. Multiply every text element by the
# inverse so the in-plot type lands at its declared size -- i.e. matching the
# 10pt caption -- while the larger canvas still buys drawing area for the many
# lines and symbols.
def _scale_text_for_page(fig, scale=1.1384615384615384):
    import matplotlib.text as _mtext
    for _t in fig.findobj(_mtext.Text):
        _s = _t.get_fontsize()
        if _s:
            _t.set_fontsize(_s * scale)

B=os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fv1"); PD="/home/qiqi/flexcompute/sa-ai/paper"
LEVEL_LW={'L0':0.8,'L1':1.6,'L2':3.2}; MESH_LS={'str':'-','cav':'--'}  # geometric ladder: finest clearly thickest
MESH_COL={'str':'C0','cav':'C1'}; MESH_MK={'str':'o','cav':'^'}

# Digitized section characteristics, Somers NASA TP-1861 Fig 11(d), R=4e6: (C_d, C_l).
EXP_POLAR=[(0.01931,-0.965),(0.01707,-0.888),(0.01474,-0.808),(0.01383,-0.735),
 (0.01284,-0.662),(0.01175,-0.570),(0.01095,-0.498),(0.01030,-0.421),(0.00970,-0.337),
 (0.00914,-0.264),(0.00855,-0.180),(0.00779,-0.092),(0.00680,0.073),(0.00634,0.149),
 (0.00623,0.222),(0.00606,0.295),(0.00600,0.375),(0.00592,0.448),(0.00598,0.521),
 (0.00600,0.593),(0.00615,0.666),(0.00611,0.750),(0.00642,0.831),(0.00684,0.903),
 (0.00747,0.980),(0.00830,1.072),(0.00909,1.148),(0.00991,1.225),(0.01069,1.298),
 (0.01224,1.432),(0.01309,1.504),(0.01488,1.611),(0.01770,1.684)]

def case_dir(mesh,level,a): return f"{B}/{mesh}{level}prop_nlf0416_Re4M_a{int(a)}"
def converged_clcd(d):
    p=f"{d}/total_forces_v2.csv"
    if not os.path.exists(p): return None
    cl,cd=[],[]
    for r in list(csv.reader(open(p)))[1:]:
        if len(r)>3:
            try: cl.append(float(r[2])); cd.append(float(r[3]))
            except ValueError: pass
    if len(cl)<3: return None
    n=len(cl); t=slice(int(0.8*n),n)
    return float(np.median(np.array(cl)[t])),float(np.median(np.array(cd)[t]))

alphas=[0,4,9,15]
mn=pickle.load(open(f"{B}/mfoil_nlf0416_Re4M.pkl",'rb'))
import json as _json
_aftD = _json.load(open(f"{PD}/data/aft_nlf0416_digitized.json"))['polar']
fig,ax=plt.subplots(figsize=(7.4, 7.4))
# AFT polars from Coder's dissertation (same digitized source as fig:nlfaft):
# nominal Ncrit=10.07 dashed, recalibrated 7.18 solid, both gray
ax.plot(_aftD['aft_ncrit10']['cd'], _aftD['aft_ncrit10']['cl'], '--',
        color='0.45', lw=1.3, zorder=2)
ax.plot(_aftD['aft_ncrit718']['cd'], _aftD['aft_ncrit718']['cl'], '-',
        color='0.45', lw=1.3, zorder=2)
ecd=[p[0] for p in EXP_POLAR]; ecl=[p[1] for p in EXP_POLAR]
ax.plot(ecd,ecl,'-',color='k',lw=1.4,zorder=1)
ax.plot(ecd,ecl,'o',mfc='none',mec='k',ms=4,zorder=1)
# mfoil e^9 reference: plot ONLY where the coupled Newton solve converged AND
# the upper surface stays attached. It does not converge at alpha=9 deg, and at
# alpha=15 deg it converges to a stalled solution (upper H>>1); both lie beyond
# the useful e^N range on this airfoil at Re=4e6 and are omitted.
def _usable(e):
    try: return bool(e.get('conv',True)) and float(np.nanmax(e['upper']['H']))<4.0
    except Exception: return bool(e.get('conv',True))
mf_ok=sorted(a for a in mn if isinstance(a,(int,float)) and _usable(mn[a]))
ax.plot([mn[a]['cd'] for a in mf_ok],[mn[a]['cl'] for a in mf_ok],
        ls=':',color='0.4',marker='s',mfc='none',ms=5,lw=1.2,zorder=2)
# xfoil e^9 fills the high-alpha cases where mfoil is unusable (alpha=9,15).
xf=pickle.load(open(f"{B}/xfoil_nlf0416_Re4M.pkl",'rb'))
xf_fill=[a for a in (9.0,15.0) if a in xf]
ax.plot([xf[a]['cd'] for a in xf_fill],[xf[a]['cl'] for a in xf_fill],
        ls='none',color='0.5',marker='D',mfc='none',ms=6,mew=1.3,zorder=2)
# Fully-turbulent SA baseline (AI_SA=0, chi_inf=3) on the structured L2 grid:
# quantifies the drag of losing the laminar run (run_turb_baselines.py).
tb_cl,tb_cd=[],[]
for a in alphas:
    f=converged_clcd(f"{B}/strL2prop_nlf0416_Re4M_turb_a{int(a)}")
    if f: tb_cl.append(f[0]); tb_cd.append(f[1])
if tb_cl: ax.plot(tb_cd,tb_cl,marker='v',mfc='none',ms=5,ls='-.',lw=1.2,color='0.55',zorder=2)
# Workshop-submittal polars (cl vs cd -- native axes), light underlay;
# AFT highlighted mediumpurple, algebraic B-C seagreen (annotated round 7).
import json as _json
_WS = _json.load(open(f'{PD}/data/workshop_nlf_submittals.json'))['drag_polar']
for _name, _ser in _WS.items():
    if _name == 'experiment' or not isinstance(_ser, dict):
        continue
    _cl = _ser.get('cl') or []
    _cd = _ser.get('cd') or []
    if len(_cl) < 1:
        continue
    _fam = _ser.get('family', '')
    if 'AFT' in _fam:
        _kw = dict(color='mediumpurple', lw=0.9, alpha=0.75, zorder=1.6)
    elif 'B-C' in _fam:
        _kw = dict(color='seagreen', lw=1.1, alpha=0.9, zorder=1.6)
    else:
        _kw = dict(color='0.82', lw=0.55, zorder=1.2)
    _o = np.argsort(np.asarray(_cl, float))
    _x = np.asarray(_cd, float)[_o]; _y = np.asarray(_cl, float)[_o]
    # break the polyline where cd jumps by >0.006 between cl-neighbors
    # (identity-ambiguous / post-stall points would otherwise draw false
    # horizontal connectors)
    _xp, _yp = list(_x), list(_y)
    for _i in range(len(_x) - 1, 0, -1):
        if abs(_x[_i] - _x[_i-1]) > 0.006:
            _xp.insert(_i, np.nan); _yp.insert(_i, np.nan)
    ax.plot(_xp, _yp, '-', marker='.', ms=2.5, **_kw)

_ofp = f'{PD}/data/openfoam_airfoil_summary.json'
assert os.path.exists(_ofp), f'missing OpenFOAM summary: {_ofp}'
# L1+L2, size by level (L0 excluded; see regen_nlf_aft_comparison)
_OF = [v for k, v in __import__('json').load(open(_ofp))['cases'].items()
       if k.startswith('nlf_') and v['level'] in ('L1', 'L2')]
for _lv, _ms in (('L1', 3.6), ('L2', 5.4)):
    _p = [v for v in _OF if v['level'] == _lv]
    ax.plot([v['cd'] for v in _p], [v['cl'] for v in _p], 'o', ms=_ms,
            mfc='none', mec='steelblue', mew=1.1, zorder=2.5)
# The 2026-08 extension incidences on the L2 pair (-12, -10, -6, -2, 2, 6, 8),
# read from the committed matrix rather than the case tree, and merged into the
# L2 polars by incidence so the curve is drawn on every solution that exists.
_EXT_TAGS = ('am12', 'am10', 'am6', 'am2', 'a2', 'a6', 'a8')
_ext = {}
_extp = f'{PD}/data/ext2026_matrix.json'
assert os.path.exists(_extp), f'missing extension matrix: {_extp}'
for _k, _v in __import__('json').load(open(_extp))['cases'].items():
    if _v.get('airfoil') == 'nlf0416' and _v.get('Re') == 4e6 \
            and _k.endswith(_EXT_TAGS):
        _ext.setdefault(_v['family'], []).append(
            (_v['alpha'], _v['CL'], _v['CD']))

for mesh in ['str','cav']:
    for level in ['L0','L1','L2']:
        pts=[]
        # negative-incidence pair (alpha=-4 zero-lift, -8 measured-range edge)
        # exists on the finest grids only
        ladder=[(-8.0, f"{B}/{mesh}{level}prop_nlf0416_Re4M_am8"),
                (-4.0, f"{B}/{mesh}{level}prop_nlf0416_Re4M_am4")] \
               + [(float(a), case_dir(mesh,level,a)) for a in alphas]
        for aa,cdir in ladder:
            f=converged_clcd(cdir)
            if f: pts.append((aa,f[0],f[1]))
        if level=='L2': pts += _ext.get(mesh, [])
        pts.sort()
        cl=[q[1] for q in pts]; cd=[q[2] for q in pts]
        if cl: ax.plot(cd,cl,marker=MESH_MK[mesh],ms=4,ls=MESH_LS[mesh],
                       lw=LEVEL_LW[level],color=MESH_COL[mesh],zorder=3)
ax.set_xlim(0,0.04); ax.set_ylim(-1.1,2.05)
ax.set_xlabel('$C_d$'); ax.set_ylabel('$C_l$'); ax.grid(alpha=0.3)
handles=[Line2D([],[],color='k',ls='-',marker='o',mfc='none',ms=4,label='Experiment (Somers TP-1861)'),
         Line2D([],[],color='0.4',ls=':',marker='s',mfc='none',ms=5,lw=1.2,label='mfoil ($e^9$, $\\alpha\\!\\leq\\!7^\\circ$)'),
         Line2D([],[],color='0.5',ls='none',marker='D',mfc='none',ms=6,mew=1.3,label='Flexcompute Foil ($e^9$, $\\alpha\\!\\geq\\!9^\\circ$)'),
         Line2D([],[],color='0.45',ls='--',lw=1.3,label='AFT (OVERFLOW), $N_\\mathrm{crit}=10.07$'),
         Line2D([],[],color='0.45',ls='-',lw=1.3,label='AFT (OVERFLOW), $N_\\mathrm{crit}=7.18$'),
         Line2D([],[],color='C0',ls='-', marker='o',ms=4,label='SA-AI, structured (O-grid)'),
         Line2D([],[],color='C1',ls='--',marker='^',ms=4,label='SA-AI, unstructured'),
         Line2D([],[],color='0.55',ls='-.',marker='v',mfc='none',ms=5,lw=1.2,label='SA, fully turbulent (str L2)'),
         Line2D([],[],color='steelblue',marker='o',mfc='none',ls='none',ms=4,label='OpenFOAM (str L1/L2 by size)'),
         Line2D([],[],color='0.82',lw=0.55,label='workshop submittals'),
         Line2D([],[],color='mediumpurple',lw=0.9,label='workshop AFT'),
         Line2D([],[],color='seagreen',lw=1.1,label='workshop B–C'),
         Line2D([],[],color='0.4',lw=LEVEL_LW['L0'],label='L0'),
         Line2D([],[],color='0.4',lw=LEVEL_LW['L1'],label='L1'),
         Line2D([],[],color='0.4',lw=LEVEL_LW['L2'],label='L2')]
ax.legend(handles=handles,fontsize=8,loc='lower right')
_scale_text_for_page(fig)
plt.tight_layout()
plt.savefig(f'{PD}/figs/nlf_polar_compare.pdf'); plt.savefig('/tmp/nlf_polar_compare.png',dpi=140)
print('wrote nlf_polar_compare.pdf')
