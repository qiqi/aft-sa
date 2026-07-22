"""fig:daepolar -> figs/daedalus_polar_sectional.pdf

Left: Daedalus wing polar, both mesh families at L0/L1/L2 (colour = family,
line weight = level, house conventions) against the AVL+XFOIL strip-theory
reference. Right: sectional lift on the finest (L2) grids at the three
incidences against the AVL distribution. Reads the case tree at
sa-ai/scripts/daedalus and the AVL work dir built by avl_compare.py."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/home/qiqi/flexcompute/sa-ai/scripts/daedalus'
sys.path.insert(0, D)
import sectional_compare as SC
from polar_compare import run_avl, rans_totals as _rt  # noqa: E402
from wing_geometry import HALF_SPAN  # noqa: E402

CASES = {
    ('str', 0): ['case_ogrid_saai', 'case_ogrid_saai_a5', 'case_ogrid_saai_a6'],
    ('str', 1): ['case_ogrid_L1_saai', 'case_ogrid_L1_saai_a5', 'case_ogrid_L1_saai_a6'],
    ('str', 2): ['case_ogrid_L2_saai_a4', 'case_ogrid_L2_saai_a5', 'case_ogrid_L2_saai_a6'],
    ('cav', 0): ['case_cavity_saai', 'case_cavity_saai_a5', 'case_cavity_saai_a6'],
    ('cav', 1): ['case_cavity_L1_saai', 'case_cavity_L1_saai_a5', 'case_cavity_L1_saai_a6'],
    ('cav', 2): ['case_cavity_L2_saai_a4', 'case_cavity_L2_saai_a5', 'case_cavity_L2_saai_a6'],
}
AVL_XFOIL = {4: (0.9758, 0.02318), 5: (1.0746, 0.02565), 6: (1.1730, 0.02837)}
ALPHAS = [4.0, 5.0, 6.0]
COL = {'str': 'C0', 'cav': 'C1'}
LW = {0: 0.8, 1: 1.6, 2: 2.4}
LAB = {'str': 'structured O-grid', 'cav': 'unstructured'}


def totals(case):
    fn = f'{D}/{case}/total_forces_v2.csv'
    hdr = open(fn).readline().split(',')
    iCL = [i for i, h in enumerate(hdr) if h.strip() == 'CL'][0]
    iCD = [i for i, h in enumerate(hdr) if h.strip() == 'CD'][0]
    f = np.genfromtxt(fn, delimiter=',', skip_header=1)
    m = f[:, 1] >= f[-1, 1] - 500
    return float(f[m, iCL].mean()), float(f[m, iCD].mean())


def main():
    fig, (axp, axs) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                                   gridspec_kw={'width_ratios': [1, 1.25]})
    # ---- left: polar ----
    for fam in ('str', 'cav'):
        for lv in (0, 1, 2):
            pts = [totals(c) for c in CASES[(fam, lv)]]
            axp.plot([p[1] for p in pts], [p[0] for p in pts], '-o',
                     color=COL[fam], lw=LW[lv], ms=2.5 + lv,
                     label=f'{LAB[fam]} L{lv}' if fam == 'str' or lv == 0 else None)
    axp.plot([v[1] for v in AVL_XFOIL.values()],
             [v[0] for v in AVL_XFOIL.values()], 's--', color='0.4', ms=5,
             lw=1.2, label='AVL$+$XFOIL ($N{=}13.6$)')
    axp.set_xlabel('$C_D$')
    axp.set_ylabel('$C_L$')
    axp.grid(alpha=0.3)
    handles, labels = axp.get_legend_handles_labels()
    # rebuild a compact legend: one entry per family + levels via weight note
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], color=COL['str'], lw=1.6, marker='o', ms=3.5,
                 label='structured O-grid (L0$\\to$L2 by weight)'),
          Line2D([], [], color=COL['cav'], lw=1.6, marker='o', ms=3.5,
                 label='unstructured (L0$\\to$L2 by weight)'),
          Line2D([], [], color='0.4', lw=1.2, ls='--', marker='s', ms=5,
                 label='AVL$+$XFOIL ($N{=}13.6$)')]
    axp.legend(handles=hl, fontsize=8, loc='lower right')
    axp.text(0.03, 0.93, '(a)', transform=axp.transAxes, fontsize=11,
             fontweight='bold')

    # ---- right: sectional cl (left axis) and cd (right axis) at L2 vs AVL+FlexFoil ----
    import pickle
    from wing_geometry import chord as _chord
    FF = pickle.load(open('/home/qiqi/flexcompute/sa-ai/flow360_ai/flexfoil_daedalus_strips.pkl', 'rb'))
    axd = axs.twinx()
    acol = {4.0: '0.15', 5.0: 'C3', 6.0: 'C2'}
    for a in ALPHAS:
        SC.ALPHA = np.deg2rad(a)
        for fam, ls in (('str', '-'), ('cav', '--')):
            case = CASES[(fam, 2)][ALPHAS.index(a)]
            e, cl, cd = SC.native_strips(case)
            axs.plot(e, cl, ls, color=acol[a], lw=1.3)
            axd.plot(e, cd, ls, color=acol[a], lw=0.7, alpha=0.65)
        cl_t, cdi_t, strips = run_avl(a)
        eta_s = strips[:, 0] / HALF_SPAN
        axs.plot(eta_s, strips[:, 2], ':', color=acol[a], lw=1.8)
        # reference sectional cd: AVL induced (cl*ai, Trefftz-rescaled) + FlexFoil profile
        ci = strips[:, 2] * strips[:, 5] if strips.shape[1] > 5 else strips[:, 2] * strips[:, 3]
        S_REF, b2 = 30.84, HALF_SPAN
        integ = 2.0 * np.trapezoid(ci * strips[:, 1], strips[:, 0]) / S_REF
        k = cdi_t / integ if integ > 0 else 1.0
        ff = FF[a]
        eta_ff = np.asarray(ff['eta'])
        cdp = np.array([st['cd'] if (st and st.get('cd') is not None) else np.nan
                        for st in ff['stations']], float)
        cd_ref = np.interp(eta_ff, eta_s, ci * k) + cdp
        mgood = np.isfinite(cd_ref)
        axd.plot(eta_ff[mgood], cd_ref[mgood], ':', color=acol[a], lw=0.9,
                 alpha=0.65)
    axd.set_ylabel(r'sectional $c_d$ (thin)')
    axd.set_ylim(0, 0.08)
    from matplotlib.lines import Line2D
    hl2 = ([Line2D([], [], color=acol[a], lw=1.5,
                   label=f'$\\alpha={a:.0f}^\\circ$') for a in ALPHAS] +
           [Line2D([], [], color='0.3', ls='-', lw=1.3, label='structured L2'),
            Line2D([], [], color='0.3', ls='--', lw=1.3, label='unstructured L2'),
            Line2D([], [], color='0.3', ls=':', lw=1.8, label='AVL ($c_l$) / +FlexFoil ($c_d$)')])
    axs.legend(handles=hl2, fontsize=8, ncol=2, loc='lower left')
    axs.set_xlabel(r'$\eta = 2y/b$')
    axs.set_ylabel('sectional $c_l$')
    axs.set_xlim(0, 1)
    axs.grid(alpha=0.3)
    axs.text(0.03, 0.93, '(b)', transform=axs.transAxes, fontsize=11,
             fontweight='bold')
    fig.tight_layout()
    fig.savefig('figs/daedalus_polar_sectional.pdf')
    print('wrote figs/daedalus_polar_sectional.pdf')


if __name__ == '__main__':
    main()
