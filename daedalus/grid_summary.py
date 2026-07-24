"""Grid-convergence summary across L0/L1/L2, both mesh families, alpha sweep:
polar with refinement trails, transition-front & bubble-length convergence,
and a totals table figure. AVL+XFOIL free-transition reference included.

Usage: python3 grid_summary.py
Writes /tmp/daedalus_mesh_views/sectional/{polar_levels,transition_levels,
totals_table}.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = '/tmp/daedalus_mesh_views/sectional'

CASES = {
    ('ogrid', 0, 4): 'case_ogrid_saai', ('ogrid', 0, 5): 'case_ogrid_saai_a5',
    ('ogrid', 0, 6): 'case_ogrid_saai_a6',
    ('cavity', 0, 4): 'case_cavity_saai', ('cavity', 0, 5): 'case_cavity_saai_a5',
    ('cavity', 0, 6): 'case_cavity_saai_a6',
    ('ogrid', 1, 4): 'case_ogrid_L1_saai', ('ogrid', 1, 5): 'case_ogrid_L1_saai_a5',
    ('ogrid', 1, 6): 'case_ogrid_L1_saai_a6',
    ('cavity', 1, 4): 'case_cavity_L1_saai', ('cavity', 1, 5): 'case_cavity_L1_saai_a5',
    ('cavity', 1, 6): 'case_cavity_L1_saai_a6',
    ('ogrid', 2, 4): 'case_ogrid_L2_saai_a4', ('ogrid', 2, 5): 'case_ogrid_L2_saai_a5',
    ('ogrid', 2, 6): 'case_ogrid_L2_saai_a6',
    ('cavity', 2, 4): 'case_cavity_L2_saai_a4', ('cavity', 2, 5): 'case_cavity_L2_saai_a5',
    ('cavity', 2, 6): 'case_cavity_L2_saai_a6',
}
DISP = {'ogrid': 'structured', 'cavity': 'unstructured'}
AVL_XFOIL = {4: (0.9758, 0.02318), 5: (1.0746, 0.02565), 6: (1.1730, 0.02837)}

# upper-surface Cf-rise front medians and bubble lengths (median over strips)
FRONT = {('ogrid', 0, 4): (0.565, 0.000), ('ogrid', 1, 4): (0.589, 0.035),
         ('ogrid', 2, 4): (0.613, 0.059),
         ('cavity', 0, 4): (0.577, 0.000), ('cavity', 1, 4): (0.577, 0.012),
         ('cavity', 2, 4): (0.601, 0.036),
         ('ogrid', 1, 5): (0.577, 0.024), ('ogrid', 2, 5): (0.589, 0.048),
         ('cavity', 1, 5): (0.577, 0.000), ('cavity', 2, 5): (0.577, 0.036),
         ('ogrid', 1, 6): (0.554, 0.024), ('ogrid', 2, 6): (0.577, 0.036),
         ('cavity', 1, 6): (0.542, 0.000), ('cavity', 2, 6): (0.554, 0.024)}
MFOIL_XTR = {4: 0.628, 5: 0.615}      # eta=0.305, N_crit=13.6


def totals(case):
    fn = f'{HERE}/{case}/total_forces_v2.csv'
    if not os.path.exists(fn):
        return None
    hdr = open(fn).readline().split(',')
    iCL = [i for i, h in enumerate(hdr) if h.strip() == 'CL'][0]
    iCD = [i for i, h in enumerate(hdr) if h.strip() == 'CD'][0]
    f = np.genfromtxt(fn, delimiter=',', skip_header=1)
    m = f[:, 1] >= f[-1, 1] - 500
    return float(f[m, iCL].mean()), float(f[m, iCD].mean())


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    T = {k: totals(c) for k, c in CASES.items()}
    T = {k: v for k, v in T.items() if v}

    # ---- polar with refinement trails ----
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cols = {'ogrid': 'C0', 'cavity': 'C1'}
    for fam in ('ogrid', 'cavity'):
        for a in (4, 5, 6):
            pts = [(T[(fam, lv, a)], lv) for lv in (0, 1, 2)
                   if (fam, lv, a) in T]
            cd = [p[0][1] for p in pts]
            cl = [p[0][0] for p in pts]
            ax.plot(cd, cl, '-', color=cols[fam], lw=0.8, alpha=0.5)
            for (CLCD, lv) in pts:
                ax.plot(CLCD[1], CLCD[0], 'o', color=cols[fam],
                        ms=4 + 3 * lv, mfc='none' if lv < 2 else cols[fam])
    ax.plot([v[1] for v in AVL_XFOIL.values()],
            [v[0] for v in AVL_XFOIL.values()], 'k-^', ms=7,
            label='AVL + XFOIL (N=13.6)')
    for fam in cols:
        ax.plot([], [], 'o-', color=cols[fam], label=f'SA-AI {DISP[fam]} (L0→L2)')
    ax.set_xlabel('$C_D$')
    ax.set_ylabel('$C_L$')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.savefig(f'{OUT}/polar_levels.png', dpi=140, bbox_inches='tight'); fig.savefig(f'{OUT}/polar_levels.pdf', bbox_inches='tight')
    plt.close(fig)

    # ---- transition front & bubble length vs level ----
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    for a, ls in ((4, '-'), (5, '--')):
        for fam in cols:
            lv = [l for l in (0, 1, 2) if (fam, l, a) in FRONT]
            fr = [FRONT[(fam, l, a)][0] for l in lv]
            bl = [FRONT[(fam, l, a)][1] for l in lv]
            axs[0].plot(lv, fr, ls, marker='o', color=cols[fam],
                        label=f'{DISP[fam]} $\\alpha$={a}$^\\circ$' if True else None)
            axs[1].plot(lv, bl, ls, marker='o', color=cols[fam])
        axs[0].axhline(MFOIL_XTR[a], color='k', ls=ls, lw=0.8)
        axs[0].text(2.05, MFOIL_XTR[a], f'mfoil $\\alpha$={a}', fontsize=7,
                    va='center')
    axs[0].set_ylabel('upper $x_{tr}/c$ (Cf-rise, median)')
    axs[1].set_ylabel('bubble length [c] (median)')
    for axx in axs:
        axx.set_xlabel('grid level')
        axx.set_xticks([0, 1, 2])
        axx.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.savefig(f'{OUT}/transition_levels.png', dpi=140, bbox_inches='tight'); fig.savefig(f'{OUT}/transition_levels.pdf', bbox_inches='tight')
    plt.close(fig)

    # ---- totals table ----
    rows = []
    for a in (4, 5, 6):
        for lv in (0, 1, 2):
            r = [f'{a}', f'L{lv}']
            for fam in ('ogrid', 'cavity'):
                v = T.get((fam, lv, a))
                r += [f'{v[0]:.4f} / {v[1]:.5f}' if v else '(running)']
            rows.append(r)
        v = AVL_XFOIL[a]
        rows.append([f'{a}', 'AVL+XFOIL', f'{v[0]:.4f} / {v[1]:.5f}', ''])
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(rows) + 1))
    ax.axis('off')
    tb = ax.table(cellText=rows,
                  colLabels=['alpha', 'level', 'O-grid CL / CD',
                             'cavity CL / CD'],
                  loc='center', cellLoc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 1.35)
    ax.set_title('SA-AI totals by grid level (last-500-step means)',
                 fontsize=11)
    fig.savefig(f'{OUT}/totals_table.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT, '(polar_levels, transition_levels, totals_table)')
