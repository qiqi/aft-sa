"""Appendix wall-anchored contour sheets for the NLF negative-incidence
pair -> paper/figs/chi_sheet_nlf0416_negalpha_{upper,lower}.pdf.

Style of regen_chi_sheets.py (velocity magnitude | log10 chi, wall-anchored),
but the negative pair exists on the finest grids only, so each sheet carries
four rows: (cavity L2, O-grid L2) x (alpha = -4, -8).

NOTE: at alpha = -8 the pressure-side (upper) transition front is an
aft-marching transient at the fixed cold-start budget these solutions used
(a front-convergence rerun advances it ~0.08c per 5000 pseudo-steps); the
sheets show that state mid-march until the front-converged recomputation
replaces the case dirs.

Run from paper/: python3 repro/cfd/regen_negalpha_sheets.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regen_chi_sheets as S

B = S.B
FIGS = S.FIGS
AF = 'nlf0416'
CFG = S.AF_SETUP[AF]
ROWS = [('cav', -4, 'cavity L2, $\\alpha=-4^\\circ$'),
        ('str', -4, 'O-grid L2, $\\alpha=-4^\\circ$'),
        ('cav', -8, 'cavity L2, $\\alpha=-8^\\circ$'),
        ('str', -8, 'O-grid L2, $\\alpha=-8^\\circ$')]


def case_dir(fam, alpha):
    return f"{B}/{fam}L2prop_nlf0416_Re4M_am{-int(alpha)}"


def sheet(side):
    m = S._mod(AF)
    L = CFG['L_up'] if side == 'upper' else CFG['L_lo']
    Re = CFG['Re']
    fig, axes = plt.subplots(4, 2, figsize=(11.5, 8.6), sharex=True,
                             sharey=True)
    for r, (fam, alpha, glabel) in enumerate(ROWS):
        case = case_dir(fam, alpha)
        x, d, chi, um = S.scan(m, AF, case, side, L)
        axU, axC = axes[r]
        cs = axU.contour(x, d*Re, um, levels=S.U_LEV, colors='k',
                         linewidths=0.6)
        axU.clabel(cs, levels=S.U_LABEL, fmt='%g', fontsize=6.5,
                   inline_spacing=2)
        logchi = np.log10(np.clip(chi, 1e-8, None))
        axC.contour(x, d*Re, logchi, levels=S.CHI_MINOR, colors='k',
                    linewidths=0.4)
        cs = axC.contour(x, d*Re, logchi, levels=S.CHI_MAJOR, colors='k',
                         linewidths=0.8)
        axC.clabel(cs, fmt=S.CHI_FMT, fontsize=6.5, inline_spacing=2)
        axU.set_xlim(0, 1); axU.set_ylim(0, L*Re)
        axU.set_ylabel(f'{glabel}\n' + r'$d\,U_\infty/\nu$', fontsize=9)
        print(f"  {fam} am{-alpha}: scanned", flush=True)
    axes[0, 0].set_title(r'$|\mathbf{u}|/U_\infty$', fontsize=10)
    axes[0, 1].set_title(r'$\log_{10}\chi$', fontsize=10)
    for c in range(2):
        axes[-1, c].set_xlabel('wall-anchor x/c')
    out = os.path.join(FIGS, f'chi_sheet_nlf0416_negalpha_{side}.pdf')
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == '__main__':
    for side in ('upper', 'lower'):
        sheet(side)
