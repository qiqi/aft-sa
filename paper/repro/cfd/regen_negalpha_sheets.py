"""Wall-anchored contour sheets for the NLF(1)-0416 negative-incidence pair
in the STANDARD six-grid per-alpha format (same as every other incidence):
paper/figs/chi_sheet_nlf0416_{am8,am4}_{upper,lower}.pdf.

(The original two 4-row L2-pair sheets predate the L0/L1 negative runs of
2026-07-26; with all six grids computed, each alpha gets the full ladder.)

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


def sheet(amtag, alabel, side):
    m = S._mod(AF)
    L = CFG['L_up'] if side == 'upper' else CFG['L_lo']
    Re = CFG['Re']
    fig, axes = plt.subplots(6, 2, figsize=(11.5, 12.5), sharex=True,
                             sharey=True)
    for r, (gname, glabel) in enumerate(S.ROWS):
        case = f"{B}/{gname}_{CFG['casetag']}_{amtag}"
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
        print(f"  {gname} {amtag}: scanned", flush=True)
    axes[0, 0].set_title(r'$|\mathbf{u}|/U_\infty$', fontsize=10)
    axes[0, 1].set_title(r'$\log_{10}\chi$', fontsize=10)
    for c in range(2):
        axes[-1, c].set_xlabel('wall-anchor x/c')
    out = os.path.join(FIGS, f'chi_sheet_nlf0416_{amtag}_{side}.pdf')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    for amtag, alabel in (('am8', '-8'), ('am4', '-4')):
        for side in ('upper', 'lower'):
            sheet(amtag, alabel, side)
