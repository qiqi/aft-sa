"""Wall-anchored contour sheets for the Eppler 387 extension incidences
(alpha = -2 and 8.5, finest L2 pair only) -> paper/figs/
chi_sheet_eppler387_{am2,a8p5}_upper.pdf, matching the Appendix-B
conventions (upper surface, L_up = 0.0335c frame).

Run from paper/: python3 repro/cfd/regen_epp_ext_sheets.py
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
AF = 'eppler387'
CFG = S.AF_SETUP[AF]

SHEETS = {
    'am2':  [('cav', 'am2', 'cavity L2, $\\alpha=-2^\\circ$'),
             ('str', 'am2', 'O-grid L2, $\\alpha=-2^\\circ$')],
    'a8p5': [('cav', 'a8p5', 'cavity L2, $\\alpha=8.5^\\circ$'),
             ('str', 'a8p5', 'O-grid L2, $\\alpha=8.5^\\circ$')],
}


def sheet(name, rows):
    m = S._mod(AF)
    L = CFG['L_up']
    Re = CFG['Re']
    fig, axes = plt.subplots(len(rows), 2, figsize=(11.5, 4.5), sharex=True,
                             sharey=True)
    for r, (fam, tag, glabel) in enumerate(rows):
        case = f"{B}/{fam}L2prop_eppler387_Re200k_{tag}"
        x, d, chi, um = S.scan(m, AF, case, 'upper', L)
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
        print(f"  {fam} {tag}: scanned", flush=True)
    axes[0, 0].set_title(r'$|\mathbf{u}|/U_\infty$', fontsize=10)
    axes[0, 1].set_title(r'$\log_{10}\chi$', fontsize=10)
    for c in range(2):
        axes[-1, c].set_xlabel('wall-anchor x/c')
    fig.tight_layout()
    out = os.path.join(FIGS, f'chi_sheet_eppler387_{name}_upper.pdf')
    fig.savefig(out)
    plt.close(fig)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    for name, rows in SHEETS.items():
        sheet(name, rows)
