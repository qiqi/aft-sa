"""Two E387 alpha=5 Cf/Cp comparison figures, each overlaying the cav (unstructured)
and str (structured) L1 meshes at two Reynolds numbers:
  Figure A: Re = 60k, 100k     Figure B: Re = 300k, 460k
Rows: -Cp and Cf (upper & lower surface).  Grey band = experimental laminar-
separation-bubble (McGhee TM-4062 Table III, oil flow) -- ONLY at the Re where
oil-flow data exist (100k, 200k, 300k); 60k and 460k have no experimental bubble
data (oil flow was run only to 300k), so no band is drawn there.
Experimental Cp (digitized from TM-4062 Fig. 22) overlaid where available.
-> figs/eppler_L1compare_lowRe.pdf, figs/eppler_L1compare_highRe.pdf
"""
import sys, os; sys.path.insert(0, '.')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import regen_eppler_v2 as R

B = "/home/qiqi/flexcompute/aft-sa/flow360"
UP, LO = R.UP_COLOR, R.LO_COLOR
# Experimental LSB band (x_LS, x_TR) at alpha=5, oil flow, TM-4062 Table III.
EXP_LSB_RE = {100: (0.35, 0.67), 200: (0.38, 0.59), 300: (0.40, 0.58)}
# Digitized experimental Cp at alpha=5 from TM-4062 Fig. 22 (filled in by
# digitize_cp_tm4062.py -> exp_cp_alpha5.json). {Re:{'upper':[[x],[cp]],'lower':...}}
import json
EXP_CP = {}
_cpf = "data/exp_cp_alpha5.json"
if os.path.exists(_cpf):
    EXP_CP = {int(k): v for k, v in json.load(open(_cpf)).items()}

def cav_dir(Rk): return f"{B}/cavL1prop_eppler387_Re200k_a5" if Rk == 200 else f"{B}/sweep_Re{Rk}k_a5"
def str_dir(Rk): return f"{B}/strL1prop_eppler387_Re200k_a5" if Rk == 200 else f"{B}/sweep_str_Re{Rk}k_a5"

def make_fig(re_list, out):
    fig, axs = plt.subplots(2, len(re_list), figsize=(5.2 * len(re_list), 7.4), sharex=True)
    for col, Rk in enumerate(re_list):
        ax_cp, ax_cf = axs[0, col], axs[1, col]
        for mesh, dfn, ls, lab in [("cav L1", cav_dir, '-', 'unstructured'),
                                    ("str L1", str_dir, '--', 'structured')]:
            try:
                (xu, cfu, cpu), (xl, cfl, cpl) = R.airfoil_walk_contour(dfn(Rk))
                ax_cp.plot(xu, -cpu, ls, lw=1.5, color=UP)
                ax_cp.plot(xl, -cpl, ls, lw=1.5, color=LO)
                ax_cf.plot(xu, cfu, ls, lw=1.5, color=UP)
                ax_cf.plot(xl, cfl, ls, lw=1.5, color=LO)
            except Exception as e:
                print(f"Re{Rk}k {mesh}: {e}")
        # experimental bubble band (only where oil-flow data exist)
        if Rk in EXP_LSB_RE:
            xls, xtr = EXP_LSB_RE[Rk]
            for a in (ax_cp, ax_cf):
                a.axvspan(xls, xtr, color='0.55', alpha=0.30, zorder=0)
        else:
            ax_cp.text(0.03, 0.05, "no exp. bubble/$C_p$ data", transform=ax_cp.transAxes,
                       ha='left', va='bottom', fontsize=8, color='0.4', style='italic')
        # experimental Cp overlay
        if Rk in EXP_CP:
            for side, c in [('upper', UP), ('lower', LO)]:
                if side in EXP_CP[Rk]:
                    x, cp = EXP_CP[Rk][side]
                    ax_cp.plot(x, -np.array(cp), 'o', ms=3.2, mfc='none', mec='k', mew=0.7, zorder=5)
        ax_cp.set_title(f"Re = {Rk}k", fontsize=12)
        ax_cp.set_ylabel(r'$-C_p$'); ax_cf.set_ylabel(r'$C_f$')
        ax_cp.grid(alpha=0.3); ax_cf.grid(alpha=0.3)
        ax_cf.set_ylim(-0.004, 0.012); ax_cf.axhline(0, color='gray', lw=0.6, alpha=0.5)
        ax_cf.set_xlim(0, 1); ax_cf.set_xlabel('$x/c$')
    # legend
    handles = [Line2D([], [], color=UP, lw=2, label='upper'),
               Line2D([], [], color=LO, lw=2, label='lower'),
               Line2D([], [], color='0.3', lw=2, ls='-', label='cav L1 (unstructured)'),
               Line2D([], [], color='0.3', lw=2, ls='--', label='str L1 (structured)'),
               Patch(facecolor='0.55', alpha=0.30, label='exp. LSB (TM-4062 oil flow)')]
    if EXP_CP:
        handles.append(Line2D([], [], marker='o', ls='none', mfc='none', mec='k', label='exp. $C_p$ (TM-4062 Fig. 22)'))
    fig.legend(handles=handles, fontsize=9, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f'Eppler 387, $\\alpha=5^\\circ$: L1 unstructured vs structured', y=0.995, fontsize=13)
    plt.tight_layout(rect=(0, 0.07, 1, 0.98))
    plt.savefig(out); plt.savefig("/tmp/" + os.path.basename(out).replace('.pdf', '.png'), dpi=120)
    print("wrote", out)

make_fig([60, 100], "figs/eppler_L1compare_lowRe.pdf")
make_fig([300, 460], "figs/eppler_L1compare_highRe.pdf")
