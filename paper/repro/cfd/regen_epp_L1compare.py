"""Two E387 alpha=5 Cf/Cp comparison figures, each overlaying the cav (unstructured)
and str (structured) L1 meshes at two Reynolds numbers:
  Figure A: Re = 60k, 100k     Figure B: Re = 300k, 460k
Rows: max chi (log, left) with mfoil N (linear, right) where mfoil data exist;
-Cp; and Cf (upper & lower surface).  Grey band = experimental laminar-
separation-bubble (McGhee TM-4062 Table III, oil flow) -- ONLY at the Re where
oil-flow data exist (100k, 200k, 300k); 60k and 460k have no experimental bubble
data (oil flow was run only to 300k), so no band is drawn there.
Experimental Cp (hand-read & verified from TM-4062 Appendix D tables, nearest-5deg
column per Re) overlaid as open circles.
-> figs/eppler_L1compare_lowRe.pdf, figs/eppler_L1compare_highRe.pdf
"""
import sys, os; sys.path.insert(0, '.')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import regen_eppler_v2 as R

B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_ai")
UP, LO = R.UP_COLOR, R.LO_COLOR
# Experimental LSB band (x_LS, x_TR) at alpha=5, oil flow, TM-4062 Table III.
EXP_LSB_RE = {100: (0.35, 0.67), 200: (0.38, 0.59), 300: (0.40, 0.58)}
# mfoil reference Cp/Cf at alpha=5 (accurate viscous-inviscid e^9 solution; no
# scanned-plot digitization). {Re_k: {'upper':{'x','cp','cf'}, 'lower':{...}}}.
import pickle
_mf = f"{B}/mfoil_eppler387_sweep_a5.pkl"
MFOIL = pickle.load(open(_mf, 'rb')) if os.path.exists(_mf) else {}
# xfoil fallback for any Re where mfoil failed to converge (low Re / big bubble).
_xf = f"{B}/xfoil_eppler387_sweep_a5.pkl"
XFOIL = pickle.load(open(_xf, 'rb')) if os.path.exists(_xf) else {}
# FlexFoil amplification envelopes (independent XFOIL-closure implementation,
# validated against mfoil N at Re=200k) for the Re where mfoil is deposed/absent
_ff = f"{B}/flexfoil_eppler387_sweep_a5.pkl"
FLEXN = pickle.load(open(_ff, 'rb')) if os.path.exists(_ff) else {}
# EXACT experimental Cp from TM-4062 Appendix D (hand-read + verified against the
# scanned tables). Nearest-to-5deg column per Re (60k has only 4.99).
import json as _json
_ct = "data/exp_cp_tables.json"
EXP_CP_TAB = _json.load(open(_ct)) if os.path.exists(_ct) else {}
RE_COL = {60: "4.99", 100: "5.01", 200: "5.05", 300: "5.00", 460: "5.01"}

def cav_dir(Rk): return f"{B}/cavL1prop_eppler387_Re200k_a5" if Rk == 200 else f"{B}/sweep_Re{Rk}k_a5"
def str_dir(Rk): return f"{B}/strL1prop_eppler387_Re200k_a5" if Rk == 200 else f"{B}/sweep_str_Re{Rk}k_a5"

def make_fig(re_list, out):
    fig, axs = plt.subplots(3, len(re_list), figsize=(5.2 * len(re_list), 10.4), sharex=True)
    for col, Rk in enumerate(re_list):
        ax_chi, ax_cp, ax_cf = axs[0, col], axs[1, col], axs[2, col]
        ax_N = ax_chi.twinx()
        NU_Re = 0.1 / (Rk * 1000.0)      # muRef = M/Re, M=0.1 (chi = nuHat/muRef)
        # ROW 0: max chi (log, left) from the CFD BL, and mfoil N (linear, right)
        for dfn, ls in [(cav_dir, '-'), (str_dir, '--')]:
            try:
                xc, mu, ml = R.max_chi_vs_x(dfn(Rk))
                ax_chi.semilogy(xc, mu / NU_Re, ls, lw=1.5, color=UP)
                ax_chi.semilogy(xc, ml / NU_Re, ls, lw=1.5, color=LO)
            except Exception as e:
                print(f"Re{Rk}k chi: {e}")
        # At Re=60k the two e^9 implementations visibly differ and XFOIL is
        # presented as the primary reference (no amplification envelope there).
        XFOIL_PRIMARY = {60}
        mf = None if Rk in XFOIL_PRIMARY else MFOIL.get(Rk)
        nsrc = FLEXN.get(Rk) if (mf is None or 'n' not in mf['upper']) else mf
        if nsrc is not None and 'n' in nsrc['upper']:
            ax_N.plot(nsrc['upper']['x'], nsrc['upper']['n'], ':', color=UP, lw=1.4, alpha=0.8)
            ax_N.plot(nsrc['lower']['x'], nsrc['lower']['n'], ':', color=LO, lw=1.4, alpha=0.8)
            ax_N.axhline(9.0, color='gray', ls=':', lw=0.6, alpha=0.6)
        else:
            ax_chi.text(0.03, 0.92, "no $e^9$ $N$", transform=ax_chi.transAxes,
                        ha='left', va='top', fontsize=8, color='0.4', style='italic')
        ax_chi.axhline(R.C_V1, color='gray', ls=':', lw=0.6, alpha=0.6)   # chi=c_v1 (handover)
        ax_chi.set_ylim(R.CHI_LO, R.CHI_HI); ax_N.set_ylim(R.N_LO, R.N_HI)
        ax_chi.set_title(f"Re = {Rk}k", fontsize=12)
        if col == 0: ax_chi.set_ylabel(r'$\chi$ (log)')
        if col == len(re_list) - 1: ax_N.set_ylabel('$e^9$ $N$ (linear)')
        ax_chi.grid(alpha=0.3)
        # ROWS 1+2: -Cp and Cf
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
            for a in (ax_chi, ax_cp, ax_cf):
                a.axvspan(xls, xtr, color='0.55', alpha=0.30, zorder=0)
        else:
            ax_cp.text(0.03, 0.05, "no exp. bubble/$C_p$ data", transform=ax_cp.transAxes,
                       ha='left', va='bottom', fontsize=8, color='0.4', style='italic')
        # mfoil (or xfoil fallback) reference Cp/Cf -- accurate e^9 viscous solution
        use_xf = Rk in XFOIL_PRIMARY and Rk in XFOIL
        ref = (XFOIL.get(Rk) if use_xf else None) or MFOIL.get(Rk) or XFOIL.get(Rk)
        reftag = ('xfoil' if (use_xf or Rk not in MFOIL) and Rk in XFOIL
                  else ('mfoil' if Rk in MFOIL else None))
        if ref is not None:
            for side, c in [('upper', UP), ('lower', LO)]:
                s = ref[side]
                ax_cp.plot(s['x'], -np.asarray(s['cp']), ':', color=c, lw=1.4, alpha=0.8, zorder=4)
                ax_cf.plot(s['x'], np.asarray(s['cf']), ':', color=c, lw=1.4, alpha=0.8, zorder=4)
        # exact experimental Cp (TM-4062 Appendix D, hand-read & verified) at nearest-5deg col
        etab = EXP_CP_TAB.get(str(Rk)); ecol = RE_COL.get(Rk)
        if etab is not None and ecol is not None:
            for side, c in [('upper', UP), ('lower', LO)]:
                if ecol in etab[side]:
                    xcs = np.asarray(etab[side]['xc']); cps = np.asarray(etab[side][ecol])
                    ax_cp.plot(xcs, -cps, 'o', mfc='none', mec=c, mew=1.1, ms=5, zorder=6)
        ax_cp.set_ylabel(r'$-C_p$'); ax_cf.set_ylabel(r'$C_f$')
        ax_cp.grid(alpha=0.3); ax_cf.grid(alpha=0.3)
        ax_cp.set_ylim(bottom=-1.0)   # -Cp >= -1 (Cp <= +1 stagnation; no unphysical range)
        ax_cf.set_ylim(-0.004, 0.012); ax_cf.axhline(0, color='gray', lw=0.6, alpha=0.5)
        ax_cf.set_xlim(0, 1); ax_cf.set_xlabel('$x/c$')
    # legend
    handles = [Line2D([], [], color=UP, lw=2, label='upper'),
               Line2D([], [], color=LO, lw=2, label='lower'),
               Line2D([], [], color='0.3', lw=2, ls='-', label='cav L1 (unstructured)'),
               Line2D([], [], color='0.3', lw=2, ls='--', label='str L1 (structured)'),
               Patch(facecolor='0.55', alpha=0.30, label='exp. LSB (TM-4062 oil flow)')]
    if MFOIL or XFOIL:
        handles.append(Line2D([], [], color='0.3', ls=':', lw=1.4, label='mfoil/xfoil ($e^9$)'))
    if EXP_CP_TAB:
        handles.append(Line2D([], [], color='0.3', ls='none', marker='o', mfc='none', mew=1.1, ms=5,
                              label=r'exp. $C_p$ (TM-4062, $\alpha\!\approx\!5^\circ$)'))
    fig.legend(handles=handles, fontsize=9, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=(0, 0.07, 1, 1.0))
    plt.savefig(out); plt.savefig("/tmp/" + os.path.basename(out).replace('.pdf', '.png'), dpi=120)
    print("wrote", out)

make_fig([60, 100], "figs/eppler_L1compare_lowRe.pdf")
make_fig([300, 460], "figs/eppler_L1compare_highRe.pdf")
