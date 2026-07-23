"""Two E387 alpha=5 Re-sweep figures with the SAME five-row suite as the
benchmark surface-distribution figures (regen_eppler_v2.make_cf_figure),
each overlaying BOTH mesh families at ALL THREE refinement levels at two
Reynolds numbers:
  Figure A: Re = 60k, 100k     Figure B: Re = 300k, 460k
Rows top-to-bottom:
  (1) max Re_Omega along a 0.01c wall-normal probe (needs
      slice_with_derived.pvtu from add_derived_to_slice.py);
  (2) max S_hat*g along the same probe (the sphere rate coordinate);
  (3) max chi (log, left) with the e^9 N envelope (linear, right);
  (4) -Cp;  (5) signed Cf.
Line conventions match the benchmark figures: upper blue / lower red,
str solid / cav dashed, L0/L1/L2 by thickness (LEVEL_LW).
Case dirs: Re=200k uses the benchmark {fam}{lvl}prop cases; L1 at other Re
uses the legacy sweep_[str_]Re{Rk}k_a5 names -- LEGACY NAMES ONLY: those 8
cases were re-converged at the committed whole-equation canon by
run_sphere_campaign.py epp_sweep (2026-07-23), same kernel as every other
curve here; L0/L2 use sweep_{fam}{lvl}_Re{Rk}k_a5
(build_eppler_resweep_levels.py).
Grey band = experimental LSB (McGhee TM-4062 Table III oil flow; only at Re
with data: 100k, 200k, 300k). Experimental Cp (TM-4062 App. D tables,
nearest-5deg column) as open circles.
-> figs/eppler_resweep_lowRe.pdf, figs/eppler_resweep_highRe.pdf
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.abspath(os.path.join(_HERE, '..', '..'))    # paper/
sys.path.insert(0, _HERE)
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import regen_eppler_v2 as R

B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fr")
UP, LO = R.UP_COLOR, R.LO_COLOR
LEVELS = ['L0', 'L1', 'L2']
FAMS = ['cav', 'str']
# Sphere-kernel onset floor k*A (pinned by tests/test_constants_consistency.py).
REOMC_FLOOR = 124.6
# Experimental LSB band (x_LS, x_TR) at alpha=5, oil flow, TM-4062 Table III.
EXP_LSB_RE = {100: (0.35, 0.67), 200: (0.38, 0.59), 300: (0.40, 0.58)}
import pickle
_mf = f"{B}/mfoil_eppler387_sweep_a5.pkl"
MFOIL = pickle.load(open(_mf, 'rb')) if os.path.exists(_mf) else {}
_xf = f"{B}/xfoil_eppler387_sweep_a5.pkl"
XFOIL = pickle.load(open(_xf, 'rb')) if os.path.exists(_xf) else {}
_ff = f"{B}/flexfoil_eppler387_sweep_a5.pkl"
FLEXN = pickle.load(open(_ff, 'rb')) if os.path.exists(_ff) else {}
import json as _json
_ct = f"{PD}/data/exp_cp_tables.json"
EXP_CP_TAB = _json.load(open(_ct)) if os.path.exists(_ct) else {}
RE_COL = {60: "4.99", 100: "5.01", 200: "5.05", 300: "5.00", 460: "5.01"}


def sweep_dir(fam, lvl, Rk):
    if Rk == 200:
        return f"{B}/{fam}{lvl}prop_eppler387_Re200k_a5"
    if lvl == 'L1':
        return (f"{B}/sweep_Re{Rk}k_a5" if fam == 'cav'
                else f"{B}/sweep_str_Re{Rk}k_a5")
    return f"{B}/sweep_{fam}{lvl}_Re{Rk}k_a5"


def for_each_case(Rk):
    for fam in FAMS:
        for lvl in LEVELS:
            yield fam, lvl, sweep_dir(fam, lvl, Rk), R.MESH_LS[fam], R.LEVEL_LW[lvl]


def make_fig(re_list, out):
    fig, axs = plt.subplots(5, len(re_list), figsize=(5.2*len(re_list), 13.6),
                            sharex=True)
    for col, Rk in enumerate(re_list):
        ax_reo, ax_P, ax_chi, ax_cp, ax_cf = (axs[i, col] for i in range(5))
        ax_N = ax_chi.twinx()
        NU_Re = 0.1/(Rk*1000.0)     # muRef = M/Re, M=0.1 (chi = nuHat/muRef)
        # ROWS 1+2: wall-normal-probe max Re_Omega and max Shat*g
        for fam, lvl, d, ls, lw in for_each_case(Rk):
            if not os.path.exists(f"{d}/slice_with_derived.pvtu"):
                print(f"Re{Rk}k {fam}{lvl}: no slice_with_derived"); continue
            try:
                xs_u, ReO_u, _, P_u = R.wallnormal_max_metrics(
                    d, side='upper', return_P=True)
                xs_l, ReO_l, _, P_l = R.wallnormal_max_metrics(
                    d, side='lower', return_P=True)
            except Exception as e:
                print(f"Re{Rk}k probe {fam}{lvl}: {e}"); continue
            ax_reo.semilogy(xs_u, ReO_u, ls=ls, lw=lw, color=UP)
            ax_reo.semilogy(xs_l, ReO_l, ls=ls, lw=lw, color=LO)
            ax_P.plot(xs_u, P_u, ls=ls, lw=lw, color=UP)
            ax_P.plot(xs_l, P_l, ls=ls, lw=lw, color=LO)
        ax_reo.axhline(REOMC_FLOOR, color='gray', ls='--', lw=0.6, alpha=0.5)
        ax_reo.set_ylim(1e2, 1e4); ax_reo.grid(alpha=0.3, which='both')
        ax_reo.set_title(f"Re = {Rk}k", fontsize=12)
        if col == 0: ax_reo.set_ylabel(r'$\max Re_\Omega$ (log)')
        ax_P.axhline(0.0, color='gray', ls=':', lw=0.6, alpha=0.5)
        ax_P.set_ylim(-0.3, 1.0); ax_P.grid(alpha=0.3)
        if col == 0: ax_P.set_ylabel(r'$\max \hat S g$')
        # ROW 3: max chi (log, left) and the e^9 N envelope (linear, right)
        for fam, lvl, d, ls, lw in for_each_case(Rk):
            if not os.path.exists(f"{d}/slice_centerSpan.pvtu"):
                continue
            try:
                xc, mu, ml = R.max_chi_vs_x(d)
            except Exception as e:
                print(f"Re{Rk}k chi {fam}{lvl}: {e}"); continue
            ax_chi.semilogy(xc, mu/NU_Re, ls=ls, lw=lw, color=UP)
            ax_chi.semilogy(xc, ml/NU_Re, ls=ls, lw=lw, color=LO)
        # At Re=60k the two e^9 implementations visibly differ and XFOIL is
        # the primary reference (no amplification envelope there).
        XFOIL_PRIMARY = {60}
        mf = None if Rk in XFOIL_PRIMARY else MFOIL.get(Rk)
        nsrc = (FLEXN.get(Rk) if (mf is None or 'upper' not in mf
                          or 'n' not in mf['upper']) else mf)
        if nsrc is not None and 'n' in nsrc['upper']:
            ax_N.plot(nsrc['upper']['x'], nsrc['upper']['n'], ':', color=UP, lw=1.4, alpha=0.8)
            ax_N.plot(nsrc['lower']['x'], nsrc['lower']['n'], ':', color=LO, lw=1.4, alpha=0.8)
            ax_N.axhline(9.0, color='gray', ls=':', lw=0.6, alpha=0.6)
        else:
            ax_chi.text(0.03, 0.92, "no $e^9$ $N$", transform=ax_chi.transAxes,
                        ha='left', va='top', fontsize=8, color='0.4', style='italic')
        ax_chi.axhline(R.C_V1, color='gray', ls=':', lw=0.6, alpha=0.6)
        ax_chi.set_ylim(R.CHI_LO, R.CHI_HI); ax_N.set_ylim(R.N_LO, R.N_HI)
        if col == 0: ax_chi.set_ylabel(r'$\chi$ (log)')
        if col == len(re_list) - 1: ax_N.set_ylabel('$e^9$ $N$ (linear)')
        ax_chi.grid(alpha=0.3)
        # ROWS 4+5: -Cp and Cf
        for fam, lvl, d, ls, lw in for_each_case(Rk):
            if not os.path.exists(f"{d}/surface_fluid_eppler387.pvtu"):
                continue
            try:
                (xu, cfu, cpu), (xl, cfl, cpl) = R.airfoil_walk_contour(d)
            except Exception as e:
                print(f"Re{Rk}k surf {fam}{lvl}: {e}"); continue
            ax_cp.plot(xu, -cpu, ls=ls, lw=lw, color=UP)
            ax_cp.plot(xl, -cpl, ls=ls, lw=lw, color=LO)
            ax_cf.plot(xu, cfu, ls=ls, lw=lw, color=UP)
            ax_cf.plot(xl, cfl, ls=ls, lw=lw, color=LO)
        # experimental bubble band (only where oil-flow data exist)
        if Rk in EXP_LSB_RE:
            xls, xtr = EXP_LSB_RE[Rk]
            for a in (ax_chi, ax_cp, ax_cf):
                a.axvspan(xls, xtr, color='0.55', alpha=0.30, zorder=0)
        else:
            ax_cp.text(0.03, 0.05, "no exp. bubble/$C_p$ data", transform=ax_cp.transAxes,
                       ha='left', va='bottom', fontsize=8, color='0.4', style='italic')
        # mfoil (or xfoil fallback) reference Cp/Cf
        use_xf = Rk in XFOIL_PRIMARY and Rk in XFOIL
        ref = (XFOIL.get(Rk) if use_xf else None) or MFOIL.get(Rk) or XFOIL.get(Rk)
        if ref is not None and 'upper' not in ref:
            ref = XFOIL.get(Rk)
        if ref is not None and 'upper' in ref:
            for side, c in [('upper', UP), ('lower', LO)]:
                s = ref[side]
                ax_cp.plot(s['x'], -np.asarray(s['cp']), ':', color=c, lw=1.4, alpha=0.8, zorder=4)
                ax_cf.plot(s['x'], np.asarray(s['cf']), ':', color=c, lw=1.4, alpha=0.8, zorder=4)
        # exact experimental Cp (TM-4062 Appendix D) at nearest-5deg column
        etab = EXP_CP_TAB.get(str(Rk)); ecol = RE_COL.get(Rk)
        if etab is not None and ecol is not None:
            for side, c in [('upper', UP), ('lower', LO)]:
                if ecol in etab[side]:
                    xcs = np.asarray(etab[side]['xc']); cps = np.asarray(etab[side][ecol])
                    ax_cp.plot(xcs, -cps, 'o', mfc='none', mec=c, mew=1.1, ms=5, zorder=6)
        if col == 0:
            ax_cp.set_ylabel(r'$-C_p$'); ax_cf.set_ylabel(r'$C_f$')
        ax_cp.grid(alpha=0.3); ax_cf.grid(alpha=0.3)
        ax_cp.set_ylim(bottom=-1.0)
        ax_cf.set_ylim(-0.004, 0.012); ax_cf.axhline(0, color='gray', lw=0.6, alpha=0.5)
        ax_cf.set_xlim(0, 1); ax_cf.set_xlabel('$x/c$')
    handles = [Line2D([], [], color=UP, lw=2, label='upper'),
               Line2D([], [], color=LO, lw=2, label='lower'),
               Line2D([], [], color='0.3', lw=2, ls='--', label='cav (unstructured)'),
               Line2D([], [], color='0.3', lw=2, ls='-', label='str (O-grid)'),
               Line2D([], [], color='0.3', lw=R.LEVEL_LW['L0'], label='L0'),
               Line2D([], [], color='0.3', lw=R.LEVEL_LW['L1'], label='L1'),
               Line2D([], [], color='0.3', lw=R.LEVEL_LW['L2'], label='L2'),
               Patch(facecolor='0.55', alpha=0.30, label='exp. LSB (TM-4062 oil flow)')]
    if MFOIL or XFOIL:
        handles.append(Line2D([], [], color='0.3', ls=':', lw=1.4, label='mfoil/xfoil ($e^9$)'))
    if EXP_CP_TAB:
        handles.append(Line2D([], [], color='0.3', ls='none', marker='o', mfc='none', mew=1.1, ms=5,
                              label=r'exp. $C_p$ (TM-4062, $\alpha\!\approx\!5^\circ$)'))
    fig.legend(handles=handles, fontsize=9, frameon=False, ncol=4,
               loc='lower center', bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=(0, 0.05, 1, 1.0))
    plt.savefig(out); plt.savefig("/tmp/" + os.path.basename(out).replace('.pdf', '.png'), dpi=120)
    print("wrote", out)


make_fig([60, 100], f"{PD}/figs/eppler_resweep_lowRe.pdf")
make_fig([300, 460], f"{PD}/figs/eppler_resweep_highRe.pdf")
