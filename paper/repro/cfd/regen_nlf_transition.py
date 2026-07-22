"""Transition-location comparison table for the NLF(1)-0416 (Sec. V).

Assembles x_tr/c on both surfaces at alpha = 0,4,9,15 deg, Re=4e6, from three
sources and prints the Table~\ref{tab:nlftrans} rows:

  * SA-AI  -- present model, solver-reported transition (xtr_history.csv),
             finest structured O-grid (L2) with the cavity L2 spread noted.
  * mfoil  -- coupled viscous-inviscid e^9 panel solver (Fidkowski 2022).
             We ALSO flag surfaces where mfoil's laminar BL separates before
             the e^N envelope reaches N_crit=9 (H peaks, N_max<9): there its
             reported "transition" is a separation point, not an e^9 onset.
  * Exp    -- Somers, NASA TP-1861, Fig. 9(d) (R=4e6): transition orifices,
             open=laminar / solid=turbulent, read as x_tr(c_l) and mapped to
             each alpha through the section c_l. Hand-digitized below. The
             upper-surface orifices do not extend forward of x/c ~ 0.30, so
             for c_l > 0.7 (alpha >= 4) the upper transition is unresolved.
             The R=2e6 oil-flow photos (Fig. 7) give the (alpha,c_l) anchors
             0->0.4, 4->0.9, 8->1.3 and corroborate the forward march.
"""
import os, csv, pickle
import numpy as np

B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fr")
ALPHAS = [0, 4, 9, 15]

# --- Somers TP-1861 Fig. 9(d), R=4e6: authoritative transition-front readings by
# incidence (re-digitized). Each entry (x_tr, extrapolated?) or None if the front
# is not resolvable. The upper orifices do not extend forward of ~0.30c, so the
# upper onset is unresolvable for alpha>=9 (marches ahead of it); the upper value
# at alpha=4 and the lower value at alpha=15 are extrapolations beyond the
# recorded orifices (lower branch is nearly flat).
EXP_BY_ALPHA = {   # alpha_deg: (upper, lower)
    0:  ((0.38, False), (0.55, False)),
    4:  ((0.31, True),  (0.62, False)),
    9:  (None,          (0.64, False)),
    15: (None,          (0.66, True)),
}
CL_MAX_EXP = 1.68        # digitized polar (Fig. 11d)
X_ORIFICE_FWD = 0.30     # forwardmost upper-surface transition orifice


def _last(path):
    with open(path) as fh:
        return list(csv.reader(fh))[-1]


def rans(case):
    cl = float(_last(os.path.join(case, "total_forces_v2.csv"))[2])
    r = _last(os.path.join(case, "xtr_history.csv"))
    return cl, float(r[1]), float(r[2])   # cl, xtr_upper, xtr_lower


def exp_str(entry):
    """Format a Somers reading as text: 'x.xx', 'x.xx*' (extrapolated), or '--'."""
    if entry is None:
        return f"{'--':>7}"
    x, extrap = entry
    return f"{x:6.2f}*" if extrap else f"{x:7.2f}"


def mfoil_side(entry, side):
    s = entry[side]
    N = np.asarray(s["n"]); H = np.asarray(s["H"])
    return entry["xtr_upper" if side == "upper" else "xtr_lower"], \
        float(np.nanmax(N)), float(np.nanmax(H))


mf = pickle.load(open(os.path.join(B, "mfoil_nlf0416_Re4M.pkl"), "rb"))

print(f"{'a':>3} {'cl_str':>7}{'cl_cav':>7} | "
      f"{'up_str':>7}{'up_cav':>7}{'up_mf':>7}{'up_exp':>7} | "
      f"{'lo_str':>7}{'lo_cav':>7}{'lo_mf':>7}{'lo_exp':>7}")
for a in ALPHAS:
    cl_s, us, ls = rans(f"{B}/strL2prop_nlf0416_Re4M_a{a}")
    cl_c, uc, lc = rans(f"{B}/cavL2prop_nlf0416_Re4M_a{a}")
    e = mf[float(a)]
    (umf, uN, uH) = mfoil_side(e, "upper")
    (lmf, lN, lH) = mfoil_side(e, "lower")
    uex, lex = EXP_BY_ALPHA.get(a, (None, None))
    # mfoil is a usable e^N reference only if its coupled solve CONVERGED and the
    # upper surface stayed attached (upper H < 4). It does not converge at a=9 and
    # converges only to a stalled state (upper H>>1) at a=15 -> "n/a" in both.
    conv = bool(e.get("conv", True)); usable = conv and (uH < 4.0)
    umfs = f"{umf:7.3f}" if usable else f"{'n/a':>7}"
    lmfs = f"{lmf:7.3f}" if usable else f"{'n/a':>7}"
    note = "" if usable else ("  [mfoil NOT CONVERGED]" if not conv
                              else "  [mfoil STALLED: upper H=%.1f]" % uH)
    print(f"{a:>3}{cl_s:7.3f}{cl_c:7.3f} | "
          f"{us:7.3f}{uc:7.3f}{umfs}{exp_str(uex)} | "
          f"{ls:7.3f}{lc:7.3f}{lmfs}{exp_str(lex)}"
          f"  cl_mf={e['cl']:.3f} conv={conv}{note}  (*=exp extrapolated)")
