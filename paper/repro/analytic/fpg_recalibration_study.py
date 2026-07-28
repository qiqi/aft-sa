"""FPG recalibration study: the rate-floor substitution P = Omega_hat*(I_hat + eps).

USER DIRECTIVE (2026-07-28): recalibrate the amplification kernel so that
Figure 4 (fig:calibrate, figs/model_calibrate.pdf) matches Drela-Giles down to
the stagnation-point H = 2.216, using the EXACT substitution

    P = Omega_hat * (I_hat + eps)          [replaces P = Omega_hat * I_hat]

EVERYWHERE the coordinate P appears -- the rate a = a_max*clip(P) AND the
onset threshold Re_Omega_c = k*softmin_2(2600, 175 + 2/P^2) -- with a single
new constant eps and no separately designed gate branch. This is an OFFLINE
study on the paper's analytic rigs: it monkeypatches fig04_shapefactor's
sphere_rate (the canonical marched instrument, grid 1600x1200) and touches
NO canonical figure and NO solver code.

Stages (each persists into figs_explore/fpg_recalibration_study.json):
    --smoke     eps=0 self-check against the 2026-07-28-1041 audit rows
    --baseline  eps=0 full family (attached + Stewartson lower branch)
    --sweep     favorable ladder H in [2.216, 2.591] over an eps list
    --final EPS full family + zero-suite + graze re-check + candidate figure
    (no flag = all stages, eps* chosen from the sweep by the minimax rule)

Scoring (favorable branch, H in [2.216, 2.591], all vs repo Drela-Giles
lib/correlations.py -- the calibration target of record; the mfoil variant
runs up to ~60% hotter in slope at stagnation H and is NOT used):
    dev_a = max |ln(s_late / s_DG)|            (panel a, envelope rate)
    dev_b = max |ln(Rt1 / Rt1_DG_N1)|          (panel b, N=1 station)
    eps_joint minimizes max(dev_a, dev_b);  eps_rate solves the beta=1 late
    secant ratio = 1 (Drela's extreme-FPG slope), interpolated on the sweep.

Zero-suite (Spalart flows, the floor's parasitic-production ledger):
Reichardt law-of-the-wall boundary layers at Re_tau in {180, 1000, 5200}
(wall units, nu=1; nuHat = SA log solution min(kappa*y+, 0.08*Re_tau) as in
sa_sustain.py): reports max over y of  a*onset*omega / (c_b1*S_tilde), the
pointwise ratio of parasitic AI production to SA production. The floor makes
P > 0 in the viscous sublayer (Shat = 1/sqrt(2), g = 0 there), so the number
is NOT structurally zero -- the onset gate (Re_Omega small near the wall) and
the log layer's strongly negative I_hat are what keep it negligible.

Run from anywhere:  python3 paper/repro/analytic/fpg_recalibration_study.py
(_saai chdir's to paper/). Runtime ~15-25 min for the full pipeline.
"""
import argparse
import json
import os

import numpy as np

import _saai  # noqa: F401  (paths + chdir to paper/)
import fig04_shapefactor as f4
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0

OUT_DIR = os.path.join('repro', 'analytic', 'figs_explore')
OUT_JSON = os.path.join(OUT_DIR, 'fpg_recalibration_study.json')
OUT_FIG = os.path.join(OUT_DIR, 'model_calibrate_candidate.png')

# canonical sphere-kernel constants, taken from the canonical rig itself
A_MAX, RAMP_W = f4.A_MAX, f4.RAMP_W
REOM_CEIL, REOM_A, REOM_B, REOM_N = f4.REOM_CEIL, f4.REOM_A, f4.REOM_B, f4.REOM_N

# the one knob of this study; [0] so the monkeypatched kernel sees updates
EPS = [0.0]
# candidate form (2026-07-28 USER REDESIGNS #1 and #2):
#   add     P = Shat*(g + eps);            gate keeps the softmin ceiling
#   sm2raw  P = Shat*sqrt(g^2 + eps^2);    gate = A + B/P^2, NO ceiling
#   sm2clip P = Shat*sqrt(clip(g,0)^2+eps^2); gate = A + B/P^2, NO ceiling
#   zc      P = Shat*sqrt(clip(g,0)^2 + clip(eps*(-Z)/R,0)^2);  NO ceiling
#   zc_ceil same P as zc; gate keeps the canonical softmin ceiling
# For the constant-eps sm2 forms the floor caps the gate blow-up
# intrinsically at A + B/(Shat*eps)^2 (Shat*eps ~ 0.029 reproduces the old
# 1851.2 ceiling), so the constant count is unchanged: drop 1851.2, add eps.
# The zc forms key the floor to the favorable curvature itself: -Z/R is
# dimensionless (same /R normalization as g = (Y-X-Z)/R), vanishes toward
# the wall like y (Z ~ y^2 u''(0)/2 while R ~ sqrt(2) y u'(0)), is positive
# throughout non-inflected FPG profiles (u'' < 0 everywhere), negative near
# an APG wall (u''(0) > 0 -> clipped to 0, inert), and zero in the free
# stream. CLIP CONVENTION: both softmax arguments are clipped at 0
# (clip(g,0) per redesign #1's sign-check verdict; clip(-Z,0) per #2).
# USER FOLLOW-UP variants (a single eps cannot serve both panels):
#   zc2  VARIANT A: rate uses P_r = zc floor with EPS (eps_r); the gate uses
#        its own P_o = zc floor with EPS_O (eps_o), no-ceiling threshold
#        124.6 + 1.424/P_o^2. Tuned jointly: eps_r -> panel (a) at
#        stagnation, eps_o -> N=1 crossing ON the DG station at low H.
#   zb   VARIANT B: rate uses the zc floor with EPS (eps_r); the gate keeps
#        the CANON form untouched -- softmin(1851.2, 124.6+1.424/P^2) on the
#        UN-floored canon P = Shat*g (canon max(P,1e-6) clip).
FORM = ['add']
EPS_O = [0.0]      # variant-A onset epsilon (form zc2 only)
# USER EXTENSION (Part III): in the B structure the ceiling is NOT sacred --
# joint (a_visc, C) optimization. CEIL_B, when set, replaces the canon
# 1851.2 in the zb gate (k-carrying units, i.e. the compiled-constant
# scale); None = canon. Results computed with it are stored under keys
# suffixed _C<value>.
CEIL_B = [None]
# re-anchoring knobs (--reanchor only; 1.0 = canonical constants)
ASCALE = [1.0]     # multiplies a_max
KSCALE = [1.0]     # multiplies the whole onset threshold (the paper's k)

FAVORABLE = [1.0, 0.55, 0.35, 0.2, 0.1, 0.05, 0.0]          # H 2.216..2.591
ADVERSE = [-0.03, -0.06, -0.09, -0.12, -0.15, -0.18, -0.1988]
LOWER = [(-0.19, -0.03), (-0.17, -0.06), (-0.15, -0.08), (-0.12, -0.10)]


def _P_and_thresh(Shat, g, Zn=None):
    """The candidate coordinate P and the gate threshold Re_Omega_c for the
    active FORM/EPS (threshold WITHOUT the anchor scale k; callers apply
    KSCALE via the stored k-carrying constants). Zn = Z/R (needed by the
    zc forms; g's own normalization)."""
    form, eps = FORM[0], EPS[0]
    softmin_gate = form in ('add', 'zc_ceil')
    if form == 'add':
        P = Shat*(g + eps)
    elif form in ('sm2raw', 'sm2clip'):
        gg = g if form == 'sm2raw' else np.clip(g, 0.0, None)
        P = Shat*np.sqrt(gg*gg + eps*eps)
    elif form == 'zc2':                      # VARIANT A: two epsilons
        gg = np.clip(g, 0.0, None)
        fl_r = np.clip(-eps*Zn, 0.0, None)
        P = Shat*np.sqrt(gg*gg + fl_r*fl_r)            # rate coordinate
        fl_o = np.clip(-EPS_O[0]*Zn, 0.0, None)
        P_o = Shat*np.sqrt(gg*gg + fl_o*fl_o)          # onset coordinate
        reomc = REOM_A + REOM_B/np.maximum(P_o, 1e-9)**2
        return P, reomc
    elif form == 'zb':                       # VARIANT B: rate-only floor
        gg = np.clip(g, 0.0, None)
        fl = np.clip(-eps*Zn, 0.0, None)
        P = Shat*np.sqrt(gg*gg + fl*fl)                # rate coordinate
        Pc = Shat*g                                    # CANON gate coordinate
        C = REOM_CEIL if CEIL_B[0] is None else CEIL_B[0]
        _pw = REOM_A + REOM_B*np.maximum(Pc, 1e-6)**(-2.0)
        reomc = (C**(-REOM_N) + _pw**(-REOM_N))**(-1.0/REOM_N)
        return P, reomc
    else:                                    # zc / zc_ceil
        gg = np.clip(g, 0.0, None)
        fl = np.clip(-eps*Zn, 0.0, None)
        P = Shat*np.sqrt(gg*gg + fl*fl)
    if softmin_gate:
        _pw = REOM_A + REOM_B*np.maximum(P, 1e-6)**(-2.0)
        reomc = (REOM_CEIL**(-REOM_N) + _pw**(-REOM_N))**(-1.0/REOM_N)
    else:
        reomc = REOM_A + REOM_B/np.maximum(P, 1e-9)**2   # NO ceiling
    return P, reomc


def sphere_rate_eps(u, dudy, yc, nu=1.0):
    """fig04_shapefactor.sphere_rate with the candidate substitution
    (FORM/EPS); identical to the canonical kernel at FORM='add', eps=0."""
    d2u = np.gradient(dudy, yc)
    X = u; Y = yc*dudy; Z = 0.5*yc**2*d2u
    R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
    Shat = Y/np.sqrt(X*X + Y*Y + 1e-30)
    g = (Y - X - Z)/R
    P, reomc = _P_and_thresh(Shat, g, Z/R)
    a = ASCALE[0]*A_MAX*np.minimum(1.0, np.clip(P, 0.0, None))
    ReOm = yc**2*np.abs(dudy)/nu
    onset = 0.5*(1.0 + np.tanh((ReOm/(KSCALE[0]*reomc) - 1.0)/RAMP_W))
    return a*onset


f4.sphere_rate = sphere_rate_eps   # march() resolves it at call time


def row(beta, guess=None, ufrac=0.0):
    H, s_early, s_late, Rt1 = f4.measures_for_beta(
        beta, verbose=False, guess=guess, ufrac=ufrac)
    d = float(dN_dRe_theta(H)); Rtc = float(Re_theta0(H))
    r = dict(beta=beta, H=H, s_early=s_early, s_late=s_late, Rt1=Rt1,
             s_DG=d, Rt1_DG=Rtc + 1.0/d, Rtc_DG=Rtc)
    print(f"  eps={EPS[0]:.4f} beta={beta:+.4f} H={H:6.3f} "
          f"early={s_early:.3e} ({s_early/d:5.2f}x) "
          f"late={s_late:.3e} ({s_late/d:5.2f}x) "
          f"Rt1={Rt1:8.1f} ({Rt1/r['Rt1_DG']:5.2f}x DG N=1)", flush=True)
    return r


def ladder(betas, eps, lower=()):
    EPS[0] = eps
    rows = [row(b) for b in betas]
    rows += [row(b, guess=g, ufrac=0.03) for b, g in lower]
    return rows


def scores(rows, Hmax=2.60):
    """Favorable-branch panel scores (log-space, worst over H <= Hmax)."""
    da = db = 0.0
    for r in rows:
        if r['H'] > Hmax:
            continue
        s = r['s_late'] if np.isfinite(r['s_late']) else r['s_early']
        da = max(da, abs(np.log(s/r['s_DG'])) if np.isfinite(s) else 99.0)
        db = max(db, abs(np.log(r['Rt1']/r['Rt1_DG']))
                 if np.isfinite(r['Rt1']) else 99.0)
    return da, db


# --------------------------------------------------------------------------
# zero-suite: parasitic production of the floor in Spalart flows
# --------------------------------------------------------------------------
CB1, CV1, KAP = 0.1355, 7.1, 0.41


def reichardt(yp):
    k, C = KAP, 7.8
    return (1/k)*np.log(1 + k*yp) + C*(1 - np.exp(-yp/11)
                                       - (yp/11)*np.exp(-yp/3))


def zero_suite(eps, Re_taus=(180.0, 1000.0, 5200.0), ny=4000, tau=4.0):
    """Parasitic production of the floored kernel in Spalart flows: max over
    y of  a*onset*omega / (c_b1*S_tilde)  on turbulent-BL mean profiles
    (wall units, nu = 1), reported RAW (kernel alone, the worst case) and
    BLENDED by the solver's handover weight (1 - sigma_t(chi; tau=4)) with
    chi = the SA equilibrium min(kappa y+, 0.08 Re_tau) -- in the solver the
    gated-max blend P = max((1-is_turb) P_ai, is_turb P_sa) suppresses the
    AI term wherever chi >> 1, and that suppression IS the operative guard.
    Profile: Reichardt law of the wall capped at u_e via a C^1 exponential
    tail (u' decays over 0.25 delta above y = Re_tau) -- a HARD cap puts a
    u'' delta at the edge that flips g positive spuriously (found the hard
    way; the tail scale is the physical wake-curvature scale, and the edge
    verdict deserves a solver-field cross-check)."""
    EPS[0] = eps
    out = []
    for Re_tau in Re_taus:
        yp = np.linspace(0.05, 1.6*Re_tau, ny)
        dy = yp[1] - yp[0]
        k, C = KAP, 7.8
        dup = 1.0/(1 + k*yp) + C*(np.exp(-yp/11)/11 - np.exp(-yp/3)/11
                                  + (yp/33)*np.exp(-yp/3))   # d reichardt/dy+
        tail = 0.25*Re_tau
        dup = np.where(yp <= Re_tau, dup,
                       np.interp(Re_tau, yp, dup)
                       * np.exp(-(yp - Re_tau)/tail))
        u = np.concatenate([[0.0], np.cumsum(0.5*(dup[1:] + dup[:-1])*dy)])
        u = u[:len(yp)] + reichardt(yp[0])
        dudy = dup
        om = np.abs(dudy) + 1e-30
        b_ai = sphere_rate_eps(u, dudy, yp)*om    # parasitic AI coefficient
        nut = np.minimum(KAP*yp, 0.08*Re_tau)     # SA log-layer solution
        chi = nut
        fv1 = chi**3/(chi**3 + CV1**3)
        fv2 = 1 - chi/(1 + chi*fv1)
        St = np.maximum(om + nut*fv2/(KAP**2*yp**2 + 1e-30), 0.3*om)
        raw = b_ai/(CB1*St)
        sig_t = np.maximum(1.0 - np.exp(-(chi - 1.0)/tau), 0.0)
        blended = (1.0 - sig_t)*raw
        i, j = int(np.argmax(raw)), int(np.argmax(blended))
        # where does the floored P go positive?
        d2u = np.gradient(dudy, yp)
        X, Y, Z = u, yp*dudy, 0.5*yp**2*d2u
        R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
        g = (Y - X - Z)/R
        Shat = Y/np.sqrt(X*X + Y*Y + 1e-30)
        Pd, _ = _P_and_thresh(Shat, g, Z/R)
        ypos = yp[Pd > (1e-12 if FORM[0] in ('add', 'zc', 'zc_ceil')
                        else 1.001*Shat*eps)]
        out.append(dict(Re_tau=Re_tau,
                        max_raw=float(raw[i]), y_plus_raw=float(yp[i]),
                        max_blended=float(blended[j]),
                        y_plus_blended=float(yp[j]),
                        y_plus_P_positive_max=float(ypos.max()) if len(ypos)
                        else 0.0))
        print(f"  zero-suite Re_tau={Re_tau:6.0f}: raw max P_AI/P_SA = "
              f"{raw[i]:.2e} at y+={yp[i]:.1f}; sigma_t-blended max = "
              f"{blended[j]:.2e} at y+={yp[j]:.1f}; floored P>0 for y+ <= "
              f"{out[-1]['y_plus_P_positive_max']:.1f}", flush=True)
    return out


# --------------------------------------------------------------------------
# graze re-check: does the substitution move the LST onset envelope?
# --------------------------------------------------------------------------
def graze_check(eps):
    """fig02_onset_graze's construction with P -> Shat*(g+eps): graze ratio
    (max Re_Omega/threshold along the profile at Drela's critical Re_theta)
    for the attached family + the separated lower-branch profile.
    Canon (eps=0): attached 0.93-1.14; separated rides far above (~4x)."""
    EPS[0] = eps
    profiles = [(0.15, None), (0.10, None), (0.05, None), (0.0, None),
                (-0.05, None), (-0.10, None), (-0.15, None), (-0.19, None),
                (-0.1988, None), (-0.19, -0.03)]
    out = []
    for beta, guess in profiles:
        fs = FalknerSkanWedge(beta, guess=guess) if guess is not None \
            else FalknerSkanWedge(beta)
        eta, u, upr = fs.eta, fs.u, fs.dudeta
        I_th = float(np.trapezoid(u*(1 - u), eta))
        H = float(np.trapezoid(1 - u, eta))/I_th
        Rtc = float(np.asarray(Re_theta0(H)))
        upp = np.gradient(upr, eta)
        X = u; Y = eta*upr; Z = 0.5*eta**2*upp
        R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
        Shat = Y/np.sqrt(X*X + Y*Y + 1e-30)
        g = (Y - X - Z)/R
        EPS[0] = eps
        P, thr_k = _P_and_thresh(Shat, g, Z/R)
        thr = thr_k/0.712        # graze convention: shape at k = 1
        ReOm = eta**2*np.abs(upr)*(Rtc/I_th)   # eta^2*|u'|*sqrt(Re_x)
        m = (P > 1e-4) & (ReOm > 1.0)
        ratio = ReOm[m]/thr[m]
        out.append(dict(beta=beta, lower=guess is not None, H=H,
                        graze=float(ratio.max())))
        print(f"  graze eps={eps:.4f} beta={beta:+.4f} H={H:5.2f} "
              f"ratio={ratio.max():.3f}", flush=True)
    return out


# --------------------------------------------------------------------------
# candidate figure (house style = fig02_model_calibrate, PNG, canon overlaid)
# --------------------------------------------------------------------------
def candidate_figure(rows_c, rows_0, eps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def cols(rows):
        H = np.array([r['H'] for r in rows])
        se = np.array([r['s_early'] for r in rows])
        sl = np.array([r['s_late'] for r in rows])
        R = np.array([r['Rt1'] for r in rows])
        o = np.argsort(H)
        return H[o], se[o], sl[o], R[o]

    H, se, sl, R = cols(rows_c)
    H0, se0, sl0, R0 = cols(rows_0)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.2, 4.3))
    fig.patch.set_facecolor('white')
    Hg = np.geomspace(2.2, 10.6, 300)
    drela_g = np.asarray(dN_dRe_theta(Hg)); Rtc_g = np.asarray(Re_theta0(Hg))

    def style(ax, ylabel, ylim):
        ax.axvspan(4.03, 11.0, color='0.92', zorder=0)
        ax.axvline(4.03, color='0.6', lw=0.9, ls=':')
        ax.set_xscale('log'); ax.set_xticks([2.2, 2.6, 3, 3.5, 4, 5, 7, 10])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(2.12, 11.0)
        ax.set_xlabel(r'shape factor $H=\delta^*/\theta$')
        ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
        ax.grid(alpha=0.3, which='both')

    axa.semilogy(Hg, drela_g, 'k--', lw=1.8, label=r'Drela--Giles $dN/dRe_\theta$')
    m0 = np.isfinite(se0) & np.isfinite(sl0)
    axa.semilogy(H0[m0], sl0[m0], '-', color='0.75', lw=1.1,
                 label=r'canon ($\epsilon=0$), late')
    m = np.isfinite(se) & np.isfinite(sl)
    axa.fill_between(H[m], se[m], sl[m], color='C0', alpha=0.15, lw=0)
    axa.semilogy(H[m], se[m], '-o', color='C0', ms=4.5, lw=1.3, mfc='white',
                 label=r'candidate early ($N\!\in\![1,5]$)')
    axa.semilogy(H[m], sl[m], '-^', color='C0', ms=4.5, lw=1.3,
                 label=r'candidate late ($N\!\in\![5,9]$)')
    style(axa, r'$dN/dRe_\theta$', (drela_g.min()/12.0, drela_g.max()*1.3))
    axa.legend(fontsize=8.0, loc='upper left')
    axa.text(0.96, 0.06, '(a)', transform=axa.transAxes, fontsize=12,
             fontweight='bold', ha='right')

    N1_g = Rtc_g + 1.0/drela_g
    axb.semilogy(Hg, Rtc_g, '--', color='0.55', lw=1.4,
                 label=r'Drela critical $Re_{\theta 0}$')
    axb.semilogy(Hg, N1_g, 'k--', lw=1.8, label=r'Drela--Giles $N\!=\!1$ station')
    axb.semilogy(H0[np.isfinite(R0)], R0[np.isfinite(R0)], '-', color='0.75',
                 lw=1.1, label=r'canon ($\epsilon=0$)')
    mR = np.isfinite(R)
    axb.semilogy(H[mR], R[mR], '-o', color='C0', ms=4.5, lw=1.3, mfc='white',
                 label=r'candidate $N\!=\!1$ crossing')
    style(axb, r'onset $Re_\theta$', (Rtc_g.min(), N1_g.max()*4.0))
    axb.legend(fontsize=8.0, loc='upper right')
    axb.text(0.96, 0.06, '(b)', transform=axb.transAxes, fontsize=12,
             fontweight='bold', ha='right')
    lab = {'add': r'$P=\hat\Omega(\hat I+\epsilon)$, softmin gate',
           'sm2raw': r'$P=\hat\Omega\sqrt{\hat I^2+\epsilon^2}$, no-ceiling gate',
           'sm2clip': r'$P=\hat\Omega\sqrt{\langle\hat I\rangle_+^2+\epsilon^2}$'
                      ', no-ceiling gate',
           'zc': r'$P=\hat\Omega\sqrt{\langle\hat I\rangle_+^2'
                 r'+\langle\epsilon(-Z)/R\rangle_+^2}$, no-ceiling gate',
           'zc_ceil': r'$P=\hat\Omega\sqrt{\langle\hat I\rangle_+^2'
                      r'+\langle\epsilon(-Z)/R\rangle_+^2}$, softmin gate',
           'zc2': r'two-$\epsilon$: rate $\epsilon_r$, gate $\epsilon_o$'
                  ' (no ceiling)',
           'zb': r'rate-only floor $\epsilon_r$; CANON gate'}
    fig.suptitle('CANDIDATE (not canon): ' + lab[FORM[0]]
                 + rf', $\epsilon={eps:g}$', fontsize=10, y=1.0)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, facecolor='white')
    print(f'wrote {OUT_FIG}', flush=True)


# --------------------------------------------------------------------------
# impact-table instruments (frozen-profile sup bound on marched mean flows,
# the 1041 audit's Part-B convention: dN/ds = max_y a*onset*|du/dy| / u)
# --------------------------------------------------------------------------
def _sup_rate(y, u, ue, nu, eps):
    """max_y of the pointwise growth rate per unit arc, on one profile."""
    EPS[0] = eps
    yc = y[1:]
    uu = np.interp(yc, y, u)
    dudy = np.gradient(uu, yc)
    b = sphere_rate_eps(uu, dudy, yc, nu=nu)*np.abs(dudy)
    with np.errstate(divide='ignore', invalid='ignore'):
        rr = b/np.maximum(uu, 1e-4*ue)
    return float(np.nanmax(rr))


def cylinder_budget(eps_list, Re_Ds=(2e6, 7e6, 2e7, 1e9),
                    seeds=(4.525, 6.485)):
    """Potential-flow cylinder nose (u_e = 2 sin theta, D = 1, U = 1):
    sup-bound e-folds N(theta) booked up to 80 deg and to the (potential)
    suction peak 90 deg, per eps. Laminar planar march (bl_march, the 1041
    Part-B instrument; measured-Cp u_e runs 3-5% below potential -- the
    potential flow is the coordinator-approved stand-in, needed anyway at
    Re_D = 1e9 where no RANS case exists)."""
    import sys
    sys.path.insert(0, os.path.join('repro', 'cfd'))
    from spheroid_a0_meanflow import bl_march
    Rcyl = 0.5
    th = np.concatenate([np.linspace(0.5, 8, 40, endpoint=False),
                         np.linspace(8, 92, 480)])*np.pi/180.0
    s = Rcyl*th
    ue = 2.0*np.sin(th)
    stations = list(Rcyl*np.linspace(4, 91, 60)*np.pi/180.0)
    out = {}
    for Re_D in Re_Ds:
        nu = 1.0/Re_D
        m = bl_march(s, ue, nu, 'planar', x_profiles=stations, geom='flat')
        for eps in eps_list:
            sup = {sx: _sup_rate(y, u, uex, nu, eps)
                   for sx, (y, u, uex) in m['profiles'].items()}
            sx = np.array(sorted(sup)); rr = np.array([sup[k] for k in sx])
            N = np.concatenate([[0.0], np.cumsum(
                0.5*(rr[1:] + rr[:-1])*np.diff(sx))])
            thd = sx/Rcyl*180/np.pi
            N80 = float(np.interp(80.0, thd, N))
            N90 = float(np.interp(90.0, thd, N))
            # sup-bound front estimates: theta where N first reaches the
            # Tu 0.2% seed thresholds (4.525 = chi=1 convention, 6.485 =
            # chi=c_v1 handover; eq:tumap numbers per the 1041 audit)
            cross = {f'N{s:g}': (float(np.interp(s, N, thd))
                                 if N[-1] >= s else None) for s in seeds}
            out[f'ReD{Re_D:g}_eps{eps:g}'] = dict(
                N80=N80, N90=N90, cross_deg=cross,
                theta_deg=[round(float(t), 2) for t in thd[::4]],
                N_sup=[round(float(n), 3) for n in N[::4]])
            cs = ' '.join(f'{k}@{v:.1f}deg' if v else f'{k}@none'
                          for k, v in cross.items())
            print(f"  cylinder Re_D={Re_D:g} eps={eps:g}: N_sup(80deg)="
                  f"{N80:.2f}  N_sup(90deg)={N90:.2f}  fronts: {cs}",
                  flush=True)
    return out


def spheroid_budget(eps_list):
    """re72a0 spheroid: sup-bound extra e-folds over the laminar run
    (axisymmetric march on the converged field's own u_e, from the committed
    a0phys cache), up to the campaign front x = 0.858."""
    import sys
    sys.path.insert(0, os.path.join('repro', 'cfd'))
    from spheroid_a0_meanflow import bl_march, spheroid_ue_march_grid
    from spheroid_a0_physics import get_data
    sw, _, nu_ref = get_data()
    xm, ue_m = spheroid_ue_march_grid(sw)
    stations = list(np.arange(0.04, 0.87, 0.02))
    m = bl_march(xm, ue_m, nu_ref, 'axi', x_profiles=stations)
    out = {}
    for eps in eps_list:
        sup = {sx: _sup_rate(y, u, uex, nu_ref, eps)
               for sx, (y, u, uex) in m['profiles'].items()}
        sx = np.array(sorted(sup)); rr = np.array([sup[k] for k in sx])
        N = np.concatenate([[0.0], np.cumsum(
            0.5*(rr[1:] + rr[:-1])*np.diff(sx))])
        Nf = float(np.interp(0.858, sx, N))
        Npk = float(np.interp(0.497, sx, N))
        dNdx = float(np.interp(0.858, sx[1:], rr[1:]))
        out[f'eps{eps:g}'] = dict(N_at_front=Nf, N_at_uepeak=Npk,
                                  dNdx_at_front=dNdx)
        print(f"  spheroid re72a0 eps={eps:g}: N_sup(front 0.858)={Nf:.2f} "
              f"(u_e peak 0.497: {Npk:.2f}); dN/dx at front={dNdx:.1f}",
              flush=True)
    return out


def hiemenz_budget(eps_list, Re_r=(4.66e5, 4.74e5)):
    """Sec VIII frozen-Hiemenz problem: sup-bound e-folds banked by the AI
    term along the attachment wedge, 0 <= x <= L = sqrt(Re_r) (FS beta=1
    similarity, delta = sqrt(nu/k) units, nu = 1)."""
    fs = FalknerSkanWedge(1.0)
    out = {}
    yg = np.linspace(1e-3, 12.0, 500)
    for Rr in Re_r:
        L = np.sqrt(Rr)
        xg = np.geomspace(1e-2, L, 400)
        for eps in eps_list:
            rr = []
            for x in xg:
                _, u, dudy, _ = fs.at(x, yg, cellCentered=False)
                rr.append(_sup_rate(np.concatenate([[0.0], yg]),
                                    np.concatenate([[0.0], u]),
                                    fs.inviscid_at(x), 1.0, eps))
            N = float(np.trapezoid(rr, xg))
            out[f'Rer{Rr:g}_eps{eps:g}'] = N
            print(f"  Hiemenz Re_r={Rr:g} (L={L:.0f}) eps={eps:g}: "
                  f"N_sup={N:.3f}", flush=True)
    return out


def boundedness(eps, Rts=(1e3, 1e4, 1e5, 1e6, 1e7)):
    """REGRESSION GATE (redesign #2): high-Re wall behavior of the frozen
    production coefficient b(y) = a*onset*|u'| on the FS beta=1 profile for
    the ACTIVE form. A constant-eps floor leaves P finite at the wall
    (Shat -> 1/sqrt(2), so P -> Shat*eps): as Re grows the gate opens ever
    deeper (Re_Omega = y^2 u'/nu), the driving peak migrates wallward and
    sup b grows with the wall shear -- the frozen-profile eigenvalue is
    unbounded at the high-Re extreme. The zc floor vanishes like y toward
    the wall (Z ~ y^2), so sup b must saturate at an interior peak.
    Reports sup_y b (normalized by U_e^2/nu-free convective scale u'_max)
    and the peak location y*/theta per Rt."""
    fs = FalknerSkanWedge(1.0)
    I_th, _ = f4.profile_ints(fs)
    EPS[0] = eps
    out = []
    for Rt in Rts:
        x = Rt/I_th                          # beta=1: U_e = x, Re_theta = I_th*x
        y = np.geomspace(1e-7, 12.0, 4000)   # eta units (delta = 1)
        _, u, dudy, _ = fs.at(x, y, cellCentered=False)
        b = sphere_rate_eps(u, dudy, y)*np.abs(dudy)
        i = int(np.argmax(b))
        theta = I_th                          # eta units
        out.append(dict(Rt=Rt, sup_b_over_upmax=float(b[i]/np.max(dudy)),
                        y_peak_over_theta=float(y[i]/theta),
                        b_wall_frac=float(b[3]/max(b[i], 1e-300))))
        print(f"  boundedness[{FORM[0]}] eps={eps:g} Rt={Rt:.0e}: "
              f"sup b/u'_max = {out[-1]['sup_b_over_upmax']:.3e} at "
              f"y*/theta = {out[-1]['y_peak_over_theta']:.3f}; "
              f"b(wall)/sup = {out[-1]['b_wall_frac']:.1e}", flush=True)
    return out


def signmap(eps):
    """Sign chart of the zc floor argument -Z/R across profile classes:
    where is the floor active (-Z > 0), how big is it at the driving band,
    and does it stay subordinate to g where the layer is inflected."""
    cases = [('FS beta=+1.0 (stagnation FPG)', ('fs', 1.0, None)),
             ('FS beta=+0.35 (FPG)', ('fs', 0.35, None)),
             ('Blasius', ('fs', 0.0, None)),
             ('FS beta=-0.15 (APG)', ('fs', -0.15, None)),
             ('Stewartson lower (separated)', ('fs', -0.19, -0.03)),
             ('tanh free-shear layer', ('tanh', None, None))]
    out = []
    for name, (kind, beta, guess) in cases:
        if kind == 'fs':
            fs = FalknerSkanWedge(beta, guess=guess) if guess is not None \
                else FalknerSkanWedge(beta)
            eta, u, upr = fs.eta, fs.u, fs.dudeta
        else:
            eta = np.linspace(0.01, 20.0, 2000)
            u = 0.5*(1.0 + np.tanh(eta - 8.0))    # wall-free shear layer
            upr = np.gradient(u, eta)
        upp = np.gradient(upr, eta)
        X, Y, Z = u, eta*upr, 0.5*eta**2*upp
        R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
        g = (Y - X - Z)/R
        mZn = -Z/R
        sh = upr > 1e-3*np.max(upr)               # the shear region
        act = float(np.mean(mZn[sh] > 0.0))
        dom = float(np.mean((eps*np.clip(mZn, 0, None) >
                             np.clip(g, 0, None))[sh]))
        out.append(dict(case=name, minus_Zn_min=float(mZn[sh].min()),
                        minus_Zn_max=float(mZn[sh].max()),
                        frac_active=act, frac_floor_dominant=dom,
                        g_max=float(g[sh].max())))
        print(f"  signmap {name}: -Z/R in [{mZn[sh].min():+.3f}, "
              f"{mZn[sh].max():+.3f}]; active {100*act:.0f}% of shear region;"
              f" floor>g on {100*dom:.0f}% (eps={eps:g}); g_max="
              f"{g[sh].max():+.3f}", flush=True)
    return out


def _secant(fun, x0, x1, target, tol, lo=1e-3, hi=1.0, iters=6):
    """Secant-solve fun(x) = target; returns (x, fun(x))."""
    f0, f1 = fun(x0) - target, fun(x1) - target
    for _ in range(iters):
        if abs(f1) < tol or abs(f1 - f0) < 1e-12:
            break
        x2 = min(max(x1 - f1*(x1 - x0)/(f1 - f0), lo), hi)
        x0, f0 = x1, f1
        x1, f1 = x2, fun(x2) - target
    return x1, f1 + target


def tune_variants(db, which):
    """USER FOLLOW-UP: joint tuning at the stagnation wedge (beta = 1).
    Variant A (form zc2): eps_r -> late secant = Drela (panel a), then
    eps_o -> N=1 crossing ON the DG station (panel b), then one re-check
    of eps_r. Variant B (form zb): eps_r -> late secant = Drela; panel (b)
    is whatever the canon gate gives (reported honestly)."""
    cache = {}

    def b1(er, eo=None):
        key = (round(er, 5), None if eo is None else round(eo, 5))
        if key not in cache:
            EPS[0] = er
            if eo is not None:
                EPS_O[0] = eo
            cache[key] = row(1.0)
        return cache[key]

    if which == 'A':
        # GATE FIRST: with a late gate the [5,9] window is rate-driven and
        # the two constants decouple; rate-first diverges (a weak gate makes
        # the late secant onset-limited -- no eps_r can reach Drela's slope,
        # found the hard way). Analytic start: opening at DG-critical needs
        # P_o ~ 5e-3 -> eps_o ~ 0.02-0.03.
        FORM[0] = 'zc2'
        er = 0.17                                     # informed rate start
        eo, _ = _secant(lambda x: b1(er, x)['Rt1']/b1(er, x)['Rt1_DG'],
                        0.02, 0.04, 1.0, 0.03, lo=0.005, hi=0.12)
        print(f"  A step 1: eps_o = {eo:.4f} (Rt1 ratio "
              f"{b1(er, eo)['Rt1']/b1(er, eo)['Rt1_DG']:.3f})", flush=True)
        er, _ = _secant(lambda x: b1(x, eo)['s_late']/b1(x, eo)['s_DG'],
                        0.15, 0.19, 1.0, 0.02, lo=0.05, hi=0.35)
        print(f"  A step 2: eps_r = {er:.4f} (late ratio "
              f"{b1(er, eo)['s_late']/b1(er, eo)['s_DG']:.3f})", flush=True)
        rr = b1(er, eo)['Rt1']/b1(er, eo)['Rt1_DG']
        if abs(rr - 1.0) > 0.05:                      # one re-check pass
            eo, _ = _secant(lambda x: b1(er, x)['Rt1']/b1(er, x)['Rt1_DG'],
                            eo, 0.85*eo, 1.0, 0.03, lo=0.005, hi=0.12)
            print(f"  A re-check: eps_o = {eo:.4f} (Rt1 ratio "
                  f"{b1(er, eo)['Rt1']/b1(er, eo)['Rt1_DG']:.3f}, late "
                  f"{b1(er, eo)['s_late']/b1(er, eo)['s_DG']:.3f})",
                  flush=True)
        EPS_O[0] = eo
        db['variantA'] = dict(eps_r=er, eps_o=eo)
    else:
        FORM[0] = 'zb'
        er, _ = _secant(lambda x: b1(x)['s_late']/b1(x)['s_DG'],
                        0.16, 0.20, 1.0, 0.02)
        r = b1(er)
        print(f"  B: eps_r = {er:.4f} (late {r['s_late']/r['s_DG']:.3f}x, "
              f"Rt1 {r['Rt1']:.0f} = {r['Rt1']/r['Rt1_DG']:.2f}x DG N=1)",
              flush=True)
        db[_ck('variantB')] = dict(eps_r=er)
    save(db)
    return db['variantA' if which == 'A' else _ck('variantB')]


def reanchor(eps, db):
    """Joint re-anchoring at fixed eps: a_max scale from the Blasius late
    secant, threshold scale k' from the Blasius N=1 crossing (grid-matched
    target 349.3); then the favorable ladder at the re-anchored constants."""
    EPS[0] = eps
    base = db['smoke'][0] if 'smoke' in db else None
    Rt1_target = base['Rt1'] if base else 349.3
    slate_target = base['s_late'] if base else 1.044e-2
    # 1) a_max scale from the late secant (nearly linear in a_max)
    ASCALE[0] = 1.0
    r = row(0.0)
    ASCALE[0] = slate_target/r['s_late']
    # 2) k' from the N=1 crossing (secant iteration, 2 steps suffice)
    k0, k1 = 1.0, 1.15
    KSCALE[0] = k0; f0 = row(0.0)['Rt1'] - Rt1_target
    KSCALE[0] = k1; f1 = row(0.0)['Rt1'] - Rt1_target
    for _ in range(4):
        if abs(f1) < 2.0 or abs(f1 - f0) < 1e-9:
            break
        k2 = k1 - f1*(k1 - k0)/(f1 - f0)
        k0, f0 = k1, f1
        KSCALE[0] = k1 = k2; f1 = row(0.0)['Rt1'] - Rt1_target
    res = dict(eps=eps, a_scale=ASCALE[0], k_scale=KSCALE[0],
               a_max_new=ASCALE[0]*A_MAX,
               blasius_Rt1_residual=f1)
    print(f"  re-anchored at eps={eps:g}: a_max x{ASCALE[0]:.3f} "
          f"(-> {ASCALE[0]*A_MAX:.4f}), k x{KSCALE[0]:.3f} "
          f"(-> {KSCALE[0]*0.712:.3f}); Blasius Rt1 residual {f1:+.1f}",
          flush=True)
    res['ladder'] = [row(b) for b in FAVORABLE]
    res['ladder'] += [row(b) for b in ADVERSE]
    da, dbv = scores(res['ladder'])
    res['dev_a'], res['dev_b'] = da, dbv
    print(f"  re-anchored scores: dev_a={da:.3f} dev_b={dbv:.3f}", flush=True)
    ASCALE[0] = KSCALE[0] = 1.0
    return res


def zb_figure(db):
    """Part III: variant-B first-class candidate Fig 4 -- BOTH panels,
    full family, at a_visc = 0.0276 (eps_r = 0.1455, panel-(a)-optimal)
    and the leaner a_visc = 0.0230 (eps_r = 0.121)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def cols(rows):
        H = np.array([r['H'] for r in rows])
        sl = np.array([r['s_late'] for r in rows])
        R = np.array([r['Rt1'] for r in rows])
        o = np.argsort(H)
        return H[o], sl[o], R[o]

    H1, sl1, R1 = cols(db['final'])                       # 0.1455, canon C
    H2, sl2, R2 = cols(db['sweep']['0.121'] + db['family_0.121'])
    H0, sl0, R0 = cols(db['baseline'])
    joint = None                       # Part III extension: retuned ceiling
    jk = [k for k in db if k.startswith('final_C')
          and not k.startswith('final_eps')]
    if jk:
        C = jk[0].split('_C')[1]
        joint = (cols(db[jk[0]]),
                 rf'joint opt: $a_\mathrm{{visc}}='
                 rf'{0.19*db["final_eps" + "_C" + C]:.4f}$, $C={C}$')
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.2, 4.3))
    fig.patch.set_facecolor('white')
    Hg = np.geomspace(2.2, 10.6, 300)
    drela_g = np.asarray(dN_dRe_theta(Hg)); Rtc_g = np.asarray(Re_theta0(Hg))

    def style(ax, ylabel, ylim):
        ax.axvspan(4.03, 11.0, color='0.92', zorder=0)
        ax.axvline(4.03, color='0.6', lw=0.9, ls=':')
        ax.set_xscale('log'); ax.set_xticks([2.2, 2.6, 3, 3.5, 4, 5, 7, 10])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(2.12, 11.0)
        ax.set_xlabel(r'shape factor $H=\delta^*/\theta$')
        ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
        ax.grid(alpha=0.3, which='both')

    axa.semilogy(Hg, drela_g, 'k--', lw=1.8, label=r'Drela--Giles $dN/dRe_\theta$')
    m = np.isfinite(sl0)
    axa.semilogy(H0[m], sl0[m], '-', color='0.75', lw=1.1, label='canon, late')
    m = np.isfinite(sl1)
    axa.semilogy(H1[m], sl1[m], '-^', color='C0', ms=4.5, lw=1.3,
                 label=r'$a_\mathrm{visc}=0.0276$, late')
    m = np.isfinite(sl2)
    axa.semilogy(H2[m], sl2[m], '-v', color='C1', ms=4.0, lw=1.1, mfc='white',
                 label=r'$a_\mathrm{visc}=0.0230$, late')
    if joint is not None:
        (Hj, slj, Rj), jlab = joint
        m = np.isfinite(slj)
        axa.semilogy(Hj[m], slj[m], '-s', color='C2', ms=4.0, lw=1.2,
                     label=jlab + ', late')
    style(axa, r'$dN/dRe_\theta$', (drela_g.min()/12.0, drela_g.max()*1.3))
    axa.legend(fontsize=8.0, loc='upper left')
    axa.text(0.96, 0.06, '(a)', transform=axa.transAxes, fontsize=12,
             fontweight='bold', ha='right')

    N1_g = Rtc_g + 1.0/drela_g
    axb.semilogy(Hg, Rtc_g, '--', color='0.55', lw=1.4,
                 label=r'Drela critical $Re_{\theta 0}$')
    axb.semilogy(Hg, N1_g, 'k--', lw=1.8, label=r'Drela--Giles $N\!=\!1$ station')
    axb.semilogy(H0[np.isfinite(R0)], R0[np.isfinite(R0)], '-', color='0.75',
                 lw=1.1, label='canon')
    axb.semilogy(H1[np.isfinite(R1)], R1[np.isfinite(R1)], '-^', color='C0',
                 ms=4.5, lw=1.3, label=r'$a_\mathrm{visc}=0.0276$')
    axb.semilogy(H2[np.isfinite(R2)], R2[np.isfinite(R2)], '-v', color='C1',
                 ms=4.0, lw=1.1, mfc='white', label=r'$a_\mathrm{visc}=0.0230$')
    if joint is not None:
        (Hj, slj, Rj), jlab = joint
        m = np.isfinite(Rj)
        axb.semilogy(Hj[m], Rj[m], '-s', color='C2', ms=4.0, lw=1.2,
                     label=jlab)
    style(axb, r'onset $Re_\theta$', (Rtc_g.min(), N1_g.max()*4.0))
    axb.legend(fontsize=8.0, loc='upper right')
    axb.text(0.96, 0.06, '(b)', transform=axb.transAxes, fontsize=12,
             fontweight='bold', ha='right')
    fig.suptitle(r'CANDIDATE (not canon), variant B: rate '
                 r'$=\mathrm{softmax}_2(a_\mathrm{inv}\hat\Omega\langle\hat I'
                 r'\rangle_+,\,a_\mathrm{visc}\hat\Omega\langle -Z\rangle_+/R)'
                 r'\cdot$ canon gate (ceiling kept, gate blind to the viscous '
                 'term)', fontsize=9, y=1.0)
    plt.tight_layout()
    fp = os.path.join(OUT_DIR, 'model_calibrate_candidate_zb.png')
    plt.savefig(fp, dpi=150, facecolor='white')
    print(f'wrote {fp}', flush=True)


def tradeoff_figure(db):
    """The record's centerpiece: eps -> Fig-4 match (both panels) and
    Blasius perturbation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    eps, da, dbv, blr, blo = [], [], [], [], []
    b0 = [r for r in db['baseline'] if r['beta'] == 0.0][0]
    for key, rows in sorted(db['sweep'].items(),
                            key=lambda kv: float(kv[0].split('_C')[0])):
        if '_C' in key:
            continue                    # retuned-ceiling rows: not this curve
        a, b = scores(rows)
        r0 = [r for r in rows if r['beta'] == 0.0][0]
        eps.append(float(key)); da.append(a); dbv.append(b)
        blr.append(r0['s_late']/b0['s_late'] - 1.0)
        blo.append(r0['Rt1']/b0['Rt1'] - 1.0)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax.plot(eps, np.exp(da), '-o', color='C0',
            label='panel (a): worst rate factor vs Drela, H<=2.6')
    ax.plot(eps, np.exp(dbv), '-s', color='C1',
            label='panel (b): worst N=1-station factor')
    ax.plot(eps, [max(np.exp(a), np.exp(b)) for a, b in zip(da, dbv)],
            'k--', lw=1.0, label='max of both')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\epsilon$'); ax.set_ylabel('worst multiplicative deviation')
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
    ax2.plot(eps, 100*np.array(blr), '-o', color='C0',
             label='Blasius late secant shift %')
    ax2.plot(eps, 100*np.array(blo), '-s', color='C1',
             label='Blasius N=1 onset shift %')
    ax2.axhspan(-5, 5, color='0.92')
    ax2.set_xscale('log'); ax2.set_xlabel(r'$\epsilon$')
    ax2.set_ylabel('Blasius anchor perturbation [%]')
    ax2.grid(alpha=0.3, which='both'); ax2.legend(fontsize=8)
    lab = {'add': r'$P=\hat\Omega(\hat I+\epsilon)$',
           'sm2raw': r'$P=\hat\Omega\sqrt{\hat I^2+\epsilon^2}$',
           'sm2clip': r'$P=\hat\Omega\sqrt{\langle\hat I\rangle_+^2+\epsilon^2}$',
           'zc': r'$P=\hat\Omega\sqrt{\langle\hat I\rangle_+^2'
                 r'+\langle\epsilon(-Z)/R\rangle_+^2}$',
           'zc_ceil': r'zc + softmin gate',
           'zc2': r'two-$\epsilon$ (A)', 'zb': r'rate-only floor (B)'}
    fig.suptitle('trade-off: ' + lab[FORM[0]], fontsize=10)
    plt.tight_layout()
    fp = os.path.join(OUT_DIR, 'fpg_recal_tradeoff.png' if FORM[0] == 'zc'
                      else f'fpg_recal_tradeoff_{FORM[0]}.png')
    plt.savefig(fp, dpi=150, facecolor='white')
    print(f'wrote {fp}', flush=True)


def _ck(base):
    """Storage-key suffix for a retuned ceiling (Part III extension)."""
    return base if CEIL_B[0] is None else f'{base}_C{CEIL_B[0]:g}'


def load():
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            return json.load(f)
    return {}


def save(db):
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = OUT_JSON + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(db, f, indent=1)
    os.replace(tmp, OUT_JSON)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--final', type=float, default=None,
                    help='run the final stage at this eps')
    ap.add_argument('--eps-list', type=float, nargs='+',
                    default=[0.005, 0.01, 0.02, 0.035, 0.06, 0.10])
    ap.add_argument('--family', type=float, default=None,
                    help='adverse+lower-branch family at this eps')
    ap.add_argument('--impact', type=float, nargs='+', default=None,
                    help='cylinder/spheroid/Hiemenz budgets at these eps')
    ap.add_argument('--reanchor', type=float, default=None,
                    help='joint (a_max, k) re-anchoring at this eps')
    ap.add_argument('--tradeoff', action='store_true')
    ap.add_argument('--form', choices=['add', 'sm2raw', 'sm2clip',
                                       'zc', 'zc_ceil', 'zc2', 'zb'],
                    default='add')
    ap.add_argument('--eps-o', type=float, default=None,
                    help='variant-A onset epsilon (form zc2)')
    ap.add_argument('--tune', choices=['A', 'B'], default=None,
                    help='joint tuning of the follow-up variants')
    ap.add_argument('--signcheck', type=float, default=None,
                    help='sm2raw sign-convention check at this eps '
                         '(zero-suite + Blasius + beta=1 rows)')
    ap.add_argument('--boundedness', type=float, default=None,
                    help='high-Re wall-boundedness gate at this eps')
    ap.add_argument('--signmap', type=float, default=None,
                    help='-Z/R sign chart across profile classes')
    ap.add_argument('--zb-figure', action='store_true',
                    help='Part III two-value variant-B candidate figure')
    ap.add_argument('--ceil', type=float, default=None,
                    help='retuned gate ceiling C for form zb (k-carrying '
                         'units; canon 1851.2); results keyed _C<value>')
    args = ap.parse_args()
    run_all = not (args.smoke or args.baseline or args.sweep
                   or args.final is not None or args.family is not None
                   or args.impact is not None or args.reanchor is not None
                   or args.tradeoff or args.signcheck is not None
                   or args.boundedness is not None
                   or args.signmap is not None or args.tune is not None
                   or args.zb_figure)
    FORM[0] = args.form
    if args.eps_o is not None:
        EPS_O[0] = args.eps_o
    if args.ceil is not None:
        assert args.form == 'zb', '--ceil is a zb (variant B) knob'
        CEIL_B[0] = args.ceil
    global OUT_JSON, OUT_FIG
    if args.form != 'add':
        OUT_JSON = os.path.join(OUT_DIR,
                                f'fpg_recalibration_{args.form}.json')
        OUT_FIG = os.path.join(
            OUT_DIR, 'model_calibrate_candidate.png' if args.form == 'zc'
            else f'model_calibrate_candidate_{args.form}.png')
    db = load()
    if args.form != 'add' and 'baseline' not in db:
        legacy = os.path.join(OUT_DIR, 'fpg_recalibration_study.json')
        if os.path.exists(legacy):
            with open(legacy) as f:
                leg = json.load(f)
            for k in ('baseline', 'smoke'):
                if k in leg:
                    db[k] = leg[k]       # canon (FORM='add', eps=0) reference
            save(db)

    if args.tune is not None:
        print(f'== variant {args.tune} joint tuning (beta = 1) ==',
              flush=True)
        tune_variants(db, args.tune)

    if args.signcheck is not None:
        print(f'== sm2raw sign-convention check at eps={args.signcheck:g} '
              '==', flush=True)
        assert FORM[0] == 'sm2raw', 'run with --form sm2raw'
        db['signcheck'] = dict(zero=zero_suite(args.signcheck),
                               rows=ladder([0.0, 1.0], args.signcheck))
        save(db)

    if args.smoke or run_all:
        print('== smoke: eps=0 vs 2026-07-28-1041 audit ==', flush=True)
        r0 = ladder([0.0, 0.2], 0.0)
        assert abs(r0[0]['Rt1'] - 351) < 4, r0[0]['Rt1']       # Blasius N=1
        assert abs(r0[1]['s_early']/r0[1]['s_DG'] - 0.50) < 0.04
        db['smoke'] = r0; save(db)

    if args.baseline or run_all:
        print('== baseline: eps=0 full family ==', flush=True)
        db['baseline'] = ladder(FAVORABLE + ADVERSE, 0.0, lower=LOWER)
        db['zero_suite_eps0'] = zero_suite(0.0)
        save(db)

    if args.sweep or run_all:
        print('== sweep: favorable ladder ==', flush=True)
        sw = db.get('sweep', {})
        for eps in args.eps_list:
            key = _ck(f'{eps:g}')
            if key in sw:
                continue
            sw[key] = ladder(FAVORABLE, eps)
            da, dbv = scores(sw[key])
            print(f'  -> eps={eps:g}: dev_a={da:.3f} dev_b={dbv:.3f} '
                  f'(log-space, worst H<=2.6)', flush=True)
            db['sweep'] = sw; save(db)
        print('== sweep summary ==')
        for key, rows in sorted(db['sweep'].items(),
                                key=lambda kv: float(kv[0].split('_C')[0])):
            da, dbv = scores(rows)
            print(f'  eps={key:>6}: dev_a={da:6.3f} dev_b={dbv:6.3f} '
                  f'max={max(da, dbv):6.3f}')

    if args.final is not None or run_all:
        if args.final is not None:
            eps = args.final
        else:
            cand = [(max(scores(r)), float(k)) for k, r in db['sweep'].items()]
            eps = min(cand)[1]
            print(f'== minimax eps from sweep: {eps:g} ==', flush=True)
        print(f'== final stage at eps={eps:g} ==', flush=True)
        db[_ck('final_eps')] = eps
        db[_ck('final')] = ladder(FAVORABLE + ADVERSE, eps, lower=LOWER)
        db[_ck('zero_suite_final')] = zero_suite(eps)
        db['graze_eps0'] = graze_check(0.0)
        db[_ck('graze_final')] = graze_check(eps)
        save(db)
        if 'baseline' not in db:
            raise SystemExit('run --baseline first for the overlay figure')
        candidate_figure(db[_ck('final')], db['baseline'], eps)
        save(db)

    if args.family is not None:
        print(f'== adverse+lower family at eps={args.family:g} ==', flush=True)
        db[_ck(f'family_{args.family:g}')] = ladder(ADVERSE, args.family,
                                                    lower=LOWER)
        save(db)

    if args.impact is not None:
        print(f'== impact budgets at eps={args.impact} ==', flush=True)
        imp = db.get(_ck('impact'), {})
        imp['cylinder'] = cylinder_budget([0.0] + list(args.impact))
        imp['spheroid_re72a0'] = spheroid_budget([0.0] + list(args.impact))
        imp['hiemenz'] = hiemenz_budget(list(args.impact))
        for eps in args.impact:
            imp[f'zero_suite_eps{eps:g}'] = zero_suite(eps)
        db[_ck('impact')] = imp
        save(db)

    if args.reanchor is not None:
        print(f'== joint re-anchoring at eps={args.reanchor:g} ==', flush=True)
        db[f'reanchor_{args.reanchor:g}'] = reanchor(args.reanchor, db)
        save(db)

    if args.boundedness is not None:
        db[f'boundedness_{args.boundedness:g}'] = boundedness(args.boundedness)
        save(db)

    if args.signmap is not None:
        db[f'signmap_{args.signmap:g}'] = signmap(args.signmap)
        save(db)

    if args.zb_figure:
        zb_figure(db)

    if args.tradeoff:
        tradeoff_figure(db)


if __name__ == '__main__':
    main()
