"""fig:shapefactor -> paper/figs/shapefactor_amplification.pdf.

dN/dRe_theta(H) from the Eq.-(transport) disturbance transport marched on
each Falkner-Skan profile vs the Drela-Giles envelope. The sphere kernel
(sphere_rate) supplies the amplification rate x onset from the indicator
sphere; there is no separate favorable-gradient treatment. The favorable
branch is amplified weakly / held near-laminar by the definite product P<~0,
so only ONE favorable curve is shown. Also exports march()/profile_ints()/
drela/sphere_rate for fig02 and fig03."""
import _saai
from _saai import SIGMA_SA, K_R
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0
from lib.aft_sources import (compute_aft_amplification_rate,
                             compute_composite_gate)

# --- sphere kernel (replaces the old rate*gate product) --------------------
# Dimensionless rate x onset on the indicator sphere; the transport source is
# this times |du/dy|.  X=u, Y=y u', Z=1/2 y^2 u'' are the (velocity, shear,
# curvature) indicators; Shat=Y/sqrt(X^2+Y^2) is the shear fraction; g is the
# Rayleigh coordinate; the definite product P=Shat*g drives both the linear
# rate (ceiling a_max at P>=1) and the onset scale Re_Omega,crit(P).
A_MAX = 0.19          # Michalke free-shear eigenvalue (fixed)
# CANONICAL MODEL CONSTANTS (paper Sec. II.D, 2026-07-23 canon).
# Laminar-diffusion reduction: c=1 is excluded structurally (the Blasius
# inception is diffusion-limited past the Drela N=1 station: unanchorable);
# below that, flatness improves smoothly as c falls while the resolvable
# nuHat band thins as sqrt(c). c = 1/6 keeps the N=9 prediction within ~7%
# of Drela at twice the molecular floor of the previous 1/12.
C_NU_AI = 1.0/6.0
# Onset threshold Re_Omega,crit = k * softmin_2(CEIL, A + B*(Sg)^-2):
# shape (CEIL, A, B) = (2600, 175, 2) fixed by the LST neutral-point graze
# (fig02_onset_graze.py, NEVER refit); the single WHOLE-EQUATION scale k
# compensates the residual laminar diffusion (the young disturbance loses to
# the drain for a while after linear theory declares growth begun) and is
# anchored by the Blasius N=1 crossing at Drela's
# Re_theta = 338, computed with the GRID-CONVERGED instrument (nx=3200,
# ny=2400: the N=1 crossing is wall-normal-resolution sensitive; ny=600
# under-resolves the thin early layer, and the short-domain marches of
# explore_k_anchor.py / explore_reomc_retune.py are worse -- treat their
# outputs as superseded). k = 0.708 at c_nu,ai = 1/6.
K_ANCHOR  = 0.712             # whole-equation scale, anchored (converged march)
REOM_CEIL = 2600.0*K_ANCHOR   # favorable-saturation ceiling
REOM_A    = 175.0*K_ANCHOR    # near-separation floor
REOM_B    = 2.0*K_ANCHOR      # inverse-square coefficient
REOM_N    = 2.0               # soft-min sharpness
RAMP_W = 0.35         # onset ramp half-width

def sphere_rate(u, dudy, yc):
    d2u = np.gradient(dudy, yc)
    X = u; Y = yc*dudy; Z = 0.5*yc**2*d2u
    R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
    Shat = Y/np.sqrt(X*X + Y*Y + 1e-30)
    g = (Y - X - Z)/R
    P = Shat*g
    a = A_MAX*np.minimum(1.0, np.clip(P, 0.0, None))   # LINEAR in P, ceiling at P>=1
    ReOm = yc**2*np.abs(dudy)
    _pw = REOM_A + REOM_B*np.maximum(P, 1e-6)**(-2.0)
    reomc = (REOM_CEIL**(-REOM_N) + _pw**(-REOM_N))**(-1.0/REOM_N)
    onset = 0.5*(1.0 + np.tanh((ReOm/reomc - 1.0)/RAMP_W))
    return a*onset       # dimensionless; caller does b = sphere_rate(...)*|du/dy|

def drela(H):
    """Drela-Giles envelope amplification rate (imported from src.correlations)."""
    return np.asarray(dN_dRe_theta(H))

def march(fs, x_max, nx=1600, ny=1200, y_top=None, seed=1.0, beta=None, k_r=K_R,
          ufrac=0.0):
    """March the disturbance transport on the FS field; return x, N(x). If beta is
    given, feed the wedge's own lambda_p(x,y) into the kernel (onset-delay cliff +
    favorable-rate factor with slope k_r; k_r=0 disables the factor). For
    reversed-flow (lower-branch) profiles the parabolic march is regularized by
    flooring the advection speed at ufrac*U_e (sensitivity below 1% for
    ufrac in [0.015, 0.06]); attached profiles use ufrac=0."""
    m = None if beta is None else beta/(2.0 - beta)
    if y_top is None:
        eta99 = np.interp(0.99, np.maximum.accumulate(fs.u), fs.eta)
        y_top = 8.0*eta99*np.sqrt(x_max/fs.inviscid_at(x_max))
    dy = y_top/ny; yc = (np.arange(ny) + 0.5)*dy; dx = x_max/nx
    nu = np.ones(ny)*seed; N = [0.0]; xs = [0.0]
    k = (C_NU_AI/SIGMA_SA)/dy**2
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, np.arange(ny + 1)*dy, cellCentered=True)
        u = np.maximum(u, max(1e-12, ufrac*fs.inviscid_at(x)))
        vp = np.clip(v, 0, None)/dy; vm = np.clip(-v, 0, None)/dy
        di = vp + vm + 2*k; lo = -(vp[1:] + k); up = -(vm[:-1] + k)
        di[0] += k; di[-1] -= k
        b = sphere_rate(u, dudy, yc)*np.abs(dudy)
        main = u/dx + di; rhs = u/dx*nu + b*nu; rhs[-1] += vm[-1]*seed
        A = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        xs.append((i + 1)*dx); N.append(float(np.log(max(nu.max()/seed, 1e-300))))
    return np.array(xs), np.array(N)

def profile_ints(fs):
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    I_ds = np.trapezoid(1 - fs.u, fs.eta)
    return I_th, I_ds/I_th

def measures_for_beta(beta, verbose=True, k_r=K_R, wedge_lambda=True,
                      guess=None, ufrac=0.0):
    """wedge_lambda=False marches with lambda_p = 0: neither cliff nor rate
    factor act (the un-modified kernel), the panel-(b) before curve. A
    negative shooting guess selects the reversed-flow (Stewartson lower)
    branch; pass ufrac>0 with it (see march)."""
    fs = FalknerSkanWedge(beta, guess=guess); I_th, H = profile_ints(fs)
    bb = beta if wedge_lambda else None
    x_max = 4e6 if beta == 0.0 else (3e5 if beta > 0 else 1.2e6)
    for _ in range(12):
        xs, N = march(fs, x_max, beta=bb, k_r=k_r, ufrac=ufrac)
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.15; continue
        if N[-1] > 14.0:
            x_max = 1.1*float(np.interp(14.0, N, xs))
            xs, N = march(fs, x_max, beta=bb, k_r=k_r, ufrac=ufrac); break
        x_max *= 3.0
    Ue = fs.inviscid_at(np.maximum(xs, 1e-12)); Rt = I_th*np.sqrt(xs*Ue)
    Rt1 = float(np.interp(1.0, N, Rt)) if N[-1] >= 1.0 else float('nan')
    Rt5 = float(np.interp(5.0, N, Rt)); Rt9 = float(np.interp(9.0, N, Rt))
    if not np.isfinite(N).all() or N[-1] < 9.0 or Rt9 <= Rt1:
        # the diagnostic-only un-factored strongly-favorable march can stall
        # below N=9 (cliff raised e^{K_lambda*lam_p} with no rate factor);
        # report NaNs and let the panel drop the point
        if verbose:
            print(f"beta={beta:+.3f}  H={H:.3f}: march stalled below N=9 "
                  f"(N_end={N[-1]:.1f}) -- NaN row", flush=True)
        return H, float('nan'), float('nan'), Rt1
    s_mean = 4.0/(Rt5 - Rt1); s_late = 4.0/(Rt9 - Rt5)  # early N in [1,5], late N in [5,9] (disjoint, visibly distinct)
    if verbose:
        d = float(drela(H))
        print(f"beta={beta:+.3f}  H={H:.3f}  Rt1={Rt1:4.0f}  Rt9={Rt9:6.0f}  "
              f"s_mean={s_mean:.4e} ({s_mean/d:4.2f}x)  s_late={s_late:.4e} ({s_late/d:4.2f}x)  "
              f"Drela={d:.4e}", flush=True)
    return H, s_mean, s_late, Rt1


def main():
    # -0.1988 is the separation-limit profile where the attached (upper) and
    # reversed-flow (lower) branches coincide -- it joins the two curves.
    adverse = [-0.1988, -0.195, -0.18, -0.15, -0.12, -0.09, -0.06, -0.03, 0.0]
    favorable = [0.05, 0.1, 0.2, 0.35, 0.55, 1.0]
    lower = [(-0.19, -0.03), (-0.17, -0.06), (-0.15, -0.08), (-0.12, -0.10)]
    oa = [measures_for_beta(b) for b in adverse]
    iBl = adverse.index(0.0)   # Blasius joins the favorable branch here
    print(f"BLASIUS ANCHORS: N=1 crossing Re_theta={oa[iBl][3]:.0f} "
          f"(target ~338), N=9 crossing implied via s_mean={oa[iBl][1]:.4e}", flush=True)
    print("favorable (sphere kernel holds these weak / near-laminar):")
    of = [measures_for_beta(b) for b in favorable]
    print("reversed-flow (Stewartson lower branch), advection floor 0.03 U_e:")
    ol = [measures_for_beta(b, guess=g, ufrac=0.03) for b, g in lower]
    ol = [oa[0]] + ol          # separation-limit profile joins the lower branch
    Ha = np.array([o[0] for o in oa]); ma = np.array([o[1] for o in oa]); la = np.array([o[2] for o in oa])
    Ra = np.array([o[3] for o in oa])
    # prepend the shared Blasius point so the favorable curve connects
    Hf = np.array([oa[iBl][0]] + [o[0] for o in of])
    mf = np.array([oa[iBl][1]] + [o[1] for o in of])
    lf = np.array([oa[iBl][2]] + [o[2] for o in of])
    Rf = np.array([oa[iBl][3]] + [o[3] for o in of])
    Hl = np.array([o[0] for o in ol]); ml = np.array([o[1] for o in ol])
    ll = np.array([o[2] for o in ol]); Rl = np.array([o[3] for o in ol])
    Hg = np.geomspace(min(Hf), max(Hl), 400)
    fig, (ax, axR) = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax.semilogy(Hg, drela(Hg), 'k--', lw=1.8, label=r'Drela--Giles envelope $dn/dRe_\theta(H)$')
    o = np.argsort(Ha)
    ax.fill_between(Ha[o], ma[o], la[o], color='C0', alpha=0.15, lw=0)
    ax.semilogy(Ha[o], la[o], '-o', color='C0', lw=1.9, ms=4.5, label=r'late-envelope rate ($N\in[5,9]$ secant)')
    ax.semilogy(Ha[o], ma[o], '-s', color='C2', lw=1.9, ms=4.5, label=r'onset-to-transition mean ($N\in[1,9]$ secant)')
    o = np.argsort(Hf)
    ax.fill_between(Hf[o], mf[o], lf[o], color='C0', alpha=0.15, lw=0)
    ax.semilogy(Hf[o], lf[o], '-o', color='C0', lw=1.9, ms=4.5)
    ax.semilogy(Hf[o], mf[o], '-s', color='C2', lw=1.9, ms=4.5,
                label=r'favorable (near-laminar, $P\!\to\!0$)')
    o = np.argsort(Hl)
    ax.fill_between(Hl[o], ml[o], ll[o], color='C3', alpha=0.12, lw=0)
    ax.semilogy(Hl[o], ll[o], '-o', color='C3', lw=1.9, ms=4.5)
    ax.semilogy(Hl[o], ml[o], '-s', color='C3', lw=1.9, ms=4.5,
                label=r'reversed flow (Stewartson lower branch): $\approx\!0.7\times$')
    iB = int(np.argmin(np.abs(Ha - 2.591)))
    for arr in (ma, la):
        ax.plot(Ha[iB], arr[iB], 'o', color='C3', ms=9, mfc='none', mew=1.6, zorder=6)
    ax.annotate('Blasius', (Ha[iB], la[iB]), (Ha[iB] + 0.03, la[iB]*1.5), fontsize=8, color='C3')
    ax.axvspan(4.03, max(Hl) + 0.3, color='0.92', zorder=0)
    ax.axvline(4.03, color='0.6', lw=0.9, ls=':')
    ax.text(4.12, 2.1e-3, 'separated\n($H>4.03$)', fontsize=7.5, va='bottom')
    ax.set_xlabel(r'shape factor $H=\delta^*/\theta$')
    ax.set_ylabel(r'$dN/dRe_\theta$')
    ax.set_xscale('log')
    ax.set_xticks([2.2, 2.6, 3, 3.5, 4, 5, 7, 10])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(min(Hf)*0.98, max(Hl)*1.04); ax.set_ylim(1.5e-3, 6e-1)
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=7.6, loc='upper left')
    ax.text(0.94, 0.06, '(a)', transform=ax.transAxes, fontsize=11, fontweight='bold')

    # ----- right panel: onset Re_theta vs H -----
    Rtc_g = np.asarray(Re_theta0(Hg))
    N1_g = Rtc_g + 1.0/np.asarray(drela(Hg))
    axR.semilogy(Hg, Rtc_g, '--', color='0.5', lw=1.5,
                 label=r'critical $Re_{\theta c}(H)$ (Drela--Giles)')
    axR.semilogy(Hg, N1_g, 'k--', lw=1.8,
                 label=r'$N\!=\!1$ station implied: $Re_{\theta c}+1/(dn/dRe_\theta)$')
    o = np.argsort(Ha)
    axR.semilogy(Ha[o], Ra[o], '-o', color='C2', lw=1.9, ms=4.5,
                 label=r'model $N\!=\!1$ crossing')
    o = np.argsort(Hf)
    axR.semilogy(Hf[o], Rf[o], '-o', color='C2', lw=1.9, ms=4.5,
                 label='favorable (near-laminar)')
    o = np.argsort(Hl)
    axR.semilogy(Hl[o], Rl[o], '-o', color='C3', lw=1.9, ms=4.5,
                 label='reversed flow (lower branch)')
    iB = int(np.argmin(np.abs(Ha - 2.591)))
    axR.plot(Ha[iB], Ra[iB], 'o', color='C3', ms=9, mfc='none', mew=1.6, zorder=6)
    axR.annotate('Blasius\n(anchor)', (Ha[iB], Ra[iB]), (Ha[iB] + 0.06, Ra[iB]*1.25),
                 fontsize=8, color='C3')
    axR.axvspan(4.03, max(Hl) + 0.3, color='0.92', zorder=0)
    axR.axvline(4.03, color='0.6', lw=0.9, ls=':')
    axR.set_xlabel(r'shape factor $H=\delta^*/\theta$')
    axR.set_ylabel(r'onset $Re_\theta$')
    axR.set_xscale('log')
    axR.set_xticks([2.2, 2.6, 3, 3.5, 4, 5, 7, 10])
    axR.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axR.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axR.set_xlim(min(Hf)*0.98, max(Hl)*1.04); axR.set_ylim(8, 2e4)
    axR.grid(alpha=0.3, which='both'); axR.legend(fontsize=7.6, loc='upper right')
    axR.text(0.03, 0.06, '(b)', transform=axR.transAxes, fontsize=11, fontweight='bold')
    plt.tight_layout(); plt.savefig('figs/shapefactor_amplification.pdf')
    print('wrote figs/shapefactor_amplification.pdf')


if __name__ == '__main__':
    main()
