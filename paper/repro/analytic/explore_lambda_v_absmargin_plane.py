"""(Gamma, Lambda_v) plane with absolute-instability proximity labels.

Falkner-Skan family (attached + Stewartson reversed-flow branch) drawn as
trajectories on the (Lambda_v, Gamma) indicator plane, each labeled by how
close its frozen profile is to ABSOLUTE instability, using the two measures
from the literature survey (notes_absolute_instability_lsb.md):

  * peak reverse flow u_rev/U_e  (2D KH absolute threshold ~ 15-20%,
    HM85 free-layer limit 13.6%);
  * the Avanci-Rodriguez-Alves geometric margin y_i - y_b: the height of
    the shear-layer inflection ABOVE the zero net mass-flux line, in theta
    units. Absolute instability onsets when the margin reaches ZERO
    (inflection point sinks into the recirculation).

Overlaid: contour LINES of the current band gate, which on this branch is
strictly a function of the two coordinates:
    Q_v(Gamma, Lambda_v) = 1 - sqrt(Gamma(2-Gamma)) / (1 + cV |Lambda_v|).

Sign convention: Lambda_v = -u'u''d^3/((u'd)^2+u^2) (Blasius all positive).
Reversed profiles are drawn from their omega=0 point outward (inside it the
profile is an attached boundary layer running backwards).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa: F401
from _saai import C_NU_AI, SIGMA_SA
from scipy.linalg import eig, eigh_tridiagonal
from scipy.optimize import minimize_scalar
from lib.correlations import Re_theta0
from explore_lsb_frozen_profile import build_profile, kernel_b, ETA_MAX

CV = 1.0
FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_explore')
ATTACHED = [(-0.1988, None), (-0.10, None), (0.0, None), (0.30, None),
            (1.00, None)]
REVERSED = [(-0.19, -0.03), (-0.17, -0.06), (-0.15, -0.08), (-0.12, -0.10)]


def indicators(pr):
    eta, u, up, upp = pr['eta'], pr['u'], pr['up'], pr['upp']
    den = (up*eta)**2 + u**2 + 1e-30
    Gam = 2*(up*eta)**2/den
    Lv = -up*upp*eta**3/den
    return Gam, Lv


def abs_margin(pr):
    """(u_rev, (y_i - y_b)/theta): peak reverse flow and the Avanci margin.
    Positive margin = inflection above the zero-mass-flux line = convective;
    absolute instability onsets at margin -> 0."""
    eta, u = pr['eta'], pr['u']
    Th = pr['Theta']
    urev = max(0.0, -float(np.min(u)))
    jm = int(np.argmin(u))
    j_i = jm + int(np.argmax(pr['up'][jm:]))
    cum = np.concatenate([[0.0], np.cumsum(0.5*(u[1:] + u[:-1])
                                           * np.diff(eta))])
    above = np.where(cum[jm:] > 0.0)[0]
    j_b = jm + (int(above[0]) if len(above) else 0)
    return urev, (float(eta[j_i]) - float(eta[j_b]))/Th


def os_wavepacket(pr, Re_th, N=420, alo=0.03, ahi=1.4):
    """PHYSICAL wavepacket at finite Reynolds number: most-amplified temporal
    ORR-SOMMERFELD mode. Fixes the Rayleigh wall-slip artifact (inviscid
    |phi'| peaks AT the wall for weakly unstable inflectional modes because
    Rayleigh allows slip) and covers Blasius/favorable profiles whose TS
    instability is viscous. No-slip clamped BCs phi = phi' = 0 via mirror
    ghost nodes in D4. Energy E = |phi'|^2 + alpha^2 |phi|^2, normalized.
    Returns (yh, E, alpha*, c*, unstable?)."""
    Th = pr['Theta']
    ymax = min(ETA_MAX/Th, 30.0)
    yh = np.linspace(0.0, ymax, N)
    h = yh[1] - yh[0]
    U = np.interp(yh*Th, pr['eta'], pr['u'], right=1.0)
    Upp = Th*Th*np.interp(yh*Th, pr['eta'], pr['upp'], right=0.0)
    n = N - 2                                   # interior nodes
    I = np.eye(n)
    D2 = (np.diag(np.ones(n-1), -1) - 2*I + np.diag(np.ones(n-1), 1))/h**2
    D4 = (np.diag(np.ones(n-2), -2) - 4*np.diag(np.ones(n-1), -1) + 6*I
          - 4*np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-2), 2))/h**4
    D4[0, 0] = 7.0/h**4                         # ghost phi(-1) = phi(1)
    D4[-1, -1] = 7.0/h**4                       # (phi'=0 at both ends)
    Ui, Uppi = U[1:-1], Upp[1:-1]

    def solve(alpha, want_vec=False):
        L2 = D2 - alpha**2*I
        A = (np.diag(Ui) @ L2 - np.diag(Uppi)
             - (D4 - 2*alpha**2*D2 + alpha**4*I)/(1j*alpha*Re_th))
        if not want_vec:
            c = eig(A, L2, right=False)
            c = c[np.isfinite(c)]
            c = c[(np.real(c) > 0.02) & (np.real(c) < 0.98)]
            return alpha*float(np.max(np.imag(c))) if len(c) else -1.0
        c, V = eig(A, L2, right=True)
        ok = np.isfinite(c) & (np.real(c) > 0.02) & (np.real(c) < 0.98)
        idx = np.where(ok)[0][int(np.argmax(np.imag(c[ok])))]
        return complex(c[idx]), V[:, idx]

    agrid = np.geomspace(alo, ahi, 16)
    wg = [solve(a) for a in agrid]
    j = int(np.argmax(wg))
    r = minimize_scalar(lambda a: -solve(a),
                        bounds=(agrid[max(j-1, 0)],
                                agrid[min(j+1, len(agrid)-1)]),
                        method='bounded', options={'xatol': 2e-3})
    astar = float(r.x)
    cstar, phi = solve(astar, want_vec=True)
    dphi = np.gradient(phi, h)
    E = np.abs(dphi)**2 + astar**2*np.abs(phi)**2
    return yh[1:-1], E/E.max(), astar, cstar, astar*cstar.imag > 0


def rayleigh_wavepacket(pr, N=600, alo=0.02, ahi=1.2):
    """PHYSICAL wavepacket of the frozen profile: the most-amplified temporal
    Rayleigh mode. Returns (yh, E, alpha*, c*) with E(y) the disturbance
    kinetic-energy density |phi'|^2 + alpha^2 |phi|^2 of the eigenfunction,
    normalized to max 1 -- 'where the wavepacket actually sits'. None if the
    profile has no inviscid instability (e.g. Blasius: viscous TS wave,
    outside Rayleigh)."""
    Th = pr['Theta']
    yh = np.linspace(0.0, ETA_MAX/Th, N)
    u = np.interp(yh*Th, pr['eta'], pr['u'], right=1.0)
    upp = Th*Th*np.interp(yh*Th, pr['eta'], pr['upp'], right=0.0)
    h = yh[1] - yh[0]
    D2 = (np.diag(np.ones(N-1), -1) - 2*np.eye(N)
          + np.diag(np.ones(N-1), 1))/h**2

    def solve(alpha, want_vec=False):
        B = D2 - alpha**2*np.eye(N)
        A = np.diag(u) @ B - np.diag(upp)
        if not want_vec:
            c = eig(A[1:-1, 1:-1], B[1:-1, 1:-1], right=False)
            return alpha*float(np.max(np.imag(c)))
        c, V = eig(A[1:-1, 1:-1], B[1:-1, 1:-1], right=True)
        i = int(np.argmax(np.imag(c)))
        return complex(c[i]), V[:, i]

    agrid = np.geomspace(alo, ahi, 20)
    wg = [solve(a) for a in agrid]
    j = int(np.argmax(wg))
    if wg[j] < 5e-4:
        return None
    r = minimize_scalar(lambda a: -solve(a),
                        bounds=(agrid[max(j-1, 0)],
                                agrid[min(j+1, len(agrid)-1)]),
                        method='bounded', options={'xatol': 2e-3})
    astar = float(r.x)
    cstar, phi = solve(astar, want_vec=True)
    dphi = np.gradient(phi, h)
    E = np.abs(dphi)**2 + astar**2*np.abs(phi)**2
    return yh[1:-1], E/E.max(), astar, cstar


def spatial_mode_weight(pr, Re_th=200.0, gate='lv', ufrac=0.03):
    """|v| of the spatial generalized eigenmode (the parabolic march's
    asymptotic dN/dx mode from the frozen-profile study) on the interior
    grid -- shows WHERE along the profile the over-amplification lives."""
    yh, u, absw, b = kernel_b(pr, Re_th, gate)
    h = yh[1] - yh[0]
    k = (C_NU_AI/SIGMA_SA)/Re_th
    uf = np.maximum(u, ufrac)
    d = (b - 2.0*k/h**2)/uf
    e = (k/h**2)/np.sqrt(uf[:-1]*uf[1:])
    _, vec = eigh_tridiagonal(d, e, select='i',
                              select_range=(len(yh)-1, len(yh)-1))
    v = np.abs(vec[:, 0])/np.sqrt(uf)
    return v/v.max()


def indicators_at(pr, yh):
    """(Gamma, Lambda_v) evaluated at theta-unit heights yh."""
    Th = pr['Theta']
    u = np.interp(yh*Th, pr['eta'], pr['u'], right=1.0)
    w = Th*np.interp(yh*Th, pr['eta'], pr['up'], right=0.0)
    dwdy = Th*Th*np.interp(yh*Th, pr['eta'], pr['upp'], right=0.0)
    den = (w*yh)**2 + u**2 + 1e-30
    return 2*(w*yh)**2/den, -w*dwdy*yh**3/den


def overlay_wavepacket(ax, pr, color, Re_th):
    yh, E, astar, cstar, unstable = os_wavepacket(pr, Re_th)
    Gam, Lv_ = indicators_at(pr, yh)
    hot = E > 0.5
    ax.scatter(Lv_[hot], Gam[hot], s=70*E[hot]**2, facecolors='none',
               edgecolors=color, linewidths=0.9, zorder=6)
    jp = int(np.argmax(E))
    ax.plot(Lv_[jp], Gam[jp], '*', ms=17, mfc='white' if unstable else '0.8',
            mec=color, mew=1.6, zorder=7)
    return Lv_[jp], Gam[jp], astar, cstar, unstable


def main():
    os.makedirs(FIGD, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 6.4))

    # ---- Q_v contour lines (the current gate on this branch) ----
    lv = np.linspace(-2.4, 1.6, 801)
    ga = np.linspace(0.0, 2.0, 401)
    LV, GA = np.meshgrid(lv, ga)
    QV = 1.0 - np.sqrt(np.clip(GA*(2 - GA), 0, None))/(1 + CV*np.abs(LV))
    cs = ax.contour(LV, GA, QV, levels=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                    colors='0.25', linewidths=0.9, linestyles=':')
    ax.clabel(cs, fmt=lambda v: f'$Q_v$={v:g}', fontsize=7.5)

    # ---- attached branch: no recirculation, convective by construction ----
    # Gamma > 0.05 floor: near (0,0) both indicators vanish (outer edge /
    # omega=0 points) and the collapsing tails draw artifact-like straight
    # segments into the origin; the gate is irrelevant there anyway
    # (production ~ |omega| -> 0).
    blues = plt.cm.winter
    for k, (beta, guess) in enumerate(ATTACHED):
        pr = build_profile(beta, guess)
        Gam, Lv_ = indicators(pr)
        m = (pr['eta'] < np.interp(0.999, np.maximum.accumulate(pr['u']),
                                   pr['eta'])) & (Gam > 0.05)
        c = blues(k/(len(ATTACHED) - 1))
        ax.plot(np.where(m, Lv_, np.nan), np.where(m, Gam, np.nan), '-',
                color=c, lw=1.5,
                label=f"$\\beta$={beta:+.2f}  H={pr['H']:.2f}  attached "
                      "(no reverse flow)")
        # amplification-relevant Reynolds number: 2x the Drela critical
        # Re_theta (capped: strongly favorable profiles push it beyond
        # what the wall-normal grid resolves)
        Re_th = float(min(2.0*Re_theta0(np.clip(pr['H'], 2.2, None)), 6000.0))
        wp = overlay_wavepacket(ax, pr, c, Re_th)
        print(f"attached beta={beta:+.3f} (H={pr['H']:.2f}, "
              f"Re_th={Re_th:.0f}): OS wavepacket peak "
              f"(Lv={wp[0]:+.2f}, Gam={wp[1]:.2f}), alpha*theta={wp[2]:.3f},"
              f" c={wp[3]:.3f}, unstable={wp[4]}", flush=True)

    # ---- Stewartson reversed branch, from the omega=0 point outward ----
    warms = plt.cm.autumn
    for k, (beta, guess) in enumerate(REVERSED):
        pr = build_profile(beta, guess)
        Gam, Lv_ = indicators(pr)
        urev, marg = abs_margin(pr)
        eta = pr['eta']
        m = (eta < np.interp(0.999, np.maximum.accumulate(pr['u']), eta)) \
            & (eta >= eta[int(np.argmin(pr['u']))]) & (Gam > 0.05)
        c = warms(k/(len(REVERSED)))
        lab = (f"$\\beta$={beta:+.2f} rev  H={pr['H']:.2f}  "
               f"$u_{{rev}}$={urev*100:.1f}%  "
               f"$(y_i{{-}}y_b)/\\theta$={marg:.1f}")
        ax.plot(np.where(m, Lv_, np.nan), np.where(m, Gam, np.nan), '--',
                color=c, lw=1.7, label=lab)
        j = m.nonzero()[0][int(np.argmax(np.abs(Lv_[m])))]
        ax.annotate(f"{urev*100:.1f}%", (Lv_[j], Gam[j]), fontsize=8,
                    color=c, textcoords='offset points', xytext=(4, 4))
        # model's spurious spatial-mode residence (frozen study): filled dots
        # + small filled circle at its peak
        w = spatial_mode_weight(pr)
        Gi, Li = Gam[1:-1], Lv_[1:-1]
        hot = w > 0.2
        ax.scatter(Li[hot], Gi[hot], s=60*w[hot]**2, color=c, alpha=0.35,
                   edgecolors='none', zorder=4)
        jp = int(np.argmax(w))
        ax.plot(Li[jp], Gi[jp], 'o', ms=7, mfc=c, mec='k', mew=0.7,
                zorder=5)
        # PHYSICAL wavepacket (OS eigenfunction energy at bubble-typical
        # Re_theta=200): open circles + big open star at its peak
        wp = overlay_wavepacket(ax, pr, c, 200.0)
        print(f"reversed beta={beta:+.3f} (H={pr['H']:.2f}): OS wavepacket "
              f"peak (Lv={wp[0]:+.2f}, Gam={wp[1]:.2f}), alpha*theta="
              f"{wp[2]:.3f}, c={wp[3]:.3f}, unstable={wp[4]}; model mode "
              f"peak (Lv={Li[jp]:+.2f}, Gam={Gi[jp]:.2f})", flush=True)

    ax.axhline(1.0, color='0.8', lw=0.6)
    ax.set_xlim(-2.4, 1.6)
    ax.set_ylim(0.0, 2.02)
    ax.set_xlabel(r'$\Lambda_v$')
    ax.set_ylabel(r'$\Gamma$')
    ax.set_title(
        'Falkner–Skan family on the gate plane '
        r'($\star$: PHYSICAL wavepacket = Orr–Sommerfeld eigenfunction '
        'energy peak (gray fill: least-stable, not unstable);\nfilled dots '
        '+ small circle: the MODEL mode; open circles: wavepacket residence '
        '$E>0.5E_{max}$. Attached at $Re_\\theta=2Re_{\\theta 0}(H)$, '
        r'reversed at $Re_\theta$=200', fontsize=9)
    ax.legend(fontsize=7.5, loc='lower left', framealpha=0.9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out = os.path.join(FIGD, 'lambda_v_absmargin_plane.png')
    plt.savefig(out, dpi=140)
    print('wrote', out)


if __name__ == '__main__':
    main()
