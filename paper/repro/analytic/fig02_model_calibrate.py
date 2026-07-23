"""fig:modelcalibrate -> paper/figs/model_calibrate.pdf.

Full-family calibration of the sphere kernel (light theme, white bg) marched
across the WHOLE Falkner-Skan family with fig04's sphere-kernel marcher:
  (a) marched rate dN/dRe_theta vs shape factor H, reported as an early
      (N in [1,5]) and a late (N in [5,9]) secant -- the local slope rises
      along the layer, so both are shown and the band between them shaded.
  (b) the model onset Re_theta (N=1 crossing) vs H.
Both against the Drela--Giles envelope. The attached (favorable+adverse) and
reversed-flow (Stewartson lower) branches are joined into ONE continuous curve
sorted by H (no per-range styling). y-limits are Drela's own range, so where
the model under-predicts the rate (or over-predicts the onset) the curve simply
runs off the panel edge.
Reuses fig04's measures_for_beta / drela / Re_theta0 (sphere kernel)."""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from fig04_shapefactor import measures_for_beta, drela, Re_theta0

XTICKS = [2.2, 2.6, 3, 3.5, 4, 5, 7, 10]
H_SEP = 4.03


def _rows(specs, **kw):
    """specs: (beta, guess). Returns (beta, H, s_early, s_late, Rt1): the early
    (N in [1,5] secant) and late (N in [5,9] secant) rates and the N=1 onset."""
    out = []
    for beta, guess in specs:
        H, s_early, s_late, Rt1 = measures_for_beta(
            beta, verbose=False, guess=guess, **kw)
        tag = 'NaN' if not (np.isfinite(s_early) and np.isfinite(Rt1)) else ''
        print(f"  beta={beta:+.4f}  H={H:6.3f}  s_early={s_early:.4e}  "
              f"s_late={s_late:.4e}  Rt1={Rt1:8.1f}  {tag}", flush=True)
        out.append((beta, H, s_early, s_late, Rt1))
    return out


def _cols(rows):
    b = np.array([r[0] for r in rows]); H = np.array([r[1] for r in rows])
    se = np.array([r[2] for r in rows])   # early rate, N in [1,5]
    sl = np.array([r[3] for r in rows])   # late rate,  N in [5,9]
    R = np.array([r[4] for r in rows])
    return b, H, se, sl, R


def main():
    attached_betas = [1.0, 0.55, 0.35, 0.2, 0.1, 0.05, 0.0,
                      -0.03, -0.06, -0.09, -0.12, -0.15, -0.18, -0.1988]
    lower = [(-0.19, -0.03), (-0.17, -0.06), (-0.15, -0.08), (-0.12, -0.10)]

    print("attached (favorable + adverse) branch:", flush=True)
    att = _rows([(b, None) for b in attached_betas])
    print("reversed-flow (Stewartson lower branch), advection floor 0.03 U_e:",
          flush=True)
    rev = _rows(lower, ufrac=0.03)

    _, Ha, sea, sla, Ra = _cols(att)
    _, Hr, ser, slr, Rr = _cols(rev)

    # ---- join the two branches into ONE curve, sorted by H (continuous across
    #      Blasius and the separation limit; single style, no per-range colors) ----
    H = np.concatenate([Ha, Hr]); se = np.concatenate([sea, ser])
    sl = np.concatenate([sla, slr]); R = np.concatenate([Ra, Rr])
    o = np.argsort(H); H, se, sl, R = H[o], se[o], sl[o], R[o]
    print(f"\nH range: {H.min():.3f} -> {H.max():.3f}", flush=True)

    # ================= figure =================
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.2, 4.3))
    fig.patch.set_facecolor('white')
    Hg = np.geomspace(2.2, 10.6, 300)
    drela_g = np.asarray(drela(Hg)); Rtc_g = np.asarray(Re_theta0(Hg))

    def style(ax, ylabel, ylim):
        ax.axvspan(H_SEP, 11.0, color='0.92', zorder=0)
        ax.axvline(H_SEP, color='0.6', lw=0.9, ls=':')
        ax.set_xscale('log'); ax.set_xticks(XTICKS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(2.12, 11.0)
        ax.set_xlabel(r'shape factor $H=\delta^*/\theta$')
        ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
        ax.grid(alpha=0.3, which='both')

    # ---- (a) rate dN/dRe_theta vs H: early + late, single color ----
    axa.semilogy(Hg, drela_g, 'k--', lw=1.8, label=r'Drela--Giles $dN/dRe_\theta$')
    m = np.isfinite(se) & np.isfinite(sl)
    axa.fill_between(H[m], se[m], sl[m], color='C0', alpha=0.15, lw=0)
    axa.semilogy(H[m], se[m], '-o', color='C0', ms=4.5, lw=1.3, mfc='white',
                 label=r'model early ($N\!\in\![1,5]$)')
    axa.semilogy(H[m], sl[m], '-^', color='C0', ms=4.5, lw=1.3,
                 label=r'model late ($N\!\in\![5,9]$)')
    # y extended below the Drela range to show the favorable-regime shortfall
    style(axa, r'$dN/dRe_\theta$', (drela_g.min()/12.0, drela_g.max()*1.3))
    axa.legend(fontsize=8.0, loc='upper left')
    axa.text(0.96, 0.06, '(a)', transform=axa.transAxes, fontsize=12,
             fontweight='bold', ha='right')

    # ---- (b) onset Re_theta (N=1) vs H, single color ----
    N1_g = Rtc_g + 1.0/drela_g   # Drela-Giles N=1 station: critical + 1/(dN/dRe_theta)
    axb.semilogy(Hg, Rtc_g, '--', color='0.55', lw=1.4,
                 label=r'Drela critical $Re_{\theta 0}$')
    axb.semilogy(Hg, N1_g, 'k--', lw=1.8,
                 label=r'Drela--Giles $N\!=\!1$ station')
    m = np.isfinite(R)
    axb.semilogy(H[m], R[m], '-o', color='C0', ms=4.5, lw=1.3, mfc='white',
                 label=r'model $N\!=\!1$ crossing')
    # y extended past the Drela range to show the favorable-regime overshoot
    style(axb, r'onset $Re_\theta$', (Rtc_g.min(), N1_g.max()*4.0))
    axb.legend(fontsize=8.0, loc='upper right')
    axb.text(0.96, 0.06, '(b)', transform=axb.transAxes, fontsize=12,
             fontweight='bold', ha='right')

    plt.tight_layout()
    plt.savefig('figs/model_calibrate.pdf', facecolor='white')
    print('wrote figs/model_calibrate.pdf', flush=True)


if __name__ == '__main__':
    main()
