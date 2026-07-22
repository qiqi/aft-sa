"""ABSOLUTE dN/dRe_theta(H) from the Eq.-11 disturbance transport solved on
each Falkner-Skan profile -- no normalization constant -- vs the Drela-Giles
envelope correlation (Sec. III.A, Fig. 3).

Replaces the single-point similarity proxy (regen_shapefactor.py), whose
absolute scale was off by a common factor C~40 (it convects at U_e, collapses
the band onto one Gamma, and omits cross-stream diffusion). Here the full
parabolic transport
    u dnu/dx + v dnu/dy = (c_nu_ai/sigma) d2nu/dy2 + a(Re_Omega, Gamma) w nu
is marched on the FS field for each beta, with the committed kernel
(Q4 gate cA=4, a_max=0.19, (s,g_c)=(8,1.055), floor=290, p=4; lambda_p=0 so the cliff
sits at the floor -- the slope is extracted far above it) and the reduced
diffusion c_nu_ai=1/12. The envelope slope is the least-squares slope of
N = ln(max_y nu / seed) vs Re_theta over the transition-relevant window
N in [3, 9]. Blasius (beta=0) reproduces the Fig.-6 solve (N ~ 14 at
Re_theta ~ 1330) as validation.

Numerics: implicit Euler in x, explicit source; sign-aware upwind for v
(favorable-gradient wedges have v < 0); uniform y grid sized to ~8 delta99 at
the outflow; domain length adapted per beta so the window is resolved.
-> figs/shapefactor_amplification.pdf (absolute comparison)
"""
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai')
from src.physics.boundary_layer import FalknerSkanWedge
from src.numerics.aft_sources import compute_aft_amplification_rate, compute_q4_gate

C_NU_AI, SIGMA_SA = 1.0/12.0, 2.0/3.0


def drela(H):
    """Drela-Giles (1987) envelope amplification rate dn/dRe_theta(H)."""
    return 0.01*np.sqrt((2.4*H - 3.7 + 2.5*np.tanh(1.5*H - 4.65))**2 + 0.25)


def march(fs, x_max, nx=800, ny=600, y_top=None, seed=1.0):
    """March the Eq.-11 transport on the FS field; return x, N(x)."""
    if y_top is None:
        eta99 = np.interp(0.99, fs.u, fs.eta)
        y_top = 8.0*eta99*np.sqrt(x_max/fs.inviscid_at(x_max))
    dy = y_top/ny
    yc = (np.arange(ny) + 0.5)*dy
    dx = x_max/nx
    nu = np.ones(ny)*seed
    N = [0.0]
    xs = [0.0]
    k = (C_NU_AI/SIGMA_SA)/dy**2
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, np.arange(ny + 1)*dy, cellCentered=True)
        u = np.maximum(u, 1e-12)
        vp = np.clip(v, 0, None)/dy
        vm = np.clip(-v, 0, None)/dy
        # advective form, sign-aware upwind; diffusion with wall Dirichlet
        # (ghost -nu[0]) and zero-gradient top; seed advected in where v<0 at top
        di = vp + vm + 2*k
        lo = -(vp[1:] + k)
        up = -(vm[:-1] + k)
        di[0] += k          # wall ghost nu_{-1} = -nu_0  ->  3k on the diagonal
        di[-1] -= k         # top ghost nu_{ny} = nu_{ny-1} (zero gradient)
        rate = np.asarray(compute_aft_amplification_rate(
            yc**2*np.abs(dudy),
            2*(dudy*yc)**2/(u**2 + (dudy*yc)**2)))
        q4 = compute_q4_gate(np.gradient(dudy, yc), np.abs(dudy), u, yc)
        b = rate*q4*np.abs(dudy)
        main = u/dx + di
        rhs = u/dx*nu + b*nu
        rhs[-1] += vm[-1]*seed          # freestream seed advected in from the top
        A = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        xs.append((i + 1)*dx)
        N.append(float(np.log(max(nu.max()/seed, 1e-300))))
    return np.array(xs), np.array(N)


def profile_ints(fs):
    I_th = np.trapezoid(fs.u*(1 - fs.u), fs.eta)
    I_ds = np.trapezoid(1 - fs.u, fs.eta)
    return I_th, I_ds/I_th


def measures_for_beta(beta, verbose=True):
    """Two envelope-rate measures from the transport solve:
    s_mean: the N=1 -> 9 mean -- onset (the observable departure) to
            transition, the window that sets the transition location;
    s_late: the [5,9] secant -- the late-envelope rate as transition nears."""
    fs = FalknerSkanWedge(beta)
    I_th, H = profile_ints(fs)
    ratio_pk = np.max(fs.eta**2*np.abs(fs.dudeta))/I_th   # Re_Omega_peak / Re_theta
    m = beta/(2.0 - beta)
    x_max = 4e6 if beta == 0.0 else (3e5 if beta > 0 else 1.2e6)
    for it in range(12):
        xs, N = march(fs, x_max)
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.15; continue
        if N[-1] > 14.0:
            x_max = 1.1*float(np.interp(14.0, N, xs))
            xs, N = march(fs, x_max)
            break
        x_max *= 3.0
    Ue = fs.inviscid_at(np.maximum(xs, 1e-12))
    Rt = I_th*np.sqrt(xs*Ue)
    Rt1 = float(np.interp(1.0, N, Rt))
    Rt5 = float(np.interp(5.0, N, Rt))
    Rt9 = float(np.interp(9.0, N, Rt))
    s_mean = 8.0/(Rt9 - Rt1)
    s_late = 4.0/(Rt9 - Rt5)
    if verbose:
        d = drela(H)
        print(f"beta={beta:+.3f}  H={H:.3f}  Rt1={Rt1:4.0f}  Rt9={Rt9:6.0f}  "
              f"s_mean={s_mean:.4e} ({s_mean/d:4.2f}x)  "
              f"s_late={s_late:.4e} ({s_late/d:4.2f}x)  Drela={d:.4e}", flush=True)
    return H, s_mean, s_late


if __name__ == '__main__':
    adverse = [-0.195, -0.18, -0.15, -0.12, -0.09, -0.06, -0.03, 0.0]
    favorable = [0.05, 0.1, 0.2, 0.35, 0.55, 1.0]
    oa = [measures_for_beta(b) for b in adverse]
    of = [measures_for_beta(b) for b in favorable]
    Ha = np.array([o[0] for o in oa])
    ma = np.array([o[1] for o in oa]); la = np.array([o[2] for o in oa])
    Hf = np.array([o[0] for o in of])
    mf = np.array([o[1] for o in of]); lf = np.array([o[2] for o in of])
    Hg = np.linspace(min(Hf), max(Ha), 300)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.semilogy(Hg, drela(Hg), 'k--', lw=1.8,
                label=r'Drela--Giles envelope $dn/dRe_\theta(H)$')
    o = np.argsort(Ha)
    ax.fill_between(Ha[o], ma[o], la[o], color='C0', alpha=0.15, lw=0)
    ax.semilogy(Ha[o], la[o], '-o', color='C0', lw=1.9, ms=4.5,
                label=r'late-envelope rate ($N\in[5,9]$ secant)')
    ax.semilogy(Ha[o], ma[o], '-s', color='C2', lw=1.9, ms=4.5,
                label=r'onset-to-transition mean ($N\in[1,9]$ secant)')
    o = np.argsort(Hf)
    ax.semilogy(Hf[o], lf[o], ':o', color='C0', lw=1.3, ms=4.5, mfc='white')
    ax.semilogy(Hf[o], mf[o], ':s', color='C2', lw=1.3, ms=4.5, mfc='white',
                label='favorable (open): rate by design unreduced\n(onset-delay cliff instead, Sec. III.B)')
    iB = int(np.argmin(np.abs(Ha - 2.591)))
    for arr in (ma, la):
        ax.plot(Ha[iB], arr[iB], 'o', color='C3', ms=9, mfc='none', mew=1.6, zorder=6)
    ax.annotate('Blasius', (Ha[iB], la[iB]), (Ha[iB] + 0.03, la[iB]*1.5),
                fontsize=8, color='C3')
    ax.axvspan(3.5, max(Ha), color='0.88', zorder=0)
    ax.text(3.52, 0.14, 'separation\n(free-shear\nceiling)', fontsize=7.5, va='top')
    ax.set_xlabel(r'shape factor $H=\delta^*/\theta$')
    ax.set_ylabel(r'$dN/dRe_\theta$ (absolute, no normalization)')
    ax.set_xlim(min(Hf) - 0.02, max(Ha) + 0.02); ax.set_ylim(3e-3, 2e-1)
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=7.6, loc='upper left')
    plt.tight_layout()
    plt.savefig('figs/shapefactor_amplification.pdf')
    plt.savefig('/tmp/shapefactor_transport.png', dpi=140)
    print('wrote figs/shapefactor_amplification.pdf (two-measure absolute version)')
