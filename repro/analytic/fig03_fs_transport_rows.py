"""fig:nuhat -> paper/figs/fs_nuHat_rows.pdf.

Disturbance transport on three Falkner-Skan layers (adverse -0.10, Blasius 0,
favorable +0.30): N=ln(nuHat) contours + envelope vs Drela-Giles. Full kernel with
each wedge's own lambda_p in the cliff. Kernel + c_nu,ai imported (via fig04)."""
import _saai
from _saai import C_NU_AI, SIGMA_SA
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from src.physics.boundary_layer import FalknerSkanWedge
from src.numerics.aft_sources import compute_aft_amplification_rate, compute_q4_gate
from fig04_shapefactor import profile_ints, drela, Re_theta0

def march_field(fs, x_max, nx=800, ny=600, beta=0.0):
    """As fig04.march but stores the full nuHat field; wedge lambda_p(x,y) in the cliff."""
    m = beta/(2.0 - beta)
    eta99 = np.interp(0.99, fs.u, fs.eta)
    y_top = 8.0*eta99*np.sqrt(x_max/fs.inviscid_at(x_max))
    dy = y_top/ny; yc = (np.arange(ny) + 0.5)*dy; dx = x_max/nx
    nu = np.ones(ny); field = [nu.copy()]; xs = [0.0]
    k = (C_NU_AI/SIGMA_SA)/dy**2
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, np.arange(ny + 1)*dy, cellCentered=True)
        u = np.maximum(u, 1e-12)
        vp = np.clip(v, 0, None)/dy; vm = np.clip(-v, 0, None)/dy
        di = vp + vm + 2*k; lo = -(vp[1:] + k); up = -(vm[:-1] + k)
        di[0] += k; di[-1] -= k
        Ue = fs.inviscid_at(x); lam = m*yc**2*Ue*Ue/(x*u)
        rate = np.asarray(compute_aft_amplification_rate(
            yc**2*np.abs(dudy), 2*(dudy*yc)**2/(u**2 + (dudy*yc)**2), lambda_p=lam))
        q4 = compute_q4_gate(np.gradient(dudy, yc), np.abs(dudy), u, yc)
        b = rate*q4*np.abs(dudy)
        main = u/dx + di; rhs = u/dx*nu + b*nu; rhs[-1] += vm[-1]
        A = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs); xs.append((i + 1)*dx); field.append(nu.copy())
    return np.array(xs), yc, np.array(field)

def size_domain(fs, x0, beta):
    x_max = x0
    for _ in range(12):
        xs, yc, fld = march_field(fs, x_max, nx=200, ny=200, beta=beta)
        N = np.log(np.maximum(fld.max(axis=1), 1e-30))
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.15; continue
        if N[-1] > 14.0:
            return 1.05*float(np.interp(14.0, N, xs))
        x_max *= 3.0
    return x_max


def main():
    ROWS = [(-0.10, 1.2e6), (0.0, 4.0e6), (0.30, 3e5)]
    fig, axs = plt.subplots(3, 2, figsize=(11.2, 12.0), layout='constrained')
    for irow, (beta, x0) in enumerate(ROWS):
        fs = FalknerSkanWedge(beta); I_th, H = profile_ints(fs)
        eta99 = np.interp(0.99, fs.u, fs.eta)
        x_max = 4.0e6 if beta == 0.0 else size_domain(fs, x0, beta)
        xs, yc, fld = march_field(fs, x_max, beta=beta)
        Ue = fs.inviscid_at(np.maximum(xs, 1e-12)); Rt = I_th*np.sqrt(xs*Ue)
        N2d = np.log(np.maximum(fld, 1e-30))
        axL, axR = axs[irow]
        Rex = xs*Ue; ReyScale = Ue
        X = np.repeat(Rex[:, None], len(yc), 1); Y = yc[None, :]*ReyScale[:, None]
        lev = np.arange(1, 15, 1)
        cs = axL.contourf(X, Y, N2d, levels=lev, cmap='viridis', extend='both')
        th = I_th*np.sqrt(xs/np.maximum(Ue, 1e-30)); d99 = eta99*np.sqrt(xs/np.maximum(Ue, 1e-30))
        axL.plot(Rex, th*ReyScale, 'w--', lw=1.1); axL.plot(Rex, d99*ReyScale, 'w-', lw=1.1)
        ymax = 1.6*float((d99*ReyScale).max()); axL.set_ylim(0, ymax); axL.set_xlim(0, Rex.max())
        axL.annotate(r'$\delta_{99}$', (0.86*Rex.max(), 1.12*float(np.interp(0.86*Rex.max(), Rex, d99*ReyScale))), color='w', fontsize=9)
        axL.annotate(r'$\theta$', (0.9*Rex.max(), 0.55*float(np.interp(0.9*Rex.max(), Rex, th*ReyScale))), color='w', fontsize=9)
        axL.set_ylabel(fr'$\beta={beta:+.2f}$ ($H={H:.2f}$)''\n'r'$Re_y$')
        if irow == 2: axL.set_xlabel(r'$Re_x$')
        axL.text(0.02, 0.95, f'({chr(97+2*irow)})', transform=axL.transAxes, fontsize=11, va='top', fontweight='bold', color='w')
        if irow == 2:
            cb = fig.colorbar(cs, ax=axs[:, 0].tolist(), fraction=0.04, pad=0.02, location='bottom')
            cb.set_label(r'$N=\ln\hat\nu$')
        env = fld.max(axis=1)
        axR.semilogy(Rt, env, 'k-', lw=1.8)
        Rtc = float(Re_theta0(H)); dr = float(drela(H))
        RtD = np.linspace(0, Rt.max(), 300); ND = np.where(RtD > Rtc, dr*(RtD - Rtc), 0.0)
        axR.semilogy(RtD, np.exp(ND), 'r--', lw=1.4)
        axR.set_xlim(0, Rt.max()); axR.set_ylim(0.5, 3e6); axR.set_ylabel(r'$\max_y \hat\nu$')
        ax2 = axR.twinx(); ax2.set_ylim(np.log(0.5), np.log(3e6)); ax2.set_ylabel(r'$N$')
        if irow == 2: axR.set_xlabel(r'$Re_\theta$')
        axR.grid(alpha=0.3, which='both')
        axR.text(0.02, 0.95, f'({chr(98+2*irow)})', transform=axR.transAxes, fontsize=11, va='top', fontweight='bold')
        if irow == 0:
            axR.legend(['transport envelope', 'Drela--Giles envelope'], fontsize=8, loc='lower right')
        print(f'beta={beta:+.2f} H={H:.2f}: x_max={x_max:.2e}, N_end={np.log(env[-1]):.1f}, '
              f'Rt_end={Rt[-1]:.0f}, Rtc={Rtc:.0f}', flush=True)
    plt.savefig('figs/fs_nuHat_rows.pdf')
    print('wrote figs/fs_nuHat_rows.pdf')


if __name__ == '__main__':
    main()
