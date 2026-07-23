"""Variant of paper Fig. 3 (fs_nuHat_rows) WITHOUT the laminar-diffusion
reduction: c_nu,ai = 1 (full molecular viscosity in the disturbance
transport). Everything else identical to fig03_fs_transport_rows (sphere
kernel, same three wedges), except the domain is re-sized adaptively for ALL
rows (the paper's hardcoded Blasius Re_x = 4e6 only reaches N ~ 5.5 at
c_nu = 1). For the II.D decision whether dropping c_nu,ai altogether (one
less constant) is worth the envelope bend.

Out: repro/analytic/figs_explore/fs_nuHat_rows_cnu1.{pdf,png}
"""
import _saai  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
import fig03_fs_transport_rows as f3
from fig04_shapefactor import profile_ints, drela, Re_theta0
from lib.boundary_layer import FalknerSkanWedge

f3.C_NU_AI = 1.0   # march_field reads the module global at call time


def main():
    ROWS = [(-0.10, 1.2e6, 8000), (0.0, 4.0e6, 10000), (0.30, 3e5, 40000)]
    fig, axs = plt.subplots(3, 2, figsize=(11.2, 8.4), layout='constrained')
    for irow, (beta, x0, ylimL) in enumerate(ROWS):
        fs = FalknerSkanWedge(beta); I_th, H = profile_ints(fs)
        eta99 = np.interp(0.99, fs.u, fs.eta)
        x_max = f3.size_domain(fs, x0, beta)      # adaptive for EVERY row
        xs, yc, fld = f3.march_field(fs, x_max, beta=beta)
        Ue = fs.inviscid_at(np.maximum(xs, 1e-12)); Rt = I_th*np.sqrt(xs*Ue)
        N2d = np.log(np.maximum(fld, 1e-30))
        axL, axR = axs[irow]
        Rex = xs*Ue; ReyScale = Ue
        X = np.repeat(Rex[:, None], len(yc), 1); Y = yc[None, :]*ReyScale[:, None]
        lev = np.arange(1, 15, 1)
        cs = axL.contour(X, Y, N2d, levels=lev, colors='k', linewidths=0.7)
        axL.clabel(cs, levels=lev[::2], fmt='%d', fontsize=6.5, inline_spacing=2)
        th = I_th*np.sqrt(xs/np.maximum(Ue, 1e-30))
        d99 = eta99*np.sqrt(xs/np.maximum(Ue, 1e-30))
        axL.plot(Rex, th*ReyScale, '--', color='0.45', lw=1.1)
        axL.plot(Rex, d99*ReyScale, '-', color='0.45', lw=1.1)
        yl = ylimL*max(1.0, np.sqrt(x_max/x0)) if beta == 0.0 else ylimL
        axL.set_ylim(0, yl); axL.set_xlim(0, Rex.max())
        axL.set_ylabel(fr'$\beta={beta:+.2f}$ ($H={H:.2f}$)''\n'r'$Re_y$')
        if irow == 2: axL.set_xlabel(r'$Re_x$')
        env = fld.max(axis=1)
        axR.semilogy(Rt, env, 'k-', lw=1.8)
        Rtc = float(Re_theta0(H)); dr = float(drela(H))
        RtD = np.linspace(0, Rt.max(), 300)
        ND = np.where(RtD > Rtc, dr*(RtD - Rtc), 0.0)
        axR.semilogy(RtD, np.exp(ND), 'r--', lw=1.4)
        axR.set_xlim(0, Rt.max()); axR.set_ylim(0.5, 3e6)
        axR.set_ylabel(r'$\max_y \hat\nu$')
        ax2 = axR.twinx(); ax2.set_ylim(np.log(0.5), np.log(3e6)); ax2.set_ylabel(r'$N$')
        if irow == 2: axR.set_xlabel(r'$Re_\theta$')
        axR.grid(alpha=0.3, which='both')
        if irow == 0:
            axR.legend(['transport envelope ($c_{\\nu}=1$)', 'Drela--Giles envelope'],
                       fontsize=8, loc='lower right')
        print(f'beta={beta:+.2f} H={H:.2f}: x_max={x_max:.2e}, '
              f'N_end={np.log(env[-1]):.1f}, Rt_end={Rt[-1]:.0f}, Rtc={Rtc:.0f}',
              flush=True)
    out = 'repro/analytic/figs_explore/fs_nuHat_rows_cnu1'
    plt.savefig(out + '.pdf'); plt.savefig(out + '.png', dpi=140)
    print(f'wrote {out}.pdf/.png')


if __name__ == '__main__':
    main()
