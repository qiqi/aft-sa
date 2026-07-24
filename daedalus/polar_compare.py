"""Postprocess the L0 SA-AI alpha sweep (4, 5, 6 deg) against AVL + XFOIL:
polar (CL vs CD and CL vs alpha), sectional cl(eta) per alpha, and upper-
surface transition location x_tr(eta) per alpha with bubble extents.

Usage: python3 polar_compare.py
Outputs: /tmp/daedalus_mesh_views/sectional/{polar,sectional_cl_sweep,
transition_eta_sweep}.png + printed table.
"""
import os
import re
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wing_geometry import chord, HALF_SPAN, C_ROOT
import sectional_compare as SC
from avl_compare import xfoil_cd, AVL_BIN, WORK

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = '/tmp/daedalus_mesh_views/sectional'
ALPHAS = [4.0, 5.0, 6.0]
NCRIT = 13.6
RE_ROOT, S_REF = 5.0e5, 30.84

RANS = {4.0: ('case_ogrid_saai', 'case_cavity_saai'),
        5.0: ('case_ogrid_saai_a5', 'case_cavity_saai_a5'),
        6.0: ('case_ogrid_saai_a6', 'case_cavity_saai_a6')}
SURF = {'ogrid': 'surface_fluid_wing.pvtu', 'cavity': 'surface_farfield_body.pvtu'}


def rans_totals(case):
    hdr = open(f'{HERE}/{case}/total_forces_v2.csv').readline().split(',')
    iCL = [i for i, h in enumerate(hdr) if h.strip() == 'CL'][0]
    iCD = [i for i, h in enumerate(hdr) if h.strip() == 'CD'][0]
    f = np.genfromtxt(f'{HERE}/{case}/total_forces_v2.csv', delimiter=',',
                      skip_header=1)
    return f[-1, iCL], f[-1, iCD]


def run_avl(alpha):
    cmds = (f'LOAD daedalus.avl\nOPER\nA A {alpha}\nX\nFT\nft_a{int(alpha)}.txt\n'
            f'FS\nfs_a{int(alpha)}.txt\n\nQUIT\n')
    # reuse cached AVL output when present (deterministic given the .avl file)
    if not (os.path.exists(f'{WORK}/ft_a{int(alpha)}.txt')
            and os.path.exists(f'{WORK}/fs_a{int(alpha)}.txt')):
        subprocess.run([AVL_BIN], input=cmds, capture_output=True, text=True,
                       cwd=WORK, timeout=300)
    ft = open(f'{WORK}/ft_a{int(alpha)}.txt').read()
    cl = float(re.search(r'CLtot\s*=\s*([\d.eE+-]+)', ft).group(1))
    cdi = float(re.search(r'CDind\s*=\s*([\d.eE+-]+)', ft).group(1))
    rows = []
    for ln in open(f'{WORK}/fs_a{int(alpha)}.txt'):
        t = ln.split()
        if len(t) >= 10:
            try:
                int(t[0])
                rows.append((float(t[1]), float(t[2]), float(t[7]), float(t[5])))
            except ValueError:
                continue
    r = np.array([x for x in rows if x[0] > 0])
    return cl, cdi, r[np.argsort(r[:, 0])]     # y, chord, cl, ai


ETAS_X = [0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.85, 0.92, 0.97]


def xfoil_sweep(strips):
    """cd, xtr_u, bubble per station at the AVL local cl."""
    out = []
    for e in ETAS_X:
        cl_loc = np.interp(e * HALF_SPAN, strips[:, 0], strips[:, 2])
        cd = xfoil_cd(f'sec_st_{int(e*100):03d}.dat',
                      RE_ROOT * chord(e) / C_ROOT, cl_loc, NCRIT, False)
        xu, xl, bu, bl = SC.xfoil_station(e, cl_loc, NCRIT)
        out.append((cd if cd else np.nan, xu, bu[0], bu[1]))
    return np.array(out)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    table = []
    avl_strips_by_a, xf_by_a, rans_x_by_a = {}, {}, {}
    for a in ALPHAS:
        cl_avl, cdi, strips = run_avl(a)
        xf = xfoil_sweep(strips)
        cd_e = xf[:, 0]
        good = np.isfinite(cd_e)
        ee = np.linspace(0.005, 0.995, 200)
        cdp = 2.0 * np.trapezoid(np.interp(ee, np.array(ETAS_X)[good], cd_e[good])
                                 * chord(ee), ee * HALF_SPAN) / S_REF
        og, cav = RANS[a]
        og_cl, og_cd = rans_totals(og)
        cav_cl, cav_cd = rans_totals(cav)
        table.append((a, og_cl, og_cd, cav_cl, cav_cd, cl_avl, cdi + cdp))
        avl_strips_by_a[a] = strips
        xf_by_a[a] = xf
        # RANS transition per mesh
        SC.ALPHA = np.deg2rad(a)
        rx = {}
        for mesh, case in (('ogrid', og), ('cavity', cav)):
            _, _, xtr, bub = SC.strip_data(case, SURF[mesh])
            rx[mesh] = (xtr['upper'], bub['upper'])
        rans_x_by_a[a] = rx
        print(f'alpha {a}: AVL CL {cl_avl:.4f} CDi {cdi:.5f} CDp {cdp:.5f}',
              flush=True)

    tab = np.array(table)
    print(f'\n{"a":>4} {"RANS ogrid CL/CD":>20} {"RANS cavity CL/CD":>20} '
          f'{"AVL+XFOIL CL/CD":>20}')
    for r in tab:
        print(f'{r[0]:4.0f} {r[1]:10.4f}/{r[2]:.5f} {r[3]:10.4f}/{r[4]:.5f} '
              f'{r[5]:10.4f}/{r[6]:.5f}')

    # ---- polar ----
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))
    axs[0].plot(tab[:, 2], tab[:, 1], 'C0-o', label='RANS O-grid SA-AI')
    axs[0].plot(tab[:, 4], tab[:, 3], 'C0--s', label='RANS cavity SA-AI')
    axs[0].plot(tab[:, 6], tab[:, 5], 'k-^', label='AVL + XFOIL (free, N=13.6)')
    axs[0].set_xlabel('$C_D$'); axs[0].set_ylabel('$C_L$')
    axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8)
    axs[1].plot(tab[:, 0], tab[:, 1], 'C0-o', label='RANS O-grid')
    axs[1].plot(tab[:, 0], tab[:, 3], 'C0--s', label='RANS cavity')
    axs[1].plot(tab[:, 0], tab[:, 5], 'k-^', label='AVL')
    axs[1].set_xlabel(r'$\alpha$ [deg]'); axs[1].set_ylabel('$C_L$')
    axs[1].grid(alpha=0.3); axs[1].legend(fontsize=8)
    fig.suptitle('Daedalus wing L0 polar: SA-AI RANS vs AVL + XFOIL strips',
                 fontsize=11)
    fig.savefig(f'{OUT}/polar.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ---- sectional cl per alpha ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cols = {4.0: 'C0', 5.0: 'C1', 6.0: 'C2'}
    for a in ALPHAS:
        SC.ALPHA = np.deg2rad(a)
        og, cav = RANS[a]
        e_n, cl_n, _ = SC.native_strips(og)
        ax.plot(e_n, cl_n, color=cols[a], ls='-', lw=1.6,
                label=rf'RANS O-grid $\alpha={a:.0f}^\circ$')
        e_c, cl_c, _ = SC.native_strips(cav)
        ax.plot(e_c, cl_c, color=cols[a], ls='--', lw=1.2)
        s = avl_strips_by_a[a]
        ax.plot(s[:, 0] / HALF_SPAN, s[:, 2], color=cols[a], ls=':', lw=1.8)
    ax.set_xlabel(r'$\eta$'); ax.set_ylabel('sectional $c_l$')
    ax.grid(alpha=0.3); ax.set_xlim(0, 1)
    ax.legend(fontsize=8, title='solid O-grid / dashed cavity / dotted AVL')
    fig.savefig(f'{OUT}/sectional_cl_sweep.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ---- transition sweep ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    eta_c = SC.ETA_C
    for k, a in enumerate(ALPHAS):
        ax = axs[k]
        rx = rans_x_by_a[a]
        for mesh, ls in (('ogrid', '-'), ('cavity', '--')):
            ax.plot(eta_c, rx[mesh][0], 'C0', ls=ls,
                    label=f'RANS {mesh}' if k == 0 else None)
        xf = xf_by_a[a]
        ax.plot(ETAS_X, xf[:, 1], 'k-o', ms=4,
                label='XFOIL $x_{tr}$' if k == 0 else None)
        ax.fill_between(ETAS_X, xf[:, 2], xf[:, 3], color='k', alpha=0.15,
                        label='XFOIL bubble' if k == 0 else None)
        ax.set_title(rf'$\alpha = {a:.0f}^\circ$')
        ax.set_xlabel(r'$\eta$'); ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axs[0].set_ylabel('$x_{tr}/c$ (upper)')
    axs[0].legend(fontsize=8, loc='lower left')
    fig.suptitle(f'Upper-surface transition vs alpha: SA-AI (Cf-rise) vs '
                 f'XFOIL (N={NCRIT})', fontsize=11)
    fig.savefig(f'{OUT}/transition_eta_sweep.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT)
