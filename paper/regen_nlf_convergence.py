"""Convergence figure for the 4 NLF(1)-0416 Re=4M cases (cav,str)×(α=0,4).

Plots residual + CL/CD history for each case in a 2x2 grid, demonstrating that
all 4 runs converge cleanly with AI_LAMINAR_SLOWDOWN=0.01 and the chi_inf=8.76e-4
freestream BC (compensated via input ratio=8.76e-6 to account for the slowdown's
BC-enforcement weakening).
"""
import csv, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 10.5, 'axes.titlesize': 10.5,
                            'axes.labelsize': 10.5, 'xtick.labelsize': 9.5,
                            'ytick.labelsize': 9.5, 'legend.fontsize': 9})

B = '/home/qiqi/flexcompute/aft-sa/flow360'
PD = '/home/qiqi/flexcompute/aft-sa/paper'

CASES = [
    ('cavprop_nlf0416_Re4M_a0p0', 'unstructured (cav), $\\alpha=0^\\circ$', 'C0'),
    ('cavprop_nlf0416_Re4M_a4p0', 'unstructured (cav), $\\alpha=4^\\circ$', 'C1'),
    ('strprop_nlf0416_Re4M_a0p0', 'O-grid (str), $\\alpha=0^\\circ$',       'C2'),
    ('strprop_nlf0416_Re4M_a4p0', 'O-grid (str), $\\alpha=4^\\circ$',       'C3'),
]

def load_residual(case):
    rows = list(csv.reader(open(f'{B}/{case}/nonlinear_residual_v2.csv')))
    h = [x.strip() for x in rows[0]]
    ip = h.index('pseudo_step')
    ic = h.index('0_cont'); inu = h.index('5_nuHat')
    s, c, n = [], [], []
    for r in rows[1:]:
        if r and len(r) > inu:
            try:
                s.append(int(float(r[ip])))
                c.append(float(r[ic]))
                n.append(float(r[inu]))
            except: pass
    return np.array(s), np.array(c), np.array(n)

def load_forces(case):
    rows = list(csv.reader(open(f'{B}/{case}/total_forces_v2.csv')))[1:]
    s, cl, cd = [], [], []
    for r in rows:
        if r and len(r) > 3:
            try:
                s.append(int(float(r[1])))
                cl.append(float(r[2]))
                cd.append(float(r[3]))
            except: pass
    return np.array(s), np.array(cl), np.array(cd)

SKIP = 50
fig, ((ax_res, ax_cl), (ax_nu, ax_cd)) = plt.subplots(2, 2, figsize=(10.5, 6.5), sharex=True)

for case, label, color in CASES:
    sr, cr, nr = load_residual(case)
    sf, cl, cd = load_forces(case)
    mr = sr >= SKIP; mf = sf >= SKIP
    ax_res.semilogy(sr[mr], cr[mr], '-', color=color, lw=1.3, label=label)
    ax_nu.semilogy (sr[mr], nr[mr], '-', color=color, lw=1.3, label=label)
    ax_cl.plot     (sf[mf], cl[mf], '-', color=color, lw=1.3, label=label)
    ax_cd.plot     (sf[mf], cd[mf], '-', color=color, lw=1.3, label=label)
    print(f'{case}: final step={sr[-1]}, cont={cr[-1]:.2e}, nuHat={nr[-1]:.2e}, CL={cl[-1]:+.4f}, CD={cd[-1]:.5f}')

ax_res.set_ylabel('$\\rho$ continuity residual')
ax_res.set_title('Continuity residual')
ax_res.grid(alpha=0.3, which='both'); ax_res.set_ylim(1e-11, 1e-4)
ax_res.legend(fontsize=8, loc='upper right', frameon=False, ncol=2)

ax_nu.set_xlabel('pseudo-step')
ax_nu.set_ylabel('$\\tilde\\nu$ residual'); ax_nu.set_title('SA $\\tilde\\nu$ residual')
ax_nu.grid(alpha=0.3, which='both'); ax_nu.set_ylim(1e-11, 1e-3)

ax_cl.set_ylabel('$C_L$'); ax_cl.set_title('Lift coefficient')
ax_cl.grid(alpha=0.3); ax_cl.set_ylim(0.3, 1.05)

ax_cd.set_xlabel('pseudo-step')
ax_cd.set_ylabel('$C_D$'); ax_cd.set_title('Drag coefficient')
ax_cd.grid(alpha=0.3); ax_cd.set_ylim(0.004, 0.012)

ax_res.set_xlim(0, 80000)
plt.suptitle('NLF(1)-0416 convergence — $Re=4\\times 10^6$, $M=0.1$, $\\chi_\\infty=8.76\\times 10^{-4}$ ($N_{crit}=9$), SA-AI with $f_{slow}=0.01$',
             fontsize=10.5, y=0.995)
plt.tight_layout(rect=(0,0,1,0.96))
plt.savefig(f'{PD}/figs/nlf_convergence.pdf')
plt.savefig('/tmp/nlf_convergence.png', dpi=140)
plt.close()
print('wrote nlf_convergence.pdf')
