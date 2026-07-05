"""Convergence plots: residual, CL, CD vs pseudo-step for cav and str."""
import csv, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

def load_residual(case):
    rows = list(csv.reader(open(f'{case}/nonlinear_residual_v2.csv')))
    h = [x.strip() for x in rows[0]]
    ip = h.index('pseudo_step'); ic = h.index('0_cont'); inu = h.index('5_nuHat')
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
    rows = list(csv.reader(open(f'{case}/total_forces_v2.csv')))
    s, cl, cd = [], [], []
    for r in rows[1:]:
        if r and len(r) > 4:
            try:
                s.append(int(float(r[1])))
                cl.append(float(r[2]))
                cd.append(float(r[3]))
            except: pass
    return np.array(s), np.array(cl), np.array(cd)

cases = [('cavprop_nlf0416_Re4M_a2p5', 'cav (unstructured)', 'C2'),
         ('strprop_nlf0416_Re4M_a2p5', 'str (O-grid)',         'C0')]

matplotlib.rcParams.update({'font.size': 11})
fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
ax_res, ax_cl, ax_resnu, ax_cd = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

SKIP_BEFORE = 1000  # skip the early CFL ramp-up transient
for case, label, color in cases:
    s_r, c_r, n_r = load_residual(case)
    s_f, cl, cd = load_forces(case)
    # Mask out early transient
    mr = s_r >= SKIP_BEFORE
    mf = s_f >= SKIP_BEFORE
    ax_res.semilogy(s_r[mr], c_r[mr], '-', color=color, lw=1.4, label=label)
    ax_resnu.semilogy(s_r[mr], n_r[mr], '-', color=color, lw=1.4, label=label)
    ax_cl.plot(s_f[mf], cl[mf], '-', color=color, lw=1.4, label=label)
    ax_cd.plot(s_f[mf], cd[mf], '-', color=color, lw=1.4, label=label)
    # Annotate final values
    print(f'{case}: final step={s_r[-1]}, cont={c_r[-1]:.2e}, nuHat={n_r[-1]:.2e}, CL={cl[-1]:+.5f}, CD={cd[-1]:.6f}')

ax_res.set_ylabel('continuity residual')
ax_res.set_title('Continuity residual'); ax_res.grid(alpha=0.3, which='both')
ax_res.legend(fontsize=10, loc='upper right')
ax_res.axhline(1e-8, color='gray', ls=':', lw=0.7)
ax_res.set_ylim(1e-10, 1e-4)

ax_resnu.set_ylabel(r'$\tilde\nu$ residual'); ax_resnu.set_xlabel('pseudo-step')
ax_resnu.set_title(r'$\tilde\nu$ residual (NOT yet converged)'); ax_resnu.grid(alpha=0.3, which='both')
ax_resnu.axhline(1e-7, color='gray', ls=':', lw=0.7)
ax_resnu.set_ylim(1e-8, 1e-3)

ax_cl.set_ylabel('$C_L$'); ax_cl.set_title('$C_L$ history')
ax_cl.grid(alpha=0.3); ax_cl.set_ylim(0.7, 0.9)

ax_cd.set_ylabel('$C_D$'); ax_cd.set_xlabel('pseudo-step')
ax_cd.set_title('$C_D$ history'); ax_cd.grid(alpha=0.3)
ax_cd.set_ylim(0.004, 0.010)

plt.suptitle('NLF(1)-0416 Re=4×10⁶ M=0.1 α=2.5° — convergence histories', fontsize=12)
plt.tight_layout(rect=(0,0,1,0.97))
plt.savefig('/tmp/convergence_nlf.png', dpi=130)
plt.close()
print('wrote /tmp/convergence_nlf.png')
