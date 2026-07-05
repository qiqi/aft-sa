"""Regenerate paper figures from the laminar-init full sweep (flow360/full_*).
SA-AI headline = m=2 log-barrier; 'orig' (original activation) shown for the
grid-convergence comparison. Adds a grid-convergence figure (cavity vs O-grid)."""
import csv, os, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 11, 'font.family': 'serif', 'axes.linewidth': 0.8})
B = "/home/qiqi/flexcompute/aft-sa/flow360"; PD = "/home/qiqi/flexcompute/aft-sa/paper"
WALL = {'naca0012': 'naca0012', 'nlf0416': 'nlf0416'}
ALPHAS = [-2, 0, 2, 4, 6, 8]

def wd(af, mesh, model, a): return f"{B}/full_{af}_{mesh}_{model}_a{a}"

def forces(d):
    rows = list(csv.reader(open(os.path.join(d, 'total_forces_v2.csv'))))
    h = [x.strip() for x in rows[0]]; last = [c.strip() for c in rows[-1] if c.strip() != '']
    g = lambda k: float(last[h.index(k)])
    return g('CL'), g('CD'), g('CDPressure'), g('CDSkinFriction')

def surf(d, af, field='cf'):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/surface_fluid_{WALL[af]}.pvtu"); r.Update()
    g = r.GetOutput(); pts = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    nm = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    arr = vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith(field))))
    v = np.linalg.norm(arr, axis=1) if (field == 'cf' and arr.ndim > 1) else arr
    x, z = pts[:, 0], pts[:, 2]; up = z > 1e-6; xs, vs = x[up], v[up]; o = np.argsort(xs); return xs[o], vs[o]

def binned(xs, vs, nb=40):
    bins = np.linspace(0, 1, nb + 1); xc = .5 * (bins[1:] + bins[:-1]); cb = np.full(nb, np.nan)
    for i in range(nb):
        s = (xs >= bins[i]) & (xs < bins[i + 1])
        if s.any(): cb[i] = np.median(vs[s])
    ok = np.isfinite(cb); return xc[ok], cb[ok]

def xtr_jump(d, af):
    xc, cb = binned(*surf(d, af, 'cf'), 50); dd = np.diff(cb); dm = .5 * (xc[1:] + xc[:-1]); w = (dm > 0.02) & (dm < 0.95)
    if not w.any(): return np.nan
    j = np.argmax(dd[w])
    return 0.02 if dd[w][j] < 0.1 * np.nanmean(cb[(xc > 0.02) & (xc < 0.95)]) else float(dm[w][j])

def resid(d):
    rows = list(csv.reader(open(f"{d}/nonlinear_residual_v2.csv"))); h = [x.strip() for x in rows[0]]
    ip, ic, inu = h.index('pseudo_step'), h.index('0_cont'), h.index('5_nuHat'); s, c, n = [], [], []
    for row in rows[1:]:
        row = [x for x in row if x.strip() != '']
        if len(row) <= inu: continue
        s.append(float(row[ip])); c.append(float(row[ic])); n.append(float(row[inu]))
    return map(np.array, (s, c, n))

mf = {af: {float(r['alpha']): r for r in csv.DictReader(open(f"{PD}/data/mfoil_{af}.csv"))}
      for af in ('naca0012', 'nlf0416')}

# ---- Table 3 (NACA0012 a=0 cavity) ----
clA, cdA, cdpA, cdfA = forces(wd('naca0012', 'cavity', 'aftsa_m2', 0))
clT, cdT, cdpT, cdfT = forces(wd('naca0012', 'cavity', 'turb', 0))
print("TABLE3 NACA0012 a=0 cavity (laminar init):")
print("  turbulent SA chi=3 : CD=%.5f CDp=%.5f CDf=%.5f" % (cdT, cdpT, cdfT))
print("  SA-AI (m=2)        : CD=%.5f CDp=%.5f CDf=%.5f  -> %.0f%% lower CD" % (cdA, cdpA, cdfA, 100 * (1 - cdA / cdT)))

# ---- Fig: NACA0012 Cf a=0 (SA-AI vs turbulent) ----
plt.figure(figsize=(3.4, 2.7))
xa, ca = binned(*surf(wd('naca0012', 'cavity', 'aftsa_m2', 0), 'naca0012', 'cf'))
xt, ct = binned(*surf(wd('naca0012', 'cavity', 'turb', 0), 'naca0012', 'cf'))
plt.plot(xt, ct, '--', color='0.5', lw=1.3, marker='s', ms=2.5, markevery=2, label='SA (turbulent)')
plt.plot(xa, ca, 'k-', lw=1.3, marker='o', ms=2.5, markevery=2, label='SA-AI')
plt.xlabel('$x/c$'); plt.ylabel('$C_f$'); plt.xlim(0, 1); plt.ylim(0, None)
plt.legend(fontsize=7.5, frameon=False); plt.tight_layout(pad=0.3); plt.savefig(f"{PD}/figs/naca0012_cf.pdf")

# ---- Fig: convergence a=0 ----
plt.figure(figsize=(3.4, 2.7))
sA, cA, nA = resid(wd('naca0012', 'cavity', 'aftsa_m2', 0)); sT, cT, nT = resid(wd('naca0012', 'cavity', 'turb', 0))
plt.semilogy(sT, cT, '-', color='0.6', lw=1.3, label='SA: continuity'); plt.semilogy(sA, cA, 'k-', lw=1.3, label='SA-AI: continuity')
plt.semilogy(sT, nT, '--', color='0.6', lw=1.1, label=r'SA: $\tilde\nu$'); plt.semilogy(sA, nA, 'k--', lw=1.1, label=r'SA-AI: $\tilde\nu$')
plt.xlabel('pseudo-time step'); plt.ylabel('RMS residual'); plt.ylim(1e-11, None)
plt.legend(fontsize=6.3, frameon=False); plt.tight_layout(pad=0.3); plt.savefig(f"{PD}/figs/convergence.pdf")

# ---- Fig: polar (NACA0012 + NLF0416) ----
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, af, ttl in [(axs[0], 'naca0012', 'NACA 0012'), (axs[1], 'nlf0416', 'NLF(1)-0416')]:
    cdT_ = [forces(wd(af, 'cavity', 'turb', a))[1] for a in ALPHAS]; clT_ = [forces(wd(af, 'cavity', 'turb', a))[0] for a in ALPHAS]
    cdA_ = [forces(wd(af, 'cavity', 'aftsa_m2', a))[1] for a in ALPHAS]; clA_ = [forces(wd(af, 'cavity', 'aftsa_m2', a))[0] for a in ALPHAS]
    cdM = [float(mf[af][a]['cd']) for a in ALPHAS]; clM = [float(mf[af][a]['cl']) for a in ALPHAS]
    ax.plot(cdT_, clT_, '^-.', color='0.6', ms=4, lw=1.0, label='SA (turbulent)')
    ax.plot(cdM, clM, 's--', color='0.4', ms=4, lw=1.1, label='mfoil ($e^N$, $N_{cr}{=}4$)')
    ax.plot(cdA_, clA_, 'ko-', ms=3.5, lw=1.3, label='SA-AI')
    ax.set_xlabel('$C_D$'); ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3)
axs[0].set_ylabel('$C_L$'); axs[0].legend(fontsize=6.6, frameon=False, loc='lower right')
plt.tight_layout(pad=0.3); plt.savefig(f"{PD}/figs/polar.pdf")

# ---- Fig: xtr vs alpha ----
alt = [0, 2, 4, 6, 8]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
xt_all = {}
for ax, af, ttl in [(axs[0], 'naca0012', 'NACA 0012'), (axs[1], 'nlf0416', 'NLF(1)-0416')]:
    xtA = [xtr_jump(wd(af, 'cavity', 'aftsa_m2', a), af) for a in alt]; xt_all[af] = xtA
    xtM = [float(mf[af][a]['xtr_up']) for a in alt]
    ax.plot(alt, [0] * len(alt), '^-.', color='0.6', ms=4, lw=1.0, label='SA (turbulent)')
    ax.plot(alt, xtM, 's--', color='0.4', ms=5, lw=1.1, label='mfoil ($e^N$)')
    ax.plot(alt, xtA, 'ko-', ms=4, lw=1.3, label='SA-AI')
    ax.set_xlabel(r'$\alpha$ (deg)'); ax.set_title(ttl, fontsize=10); ax.set_ylim(0, 0.8); ax.grid(alpha=0.3)
axs[0].set_ylabel('suction-side $x_{tr}/c$'); axs[0].legend(fontsize=6.6, frameon=False)
plt.tight_layout(pad=0.3); plt.savefig(f"{PD}/figs/xtr_alpha.pdf")

# ---- NEW Fig: grid convergence (NACA0012 cavity vs O-grid) ----
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
xc1, cc1 = binned(*surf(wd('naca0012', 'cavity', 'aftsa_m2', 0), 'naca0012', 'cf'))
xo1, co1 = binned(*surf(wd('naca0012', 'ogrid', 'aftsa_m2', 0), 'naca0012', 'cf'))
axs[0].plot(xc1, cc1, 'k-', lw=1.3, marker='o', ms=2.5, markevery=2, label='cavity (unstructured)')
axs[0].plot(xo1, co1, '--', color='C3', lw=1.3, marker='^', ms=3, markevery=2, label='C-grid (structured)')
axs[0].set_xlabel('$x/c$'); axs[0].set_ylabel('$C_f$'); axs[0].set_xlim(0, 1); axs[0].set_ylim(0, None)
axs[0].set_title(r'$C_f$, $\alpha=0^\circ$', fontsize=10); axs[0].legend(fontsize=7, frameon=False)
xtc = [xtr_jump(wd('naca0012', 'cavity', 'aftsa_m2', a), 'naca0012') for a in alt]
xto = [xtr_jump(wd('naca0012', 'ogrid', 'aftsa_m2', a), 'naca0012') for a in alt]
axs[1].plot(alt, xtc, 'ko-', ms=4, lw=1.3, label='cavity')
axs[1].plot(alt, xto, '^--', color='C3', ms=5, lw=1.2, label='C-grid')
axs[1].set_xlabel(r'$\alpha$ (deg)'); axs[1].set_ylabel('$x_{tr}/c$'); axs[1].set_ylim(0, 0.8); axs[1].grid(alpha=0.3)
axs[1].set_title('transition vs.\\ incidence', fontsize=10); axs[1].legend(fontsize=7, frameon=False)
plt.tight_layout(pad=0.3); plt.savefig(f"{PD}/figs/grid_convergence.pdf")

print("\nSA-AI transition x/c (cavity):")
for af in ('naca0012', 'nlf0416'):
    print("  %-9s" % af, ["%.2f" % x for x in xt_all[af]], " mfoil", ["%.2f" % float(mf[af][a]['xtr_up']) for a in alt])
print("grid-conv NACA0012 cavity vs O-grid xtr:", ["%.2f" % x for x in xtc], "|", ["%.2f" % x for x in xto])
print("wrote naca0012_cf, convergence, polar, xtr_alpha, grid_convergence")
