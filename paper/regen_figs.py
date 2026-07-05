"""Regenerate Figs 2-5 + Table 2 numbers from the 10k-step runs.
Fig2 NACA0012 Cf (SA-AI vs ONE turbulent SA); Fig3 convergence (10k);
Fig4 polar (SA-AI, turbulent SA, mfoil e^N); Fig5 transition vs alpha."""
import csv, os, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 11, 'font.family': 'serif', 'axes.linewidth': 0.8})
B = "/home/qiqi/flexcompute/aft-sa/flow360"
PD = "/home/qiqi/flexcompute/aft-sa/paper"
wall = {'naca0012': 'naca0012', 'nlf0416': 'nlf0416'}
alphas = [-2, 0, 2, 4, 6, 8]

def case(model, af, a):   # 10k case dir
    return f"{B}/run10k_{model}_{af}_a{a}"

def forces(wd):
    rows = list(csv.reader(open(os.path.join(wd, 'total_forces_v2.csv'))))
    h = [x.strip() for x in rows[0]]; last = [c.strip() for c in rows[-1] if c.strip() != '']
    g = lambda k: float(last[h.index(k)])
    return g('CL'), g('CD'), g('CDPressure'), g('CDSkinFriction')

def cf_xy(wd, af):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{wd}/surface_fluid_{wall[af]}.pvtu"); r.Update()
    g = r.GetOutput(); pts = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    nm = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    a = vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cf'))))
    cf = np.linalg.norm(a, axis=1) if a.ndim > 1 else a
    x, z = pts[:, 0], pts[:, 2]; up = z > 1e-6
    xs, cfs = x[up], cf[up]; o = np.argsort(xs); return xs[o], cfs[o]

def binned(xs, cfs, nb=45):
    bins = np.linspace(0, 1, nb + 1); xc = 0.5 * (bins[1:] + bins[:-1]); cb = np.full(nb, np.nan)
    for i in range(nb):
        s = (xs >= bins[i]) & (xs < bins[i + 1])
        if s.any(): cb[i] = np.median(cfs[s])
    ok = np.isfinite(cb); return xc[ok], cb[ok]

def xtr_jump(xs, cfs):
    xc, cb = binned(xs, cfs, 50); d = np.diff(cb); dm = 0.5 * (xc[1:] + xc[:-1])
    w = (dm > 0.03) & (dm < 0.90)
    if not w.any(): return np.nan
    j = np.argmax(d[w])
    if d[w][j] < 0.1 * np.nanmean(cb[(xc > 0.03) & (xc < 0.90)]): return 0.03
    return float(dm[w][j])

def resid(wd):
    s, c, n = [], [], []
    rows = list(csv.reader(open(f"{wd}/nonlinear_residual_v2.csv")))
    h = [x.strip() for x in rows[0]]; ic, inu, ip = h.index('0_cont'), h.index('5_nuHat'), h.index('pseudo_step')
    for row in rows[1:]:
        row = [c2 for c2 in row if c2.strip() != '']
        if len(row) <= inu: continue
        s.append(float(row[ip])); c.append(float(row[ic])); n.append(float(row[inu]))
    return np.array(s), np.array(c), np.array(n)

def loadm(p): return {float(r['alpha']): r for r in csv.DictReader(open(p))}
mf = {'naca0012': loadm(f"{PD}/data/mfoil_naca0012.csv"), 'nlf0416': loadm(f"{PD}/data/mfoil_nlf0416.csv")}

# ---- gather SA-AI + turbulent forces & SA-AI transition ----
saaf, turb, xtr = {}, {}, {}
for af in ('naca0012', 'nlf0416'):
    for a in alphas:
        saaf[(af, a)] = forces(case('aftsa', af, a))
        turb[(af, a)] = forces(case('turb', af, a))
        xs, cfs = cf_xy(case('aftsa', af, a), af); xtr[(af, a)] = xtr_jump(xs, cfs)

# ---- Table 2 numbers (NACA0012 a=0) ----
clA, cdA, cdpA, cdfA = saaf[('naca0012', 0)]
clT, cdT, cdpT, cdfT = turb[('naca0012', 0)]
print("TABLE2 NACA0012 a=0 (10k):")
print("  turbulent SA chi=3 : CD=%.5f CDp=%.5f CDf=%.5f" % (cdT, cdpT, cdfT))
print("  SA-AI chi=0.02     : CD=%.5f CDp=%.5f CDf=%.5f" % (cdA, cdpA, cdfA))

# ---- Fig 2: NACA0012 Cf, SA-AI vs ONE turbulent SA ----
plt.figure(figsize=(3.4, 2.7))
xa, ca = binned(*cf_xy(case('aftsa', 'naca0012', 0), 'naca0012'))
xt, ct = binned(*cf_xy(case('turb', 'naca0012', 0), 'naca0012'))
plt.plot(xt, ct, '--', color='0.5', lw=1.3, marker='s', ms=2.5, markevery=2, label='SA (turbulent)')
plt.plot(xa, ca, 'k-', lw=1.3, marker='o', ms=2.5, markevery=2, label='SA-AI')
plt.xlabel('$x/c$'); plt.ylabel('$C_f$'); plt.xlim(0, 1); plt.ylim(0, None)
plt.legend(fontsize=7.5, frameon=False); plt.tight_layout(pad=0.3); plt.savefig(f"{PD}/figs/naca0012_cf.pdf")

# ---- Fig 3: convergence (10k) ----
plt.figure(figsize=(3.4, 2.7))
sA, cA, nA = resid(case('aftsa', 'naca0012', 0)); sT, cT, nT = resid(case('turb', 'naca0012', 0))
plt.semilogy(sT, cT, '-', color='0.6', lw=1.3, label='SA: continuity')
plt.semilogy(sA, cA, 'k-', lw=1.3, label='SA-AI: continuity')
plt.semilogy(sT, nT, '--', color='0.6', lw=1.1, label=r'SA: $\tilde\nu$')
plt.semilogy(sA, nA, 'k--', lw=1.1, label=r'SA-AI: $\tilde\nu$')
plt.xlabel('pseudo-time step'); plt.ylabel('RMS residual'); plt.ylim(1e-11, None)
plt.legend(fontsize=6.3, frameon=False); plt.tight_layout(pad=0.3); plt.savefig(f"{PD}/figs/convergence.pdf")
print("CONV final: SA cont/nu=%.1e/%.1e  SA-AI cont/nu=%.1e/%.1e" % (cT[-1], nT[-1], cA[-1], nA[-1]))

# ---- Fig 4: polar ----
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, af, ttl in [(axs[0], 'naca0012', 'NACA 0012'), (axs[1], 'nlf0416', 'NLF(1)-0416')]:
    cdT_ = [turb[(af, a)][1] for a in alphas]; clT_ = [turb[(af, a)][0] for a in alphas]
    cdM = [float(mf[af][a]['cd']) for a in alphas]; clM = [float(mf[af][a]['cl']) for a in alphas]
    cdA_ = [saaf[(af, a)][1] for a in alphas]; clA_ = [saaf[(af, a)][0] for a in alphas]
    ax.plot(cdT_, clT_, '^-.', color='0.6', ms=4, lw=1.0, label='SA (turbulent)')
    ax.plot(cdM, clM, 's--', color='0.4', ms=4, lw=1.1, label='mfoil ($e^N$, $N_{cr}{=}4$)')
    ax.plot(cdA_, clA_, 'ko-', ms=3.5, lw=1.3, label='SA-AI')
    ax.set_xlabel('$C_D$'); ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3)
axs[0].set_ylabel('$C_L$'); axs[0].legend(fontsize=6.6, frameon=False, loc='lower right')
plt.tight_layout(pad=0.3); plt.savefig(f"{PD}/figs/polar.pdf")

# ---- Fig 5: transition vs alpha (suction side, alpha>=0) ----
alt = [0, 2, 4, 6, 8]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, af, ttl in [(axs[0], 'naca0012', 'NACA 0012'), (axs[1], 'nlf0416', 'NLF(1)-0416')]:
    xtM = [float(mf[af][a]['xtr_up']) for a in alt]; xtA = [xtr[(af, a)] for a in alt]
    ax.plot(alt, [0]*len(alt), '^-.', color='0.6', ms=4, lw=1.0, label='SA (turbulent)')
    ax.plot(alt, xtM, 's--', color='0.4', ms=5, lw=1.1, label='mfoil ($e^N$)')
    ax.plot(alt, xtA, 'ko-', ms=4, lw=1.3, label='SA-AI')
    ax.set_xlabel(r'$\alpha$ (deg)'); ax.set_title(ttl, fontsize=10); ax.set_ylim(0, 0.8); ax.grid(alpha=0.3)
axs[0].set_ylabel('suction-side $x_{tr}/c$'); axs[0].legend(fontsize=6.6, frameon=False)
plt.tight_layout(pad=0.3); plt.savefig(f"{PD}/figs/xtr_alpha.pdf")

print("\nSA-AI transition x/c (alpha 0..8):")
for af in ('naca0012', 'nlf0416'):
    print("  %-8s" % af, ["%.2f" % xtr[(af, a)] for a in alt], "  mfoil", ["%.2f" % float(mf[af][a]['xtr_up']) for a in alt])
print("wrote naca0012_cf.pdf, convergence.pdf, polar.pdf, xtr_alpha.pdf")
