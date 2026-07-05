"""Paper figures with BOTH meshes overlaid (cavity unstructured + structured O-grid).
Fig5 cf(a=0), Fig6 convergence, Fig8 cp_cf(a=4), Fig9 polar, Fig10 xtr-vs-alpha;
Fig7 (grid_convergence.pdf) = grid-refinement study (separate, needs refine data)."""
import csv, os, json, pickle, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 11, 'font.family': 'serif', 'axes.linewidth': 0.8})
B = "/home/qiqi/flexcompute/aft-sa/flow360"; PD = "/home/qiqi/flexcompute/aft-sa/paper"
ALPHAS = [-2, 0, 2, 4, 6, 8]; ALT = [0, 2, 4, 6, 8]
CAV, STR = 'cavity', 'ogrid'
def wd(af, mesh, model, a): return f"{B}/full_{af}_{mesh}_{model}_a{a}"
def forces(d):
    rows = list(csv.reader(open(os.path.join(d, 'total_forces_v2.csv')))); h = [x.strip() for x in rows[0]]
    last = [c.strip() for c in rows[-1] if c.strip() != '']; g = lambda k: float(last[h.index(k)]); return g('CL'), g('CD')
def surf(d, af, field='cf'):
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/surface_fluid_{af}.pvtu"); r.Update()
    g = r.GetOutput(); pts = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    nm = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    arr = vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith(field))))
    v = np.linalg.norm(arr, axis=1) if (field == 'cf' and arr.ndim > 1) else arr
    x, z = pts[:, 0], pts[:, 2]; up = z > 1e-6; xs, vs = x[up], v[up]; o = np.argsort(xs); return xs[o], vs[o]
def binned(xs, vs, nb=45):
    bins = np.linspace(0, 1, nb + 1); xc = .5 * (bins[1:] + bins[:-1])
    cb = np.array([np.median(vs[(xs >= bins[i]) & (xs < bins[i+1])]) if ((xs >= bins[i]) & (xs < bins[i+1])).any() else np.nan for i in range(nb)])
    ok = np.isfinite(cb); return xc[ok], cb[ok]
def xtr_jump(d, af):
    xc, cb = binned(*surf(d, af, 'cf'), 50); dd = np.diff(cb); dm = .5 * (xc[1:] + xc[:-1]); w = (dm > 0.02) & (dm < 0.95)
    if not w.any(): return np.nan
    j = np.argmax(dd[w]); return 0.02 if dd[w][j] < 0.1 * np.nanmean(cb[(xc > 0.02) & (xc < 0.95)]) else float(dm[w][j])
def resid(d):
    rows = list(csv.reader(open(f"{d}/nonlinear_residual_v2.csv"))); h = [x.strip() for x in rows[0]]
    ip, ic, inu = h.index('pseudo_step'), h.index('0_cont'), h.index('5_nuHat'); s, c, n = [], [], []
    for row in rows[1:]:
        row = [x for x in row if x.strip() != '']
        if len(row) <= inu: continue
        s.append(float(row[ip])); c.append(float(row[ic])); n.append(float(row[inu]))
    return map(np.array, (s, c, n))
mf = {af: {float(r['alpha']): r for r in csv.DictReader(open(f"{PD}/data/mfoil_{af}.csv"))} for af in ('naca0012', 'nlf0416')}

# ---- Fig 5: Cf a=0, both meshes + mfoil ----
plt.figure(figsize=(3.6, 2.8))
mf0 = pickle.load(open(f"{PD}/data/mfoil_surf_naca0012.pkl", 'rb'))[0.0]
xc, cc = binned(*surf(wd('naca0012', CAV, 'aftsa_m2', 0), 'naca0012')); xo, co = binned(*surf(wd('naca0012', STR, 'aftsa_m2', 0), 'naca0012'))
xt, ct = binned(*surf(wd('naca0012', CAV, 'turb', 0), 'naca0012'))
plt.plot(xt, ct, '-', color='0.6', lw=1.2, label='SA (turbulent)')
plt.plot(mf0['x_upper'], mf0['cf_upper'], '-', color='0.35', lw=1.1, label='mfoil ($e^N$)')
plt.plot(xc, cc, 'k-', lw=1.3, marker='o', ms=2.5, markevery=2, label='SA-AI, cavity')
plt.plot(xo, co, '--', color='C3', lw=1.3, marker='^', ms=3, markevery=2, label='SA-AI, O-grid')
plt.xlabel('$x/c$'); plt.ylabel('$C_f$'); plt.xlim(0, 1); plt.ylim(0, None)
plt.legend(fontsize=7.5, frameon=False); plt.tight_layout(pad=0.3); plt.savefig(f"{PD}/figs/naca0012_cf.pdf")

# ---- Fig 6: convergence, both meshes ----
plt.figure(figsize=(3.6, 2.8))
sC, cC, nC = resid(wd('naca0012', CAV, 'aftsa_m2', 0)); sO, cO, nO = resid(wd('naca0012', STR, 'aftsa_m2', 0))
plt.semilogy(sC, cC, 'k-', lw=1.3, label='cavity: cont.'); plt.semilogy(sC, nC, 'k--', lw=1.0, label=r'cavity: $\tilde\nu$')
plt.semilogy(sO, cO, '-', color='C3', lw=1.3, label='O-grid: cont.'); plt.semilogy(sO, nO, '--', color='C3', lw=1.0, label=r'O-grid: $\tilde\nu$')
plt.xlabel('pseudo-time step'); plt.ylabel('RMS residual'); plt.ylim(1e-11, None)
plt.legend(fontsize=6.5, frameon=False); plt.tight_layout(pad=0.3); plt.savefig(f"{PD}/figs/convergence.pdf")

# ---- Fig 8: cp_cf a=4, both meshes ----
mfs = pickle.load(open(f"{PD}/data/mfoil_surf_naca0012.pkl", 'rb'))[4.0]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
xpc, cpc = surf(wd('naca0012', CAV, 'aftsa_m2', 4), 'naca0012', 'cp'); xpo, cpo = surf(wd('naca0012', STR, 'aftsa_m2', 4), 'naca0012', 'cp')
axs[0].plot(mfs['x_upper'], -mfs['cp_upper'], '-', color='0.4', lw=1.1, label='mfoil ($e^N$)')
axs[0].plot(xpc, -cpc, 'k.', ms=2, label='SA-AI, cavity'); axs[0].plot(xpo, -cpo, '.', color='C3', ms=2, label='SA-AI, O-grid')
axs[0].set_xlabel('$x/c$'); axs[0].set_ylabel('$-C_p$'); axs[0].legend(fontsize=7, frameon=False); axs[0].set_title(r'$C_p$, $\alpha=4^\circ$', fontsize=10)
xcc, ccc = binned(*surf(wd('naca0012', CAV, 'aftsa_m2', 4), 'naca0012')); xoo, coo = binned(*surf(wd('naca0012', STR, 'aftsa_m2', 4), 'naca0012'))
xtt, ctt = binned(*surf(wd('naca0012', CAV, 'turb', 4), 'naca0012'))
axs[1].plot(mfs['x_upper'], mfs['cf_upper'], '-', color='0.4', lw=1.1, label='mfoil')
axs[1].plot(xtt, ctt, '-', color='0.6', lw=1.0, label='SA (turb.)')
axs[1].plot(xcc, ccc, 'ko-', ms=2.5, lw=1.1, markevery=2, label='SA-AI, cavity')
axs[1].plot(xoo, coo, '^--', color='C3', ms=2.5, lw=1.1, markevery=2, label='SA-AI, O-grid')
axs[1].set_xlabel('$x/c$'); axs[1].set_ylabel('$C_f$'); axs[1].set_ylim(0, None); axs[1].legend(fontsize=6.6, frameon=False); axs[1].set_title(r'$C_f$, $\alpha=4^\circ$', fontsize=10)
plt.tight_layout(pad=0.4); plt.savefig(f"{PD}/figs/naca0012_cp_cf_a4.pdf")

# ---- Fig 9: polar, both meshes ----
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, af, ttl, hasstr in [(axs[0], 'naca0012', 'NACA 0012', True), (axs[1], 'nlf0416', 'NLF(1)-0416', True)]:
    cdT = [forces(wd(af, CAV, 'turb', a))[1] for a in ALPHAS]; clT = [forces(wd(af, CAV, 'turb', a))[0] for a in ALPHAS]
    cdA = [forces(wd(af, CAV, 'aftsa_m2', a))[1] for a in ALPHAS]; clA = [forces(wd(af, CAV, 'aftsa_m2', a))[0] for a in ALPHAS]
    cdM = [float(mf[af][a]['cd']) for a in ALPHAS]; clM = [float(mf[af][a]['cl']) for a in ALPHAS]
    ax.plot(cdT, clT, '^-.', color='0.6', ms=4, lw=1.0, label='SA (turbulent)')
    ax.plot(cdM, clM, 's--', color='0.4', ms=4, lw=1.1, label='mfoil ($e^N$)')
    ax.plot(cdA, clA, 'ko-', ms=3.5, lw=1.3, label='SA-AI, cavity')
    if hasstr:
        cdO = [forces(wd(af, STR, 'aftsa_m2', a))[1] for a in ALPHAS]; clO = [forces(wd(af, STR, 'aftsa_m2', a))[0] for a in ALPHAS]
        ax.plot(cdO, clO, '^--', color='C3', ms=4, lw=1.1, label='SA-AI, O-grid')
    ax.set_xlabel('$C_D$'); ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3)
axs[0].set_ylabel('$C_L$'); axs[0].legend(fontsize=6.4, frameon=False, loc='lower right')
plt.tight_layout(pad=0.3); plt.savefig(f"{PD}/figs/polar.pdf")

# ---- Fig 10: xtr vs alpha, both meshes (NACA0012) + mfoil ----
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, af, ttl, hasstr in [(axs[0], 'naca0012', 'NACA 0012', True), (axs[1], 'nlf0416', 'NLF(1)-0416', True)]:
    xtA = [xtr_jump(wd(af, CAV, 'aftsa_m2', a), af) for a in ALT]; xtM = [float(mf[af][a]['xtr_up']) for a in ALT]
    ax.plot(ALT, xtM, 's--', color='0.4', ms=5, lw=1.1, label='mfoil ($e^N$)')
    ax.plot(ALT, xtA, 'ko-', ms=4, lw=1.3, label='SA-AI, cavity')
    if hasstr:
        xtO = [xtr_jump(wd(af, STR, 'aftsa_m2', a), af) for a in ALT]
        ax.plot(ALT, xtO, '^--', color='C3', ms=5, lw=1.1, label='SA-AI, O-grid')
    ax.set_xlabel(r'$\alpha$ (deg)'); ax.set_title(ttl, fontsize=10); ax.set_ylim(0, 0.8); ax.grid(alpha=0.3)
axs[0].set_ylabel('suction-side $x_{tr}/c$'); axs[0].legend(fontsize=6.4, frameon=False)
plt.tight_layout(pad=0.3); plt.savefig(f"{PD}/figs/xtr_alpha.pdf")
print("wrote naca0012_cf, convergence, naca0012_cp_cf_a4, polar, xtr_alpha (both meshes)")
