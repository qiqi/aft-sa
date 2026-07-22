"""Regenerate NACA figures:
  fig:cf = naca0012_cf.pdf — 4 panels: top row Cp at alpha=0 and 4, bottom row Cf at alpha=0 and 4.
                              Both surfaces (color=side), four solver/mesh combinations (linestyle):
                              SA-AI O-grid (solid), SA-AI unstructured (dashed), mfoil e^N (dotted),
                              SA turbulent O-grid (dash-dot). Replaces and subsumes naca0012_cp_cf_a4.pdf.
  fig:conv = convergence.pdf — alpha=4 grid-refinement residual study, 2 panels (left=O-grid, right=unstructured),
                                3 mesh levels each, continuity + nuHat residuals.
"""
import csv,os,pickle,numpy as np,vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.rcParams.update({'font.size': 11, 'axes.titlesize': 11, 'axes.labelsize': 11,
                            'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10})
B="/home/qiqi/flexcompute/sa-ai/flow360";PD="/home/qiqi/flexcompute/sa-ai/paper"

def wd(af,mesh,model,a):
    # Paper convention: 'cavity' = unstructured L1 (108k nodes, TEprop metric);
    # 'ogrid' = structured L1 (proper_struct_L1, 127k nodes).
    if mesh=='cavity' and model=='aftsa_m2': return f"{B}/cavprop_{af}_a{a}"
    if mesh=='cavity' and model=='turb':     return f"{B}/cavprop_{af}_turb_a{a}"
    if mesh=='ogrid'  and model=='aftsa_m2': return f"{B}/strprop_{af}_a{a}"
    if mesh=='ogrid'  and model=='turb':     return f"{B}/strprop_{af}_turb_a{a}"
    return f"{B}/full_{af}_{mesh}_{model}_a{a}"

def contour(d,af):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/surface_fluid_{af}.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData()
    nm=[pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    cf=vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cf'))));cf=np.linalg.norm(cf,axis=1) if cf.ndim>1 else cf
    cp=vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cp'))))
    x,y,z=p[:,0],p[:,1],p[:,2];s=np.abs(y-y.min())<1e-6;X,Z,CF,CP=x[s],z[s],cf[s],cp[s];n=len(X);pts=np.column_stack([X,Z])
    st=int(np.argmin(X));o=[st];u=np.zeros(n,bool);u[st]=True
    for _ in range(n-1):
        c=o[-1];dd=np.sum((pts-pts[c])**2,1);dd[u]=1e9;nx=int(np.argmin(dd));o.append(nx);u[nx]=True
    o=np.array(o);xo,zo,cfo,cpo=X[o],Z[o],CF[o],CP[o]
    mid = n // 2
    window = slice(max(0, mid - n//8), min(n, mid + n//8))
    te = window.start + int(np.argmax(xo[window]))
    b1,b2=slice(0,te),slice(te+1,n)
    up,lo=(b1,b2) if zo[b1].mean()>=zo[b2].mean() else (b2,b1)
    def srt(sl):
        xx=xo[sl];oo=np.argsort(xx);xx,cfs,cps=xx[oo],cfo[sl][oo],cpo[sl][oo]
        if len(xx)>2 and xx[0]<0.001 and xx[1]>2*xx[0]: xx,cfs,cps=xx[1:],cfs[1:],cps[1:]
        return xx,cfs,cps
    return srt(up),srt(lo)

def resid(d):
    rows=list(csv.reader(open(f"{d}/nonlinear_residual_v2.csv")));h=[x.strip() for x in rows[0]]
    ip,ic,inu=h.index('pseudo_step'),h.index('0_cont'),h.index('5_nuHat');s,c,n=[],[],[]
    for r in rows[1:]:
        r=[x for x in r if x.strip()!='']
        if len(r)>inu:
            try:s.append(float(r[ip]));c.append(float(r[ic]));n.append(float(r[inu]))
            except:pass
    return map(np.array,(s,c,n))

# ---- fig:cf -- NACA0012 4 panels (top row Cp, bottom row Cf; cols alpha=0, alpha=4) ----
mfs = pickle.load(open(f"{PD}/data/mfoil_surf_naca0012.pkl",'rb'))
UP, LO = 'C0', 'C3'   # upper / lower colors

fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.0), sharex=True)
for col, a in enumerate([0, 4]):
    ax_cp = axs[0, col]; ax_cf = axs[1, col]
    # solver/mesh entries: (mesh, model, ls, lw, alpha, label)
    sols = [
        ('ogrid',  'aftsa_m2', '-',  1.4, 1.0, 'SA-AI, O-grid'),
        ('cavity', 'aftsa_m2', '--', 1.4, 1.0, 'SA-AI, unstructured'),
        ('ogrid',  'turb',     '-.', 1.0, 0.85,'SA (turbulent), O-grid'),
    ]
    for mesh, model, ls, lw, al, _ in sols:
        try:
            (xu, cfu, cpu), (xl, cfl, cpl) = contour(wd('naca0012', mesh, model, a), 'naca0012')
            ax_cp.plot(xu, -cpu, ls, color=UP, lw=lw, alpha=al)
            ax_cp.plot(xl, -cpl, ls, color=LO, lw=lw, alpha=al)
            ax_cf.plot(xu,  cfu, ls, color=UP, lw=lw, alpha=al)
            ax_cf.plot(xl,  cfl, ls, color=LO, lw=lw, alpha=al)
        except Exception as e:
            print(f"  skip ({mesh},{model},a={a}): {e}")
    # mfoil dotted reference (both surfaces)
    m = mfs[float(a)]
    ax_cp.plot(m['x_upper'], -np.asarray(m['cp_upper']), ':', color=UP, lw=1.4)
    ax_cp.plot(m['x_lower'], -np.asarray(m['cp_lower']), ':', color=LO, lw=1.4)
    ax_cf.plot(m['x_upper'],  np.asarray(m['cf_upper']), ':', color=UP, lw=1.4)
    ax_cf.plot(m['x_lower'],  np.asarray(m['cf_lower']), ':', color=LO, lw=1.4)
    ax_cp.set_title(rf'$\alpha={a}^\circ$', fontsize=10)
    ax_cf.set_xlabel('$x/c$')
    ax_cp.set_xlim(0, 1); ax_cf.set_xlim(0, 1)
    ax_cf.set_ylim(0, 0.012)
    ax_cp.grid(alpha=0.3); ax_cf.grid(alpha=0.3)
axs[0,0].set_ylabel('$-C_p$'); axs[1,0].set_ylabel('$C_f$')

# Legend: 2 entries for surface color, 4 for solver/mesh linestyle (single composite legend on upper-right axes)
surfh = [Line2D([],[],color=UP, lw=2, label='upper'),
         Line2D([],[],color=LO, lw=2, label='lower')]
solh  = [Line2D([],[],color='0.3', ls='-',  lw=1.4, label='SA-AI, O-grid'),
         Line2D([],[],color='0.3', ls='--', lw=1.4, label='SA-AI, unstructured'),
         Line2D([],[],color='0.3', ls='-.', lw=1.0, alpha=0.85, label='SA (turb.), O-grid'),
         Line2D([],[],color='0.3', ls=':',  lw=1.4, label=r'mfoil ($e^N$)')]
axs[0,1].legend(handles=surfh, fontsize=8, frameon=False, loc='upper right')
axs[1,1].legend(handles=solh,  fontsize=8, frameon=False, loc='upper right')
plt.tight_layout(pad=0.4)
plt.savefig(f"{PD}/figs/naca0012_cf.pdf"); plt.savefig('/tmp/naca0012_cf.png', dpi=130)
plt.close()
print("wrote naca0012_cf.pdf (4 panels Cp+Cf, alpha=0 and 4)")

# ---- fig:conv -- alpha=4 grid-refinement residual study, 2 panels (monotone B&W) ----
# Distinguish six lines per panel by thickness (level) and dash pattern (residual type):
# thin = L0, medium = L1, thick = L2; solid = continuity, dashed = nuHat.
LEVELS = ['L0', 'L1', 'L2']
LVL_LW = {'L0': 0.7, 'L1': 1.2, 'L2': 1.8}
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
for ax, family, dirs in [
    (axs[0], 'O-grid',       [f"{B}/proper_str_{l}"        for l in LEVELS]),
    (axs[1], 'unstructured', [f"{B}/proper_cav_{l}_TEprop" for l in LEVELS]),
]:
    for lvl, d in zip(LEVELS, dirs):
        try:
            s, cR, nR = resid(d)
            ax.semilogy(s, cR, '-',  color='k', lw=LVL_LW[lvl])
            ax.semilogy(s, nR, '--', color='k', lw=LVL_LW[lvl])
        except Exception as e:
            print(f"  resid skip {d}: {e}")
    ax.set_xlabel('pseudo-time step')
    ax.set_title(family)
    ax.set_ylim(1e-11, None)
    ax.grid(alpha=0.3)
axs[0].set_ylabel('RMS residual')
plt.tight_layout(pad=0.4)
plt.savefig(f"{PD}/figs/convergence.pdf"); plt.savefig('/tmp/convergence.png', dpi=130)
plt.close()
print("wrote convergence.pdf (alpha=4 grid refinement, monotone)")

print("REGEN NACA DONE")
