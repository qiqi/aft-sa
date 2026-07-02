"""Regenerate gated-max NLF figures with CONTOUR-WALK upper/lower split (not sign-of-z, which
interleaves the cambered NLF's surfaces aft). Updates polar.pdf, xtr_alpha.pdf, nlf0416_cp_cf_a0.pdf."""
import csv,os,pickle,numpy as np,vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
B="/home/qiqi/flexcompute/aft-sa/flow360";PD="/home/qiqi/flexcompute/aft-sa/paper"
ALPHAS=[-2,0,2,4,6,8];ALT=[0,2,4,6,8]
def wd(af,mesh,model,a):
    # Paper convention: 'cavity' = unstructured L1 (108k nodes, TEprop metric);
    # 'ogrid' = structured L1 (proper_struct_L1, 127k nodes).
    if mesh=='cavity' and model=='aftsa_m2': return f"{B}/cavprop_{af}_a{a}"
    if mesh=='cavity' and model=='turb':     return f"{B}/cavprop_{af}_turb_a{a}"
    if mesh=='ogrid'  and model=='aftsa_m2': return f"{B}/strprop_{af}_a{a}"
    if mesh=='ogrid'  and model=='turb':     return f"{B}/strprop_{af}_turb_a{a}"
    return f"{B}/full_{af}_{mesh}_{model}_a{a}"
def forces(d):
    rows=list(csv.reader(open(os.path.join(d,'total_forces_v2.csv'))));h=[x.strip() for x in rows[0]]
    last=[c.strip() for c in rows[-1] if c.strip()!=''];g=lambda k:float(last[h.index(k)]);return g('CL'),g('CD')
def contour(d,af):
    """Return upper (x,Cf,Cp) and lower (x,Cf,Cp) via nearest-neighbor loop walk on one span plane."""
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/surface_fluid_{af}.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData()
    nm=[pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    cf=vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cf'))))
    cf=np.linalg.norm(cf,axis=1) if cf.ndim>1 else cf
    cp=vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cp'))))
    x,y,z=p[:,0],p[:,1],p[:,2];s=np.abs(y-y.min())<1e-6  # one span plane
    X,Z,CF,CP=x[s],z[s],cf[s],cp[s];n=len(X);pts=np.column_stack([X,Z])
    st=int(np.argmin(X));o=[st];u=np.zeros(n,bool);u[st]=True
    for _ in range(n-1):
        c=o[-1];dd=np.sum((pts-pts[c])**2,1);dd[u]=1e9;nx=int(np.argmin(dd));o.append(nx);u[nx]=True
    o=np.array(o);xo,zo,cfo,cpo=X[o],Z[o],CF[o],CP[o]
    # Robust TE-split (handles both open-TE O-grid and closed-TE unstructured contours).
    mid = n // 2
    window = slice(max(0, mid - n//8), min(n, mid + n//8))
    te = window.start + int(np.argmax(xo[window]))
    # Split EXCLUSIVE of the TE-shared point (which otherwise appears in both surfaces and
    # produces a spurious Cf spike at x=1 on airfoils with finite-thickness TE, e.g. NLF).
    b1,b2=slice(0,te),slice(te+1,n)
    if zo[b1].mean()>=zo[b2].mean(): up,lo=b1,b2
    else: up,lo=b2,b1
    def srt(sl):
        xx=xo[sl];oo=np.argsort(xx);xx,cfs,cps=xx[oo],cfo[sl][oo],cpo[sl][oo]
        # Drop a single LE outlier if present (walk-start point sometimes lands on the
        # opposite surface for cambered LEs with very close upper/lower points).
        if len(xx)>2 and xx[0]<0.001 and xx[1]>2*xx[0]: xx,cfs,cps=xx[1:],cfs[1:],cps[1:]
        return xx,cfs,cps
    return srt(up),srt(lo)
def xtr_up(d,af):
    (xu,cfu,_),_=contour(d,af);k=np.ones(7)/7;cm=np.convolve(cfu,k,'same')
    dc=np.diff(cm);dm=.5*(xu[1:]+xu[:-1]);w=(dm>0.05)&(dm<0.9)
    if not w.any():return np.nan
    j=np.argmax(dc[w]);return float(dm[w][j]) if dc[w][j]>0.1*np.nanmean(cm[(xu>0.05)&(xu<0.9)]) else 0.05
mf={af:{float(r['alpha']):r for r in csv.DictReader(open(f"{PD}/data/mfoil_{af}.csv"))} for af in ('naca0012','nlf0416')}

# ---- polar.pdf ----
fig,axs=plt.subplots(1,2,figsize=(6.6,2.9))
for ax,af,ttl in [(axs[0],'naca0012','NACA 0012'),(axs[1],'nlf0416','NLF(1)-0416')]:
    def col(mesh,model,k):return [forces(wd(af,mesh,model,a))[k] for a in ALPHAS]
    try:ax.plot(col('cavity','turb',1),col('cavity','turb',0),'^-.',color='0.6',ms=4,lw=1.0,label='SA (turbulent)')
    except:pass
    ax.plot([float(mf[af][a]['cd']) for a in ALPHAS],[float(mf[af][a]['cl']) for a in ALPHAS],'s--',color='0.4',ms=4,lw=1.1,label='mfoil ($e^N$)')
    # Both grids equally prominent (the new TEprop unstructured converges tightly).
    # Monotone polar: black for both SA-AF lines, distinguish by linestyle + marker.
    try:ax.plot(col('ogrid','aftsa_m2',1),col('ogrid','aftsa_m2',0),'^-',color='k',ms=4,lw=1.4,label='SA-AF, O-grid')
    except Exception as e:print('ogrid polar',af,e)
    ax.plot(col('cavity','aftsa_m2',1),col('cavity','aftsa_m2',0),'o--',color='k',ms=4,lw=1.4,alpha=1.0,label='SA-AF, unstructured')
    ax.set_xlabel('$C_D$');ax.set_title(ttl,fontsize=10);ax.grid(alpha=0.3)
axs[0].set_ylabel('$C_L$');axs[0].legend(fontsize=6.4,frameon=False,loc='lower right')
plt.tight_layout(pad=0.3);plt.savefig(f"{PD}/figs/polar.pdf");print("wrote polar.pdf")

# ---- xtr_alpha.pdf (transition BAND vs alpha, both surfaces, both grids + mfoil) ----
def tband_surface(x_surf, cf_surf):
    """Given (x, Cf) for one surface, return (x10, x50, x90) of the Cf laminar->turbulent rise.
    Returns NaNs if no clear transition is detected (either flow stays laminar to TE, or the
    detected 'rise' is just the LE Cf spike or marginal, with cfturb/cflam < 1.5)."""
    if len(cf_surf) < 10: return (np.nan, np.nan, np.nan)
    k = np.ones(7)/7; cm = np.convolve(cf_surf, k, 'same')
    m = (x_surf > 0.03) & (x_surf < 0.95); xx, cc = x_surf[m], cm[m]
    if len(xx) < 5: return (np.nan, np.nan, np.nan)
    dcf = np.diff(cc); dm = 0.5*(xx[1:] + xx[:-1]); w = (dm > 0.05) & (dm < 0.9)
    if not w.any() or dcf[w].max() <= 0: return (np.nan, np.nan, np.nan)
    x50 = float(dm[np.where(w)[0][int(np.argmax(dcf[w]))]])
    # Reject if the detected x50 is too close to the LE (the algorithm has picked up the LE
    # Cf spike rather than a real laminar-to-turbulent rise).
    if x50 < 0.07: return (np.nan, np.nan, np.nan)
    cflam = cc[xx <= x50].min(); ipk = int(np.argmax(cc[xx >= x50])); cfturb = cc[xx >= x50][ipk]
    # Reject marginal rises (cfturb/cflam < 1.5) — this is below a true laminar-to-turbulent ratio.
    if cfturb <= cflam or cfturb / max(cflam, 1e-12) < 1.5: return (np.nan, np.nan, np.nan)
    lo = cflam + 0.1*(cfturb - cflam); hi = cflam + 0.9*(cfturb - cflam)
    xdip = xx[xx <= x50][int(np.argmin(cc[xx <= x50]))]; xpk = xx[xx >= x50][ipk]
    rise = (xx >= xdip) & (xx <= xpk); xr, cr = xx[rise], cc[rise]
    cx = lambda l: (float(xr[np.where(cr >= l)[0][0]]) if (cr >= l).any() else np.nan)
    return (cx(lo), x50, cx(hi))

def tbands_both(d, af):
    """Return (upper, lower) transition bands; each is (x10, x50, x90)."""
    (xu, cfu, _), (xl, cfl, _) = contour(d, af)
    return tband_surface(xu, cfu), tband_surface(xl, cfl)

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4))
UP, LO = 'C0', 'C3'   # upper/lower colors
# Marker convention: triangle for the unstructured (triangular cells), circle for the O-grid
# (circular topology), square for mfoil.
for ax, af, ttl in [(axs[0], 'naca0012', 'NACA 0012'), (axs[1], 'nlf0416', 'NLF(1)-0416')]:
    # mfoil reference (dotted), both surfaces
    ax.plot(ALPHAS, [float(mf[af][a]['xtr_up']) for a in ALPHAS], ':', color=UP, lw=1.4, marker='s', ms=4)
    ax.plot(ALPHAS, [float(mf[af][a]['xtr_lo']) for a in ALPHAS], ':', color=LO, lw=1.4, marker='s', ms=4)
    # SA-AF on both grids, both surfaces, with x10..x90 band; span the full alpha range.
    for mesh, ls, mk in [('ogrid', '-', 'o'), ('cavity', '--', '^')]:
        bands = [tbands_both(wd(af, mesh, 'aftsa_m2', a), af) for a in ALPHAS]
        x10u = np.array([b[0][0] for b in bands]); x50u = np.array([b[0][1] for b in bands]); x90u = np.array([b[0][2] for b in bands])
        x10l = np.array([b[1][0] for b in bands]); x50l = np.array([b[1][1] for b in bands]); x90l = np.array([b[1][2] for b in bands])
        ax.fill_between(ALPHAS, x10u, x90u, color=UP, alpha=0.12, lw=0)
        ax.fill_between(ALPHAS, x10l, x90l, color=LO, alpha=0.12, lw=0)
        ax.plot(ALPHAS, x50u, ls, color=UP, marker=mk, ms=4, lw=1.2)
        ax.plot(ALPHAS, x50l, ls, color=LO, marker=mk, ms=4, lw=1.2)
    ax.set_xlabel(r'$\alpha$ (deg)'); ax.set_title(ttl, fontsize=10)
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.3)
axs[0].set_ylabel('$x_{tr}/c$')
# Composite legend
from matplotlib.lines import Line2D
surfh = [Line2D([],[],color=UP, lw=2, label='upper'), Line2D([],[],color=LO, lw=2, label='lower')]
solh  = [Line2D([],[],color='0.3', ls='-',  marker='o', ms=4, label='SA-AF, O-grid'),
         Line2D([],[],color='0.3', ls='--', marker='^', ms=4, label='SA-AF, unstructured'),
         Line2D([],[],color='0.3', ls=':',  marker='s', ms=4, label='mfoil ($e^N$)')]
axs[0].legend(handles=surfh, fontsize=8, frameon=False, loc='upper right')
axs[1].legend(handles=solh,  fontsize=7.5, frameon=False, loc='center right')
plt.tight_layout(pad=0.4)
plt.savefig(f"{PD}/figs/xtr_alpha.pdf"); print("wrote xtr_alpha.pdf (both surfaces, band)")

# ---- nlf0416_cp_cf_a0.pdf (both surfaces, both grids, color=side) ----
mfs_full=pickle.load(open(f"{B}/mfoil_nlf0416_Re4M.pkl",'rb'))[0.0]  # Re=4M
mfs={'x_upper':mfs_full['upper']['x'],'cp_upper':mfs_full['upper']['cp'],'cf_upper':mfs_full['upper']['cf'],
     'x_lower':mfs_full['lower']['x'],'cp_lower':mfs_full['lower']['cp'],'cf_lower':mfs_full['lower']['cf']}
fig,axs=plt.subplots(1,2,figsize=(7.4,3.1));UP,LO='C0','C3'
for mesh,ls,lw,al in [('ogrid','-',1.4,1.0),('cavity','--',1.4,1.0)]:
    try:
        (xu,cfu,cpu),(xl,cfl,cpl)=contour(wd('nlf0416',mesh,'aftsa_m2',0),'nlf0416')
        axs[0].plot(xu,-cpu,ls,color=UP,lw=lw,alpha=al);axs[0].plot(xl,-cpl,ls,color=LO,lw=lw,alpha=al)
        axs[1].plot(xu,cfu,ls,color=UP,lw=lw,alpha=al);axs[1].plot(xl,cfl,ls,color=LO,lw=lw,alpha=al)
    except Exception as e:print('cpcf',mesh,e)
axs[0].plot(mfs['x_upper'],-np.asarray(mfs['cp_upper']),':',color=UP,lw=1.4);axs[0].plot(mfs['x_lower'],-np.asarray(mfs['cp_lower']),':',color=LO,lw=1.4)
axs[1].plot(mfs['x_upper'],np.asarray(mfs['cf_upper']),':',color=UP,lw=1.4);axs[1].plot(mfs['x_lower'],np.asarray(mfs['cf_lower']),':',color=LO,lw=1.4)
axs[0].set_xlabel('$x/c$');axs[0].set_ylabel('$-C_p$');axs[0].set_xlim(0,1);axs[0].set_title(r'$C_p$, $\alpha=0^\circ$',fontsize=10)
axs[1].set_xlabel('$x/c$');axs[1].set_ylabel('$C_f$');axs[1].set_xlim(0,1);axs[1].set_ylim(0,None);axs[1].set_title(r'$C_f$, $\alpha=0^\circ$',fontsize=10)
from matplotlib.lines import Line2D
surfh = [Line2D([],[],color=UP, lw=2, label='upper'),
         Line2D([],[],color=LO, lw=2, label='lower')]
solh  = [Line2D([],[],color='0.3',ls='-', label='SA-AF, O-grid'),
         Line2D([],[],color='0.3',ls='--',label='SA-AF, unstructured'),
         Line2D([],[],color='0.3',ls=':', label='mfoil ($e^N$)')]
axs[0].legend(handles=surfh, fontsize=8, frameon=False, loc='upper right')
axs[1].legend(handles=solh,  fontsize=8, frameon=False, loc='upper right')
plt.tight_layout(pad=0.4);plt.savefig(f"{PD}/figs/nlf0416_cp_cf_a0.pdf");print("wrote nlf0416_cp_cf_a0.pdf")
print("REGEN DONE")
