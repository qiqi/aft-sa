"""f=0.03 geom mechanism: (1) global feedback (LE Cp vs CL), (2) receptivity zone (where near-wall
chi/nuHat oscillates over the cycle), (3) sSlow=0.03^(1-is_turb) there."""
import json,glob,os,csv,numpy as np,vtk
from vtk.util.numpy_support import vtk_to_numpy
F="/home/qiqi/flexcompute/aft-sa/flow360";NU0=2e-7;FSLOW=0.03;swC=1.0;swW=4.0
def isturb(chi): return np.where(chi<=swC,0.0,1.0-np.exp(-(chi-swC)/swW))
def sslow(chi): return FSLOW**(1.0-isturb(chi))
steps=sorted([int(os.path.basename(p).split('_')[1]) for p in glob.glob(f"{F}/f03_*") if os.path.isdir(p) and os.path.basename(p).split('_')[1].isdigit()])
res=json.load(open(f"{F}/run_f03ladder_results.json"))
def cp_upper(d):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/surface_fluid_nlf0416.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());cp=vtk_to_numpy(g.GetPointData().GetArray('Cp'))
    y=p[:,1];s=np.abs(y+0.0)<1e-3;X,Z,Cp=p[s][:,0],p[s][:,2],cp[s];n=len(X);pts=np.column_stack([X,Z])
    st=int(np.argmin(X));o=[st];u=np.zeros(n,bool);u[st]=True
    for _ in range(n-1):
        c=o[-1];dd=np.sum((pts-pts[c])**2,1);dd[u]=1e9;nx=int(np.argmin(dd));o.append(nx);u[nx]=True
    o=np.array(o);xo,zo,co=X[o],Z[o],Cp[o];te=int(np.argmax(xo));b1,b2=slice(0,te+1),slice(te,n)
    up=b1 if zo[b1].mean()>zo[b2].mean() else b2;xu,cu=xo[up],co[up];oo=np.argsort(xu);return xu[oo],cu[oo]
def nearwall_chi(d,xst):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/volume.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData()
    nuh=vtk_to_numpy(pd.GetArray('nuHat'));rho=vtk_to_numpy(pd.GetArray('rho'))
    X,Y,Z=p[:,0],p[:,1],p[:,2];sp=np.abs(Y+0.0)<1e-3
    X,Z,nuh,rho=X[sp],Z[sp],nuh[sp],rho[sp];nu=NU0/np.maximum(rho,0.3)
    rr=vtk.vtkXMLPUnstructuredGridReader();rr.SetFileName(f"{d}/surface_fluid_nlf0416.pvtu");rr.Update()
    s2=vtk_to_numpy(rr.GetOutput().GetPoints().GetData());ss=np.abs(s2[:,1]+0.0)<1e-3;sx,sz=s2[ss][:,0],s2[ss][:,2]
    out=[]
    for x0 in xst:
        col=(np.abs(X-x0)<0.01)&(Z>0)
        Xc,Zc,nc,nuc=X[col],Z[col],nuh[col],nu[col]
        dm=np.full(len(Xc),1e9)
        for i in range(len(sx)):dm=np.minimum(dm,(Xc-sx[i])**2+(Zc-sz[i])**2)
        dw=np.sqrt(dm);nw=(dw>0.0008)&(dw<0.004)
        out.append(float((nc[nw]/nuc[nw]).max()) if nw.any() else np.nan)  # peak near-wall chi
    return np.array(out)
xst=np.array([0.10,0.15,0.20,0.25,0.30,0.35])
CL=np.array([res[str(s)]['CL'] for s in steps]);cpLE=[]
chi_t=[]
for s in steps:
    d=f"{F}/f03_{s}";xu,cu=cp_upper(d);cpLE.append(float(np.interp(0.03,xu,cu)))
    chi_t.append(nearwall_chi(d,xst))
cpLE=np.array(cpLE);chi_t=np.array(chi_t)  # [nsnap, nx]
print("=== (1) GLOBAL FEEDBACK: corr(LE Cp @x0.03, CL) = %.2f  (f=1 was -0.94) ==="%np.corrcoef(cpLE,CL)[0,1])
print("\n=== (2) RECEPTIVITY: near-wall chi oscillation by station ===")
print("  x      chi_mean  chi_swing(max-min)  is_turb(mean)  sSlow=0.03^(1-isturb)")
for j,x0 in enumerate(xst):
    c=chi_t[:,j];cm=np.nanmean(c);sw=np.nanmax(c)-np.nanmin(c);it=float(isturb(np.array([cm]))[0]);ss=float(sslow(np.array([cm]))[0])
    print("  %.2f   %6.2f    %6.2f             %.2f          %.3f"%(x0,cm,sw,it,ss))
# the receptivity station = max chi swing
j=int(np.nanargmax([np.nanmax(chi_t[:,k])-np.nanmin(chi_t[:,k]) for k in range(len(xst))]))
print("\n  => receptivity (max chi swing) at x=%.2f: chi_mean=%.2f, is_turb=%.2f, sSlow=%.3f (floor f=0.03)"%(
    xst[j],np.nanmean(chi_t[:,j]),float(isturb(np.array([np.nanmean(chi_t[:,j])]))[0]),float(sslow(np.array([np.nanmean(chi_t[:,j])]))[0])))
