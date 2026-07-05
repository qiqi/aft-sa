"""Self-contained f=0.03 geom mechanism analysis on completed snapshots."""
import glob,os,csv,numpy as np,vtk
from vtk.util.numpy_support import vtk_to_numpy
F="/home/qiqi/flexcompute/aft-sa/flow360";NU0=2e-7;FSLOW=0.03;swC=1.0;swW=4.0
def isturb(chi): return 0.0 if chi<=swC else 1.0-np.exp(-(chi-swC)/swW)
def sslow(chi): return FSLOW**(1.0-isturb(chi))
def CLof(d):
    rows=list(csv.reader(open(f"{d}/total_forces_v2.csv")));h=[x.strip() for x in rows[0]];last=[c for c in rows[-1] if c.strip()!='']
    return float(last[h.index('CL')]),float(last[h.index('CD')]),int(float(last[h.index('pseudo_step')]))
def cp_at(d,x0):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/surface_fluid_nlf0416.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());cp=vtk_to_numpy(g.GetPointData().GetArray('Cp'))
    y=p[:,1];s=np.abs(y+0.0)<1e-3;X,Z,Cp=p[s][:,0],p[s][:,2],cp[s];n=len(X);pts=np.column_stack([X,Z])
    st=int(np.argmin(X));o=[st];u=np.zeros(n,bool);u[st]=True
    for _ in range(n-1):
        c=o[-1];dd=np.sum((pts-pts[c])**2,1);dd[u]=1e9;nx=int(np.argmin(dd));o.append(nx);u[nx]=True
    o=np.array(o);xo,zo,co=X[o],Z[o],Cp[o];te=int(np.argmax(xo));b1,b2=slice(0,te+1),slice(te,n)
    up=b1 if zo[b1].mean()>zo[b2].mean() else b2;xu,cu=xo[up],co[up];oo=np.argsort(xu);return float(np.interp(x0,xu[oo],cu[oo]))
def chi_prof(d,xst):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/volume.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData()
    nuh=vtk_to_numpy(pd.GetArray('nuHat'));rho=vtk_to_numpy(pd.GetArray('rho'))
    X,Y,Z=p[:,0],p[:,1],p[:,2];sp=np.abs(Y+0.0)<1e-3;X,Z,nuh,rho=X[sp],Z[sp],nuh[sp],rho[sp];nu=NU0/np.maximum(rho,0.3)
    rr=vtk.vtkXMLPUnstructuredGridReader();rr.SetFileName(f"{d}/surface_fluid_nlf0416.pvtu");rr.Update()
    s2=vtk_to_numpy(rr.GetOutput().GetPoints().GetData());ss=np.abs(s2[:,1]+0.0)<1e-3;sx,sz=s2[ss][:,0],s2[ss][:,2]
    out=[]
    for x0 in xst:
        col=(np.abs(X-x0)<0.01)&(Z>0);Xc,Zc,nc,nuc=X[col],Z[col],nuh[col],nu[col]
        dm=np.full(len(Xc),1e9)
        for i in range(len(sx)):dm=np.minimum(dm,(Xc-sx[i])**2+(Zc-sz[i])**2)
        dw=np.sqrt(dm);nw=(dw>0.0008)&(dw<0.004)
        out.append(float((nc[nw]/nuc[nw]).max()) if nw.any() else np.nan)
    return np.array(out)
# completed snapshots = dirs with a non-empty volume.pvtu
dirs=[]
for d in sorted(glob.glob(f"{F}/f03_4*")):
    if os.path.isdir(d) and os.path.exists(f"{d}/volume.pvtu") and os.path.getsize(f"{d}/volume.pvtu")>0:
        try:
            cl,cd,ps=CLof(d)
            if ps>=int(os.path.basename(d).split('_')[1])-5: dirs.append((d,cl,cd,ps))
        except:pass
dirs=sorted(dirs,key=lambda t:t[3])
print("snapshots used:",[os.path.basename(d) for d,_,_,_ in dirs])
xst=np.array([0.10,0.15,0.20,0.25,0.30,0.35]);CL=np.array([t[1] for t in dirs]);CD=np.array([t[2] for t in dirs])
cpLE=np.array([cp_at(d,0.03) for d,_,_,_ in dirs])
chi=np.array([chi_prof(d,xst) for d,_,_,_ in dirs])
print("(1) GLOBAL FEEDBACK: corr(LE Cp@0.03, CL)=%.2f ; corr(CD,CL)=%.2f  [f=1: LECp/CL=-0.94, CD/CL quad]"%(np.corrcoef(cpLE,CL)[0,1],np.corrcoef(CD,CL)[0,1]))
print("    CL range %.4f-%.4f ; LE Cp range %.4f-%.4f (%.1f%% swing)"%(CL.min(),CL.max(),cpLE.min(),cpLE.max(),100*(cpLE.max()-cpLE.min())/abs(cpLE.mean())))
print("\n(2)+(3) RECEPTIVITY by station (near-wall peak chi over the snapshots):")
print("  x      chi_mean  chi_swing   is_turb(mn)  sSlow=0.03^(1-isturb)")
for j,x0 in enumerate(xst):
    c=chi[:,j];cm=np.nanmean(c);sw=np.nanmax(c)-np.nanmin(c)
    print("  %.2f   %6.2f    %6.2f      %.2f         %.3f"%(x0,cm,sw,isturb(cm),sslow(cm)))
j=int(np.nanargmax([np.nanmax(chi[:,k])-np.nanmin(chi[:,k]) for k in range(len(xst))]))
cm=np.nanmean(chi[:,j])
print("\n  => RECEPTIVITY (max chi swing) at x=%.2f: chi_mean=%.2f, is_turb=%.2f, sSlow=%.3f"%(xst[j],cm,isturb(cm),sslow(cm)))
print("     (floor sSlow at chi<1 = f = 0.030; if sSlow here >> 0.03, the slowdown is NOT reaching the receptivity)")
