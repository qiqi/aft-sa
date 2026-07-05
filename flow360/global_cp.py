"""Test LOCAL vs GLOBAL: does upper Cp oscillate in phase from LE to TE, tracking CL (circulation)?
Global circulation feedback => all stations swing together ~proportional to dCL/CL, incl. the LE
(far upstream of transition at x~0.3, where a LOCAL influence could not reach)."""
import json,glob,os,csv,numpy as np,vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
B="/home/qiqi/flexcompute/aft-sa/flow360"
def cp_upper(d):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/surface_fluid_nlf0416.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());cp=vtk_to_numpy(g.GetPointData().GetArray('Cp'))
    y=p[:,1];s=np.abs(y+0.0)<1e-3;X,Z,Cp=p[s][:,0],p[s][:,2],cp[s];n=len(X);pts=np.column_stack([X,Z])
    st=int(np.argmin(X));order=[st];used=np.zeros(n,bool);used[st]=True
    for _ in range(n-1):
        cur=order[-1];dd=np.sum((pts-pts[cur])**2,1);dd[used]=1e9;nx=int(np.argmin(dd));order.append(nx);used[nx]=True
    order=np.array(order);xo,zo,co=X[order],Z[order],Cp[order];te=int(np.argmax(xo))
    b1,b2=slice(0,te+1),slice(te,n);up=b1 if zo[b1].mean()>zo[b2].mean() else b2
    xu,cu=xo[up],co[up];o=np.argsort(xu);return xu[o],cu[o]
def CL(d):
    rows=list(csv.reader(open(f"{d}/total_forces_v2.csv")));h=[x.strip() for x in rows[0]]
    last=[c.strip() for c in rows[-1] if c.strip()!=''];return float(last[h.index('CL')])
steps=sorted([int(os.path.basename(p).split('_')[1]) for p in glob.glob(f"{B}/snap_*") if os.path.isdir(p) and os.path.basename(p).split('_')[1].isdigit()])
stns=[0.03,0.05,0.10,0.25,0.45,0.80]
cps={s:[] for s in stns};cls=[]
for st in steps:
    d=f"{B}/snap_{st}";xu,cu=cp_upper(d);cls.append(CL(d))
    for s in stns:cps[s].append(float(np.interp(s,xu,cu)))
cls=np.array(cls)
print("station   Cp swing(%)   corr(Cp,CL)")
for s in stns:
    a=np.array(cps[s]);sw=100*(a.max()-a.min())/abs(a.mean());cc=np.corrcoef(a,cls)[0,1]
    print("  x=%.2f   %5.1f%%      %+.2f"%(s,sw,cc))
print("CL: %.4f .. %.4f  (swing %.1f%% -> effective dAlpha ~ %.2f deg @ dCL/dA~0.11/deg)"%(cls.min(),cls.max(),100*(cls.max()-cls.min())/cls.mean(),(cls.max()-cls.min())/0.11))
# plot
fig,ax=plt.subplots(1,2,figsize=(11,4))
st=np.array(steps)
for s in stns:
    a=np.array(cps[s]);ax[0].plot(st,(a-a.mean())/abs(a.mean())*100,'o-',ms=3,label='x=%.2f'%s)
ax[0].plot(st,(cls-cls.mean())/cls.mean()*100,'k--',lw=2,label='CL')
ax[0].set_xlabel('pseudo-step');ax[0].set_ylabel('% deviation from mean');ax[0].legend(fontsize=7,frameon=False,ncol=2);ax[0].set_title('upper Cp at stations vs CL: in phase = GLOBAL')
for s in stns:
    a=np.array(cps[s]);ax[1].plot(cls,a,'o',ms=4,label='x=%.2f'%s)
ax[1].set_xlabel('CL');ax[1].set_ylabel('upper Cp');ax[1].legend(fontsize=7,frameon=False,ncol=2);ax[1].set_title('Cp vs CL (tight line = circulation-driven)')
plt.tight_layout();plt.savefig(f"{B}/diagnostics/global_cp.png",dpi=130);print("wrote diagnostics/global_cp.png")
