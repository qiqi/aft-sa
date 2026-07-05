import json,numpy as np,vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
B="/home/qiqi/flexcompute/aft-sa/flow360";steps=list(range(13000,15001,250))
res=json.load(open(f"{B}/run_snaps_results.json"))
def vol(d):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/volume.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData()
    mr=vtk_to_numpy(pd.GetArray('mutRatio'));vel=vtk_to_numpy(pd.GetArray('velocity'))
    return p[:,0],p[:,2],mr,vel
def surf_xtr(d):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/surface_fluid_nlf0416.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());cf=vtk_to_numpy(g.GetPointData().GetArray('Cf'))
    x,z=p[:,0],p[:,2];u=z>1e-6;xs=x[u];o=np.argsort(xs);xs,cfs=xs[o],cf[u][o]
    k=np.ones(7)/7;cfm=np.convolve(cfs,k,mode='same');dcf=np.diff(cfm);dm=0.5*(xs[1:]+xs[:-1])
    w=(dm>0.1)&(dm<0.9);return float(dm[w][np.argmax(dcf[w])])
rows=[]
for s in steps:
    d=f"{B}/snap_{s}";x,z,mr,vel=vol(d)
    up=(z>0)&(z<0.15)&(x>0)&(x<1.1)
    mrpk=float(mr[up].max());xmr=float(x[up][np.argmax(mr[up])])
    # turbulent front: most-forward x where mutRatio>3 in a near-wall band
    nw=(z>0)&(z<0.05)&(x>0.05)&(x<1.0);turb=nw&(mr>3)
    xfront=float(x[turb].min()) if turb.any() else np.nan
    # upper-surface rear reversed flow (separation), x>0.3
    rev=(z>0)&(z<0.04)&(x>0.3)&(x<1.0)&(vel[:,0]<0)
    sepn=int(rev.sum());sx0=float(x[rev].min()) if sepn else np.nan;sx1=float(x[rev].max()) if sepn else np.nan
    rows.append(dict(step=s,CD=res[str(s)]['CD'],CL=res[str(s)]['CL'],xtr=surf_xtr(d),
                     mrpk=mrpk,xmr=xmr,xfront=xfront,sepn=sepn,sx0=sx0,sx1=sx1))
print("step    CD      x_tr  mutRatio_pk@x   turbfront  rear-sep[x0-x1](n)")
for r in rows:
    sep=f"{r['sx0']:.2f}-{r['sx1']:.2f}({r['sepn']})" if r['sepn'] else "none"
    ff=f"{r['xfront']:.2f}" if r['xfront']==r['xfront'] else "  - "
    print("%d %.5f  %.3f  %6.1f@%.2f   %s        %s"%(r['step'],r['CD'],r['xtr'],r['mrpk'],r['xmr'],ff,sep))
json.dump(rows,open(f"{B}/mech_rows2.json","w"),indent=1)
# plot the cycle
st=np.array([r['step'] for r in rows]);CD=np.array([r['CD'] for r in rows])
xtr=np.array([r['xtr'] for r in rows]);mrpk=np.array([r['mrpk'] for r in rows]);sepn=np.array([r['sepn'] for r in rows])
fig,ax=plt.subplots(2,1,figsize=(7,5),sharex=True)
ax[0].plot(st,CD*1000,'ko-',label='CD x1000');ax[0].set_ylabel('CD x1000');axb=ax[0].twinx()
axb.plot(st,xtr,'s--',color='C3',label='x_tr');axb.set_ylabel('x_tr/c',color='C3');axb.set_ylim(0,1)
ax[0].set_title('NLF O-grid a=4 limit cycle: drag vs transition location')
ax[1].plot(st,mrpk,'^-',color='C0',label='peak mutRatio');ax[1].set_ylabel('peak mutRatio',color='C0')
axc=ax[1].twinx();axc.plot(st,sepn,'v--',color='C2',label='rear reversed-flow pts');axc.set_ylabel('rear sep pts',color='C2')
ax[1].set_xlabel('pseudo-step')
plt.tight_layout();plt.savefig('/tmp/mech_cycle.png',dpi=120);print("wrote /tmp/mech_cycle.png")
