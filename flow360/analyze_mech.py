import json,numpy as np,vtk
from vtk.util.numpy_support import vtk_to_numpy
B="/home/qiqi/flexcompute/aft-sa/flow360"
steps=list(range(13000,15001,250))
res=json.load(open(f"{B}/run_snaps_results.json"))
def surf_upper(d):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/surface_fluid_nlf0416.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData()
    cf=vtk_to_numpy(pd.GetArray('Cf'));cp=vtk_to_numpy(pd.GetArray('Cp'))
    x,z=p[:,0],p[:,2];u=z>1e-6;xs=x[u];o=np.argsort(xs)
    return xs[o],cf[u][o],cp[u][o]
def smooth(v,w=5):
    k=np.ones(w)/w;return np.convolve(v,k,mode='same')
def analyze_surf(d):
    x,cf,cp=surf_upper(d)
    cfs=smooth(cf,7)
    # x_tr = steepest Cf rise in [0.1,0.9]
    win=(x[:-1]>0.1)&(x[:-1]<0.9);dcf=np.diff(cfs);dm=0.5*(x[1:]+x[:-1])
    xtr=float(dm[win][np.argmax(dcf[win])]) if win.any() else np.nan
    # laminar Cf minimum upstream of xtr (bubble depth)
    lam=(x>0.05)&(x<xtr);cfmin=float(cfs[lam].min()) if lam.any() else np.nan
    xcfmin=float(x[lam][np.argmin(cfs[lam])]) if lam.any() else np.nan
    cfturb=float(cfs[(x>xtr)&(x<0.95)].max()) if ((x>xtr)&(x<0.95)).any() else np.nan
    return xtr,cfmin,xcfmin,cfturb
def analyze_vol(d):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/volume.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData()
    mut=vtk_to_numpy(pd.GetArray('mut'));nu=vtk_to_numpy(pd.GetArray('nuHat'));vel=vtk_to_numpy(pd.GetArray('velocity'))
    x,y,z=p[:,0],p[:,1],p[:,2]
    # restrict to near-airfoil upper region (0<x<1.1, 0<z<0.15)
    band=(x>0.0)&(x<1.1)&(z>0)&(z<0.15)
    im=np.argmax(mut[band]);xb=x[band]
    mutmax=float(mut[band][im]);xmut=float(xb[im])
    numax=float(nu[band].max())
    # reversed streamwise flow (u<0) near wall (z<0.05) on upper surface = separation extent
    nw=(x>0.0)&(x<1.0)&(z>0)&(z<0.04)&(vel[:,0]<0)
    xsep=x[nw]
    sep0=float(xsep.min()) if nw.any() else np.nan
    sep1=float(xsep.max()) if nw.any() else np.nan
    nrev=int(nw.sum())
    return mutmax,xmut,numax,sep0,sep1,nrev
rows=[]
for s in steps:
    d=f"{B}/snap_{s}"
    xtr,cfmin,xcfmin,cfturb=analyze_surf(d)
    mutmax,xmut,numax,sep0,sep1,nrev=analyze_vol(d)
    rows.append(dict(step=s,CD=res[str(s)]['CD'],CL=res[str(s)]['CL'],xtr=xtr,
                     cfmin=cfmin,xcfmin=xcfmin,cfturb=cfturb,mutmax=mutmax,xmut=xmut,
                     numax=numax,sep0=sep0,sep1=sep1,nrev=nrev))
print("step    CD      CL     x_tr  Cf_lammin@x   Cf_turb  mut_max@x    nuHmax  sep[x0-x1](n)")
for r in rows:
    sep=f"{r['sep0']:.2f}-{r['sep1']:.2f}({r['nrev']})" if r['nrev']>0 else "none"
    print("%d %.5f %.4f  %.3f  %.4f@%.2f  %.4f  %.1f@%.3f  %.4f  %s"%(
        r['step'],r['CD'],r['CL'],r['xtr'],r['cfmin'],r['xcfmin'],r['cfturb'],
        r['mutmax'],r['xmut'],r['numax'],sep))
json.dump(rows,open(f"{B}/mech_rows.json","w"),indent=1)
