"""At the forward-trip trigger region (x~0.2-0.35), probe across the cycle what changes:
surface Cp/Cf, and near-wall vorticity, edge velocity, nuHat, mut. Tests whether the
relaminarization is driven by upstream PRESSURE influence (Cp/edge-vel change) or by the
near-wall profile (vorticity/shear), or by nuHat transport."""
import json,numpy as np,vtk
from vtk.util.numpy_support import vtk_to_numpy
B="/home/qiqi/flexcompute/aft-sa/flow360";steps=list(range(13000,15001,250))
res=json.load(open(f"{B}/run_snaps_results.json"))
def surf(d):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/surface_fluid_nlf0416.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData()
    cf=vtk_to_numpy(pd.GetArray('Cf'));cp=vtk_to_numpy(pd.GetArray('Cp'))
    x,z=p[:,0],p[:,2];u=z>1e-6;o=np.argsort(x[u]);return x[u][o],cf[u][o],cp[u][o],z[u][o]
def vol(d):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/volume.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData()
    vort=vtk_to_numpy(pd.GetArray('vorticity'));vel=vtk_to_numpy(pd.GetArray('velocity'))
    nu=vtk_to_numpy(pd.GetArray('nuHat'));mut=vtk_to_numpy(pd.GetArray('mut'))
    vm=np.linalg.norm(vort,axis=1) if vort.ndim>1 else np.abs(vort)
    return p[:,0],p[:,2],vm,np.linalg.norm(vel,axis=1),nu,mut
def probe(d,x0):
    xs,cf,cp,zs=surf(d)
    # surface values at x0 (upper)
    cp0=float(np.interp(x0,xs,cp));cf0=float(np.interp(x0,xs,cf));zw=float(np.interp(x0,xs,zs))
    X,Z,VM,V,NU,MUT=vol(d)
    col=(np.abs(X-x0)<0.008)&(Z>zw-0.002)
    zc=Z[col]-zw;o=np.argsort(zc);zc=zc[o]
    vmc=VM[col][o];vc=V[col][o];nuc=NU[col][o];mc=MUT[col][o]
    nw=(zc>0.0005)&(zc<0.0025)  # near-wall band ~ where Re_Omega amplification peaks
    vort_nw=float(vmc[nw].mean()) if nw.any() else np.nan
    nu_nw=float(nuc[nw].mean()) if nw.any() else np.nan
    mut_nw=float(mc[nw].mean()) if nw.any() else np.nan
    vedge=float(vc[(zc>0.01)&(zc<0.03)].mean())  # BL-edge speed (pressure proxy)
    d_nw=0.0015;nu_lam=1e-6;reOm=d_nw*d_nw*vort_nw/nu_lam
    return cp0,cf0,vort_nw,vedge,nu_nw,mut_nw,reOm
print("step   phase     Cp@.25  Cf@.25  vort_nw  Vedge   nuHat_nw   mut_nw    Re_Omega")
rows=[]
for s in steps:
    d=f"{B}/snap_{s}";cp0,cf0,vort,ve,nu,mut,reOm=probe(d,0.25)
    ph='TURB' if res[str(s)]['CD']>0.0115 else 'lam'
    print("%d  %-5s  %+.4f  %.4f  %6.1f  %.4f  %.2e  %.2e  %.0f"%(s,ph,cp0,cf0,vort,ve,nu,mut,reOm))
    rows.append(dict(step=s,cp=cp0,cf=cf0,vort=vort,vedge=ve,nu=nu,mut=mut,reOm=reOm,CD=res[str(s)]['CD']))
json.dump(rows,open(f"{B}/trigger_rows.json","w"),indent=1)
# range of each quantity across the cycle (what oscillates most?)
import numpy as np
for k in ['cp','vedge','vort','nu','mut','reOm']:
    a=np.array([r[k] for r in rows]);print("  %-6s range %.3e .. %.3e  (swing %.0f%%)"%(k,a.min(),a.max(),100*(a.max()-a.min())/abs(np.mean(a)+1e-30)))
