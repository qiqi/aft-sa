"""Define transition as a BAND: x10..x90 of the laminar->turbulent Cf rise (x50 = steepest rise =
the current 'point'). Report band widths for NACA + NLF, both grids, across alpha."""
import numpy as np,vtk
from vtk.util.numpy_support import vtk_to_numpy
B="/home/qiqi/flexcompute/aft-sa/flow360"
def wd(af,mesh,a): return f"{B}/full_{af}_{mesh}_aftsa_m2_a{a}"
def contour_upper(d,af):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/surface_fluid_{af}.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData()
    nm=[pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    cf=vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cf'))));cf=np.linalg.norm(cf,axis=1) if cf.ndim>1 else cf
    x,y,z=p[:,0],p[:,1],p[:,2];s=np.abs(y-y.min())<1e-6;X,Z,CF=x[s],z[s],cf[s];n=len(X);pts=np.column_stack([X,Z])
    st=int(np.argmin(X));o=[st];u=np.zeros(n,bool);u[st]=True
    for _ in range(n-1):
        c=o[-1];dd=np.sum((pts-pts[c])**2,1);dd[u]=1e9;nx=int(np.argmin(dd));o.append(nx);u[nx]=True
    o=np.array(o);xo,zo,cfo=X[o],Z[o],CF[o];te=int(np.argmax(xo));b1,b2=slice(0,te+1),slice(te,n)
    up=b1 if zo[b1].mean()>=zo[b2].mean() else b2;xu,cu=xo[up],cfo[up];oo=np.argsort(xu);return xu[oo],cu[oo]
def band(d,af):
    xu,cu=contour_upper(d,af);k=np.ones(7)/7;cm=np.convolve(cu,k,'same')
    m=(xu>0.03)&(xu<0.95);xx,cc=xu[m],cm[m]
    dc=np.diff(cc);dm=.5*(xx[1:]+xx[:-1]);w=(dm>0.05)&(dm<0.9)
    if not w.any() or dc[w].max()<=0: return None
    i50=np.where(w)[0][np.argmax(dc[w])];x50=dm[i50]
    cflam=cc[xx<=x50].min();           # laminar dip before transition
    ipk=np.argmax(cc[xx>=x50]);cfturb=cc[xx>=x50][ipk]  # turbulent peak after
    if cfturb<=cflam: return None
    lo=cflam+0.1*(cfturb-cflam);hi=cflam+0.9*(cfturb-cflam)
    rise=(xx>=xx[xx<=x50][np.argmin(cc[xx<=x50])])&(xx<=xx[xx>=x50][ipk])  # between dip and peak
    xr,cr=xx[rise],cc[rise]
    def crossx(level):
        idx=np.where(cr>=level)[0]
        return float(xr[idx[0]]) if len(idx) else np.nan
    return crossx(lo),x50,crossx(hi)
for af in ['naca0012','nlf0416']:
    print(f"=== {af} (x10 / x50=point / x90 ; width) ===")
    for a in [0,2,4,6,8]:
        for mesh in ['cavity','ogrid']:
            try:
                b=band(wd(af,mesh,a),af)
                if b: print("  a=%d %-6s  %.3f / %.3f / %.3f   width=%.3f"%(a,mesh,b[0],b[1],b[2],b[2]-b[0]))
            except Exception as e:print("  a=%d %s ERR"%(a,mesh))
