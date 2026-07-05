"""Verify the displacement-thickness mechanism: delta*, theta, H=delta*/theta vs x through
transition (tripped vs laminar). Mechanism claim: H/delta* drops sharply at x_tr while theta
is continuous -> sudden displacement squash -> streamline deflection -> upstream pressure."""
import numpy as np,vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
B="/home/qiqi/flexcompute/aft-sa/flow360"
def load(d):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{B}/{d}/volume.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData()
    vel=vtk_to_numpy(pd.GetArray('velocity'))
    X,Y,Z=p[:,0],p[:,1],p[:,2];sp=np.abs(Y+0.0)<1e-3
    rr=vtk.vtkXMLPUnstructuredGridReader();rr.SetFileName(f"{B}/{d}/surface_fluid_nlf0416.pvtu");rr.Update()
    s2=vtk_to_numpy(rr.GetOutput().GetPoints().GetData());ss=np.abs(s2[:,1]+0.0)<1e-3;sx,sz=s2[ss][:,0],s2[ss][:,2]
    return X[sp],Z[sp],np.linalg.norm(vel[sp],axis=1),vel[sp],sx,sz
def blparams(X,Z,Vmag,xstns,sx,sz):
    Hs=[];dstar=[];theta=[]
    for x0 in xstns:
        col=(np.abs(X-x0)<0.004)&(Z>0)
        if col.sum()<6:Hs.append(np.nan);dstar.append(np.nan);theta.append(np.nan);continue
        Xc,Zc,Vc=X[col],Z[col],Vmag[col]
        dm=np.full(len(Xc),1e9)
        for i in range(len(sx)):dm=np.minimum(dm,(Xc-sx[i])**2+(Zc-sz[i])**2)
        dwall=np.sqrt(dm);o=np.argsort(dwall);y=dwall[o];u=Vc[o]
        keep=y<0.1;y,u=y[keep],u[keep]
        if len(y)<6:Hs.append(np.nan);dstar.append(np.nan);theta.append(np.nan);continue
        # BL edge = wall-normal location of the velocity PEAK (suction region: |V| peaks at edge, drops outward)
        iedge=int(np.argmax(u));ue=u[iedge]
        if ue<=0 or iedge<3:Hs.append(np.nan);dstar.append(np.nan);theta.append(np.nan);continue
        ye,ui=y[:iedge+1],u[:iedge+1];r=np.clip(ui/ue,0,1)
        ds=np.trapz(1-r,ye);th=np.trapz(r*(1-r),ye)
        dstar.append(ds);theta.append(th);Hs.append(ds/th if th>1e-9 else np.nan)
    return np.array(dstar),np.array(theta),np.array(Hs)
xst=np.arange(0.10,0.62,0.03)
fig,ax=plt.subplots(1,3,figsize=(13,3.6))
for d,lab,c in [('snap_14000','laminar phase','C0'),('snap_14750','tripped phase','C3')]:
    X,Z,Vm,V,sx,sz=load(d);ds,th,H=blparams(X,Z,Vm,xst,sx,sz)
    ax[0].plot(xst,ds*1000,'o-',color=c,label=lab);ax[1].plot(xst,th*1000,'o-',color=c,label=lab);ax[2].plot(xst,H,'o-',color=c,label=lab)
ax[0].set_ylabel('$\\delta^*\\times10^3$');ax[0].set_title('displacement thickness');ax[0].legend(fontsize=8,frameon=False)
ax[1].set_ylabel('$\\theta\\times10^3$');ax[1].set_title('momentum thickness (continuous?)')
ax[2].set_ylabel('H=$\\delta^*/\\theta$');ax[2].set_title('shape factor')
for a in ax:a.set_xlabel('x/c');a.grid(alpha=0.3)
plt.tight_layout();plt.savefig(f"{B}/diagnostics/bl_params.png",dpi=130);print("wrote diagnostics/bl_params.png")
# print the jump
for d in ['snap_14750']:
    X,Z,Vm,V,sx,sz=load(d);ds,th,H=blparams(X,Z,Vm,xst,sx,sz)
    print("tripped: x, delta*, theta, H"); 
    for i,x in enumerate(xst):print("  %.2f  d*=%.4f th=%.4f H=%.2f"%(x,ds[i],th[i],H[i]))
