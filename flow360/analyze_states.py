import numpy as np,vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
B="/home/qiqi/flexcompute/aft-sa/flow360"
def surf(d):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/surface_fluid_nlf0416.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData()
    cf=vtk_to_numpy(pd.GetArray('Cf'));cp=vtk_to_numpy(pd.GetArray('Cp'))
    x,z=p[:,0],p[:,2];u=z>1e-6;o=np.argsort(x[u]);return x[u][o],cf[u][o],cp[u][o]
# wall-normal vorticity profile at a rear station x0, upper surface
def prof(d,x0):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/volume.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData()
    vort=vtk_to_numpy(pd.GetArray('vorticity'));vel=vtk_to_numpy(pd.GetArray('velocity'))
    vm=np.linalg.norm(vort,axis=1) if vort.ndim>1 else np.abs(vort)
    x,z=p[:,0],p[:,2];sel=(np.abs(x-x0)<0.01)&(z>0)&(z<0.12)
    zz=z[sel];o=np.argsort(zz);return zz[o],vm[sel][o],vel[sel,0][o]
LAM,TURB="snap_14000","snap_14750"
fig,ax=plt.subplots(1,3,figsize=(11,3.2))
for d,lab,c in [(LAM,'laminar phase (step 14000, $x_{tr}$≈0.90)','C0'),(TURB,'turbulent phase (step 14750, $x_{tr}$≈0.28)','C3')]:
    x,cf,cp=surf(d)
    ax[0].plot(x,-cp,color=c,lw=1.3,label=lab);ax[1].plot(x,cf,color=c,lw=1.3)
ax[0].set_xlabel('x/c');ax[0].set_ylabel('-Cp');ax[0].set_xlim(0,1);ax[0].legend(fontsize=7,frameon=False);ax[0].set_title('pressure')
ax[1].set_xlabel('x/c');ax[1].set_ylabel('Cf');ax[1].set_xlim(0,1);ax[1].set_ylim(-0.001,0.008);ax[1].axhline(0,color='0.6',lw=0.6);ax[1].set_title('skin friction')
# rear-station vorticity profiles (x=0.8)
for d,lab,c in [(LAM,'laminar',  'C0'),(TURB,'turbulent','C3')]:
    zz,vm,u=prof(d,0.8)
    ax[2].plot(vm,zz,color=c,lw=1.3,label=lab)
ax[2].set_xlabel('|vorticity| at x/c=0.8');ax[2].set_ylabel('z (wall-normal)');ax[2].set_ylim(0,0.08);ax[2].legend(fontsize=7,frameon=False);ax[2].set_title('rear BL profile')
plt.tight_layout();plt.savefig('/tmp/mech_states.png',dpi=120);print("wrote /tmp/mech_states.png")
# quantify: near-wall vorticity at x=0.8 (proxy for Re_Omega/amplification at the rear trigger)
for d,lab in [(LAM,'laminar(14000)'),(TURB,'turb(14750)')]:
    zz,vm,u=prof(d,0.8)
    nw=zz<0.01
    print(f"{lab}: near-wall(z<0.01) |vort| max={vm[nw].max():.1f} mean={vm[nw].mean():.1f}; min streamwise u in BL={u.min():.3f} (neg=reversed)")
