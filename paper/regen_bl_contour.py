"""Unrolled boundary-layer contours for the Eppler 387.
x-axis: s = arc length along the surface (TE -> around LE -> TE, lower then upper).
y-axis: wall-normal distance (from the connectivity walk + normal-probe shoot).
Panels: Re_Omega, Gamma, omega, the nondim rate a(Re_Omega,Gamma), and chi=nuHat/nu.
Usage: python regen_bl_contour.py <case_dir> <out.png> [nu]
"""
import sys, os; sys.path.insert(0,'.')
import numpy as np, vtk, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import regen_eppler_v2 as R

CASE=sys.argv[1]; OUT=sys.argv[2]; NU=float(sys.argv[3]) if len(sys.argv)>3 else 5e-7
L_PROBE=0.04; NP=140

Xm,Zm,up_idx,lo_idx = R.walk_contour_xz(CASE)
lo=np.array(lo_idx); lo=lo[np.argsort(-Xm[lo])]   # lower: TE(x=1) -> LE(x=0)
up=np.array(up_idx); up=up[np.argsort(Xm[up])]    # upper: LE(x=0) -> TE(x=1)
full=np.concatenate([lo,up]); xs=Xm[full]; zs=Zm[full]
s=np.concatenate([[0],np.cumsum(np.hypot(np.diff(xs),np.diff(zs)))])
s_LE=s[np.argmin(xs)]
# outward normal: perpendicular to tangent, z-sign = away from camber
tx=np.gradient(xs); tz=np.gradient(zs); tn=np.hypot(tx,tz)+1e-30; tx/=tn; tz/=tn
nx=tz.copy(); nz=-tx.copy()
zc_up,zc_lo,_=R.airfoil_surfaces(CASE); cam=0.5*(zc_up(xs)+zc_lo(xs))
outsgn=np.sign(zs-cam); outsgn[outsgn==0]=1
flip=np.sign(nz)!=outsgn; nx[flip]*=-1; nz[flip]*=-1
# probe grid
M=len(xs); dists=np.linspace(1e-6,L_PROBE,NP); y0=R.slice_y_plane(CASE)
pts=np.empty((M*NP,3))
for j in range(NP):
    pts[j*M:(j+1)*M,0]=xs+dists[j]*nx; pts[j*M:(j+1)*M,1]=y0; pts[j*M:(j+1)*M,2]=zs+dists[j]*nz
vp=vtk.vtkPoints(); vp.SetData(numpy_to_vtk(pts,deep=True)); poly=vtk.vtkPolyData(); poly.SetPoints(vp)
r=vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{CASE}/slice_centerSpan.pvtu"); r.Update()
pr=vtk.vtkProbeFilter(); pr.SetInputData(poly); pr.SetSourceData(r.GetOutput()); pr.Update()
pd=pr.GetOutput().GetPointData()
nh=vtk_to_numpy(pd.GetArray('nuHat')); vm=vtk_to_numpy(pd.GetArray('vorticityMagnitude'))
U=np.linalg.norm(vtk_to_numpy(pd.GetArray('velocity')),axis=1)
valid=vtk_to_numpy(pr.GetValidPoints()); mask=np.zeros(M*NP,bool); mask[valid]=True
rs=lambda a: np.where(mask,a,np.nan).reshape(NP,M)
D=np.repeat(dists[:,None],M,axis=1)
om=rs(vm); chi=rs(nh)/NU; Uu=rs(U)
ReO=D*D*om/NU; omd=om*D; Gam=2*omd**2/(Uu**2+omd**2+1e-30); a=R.rate(ReO,Gam)

S,Y=np.meshgrid(s,dists)
panels=[('$Re_\\Omega$ (log)',ReO,LogNorm(1e0,1e4),'viridis',R.RE_OMEGA_FLOOR),
        ('$\\omega$ (log)',om,LogNorm(1e-2,1e2),'viridis',None),
        ('$\\Gamma$',Gam,None,'coolwarm',R.G_C),
        ('$a$ (rate)',a,None,'inferno',None),
        ('$\\chi=\\tilde\\nu/\\nu$ (log)',chi,LogNorm(1e-4,1e2),'magma',1.0)]
fig,axs=plt.subplots(len(panels),1,figsize=(12,13),sharex=True)
for ax,(name,F,norm,cmap,mark) in zip(axs,panels):
    if norm is None:
        vmax=2.05 if 'Gamma' in name else (0.15 if name=='$a$ (rate)' else np.nanmax(F))
        pc=ax.pcolormesh(S,Y,F,cmap=cmap,vmin=0,vmax=vmax,shading='auto')
    else:
        pc=ax.pcolormesh(S,Y,F,cmap=cmap,norm=norm,shading='auto')
    fig.colorbar(pc,ax=ax,pad=0.01,label=name)
    if mark is not None and norm is not None:
        ax.contour(S,Y,F,levels=[mark],colors='w',linewidths=1.0)
    elif mark is not None:
        ax.contour(S,Y,F,levels=[mark],colors='k',linewidths=0.8)
    ax.axvline(s_LE,color='r',ls='--',lw=0.8)
    ax.set_ylabel('wall dist')
    ax.set_title(name,fontsize=9,loc='left')
axs[-1].set_xlabel('s (arc length: TE $\\to$ LE $\\to$ TE); red dashed = LE')
fig.suptitle(f'BL contours (unrolled): {os.path.basename(CASE)}',y=0.995,fontsize=12)
plt.tight_layout(rect=(0,0,1,0.99)); plt.savefig(OUT,dpi=120); print('wrote',OUT,'| M=%d s_LE=%.3f'%(M,s_LE))
