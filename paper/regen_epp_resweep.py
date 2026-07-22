"""Eppler 387, alpha=5, Reynolds sweep -- 5-row layout, one column per Re.
ALL of Re_Omega, Gamma, chi are computed by the SAME connectivity-walk +
wall-normal-probe method (on slice_centerSpan.pvtu, which carries velocity,
vorticityMagnitude, nuHat, wallDistance), so the upper/lower curves are
consistent and MEET at the LE. Row 3 chi = max_normal(nuHat)/nu.
-> /tmp/epp_resweep.png
"""
import sys, os; sys.path.insert(0, '.')
import numpy as np, vtk, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import regen_eppler_v2 as R

B="/home/qiqi/flexcompute/sa-ai/flow360"
MODE=sys.argv[1] if len(sys.argv)>1 else "orig"
def case_dir(Rk):
    if Rk==200: return f"{B}/cavL1prop_eppler387_Re200k_a5"
    if MODE=="sweep": return f"{B}/sweep_Re{Rk}k_a5"   # staged-fSlow, AI_SA=1 (2026-07-05 corrected sweep)
    if MODE in ("warm","cold"): return f"{B}/{MODE}_eppler_Re{Rk}k_a5"
    return f"{B}/cavL1prop_eppler387_Re{Rk}k_a5"
COLS=[(60,"Re60k",1.6667e-6),(100,"Re100k",1e-6),(200,"Re200k",5e-7),(300,"Re300k",3.3333e-7),(460,"Re460k",2.1739e-7)]
UP,LO=R.UP_COLOR,R.LO_COLOR

def normal_metrics(case_d, side, nu, L_probe=0.03, n_probe=120):
    """Connectivity walk + outward normal probe on slice_centerSpan. Returns
    (x, ReO_max, Gam_max, chi_max) along the surface, max taken over the probe line."""
    Xm,Zm,up_idx,lo_idx = R.walk_contour_xz(case_d)
    idx = up_idx if side=='upper' else lo_idx
    xs=Xm[idx]; zs=Zm[idx]
    tx=np.gradient(xs); tz=np.gradient(zs); s=np.hypot(tx,tz)+1e-30; tx/=s; tz/=s
    nx=tz; nz=-tx
    if side=='upper' and np.mean(nz)<0: nx,nz=-nx,-nz
    if side=='lower' and np.mean(nz)>0: nx,nz=-nx,-nz
    M=len(xs); dists=np.linspace(1e-6,L_probe,n_probe); y0=R.slice_y_plane(case_d)
    pts=np.empty((M*n_probe,3))
    for j in range(n_probe):
        pts[j*M:(j+1)*M,0]=xs+dists[j]*nx; pts[j*M:(j+1)*M,1]=y0; pts[j*M:(j+1)*M,2]=zs+dists[j]*nz
    vp=vtk.vtkPoints(); vp.SetData(numpy_to_vtk(pts,deep=True)); poly=vtk.vtkPolyData(); poly.SetPoints(vp)
    r=vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{case_d}/slice_centerSpan.pvtu"); r.Update()
    pr=vtk.vtkProbeFilter(); pr.SetInputData(poly); pr.SetSourceData(r.GetOutput()); pr.Update()
    o=pr.GetOutput(); pd=o.GetPointData()
    nh=vtk_to_numpy(pd.GetArray('nuHat')); vm=vtk_to_numpy(pd.GetArray('vorticityMagnitude'))
    vel=vtk_to_numpy(pd.GetArray('velocity')); U=np.linalg.norm(vel,axis=1)
    wd=dists[np.arange(M*n_probe)//M]  # probe distance = wall distance along the ray
    valid=vtk_to_numpy(pr.GetValidPoints()); mask=np.zeros(M*n_probe,bool); mask[valid]=True
    ReO=wd*wd*vm/nu; omd=vm*wd; Gam=2*omd**2/(U**2+omd**2+1e-30); chi=nh/nu
    def mx(a): return np.nanmax(np.where(mask,a,np.nan).reshape(n_probe,M),axis=0)
    order=np.argsort(xs)
    return xs[order], mx(ReO)[order], mx(Gam)[order], mx(chi)[order]

fig,axs=plt.subplots(5,len(COLS),figsize=(4.4*len(COLS),13),sharex=True)
for col,(Rk,tag,nu) in enumerate(COLS):
    d=case_dir(Rk)
    ax_reo,ax_gam,ax_n,ax_cp,ax_cf=axs[0,col],axs[1,col],axs[2,col],axs[3,col],axs[4,col]
    for side,c in [('upper',UP),('lower',LO)]:
        try:
            x,ReO,Gam,chi=normal_metrics(d,side,nu)
            ax_reo.semilogy(x,ReO,'-',lw=1.5,color=c); ax_gam.plot(x,Gam,'-',lw=1.5,color=c)
            ax_n.semilogy(x,chi,'-',lw=1.5,color=c)
        except Exception as e: print(f"{tag} {side}: {e}")
    ax_reo.axhline(R.RE_OMEGA_FLOOR,color='gray',ls='--',lw=0.6,alpha=0.5); ax_reo.set_ylim(1e2,1e4)
    ax_reo.grid(alpha=0.3,which='both'); ax_reo.set_title(f"Re = {Rk}k",fontsize=11)
    ax_gam.axhline(R.G_C,color='gray',ls=':',lw=0.7,alpha=0.7); ax_gam.set_ylim(0,2.05); ax_gam.grid(alpha=0.3)
    ax_n.axhline(R.C_V1,color='gray',ls=':',lw=0.6,alpha=0.6); ax_n.axhline(1.0,color='gray',lw=0.5,alpha=0.4)
    ax_n.axhline(8.76e-4,color='green',ls=':',lw=0.8,alpha=0.7)  # seed chi_inf
    ax_n.set_ylim(R.CHI_LO,R.CHI_HI); ax_n.grid(alpha=0.3,which='both')
    if os.path.exists(f"{d}/surface_fluid_eppler387.pvtu"):
        try:
            (xu,cfu,cpu),(xl,cfl,cpl)=R.airfoil_walk_contour(d)
            ax_cp.plot(xu,-cpu,'-',lw=1.5,color=UP); ax_cp.plot(xl,-cpl,'-',lw=1.5,color=LO)
            ax_cf.plot(xu,cfu,'-',lw=1.5,color=UP); ax_cf.plot(xl,cfl,'-',lw=1.5,color=LO)
        except Exception as e: print(f"{tag} surf: {e}")
    if Rk==200:
        xls,xtr=R.EXP_LSB[5]
        for a in (ax_cp,ax_cf): a.axvspan(xls,xtr,color='0.55',alpha=0.30,zorder=0)
    ax_cf.set_ylim(-0.004,0.012); ax_cf.axhline(0,color='gray',lw=0.6,alpha=0.5)
    for a in (ax_reo,ax_gam,ax_n,ax_cp,ax_cf): a.set_xlim(0,1)
    ax_cp.grid(alpha=0.3); ax_cf.grid(alpha=0.3); ax_cf.set_xlabel('$x/c$')
for r,lab in enumerate([r'$\max Re_\Omega$',r'$\max\Gamma$',r'$\chi=\tilde\nu/\nu$ (log)',r'$-C_p$',r'$C_f$']):
    axs[r,0].set_ylabel(lab)
axs[0,-1].legend(handles=[Line2D([],[],color=UP,lw=2,label='upper'),Line2D([],[],color=LO,lw=2,label='lower')],fontsize=9,frameon=False,loc='upper left')
axs[2,-1].legend(handles=[Line2D([],[],color='green',ls=':',label=r'seed $\chi_\infty$')],fontsize=8,frameon=False,loc='lower right')
fig.suptitle(f'Eppler 387, alpha=5, Re sweep ({MODE}, fSlow=1) — chi via normal-probe walk',fontsize=13,y=0.995)
plt.tight_layout(rect=(0,0,1,0.985)); plt.savefig(f'/tmp/epp_resweep_{MODE}.png',dpi=110); print(f'wrote /tmp/epp_resweep_{MODE}.png')
