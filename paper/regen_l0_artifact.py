"""Sec. V.A figure (figs/l0_artifact.pdf): the coarse-mesh (L0) transition artifact.

Through the two cells preceding transition, the amplification factor chi=max_y nu~/nu
on the structured L0 mesh stalls and dips instead of rising monotonically. The cause
is a collapse of the profile-fullness indicator Gamma at the nu~-peak point -- driven
by the under-resolved transition jump propagating upstream -- which, through the
Gamma-sigmoid gate (g_c=1.572), collapses the amplification rate a. L2 (converged)
holds Gamma~1.1 and rises monotonically. NLF(1)-0416, alpha=4 deg, upper surface.
"""
import os
import vtk, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy as v2n
B=os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_ai"); CHI=8.76e-4; GC=1.005  # = canonical g_c (tests/test_constants_consistency.py)
def load(fn):
    r=vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(fn); r.Update(); return r.GetOutput()
def surfaces(case):
    s=load(f"{B}/{case}/surface_fluid_nlf0416_proc0.vtu"); P=v2n(s.GetPoints().GetData()); x,y,z=P[:,0],P[:,1],P[:,2]; m=np.isclose(y,y.min())
    xs,zs=x[m],z[m]; b=np.linspace(0,1,140); xc=[];zu=[];zl=[]
    for i in range(len(b)-1):
        k=(xs>=b[i])&(xs<b[i+1])
        if k.sum(): xc.append(0.5*(b[i]+b[i+1])); zu.append(zs[k].max()); zl.append(zs[k].min())
    return np.array(xc),np.array(zu),np.array(zl)
def track(case,side='upper',x0=0.14,x1=0.30):
    sl=load(f"{B}/{case}/slice_with_derived_0.vtu"); Q=v2n(sl.GetPoints().GetData())
    nh=v2n(sl.GetPointData().GetArray("nuHat")); wd=v2n(sl.GetPointData().GetArray("wallDistance"))
    G=v2n(sl.GetPointData().GetArray("Gamma")); ar=v2n(sl.GetPointData().GetArray("amp_rate"))
    x,z=Q[:,0],Q[:,2]; nhi=np.median(nh[wd>5]); xc,zu,zl=surfaces(case); cam=0.5*(np.interp(x,xc,zu)+np.interp(x,xc,zl))
    sm=(z>cam) if side=='upper' else (z<cam); X=[];C=[];Gm=[];A=[]
    for xi in np.arange(x0,x1,0.007):
        m=(np.abs(x-xi)<0.006)&sm&(wd>0)&(wd<0.012)
        if m.sum()<2: continue
        j=np.argmax(nh[m]); X.append(xi); C.append(nh[m][j]/nhi*CHI); Gm.append(G[m][j]); A.append(ar[m][j])
    return map(np.array,(X,C,Gm,A))
D={L:track(f"strL{L[1]}prop_nlf0416_Re4M_a4") for L in ['L0','L2']}
plt.rcParams.update({'font.size':12,'axes.labelsize':13})
fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(12.6,3.7))
sty={'L0':dict(color='k',ls='-',marker='o',ms=4),'L2':dict(color='0.55',ls='--',marker='s',ms=4,mfc='white')}
for L in ['L0','L2']:
    X,C,Gm,A=list(D[L]) if False else track(f"strL{L[1]}prop_nlf0416_Re4M_a4")
    a1.semilogy(X,C,lw=1.8,**sty[L],label=f'str {L}')
    a2.plot(X,Gm,lw=1.8,**sty[L]); a3.plot(X,A,lw=1.8,**sty[L])
for ax in (a1,a2,a3): ax.axvspan(0.20,0.24,color='navajowhite',alpha=0.5,lw=0); ax.set_xlim(0.14,0.30); ax.grid(alpha=0.25,which='both'); ax.set_xlabel('$x/c$')
a1.axhline(1,color='0.6',ls=':',lw=0.8); a1.set_ylabel(r'$\chi=\max_y\tilde\nu/\nu$'); a1.legend(fontsize=9,loc='upper left'); a1.set_ylim(1e-3,1e2)
a2.axhline(GC,color='0.6',ls=':',lw=0.8); a2.text(0.105,GC+0.03,r'$g_c$',fontsize=9,color='0.5'); a2.set_ylabel(r'$\Gamma$ at $\tilde\nu$-peak'); a2.set_ylim(0,1.7)
a3.set_ylabel(r'amplification rate $a$ at $\tilde\nu$-peak'); a3.set_ylim(0,0.016)
for ax,l in zip((a1,a2,a3),'abc'): ax.text(0.04,0.94,f'({l})',transform=ax.transAxes,fontsize=13,fontweight='bold',va='top')
plt.tight_layout(); plt.savefig('figs/l0_artifact.pdf'); plt.savefig('/tmp/l0_artifact.png',dpi=150)
print("wrote figs/l0_artifact.pdf")
