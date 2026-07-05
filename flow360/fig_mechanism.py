import json,numpy as np,vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
B="/home/qiqi/flexcompute/aft-sa/flow360"
cyc=json.load(open(f"{B}/mech_rows2.json"));trg=json.load(open(f"{B}/trigger_rows.json"))
st=np.array([r['step'] for r in cyc]);CD=np.array([r['CD'] for r in cyc])*1000
xtr=np.array([r['xtr'] for r in cyc]);mr=np.array([r['mrpk'] for r in cyc])
def surf(d):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/surface_fluid_nlf0416.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData()
    cf=vtk_to_numpy(pd.GetArray('Cf'));x,z=p[:,0],p[:,2];u=z>1e-6;o=np.argsort(x[u]);return x[u][o],cf[u][o]
fig,ax=plt.subplots(1,3,figsize=(12,3.4))
# A: cycle
ax[0].plot(st,CD,'ko-',lw=1.3,label='$C_D\\times10^3$');ax[0].set_ylabel('$C_D\\times10^3$');ax[0].set_xlabel('pseudo-step')
axb=ax[0].twinx();axb.plot(st,xtr,'s--',color='C3',lw=1.3);axb.set_ylabel('$x_{tr}/c$',color='C3');axb.set_ylim(0,1);axb.tick_params(axis='y',colors='C3')
ax[0].set_title('(a) one period: drag $\\leftrightarrow$ transition',fontsize=10)
ax[0].annotate('TURB\n(fwd trip)',xy=(13000,12.5),fontsize=7,color='0.3');ax[0].annotate('LAM\n(aft)',xy=(13900,10.1),fontsize=7,color='0.3')
# B: two states Cf
for d,lab,c in [('snap_14000','laminar phase ($x_{tr}{\\approx}0.9$)','C0'),('snap_14750','turbulent phase ($x_{tr}{\\approx}0.28$)','C3')]:
    x,cf=surf(d);ax[1].plot(x,cf,color=c,lw=1.3,label=lab)
ax[1].set_xlim(0,1);ax[1].set_ylim(-0.0005,0.007);ax[1].axhline(0,color='0.7',lw=0.5)
ax[1].set_xlabel('$x/c$');ax[1].set_ylabel('$C_f$ (upper)');ax[1].legend(fontsize=7,frameon=False);ax[1].set_title('(b) two bistable states',fontsize=10)
# C: what swings at the trigger (normalized to mean) -> pressure flat, nuHat/mut huge
def nrm(k):
    a=np.array([r[k] for r in trg]);return a/np.mean(a)
sx=np.array([r['step'] for r in trg])
ax[2].plot(sx,nrm('cp'),'o-',color='C2',lw=1.3,label='$C_p$ (2%)')
ax[2].plot(sx,nrm('reOm'),'s-',color='C4',lw=1.3,label='$Re_\\Omega$ (7%)')
ax[2].plot(sx,nrm('nu'),'^-',color='C1',lw=1.5,label='$\\tilde\\nu$ (236%)')
ax[2].plot(sx,nrm('mut'),'v-',color='C3',lw=1.5,label='$\\mu_t$ (769%)')
ax[2].set_yscale('log');ax[2].set_xlabel('pseudo-step');ax[2].set_ylabel('value / cycle-mean')
ax[2].legend(fontsize=7,frameon=False,loc='upper left');ax[2].set_title('(c) trigger inputs @ $x{=}0.25$: $\\tilde\\nu$ swings, $p$ frozen',fontsize=9.5)
plt.tight_layout();plt.savefig(f"{B}/diagnostics/mechanism_summary.png",dpi=130)
print("wrote diagnostics/mechanism_summary.png")
