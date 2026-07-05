"""Convergence plots: nuHat & continuity residual + CD vs pseudo-step, for the slowdown runs,
baseline (f=1 limit cycle), and the evalFreq=50 route."""
import csv,os,numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
B="/home/qiqi/flexcompute/aft-sa/flow360"
def load(d):
    if not os.path.exists(f"{B}/{d}/nonlinear_residual_v2.csv"):return None
    rr=list(csv.reader(open(f"{B}/{d}/nonlinear_residual_v2.csv")));rh=[x.strip() for x in rr[0]]
    ip,ic,inu=rh.index('pseudo_step'),rh.index('0_cont'),rh.index('5_nuHat');s,c,n=[],[],[]
    for r in rr[1:]:
        r=[x for x in r if x.strip()!='']
        if len(r)>inu:
            try:s.append(float(r[ip]));c.append(float(r[ic]));n.append(float(r[inu]))
            except:pass
    fr=list(csv.reader(open(f"{B}/{d}/total_forces_v2.csv")));fh=[x.strip() for x in fr[0]];fip,icd=fh.index('pseudo_step'),fh.index('CD');fs,cd=[],[]
    for r in fr[1:]:
        r=[x for x in r if x.strip()!='']
        if len(r)>icd:
            try:fs.append(float(r[fip]));cd.append(float(r[icd]))
            except:pass
    return map(np.array,(s,c,n,fs,cd))
cases=[('slow_1.0','f=1.0 (off): limit cycle','C3','-'),
       ('slow_0.03','f=0.03 (33x, arith)','C1','-'),
       ('slow_0.01','f=0.01 (100x, arith)','C0','-'),
       ('lim_B_ef50','evalFreq=50','k','--')]
for extra in ['slow2_0.005','slow2_0.008']:
    if os.path.exists(f"{B}/{extra}/total_forces_v2.csv"):
        cases.append((extra,extra.replace('slow2_','f=')+' (arith)','C2','-'))
fig,ax=plt.subplots(2,1,figsize=(9,7),sharex=True)
for d,lab,c,ls in cases:
    r=load(d)
    if r is None:continue
    s,cont,nu,fs,cd=r
    ax[0].semilogy(s,nu,ls,color=c,lw=1.2,label=lab)
    ax[1].plot(fs,cd,ls,color=c,lw=1.0)
ax[0].set_ylabel(r'$\tilde\nu$ residual (RMS)');ax[0].legend(fontsize=8,frameon=False);ax[0].set_title('Convergence: O-grid NLF $\\alpha=4$ — laminar slowdown vs evalFreq')
ax[0].grid(alpha=0.3,which='both')
ax[1].set_ylabel('$C_D$');ax[1].set_xlabel('pseudo-step');ax[1].grid(alpha=0.3);ax[1].set_ylim(0.009,0.014)
plt.tight_layout();plt.savefig(f"{B}/diagnostics/convergence_slowdown.png",dpi=130);print("wrote diagnostics/convergence_slowdown.png")
