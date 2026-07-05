"""Confirm: (1) clean Limit A via SLOW NS CFL (not high SA CFL); (2) generalize the ef=50 fix
to a=6,8. If clean Limit A still oscillates -> asymmetric (only mean-flow-fast stabilizes)."""
import os,json,shutil,threading,csv,numpy as np,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env;from rans.solve import run_solver
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
def stage(wd,evalfreq,cflmult,nscfl,steps,alpha):
    shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
    for f in os.listdir(f"{F}/base_nlf_ogrid"):
        if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
        s,d=f"{F}/base_nlf_ogrid/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    c=json.load(open(f"{wd}/Flow360.json"));c['freestream']['alphaAngle']=float(alpha);c['timeStepping']['maxPseudoSteps']=int(steps)
    c['turbulenceModelSolver']['equationEvalFrequency']=int(evalfreq);c['turbulenceModelSolver']['CFLMultiplier']=float(cflmult)
    if nscfl is not None:c['timeStepping']['CFL']['max']=float(nscfl)
    json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1);return wd
def stats(wd):
    rows=list(csv.reader(open(f"{wd}/total_forces_v2.csv")));h=[x.strip() for x in rows[0]];ip,icd=h.index('pseudo_step'),h.index('CD');s,cd=[],[]
    for r in rows[1:]:
        r=[c for c in r if c.strip()!='']
        if len(r)>icd:
            try:s.append(float(r[ip]));cd.append(float(r[icd]))
            except:pass
    s,cd=np.array(s),np.array(cd);m=s>0.6*s.max()
    rr=list(csv.reader(open(f"{wd}/nonlinear_residual_v2.csv")));rh=[x.strip() for x in rr[0]];rl=[c for c in rr[-1] if c.strip()!='']
    return cd[m].mean(),cd[m].std(),float(rl[rh.index('0_cont')])
res={};lock=threading.Lock();sem=threading.Semaphore(8)
def run(tag,ef,cm,nscfl,steps,alpha,gpu):
    with sem:
        try:
            wd=stage(f"{F}/l2_{tag}",ef,cm,nscfl,steps,alpha);env,find=make_env();env["AI_SA"]="1"
            run_solver(wd,find,env,gpu=gpu,timeout=9000);m,sd,rc=stats(wd)
            with lock:res[tag]=dict(CDmean=round(m,5),CDstd_pct=round(100*sd/max(m,1e-9),2),cont=rc)
            print("done %-14s CDmean=%.5f CDstd=%.2f%% cont=%.1e"%(tag,m,100*sd/max(m,1e-9),rc),flush=True)
        except Exception as e:
            with lock:res[tag]=dict(err=str(e)[:70]);print("FAIL %s %s"%(tag,e),flush=True)
# (tag, evalFreq, CFLmult, NS CFL max, steps, alpha)
jobs=[("limitA_clean_a4",1,8,3.0,30000,4),   # SLOW NS, faster SA -> clean-ish Limit A
      ("gen_a6_ef50",50,1,None,45000,6),       # generalize fix to a=6
      ("gen_a8_ef50",50,1,None,45000,8)]       # generalize fix to a=8
ts=[threading.Thread(target=run,args=(t,ef,cm,nc,st,al,i%8)) for i,(t,ef,cm,nc,st,al) in enumerate(jobs)]
for t in ts:t.start()
for t in ts:t.join()
json.dump(res,open(f"{F}/run_limits2_results.json","w"),indent=1);print("LIMITS2 DONE:",json.dumps(res))
