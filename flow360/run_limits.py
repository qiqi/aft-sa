"""Test the two timescale limits of the SA-AI <-> NS coupling on the O-grid NLF a4 cycle.
Toward Limit A (SA converged per NS step): high turb CFLMultiplier, evalFreq=1.
Toward Limit B (NS converged per SA step): high equationEvalFrequency.
If a limit goes steady -> the breathing is a two-timescale COUPLING artifact, not physical."""
import os,json,shutil,threading,csv,numpy as np,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env;from rans.solve import run_solver
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
def stage(wd,evalfreq,cflmult,steps,alpha=4):
    shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
    for f in os.listdir(f"{F}/base_nlf_ogrid"):
        if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
        s,d=f"{F}/base_nlf_ogrid/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    c=json.load(open(f"{wd}/Flow360.json"));c['freestream']['alphaAngle']=float(alpha);c['timeStepping']['maxPseudoSteps']=int(steps)
    c['turbulenceModelSolver']['equationEvalFrequency']=int(evalfreq)
    c['turbulenceModelSolver']['CFLMultiplier']=float(cflmult)
    json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1);return wd
def cdstats(wd):
    rows=list(csv.reader(open(f"{wd}/total_forces_v2.csv")));h=[x.strip() for x in rows[0]];ip,icd=h.index('pseudo_step'),h.index('CD');s,cd=[],[]
    for r in rows[1:]:
        r=[c for c in r if c.strip()!='']
        if len(r)>icd:
            try:s.append(float(r[ip]));cd.append(float(r[icd]))
            except:pass
    s,cd=np.array(s),np.array(cd);m=s>0.6*s.max();sm,cm=s[m],cd[m]
    x=cm-cm.mean();zc=np.where((x[:-1]<0)&(x[1:]>=0))[0];per=np.mean(np.diff(sm[zc])) if len(zc)>1 else np.nan
    return cm.mean(),cm.std(),per
res={};lock=threading.Lock();sem=threading.Semaphore(8)
def run(tag,ef,cm,steps,gpu):
    with sem:
        try:
            wd=stage(f"{F}/lim_{tag}",ef,cm,steps);env,find=make_env();env["AI_SA"]="1"
            run_solver(wd,find,env,gpu=gpu,timeout=9000);m,sd,per=cdstats(wd)
            with lock:res[tag]=dict(CDmean=round(m,5),CDstd_pct=round(100*sd/max(m,1e-9),2),period=round(per,0) if per==per else None)
            print("done %-12s CDmean=%.5f CDstd=%.2f%% period=%s (evalFreq=%d CFLmult=%g)"%(tag,m,100*sd/max(m,1e-9),per,ef,cm),flush=True)
        except Exception as e:
            with lock:res[tag]=dict(err=str(e)[:70]);print("FAIL %s %s"%(tag,e),flush=True)
# (tag, equationEvalFrequency, turb CFLMultiplier, steps)
jobs=[("base",4,2,20000),("A_cfl20",1,20,20000),("A_cfl100",1,100,20000),
      ("B_ef20",20,1,30000),("B_ef50",50,1,45000)]
ts=[threading.Thread(target=run,args=(t,ef,cm,st,i%8)) for i,(t,ef,cm,st) in enumerate(jobs)]
for t in ts:t.start()
for t in ts:t.join()
json.dump(res,open(f"{F}/run_limits_results.json","w"),indent=1);print("LIMITS DONE:",json.dumps(res))
