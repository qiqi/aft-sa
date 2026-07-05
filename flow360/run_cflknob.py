import os,json,shutil,threading,csv,numpy as np,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver,extract_forces
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
def stage(wd,turbcfl,evalfreq,nscflmax,steps,alpha=4):
    shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
    for f in os.listdir(f"{F}/base_nlf_ogrid"):
        if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
        s,d=f"{F}/base_nlf_ogrid/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    c=json.load(open(f"{wd}/Flow360.json"));c['freestream']['alphaAngle']=float(alpha);c['timeStepping']['maxPseudoSteps']=steps
    c['turbulenceModelSolver']['CFLMultiplier']=turbcfl; c['turbulenceModelSolver']['equationEvalFrequency']=evalfreq
    c['timeStepping']['CFL']['max']=float(nscflmax)
    json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1);return wd
def cdtail(wd):
    rows=list(csv.reader(open(f"{wd}/total_forces_v2.csv")));h=[x.strip() for x in rows[0]];ip,icd=h.index('pseudo_step'),h.index('CD');s,cd=[],[]
    for r in rows[1:]:
        r=[c for c in r if c.strip()!='']
        if len(r)>icd:
            try:s.append(float(r[ip]));cd.append(float(r[icd]))
            except:pass
    s,cd=np.array(s),np.array(cd);m=s>0.7*s.max();return cd[m].mean(),cd[m].std()
res={};lock=threading.Lock();sem=threading.Semaphore(8)
def run(tag,tcfl,ef,nscfl,gpu):
    with sem:
        try:
            wd=stage(f"{F}/knob_{tag}",tcfl,ef,nscfl,15000)
            env,find=make_env();env["AI_SA"]="1"
            run_solver(wd,find,env,gpu=gpu,timeout=5400);f=extract_forces(wd);m,sd=cdtail(wd)
            with lock:res[tag]=dict(CDmean=round(m,5),CDstd_pct=round(100*sd/max(m,1e-9),1))
            print("done %-16s CDmean=%.5f  CDstd=%.1f%%"%(tag,m,100*sd/max(m,1e-9)),flush=True)
        except Exception as e:
            with lock:res[tag]=dict(err=str(e)[:70]);print("FAIL %s %s"%(tag,e),flush=True)
# (tag, turb.CFLMultiplier, equationEvalFreq, ns CFL max)
jobs=[("base_2.0",2.0,4,10000),("cflm1.0",1.0,4,10000),("cflm0.5",0.5,4,10000),
      ("cflm0.25",0.25,4,10000),("evalfreq1",2.0,1,10000),("cflm0.5_cfl200",0.5,4,200)]
ts=[threading.Thread(target=run,args=(t,tc,ef,nc,i%8)) for i,(t,tc,ef,nc) in enumerate(jobs)]
for t in ts:t.start()
for t in ts:t.join()
json.dump(res,open(f"{F}/run_cflknob_results.json","w"),indent=1)
print("KNOB TEST DONE:",json.dumps(res))
