import os,json,shutil,threading,csv,numpy as np,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver,extract_forces
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
def stage(wd,steps,alpha=4):
    shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
    for f in os.listdir(f"{F}/base_nlf_ogrid"):
        if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
        s,d=f"{F}/base_nlf_ogrid/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    c=json.load(open(f"{wd}/Flow360.json"));c['freestream']['alphaAngle']=float(alpha);c['timeStepping']['maxPseudoSteps']=steps
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
def run(tag,rs,gpu):
    with sem:
        try:
            wd=stage(f"{F}/rs_{tag}",15000)
            env,find=make_env();env["AI_SA"]="1";env["AI_RATESCALE"]=str(rs)
            run_solver(wd,find,env,gpu=gpu,timeout=5400);f=extract_forces(wd);m,sd=cdtail(wd)
            with lock:res[tag]=dict(CDmean=round(m,5),CDstd_pct=round(100*sd/max(m,1e-9),1))
            print("done rateScale=%-6s CDmean=%.5f CDstd=%.1f%%"%(tag,m,100*sd/max(m,1e-9)),flush=True)
        except Exception as e:
            with lock:res[tag]=dict(err=str(e)[:70]);print("FAIL %s %s"%(tag,e),flush=True)
jobs=[("0.05",0.05),("0.02",0.02),("0.01",0.01),("0.005",0.005)]
ts=[threading.Thread(target=run,args=(t,rs,i%8)) for i,(t,rs) in enumerate(jobs)]
for t in ts:t.start()
for t in ts:t.join()
print("RATESCALE TEST DONE:",json.dumps(res))
