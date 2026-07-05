"""Sample the O-grid NLF a4 limit cycle in pseudo-time: 9 runs to maxPseudoSteps in
[13000..15000], each dumping final surface (Cf,Cp) + volume (mut,nuHat,vorticity,U)."""
import os,json,shutil,threading,csv,numpy as np,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
def stage(wd,steps):
    shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
    for f in os.listdir(f"{F}/base_nlf_ogrid"):
        if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
        s,d=f"{F}/base_nlf_ogrid/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    c=json.load(open(f"{wd}/Flow360.json"));c['freestream']['alphaAngle']=4.0;c['timeStepping']['maxPseudoSteps']=int(steps)
    c['volumeOutput']['outputFields']=["primitiveVars","mut","mutRatio","nuHat","vorticity"]
    json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1);return wd
def cdcl(wd):
    rows=list(csv.reader(open(f"{wd}/total_forces_v2.csv")));h=[x.strip() for x in rows[0]]
    last=[c.strip() for c in rows[-1] if c.strip()!=''];g=lambda k:float(last[h.index(k)]);return g('CL'),g('CD')
lock=threading.Lock();sem=threading.Semaphore(8);res={}
def run(step,gpu):
    with sem:
        try:
            wd=stage(f"{F}/snap_{step}",step)
            env,find=make_env();env["AI_SA"]="1"
            run_solver(wd,find,env,gpu=gpu,timeout=6000);cl,cd=cdcl(wd)
            with lock:res[step]=dict(CL=round(cl,5),CD=round(cd,6))
            print("done step=%d CL=%.4f CD=%.5f"%(step,cl,cd),flush=True)
        except Exception as e:
            with lock:res[step]=dict(err=str(e)[:70]);print("FAIL %d %s"%(step,e),flush=True)
steps=list(range(13000,15001,250))  # 9 snapshots
ts=[threading.Thread(target=run,args=(s,i%8)) for i,s in enumerate(steps)]
for t in ts:t.start()
for t in ts:t.join()
json.dump(res,open(f"{F}/run_snaps_results.json","w"),indent=1)
print("SNAPS DONE:",json.dumps(res))
