"""f=0.03 GEOM snapshot ladder over one period (~1958 steps): steps 40000-42000 step 250.
Volume fields for receptivity + sSlow analysis."""
import os,json,shutil,threading,csv,numpy as np,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env;from rans.solve import run_solver
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
def stage(wd,steps):
    shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
    for f in os.listdir(f"{F}/base_nlf_ogrid"):
        if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
        s,d=f"{F}/base_nlf_ogrid/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    c=json.load(open(f"{wd}/Flow360.json"));c['freestream']['alphaAngle']=4.0;c['timeStepping']['maxPseudoSteps']=int(steps)
    c['volumeOutput']['outputFields']=["primitiveVars","mut","mutRatio","nuHat","vorticity"]
    json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1);return wd
lock=threading.Lock();sem=threading.Semaphore(8);res={}
def run(step,gpu):
    with sem:
        try:
            wd=stage(f"{F}/f03_{step}",step);env,find=make_env();env["AI_SA"]="1";env["AI_LAMINAR_SLOWDOWN"]="0.03"
            try: run_solver(wd,find,env,gpu=gpu,timeout=10000)
            except Exception as e: pass
            rows=list(csv.reader(open(f"{wd}/total_forces_v2.csv")));h=[x.strip() for x in rows[0]];last=[c for c in rows[-1] if c.strip()!='']
            cd=float(last[h.index('CD')]);cl=float(last[h.index('CL')])
            with lock:res[step]=dict(CD=round(cd,6),CL=round(cl,5));print("done step=%d CD=%.5f CL=%.4f"%(step,cd,cl),flush=True)
        except Exception as e:print("FAIL %d %s"%(step,e),flush=True)
steps=list(range(40000,42001,250))  # 9 snapshots over ~1 period
ts=[threading.Thread(target=run,args=(s,i%8)) for i,s in enumerate(steps)]
for t in ts:t.start()
for t in ts:t.join()
json.dump(res,open(f"{F}/run_f03ladder_results.json","w"),indent=1);print("F03 LADDER DONE:",json.dumps(res))
