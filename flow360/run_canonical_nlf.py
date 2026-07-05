"""Regenerate canonical NLF case dirs with the adopted gated-max model (now default), both grids,
all alpha -> full_nlf0416_{cavity,ogrid}_aftsa_m2_a{a}. GPUs 1-7. For paper figure regeneration."""
import os,json,shutil,threading,csv,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env;from rans.solve import run_solver
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
def stage(base,wd,alpha,steps):
    shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
    for f in os.listdir(base):
        if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
        s,d=f"{base}/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    c=json.load(open(f"{wd}/Flow360.json"));c['freestream']['alphaAngle']=float(alpha);c['timeStepping']['maxPseudoSteps']=int(steps)
    json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1);return wd
res={};lock=threading.Lock();sem=threading.Semaphore(7)
def run(grid,base,a,steps,gpu):
    with sem:
        tag=f"{grid}_a{a}";wd=f"{F}/full_nlf0416_{grid}_aftsa_m2_a{a}"
        try:
            stage(base,wd,a,steps);env,find=make_env();env["AI_SA"]="1"  # gated max = default
            try: run_solver(wd,find,env,gpu=gpu,timeout=9000)
            except: pass
            rows=list(csv.reader(open(f"{wd}/total_forces_v2.csv")));h=[x.strip() for x in rows[0]];last=[c for c in rows[-1] if c.strip()!='']
            with lock:res[tag]=dict(CL=round(float(last[h.index('CL')]),4),CD=round(float(last[h.index('CD')]),5))
            print("done %-12s CL=%.4f CD=%.5f"%(tag,float(last[h.index('CL')]),float(last[h.index('CD')])),flush=True)
        except Exception as e:print("FAIL %s %s"%(tag,e),flush=True)
jobs=[]
for a in [-2,0,2,4,6,8]:
    jobs.append(("cavity",f"{F}/base_nlf_cavity",a,15000))
    jobs.append(("ogrid",f"{F}/base_nlf_ogrid",a,25000))
ts=[threading.Thread(target=run,args=(g,b,a,st,1+(i%7))) for i,(g,b,a,st) in enumerate(jobs)]
for t in ts:t.start()
for t in ts:t.join()
json.dump(res,open(f"{F}/run_canonical_nlf_results.json","w"),indent=1);print("CANONICAL NLF DONE:",json.dumps(res))
