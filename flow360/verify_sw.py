"""Verify AI_SWITCHWIDTH hook: extreme sharp (0.5) vs gentle (64) handover on NACA0012
cavity a4 (robust transition) MUST differ if the env var is read."""
import os,json,shutil,threading,csv,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver,extract_forces
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
BASE=f"{F}/full_naca0012_cavity_aftsa_m2_a4"  # existing converged NACA cavity a4 dir as template
def stage(wd,steps=6000,alpha=4):
    shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
    for f in os.listdir(BASE):
        if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
        s,d=f"{BASE}/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    c=json.load(open(f"{wd}/Flow360.json"));c['freestream']['alphaAngle']=float(alpha);c['timeStepping']['maxPseudoSteps']=steps
    json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1);return wd
res={};lock=threading.Lock()
def run(sw,gpu):
    try:
        wd=stage(f"{F}/vsw_{sw}")
        env,find=make_env();env["AI_SA"]="1";env["AI_SWITCHWIDTH"]=str(sw)
        run_solver(wd,find,env,gpu=gpu,timeout=4000);f=extract_forces(wd)
        with lock:res[str(sw)]=round(f['CD'],6);print("done switchWidth=%-5s CD=%.6f"%(sw,f['CD']),flush=True)
    except Exception as e:print("FAIL sw=%s %s"%(sw,e),flush=True)
ts=[threading.Thread(target=run,args=(sw,i)) for i,sw in enumerate([0.5,4,64])]
for t in ts:t.start()
for t in ts:t.join()
print("VERIFY DONE:",json.dumps(res),"-> hook works if CDs differ")
