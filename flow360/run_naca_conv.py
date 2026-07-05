"""Re-run NACA a0 + a4 (both grids) with a high step cap so the absTol early-stop (NS 1e-9,
turb 1e-8) is actually reached -- for Fig 6 (convergence) and Fig 5 (cf at a0 & a4)."""
import os,json,threading,csv,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env;from rans.solve import run_solver
F="/home/qiqi/flexcompute/aft-sa/flow360"
def clean(d):
    for f in os.listdir(d):
        p=os.path.join(d,f)
        if f.endswith(('.pvtu','.vtu','.pvd','.sock','.gltf')) or f.endswith('_v2.csv') or f.startswith('surface_forces') or f.endswith('.log'):
            try:os.remove(p)
            except:pass
        if f in('ipc_data','restartOutput'):
            import shutil;shutil.rmtree(p,ignore_errors=True)
res={};lock=threading.Lock();sem=threading.Semaphore(7)
def run(d,gpu):
    with sem:
        wd=f"{F}/{d}"
        try:
            clean(wd);c=json.load(open(f"{wd}/Flow360.json"));c['timeStepping']['maxPseudoSteps']=25000;json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1)
            env,find=make_env();env["AI_SA"]="1"
            try:run_solver(wd,find,env,gpu=gpu,timeout=9000)
            except:pass
            rr=list(csv.reader(open(f"{wd}/nonlinear_residual_v2.csv")));h=[x.strip() for x in rr[0]];last=[c for c in rr[-1] if c.strip()!='']
            with lock:res[d]=dict(step=int(float(last[h.index('pseudo_step')])),cont=float(last[h.index('0_cont')]),nuHat=float(last[h.index('5_nuHat')]))
            print("done %-30s step=%d cont=%.1e nuHat=%.1e"%(d,res[d]['step'],res[d]['cont'],res[d]['nuHat']),flush=True)
        except Exception as e:print("FAIL %s %s"%(d,e),flush=True)
jobs=[f"full_naca0012_{g}_aftsa_m2_a{a}" for a in [0,4] for g in ['cavity','ogrid']]
ts=[threading.Thread(target=run,args=(d,1+i)) for i,d in enumerate(jobs)]
for t in ts:t.start()
for t in ts:t.join()
print("NACA CONV DONE:",json.dumps(res))
