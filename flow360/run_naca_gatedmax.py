"""Re-run NACA sweep + Fig-7 refinement meshes IN PLACE with the gated-max model (now default),
so every NACA figure is single-provenance gated-max. Clears only output; preserves mesh+config."""
import os,json,glob,threading,csv,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env;from rans.solve import run_solver
F="/home/qiqi/flexcompute/aft-sa/flow360"
CLEAR=('.pvtu','.vtu','.pvd','.sock','.gltf')
def clean(d):
    for f in os.listdir(d):
        p=os.path.join(d,f)
        if any(f.endswith(x) for x in CLEAR) or f.endswith('_v2.csv') or f.startswith('surface_forces') or f.endswith('.log'):
            try:os.remove(p)
            except:pass
        if f in ('ipc_data','restartOutput'):
            import shutil;shutil.rmtree(p,ignore_errors=True)
def setsteps(d,steps):
    c=json.load(open(f"{d}/Flow360.json"));c['timeStepping']['maxPseudoSteps']=int(steps);json.dump(c,open(f"{d}/Flow360.json","w"),indent=1)
res={};lock=threading.Lock();sem=threading.Semaphore(7)
def run(d,steps,gpu):
    with sem:
        wd=f"{F}/{d}"
        try:
            clean(wd);setsteps(wd,steps);env,find=make_env();env["AI_SA"]="1"  # gated max = default
            try: run_solver(wd,find,env,gpu=gpu,timeout=9000)
            except: pass
            rows=list(csv.reader(open(f"{wd}/total_forces_v2.csv")));h=[x.strip() for x in rows[0]];last=[c for c in rows[-1] if c.strip()!='']
            with lock:res[d]=dict(CL=round(float(last[h.index('CL')]),4),CD=round(float(last[h.index('CD')]),5))
            print("done %-34s CL=%.4f CD=%.5f"%(d,float(last[h.index('CL')]),float(last[h.index('CD')])),flush=True)
        except Exception as e:print("FAIL %s %s"%(d,str(e)[:50]),flush=True)
jobs=[]
for a in [-2,0,2,4,6,8]:
    jobs.append((f"full_naca0012_cavity_aftsa_m2_a{a}",8000))
    jobs.append((f"full_naca0012_ogrid_aftsa_m2_a{a}",8000))
for d in sorted(glob.glob(f"{F}/refineA4_*")):
    if os.path.isdir(d): jobs.append((os.path.basename(d),12000))
ts=[threading.Thread(target=run,args=(d,st,1+(i%7))) for i,(d,st) in enumerate(jobs)]
for t in ts:t.start()
for t in ts:t.join()
json.dump(res,open(f"{F}/run_naca_gatedmax_results.json","w"),indent=1);print("NACA GATEDMAX DONE:",json.dumps(res))
