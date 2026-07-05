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
        if f in ('ipc_data','restartOutput'):
            import shutil;shutil.rmtree(p,ignore_errors=True)
res={};lock=threading.Lock()
def run(d,gpu):
    wd=f"{F}/{d}"
    try:
        clean(wd);c=json.load(open(f"{wd}/Flow360.json"));c['timeStepping']['maxPseudoSteps']=25000;json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1)
        env,find=make_env();env["AI_SA"]="1"
        try:run_solver(wd,find,env,gpu=gpu,timeout=9000)
        except:pass
        rows=list(csv.reader(open(f"{wd}/total_forces_v2.csv")));h=[x.strip() for x in rows[0]];ip,icd=h.index('pseudo_step'),h.index('CD');s,cd=[],[]
        for r in rows[1:]:
            r=[c for c in r if c.strip()!='']
            if len(r)>icd:
                try:s.append(float(r[ip]));cd.append(float(r[icd]))
                except:pass
        import numpy as np;s,cd=np.array(s),np.array(cd);m=s>0.7*s.max()
        print("done %s CDmean=%.5f CDstd=%.2f%%"%(d,cd[m].mean(),100*cd[m].std()/cd[m].mean()),flush=True)
    except Exception as e:print("FAIL %s %s"%(d,e),flush=True)
ts=[threading.Thread(target=run,args=(d,1+i)) for i,d in enumerate(['refineA4_struct_sxxfine','refineA4_cavity_xxfine'])]
for t in ts:t.start()
for t in ts:t.join();print("FINEST DONE")
