import os, sys, json, shutil, threading, csv
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver, extract_forces
B="/home/qiqi/flexcompute/aft-sa/flow360"
SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
bases={'naca0012':f"{B}/out_naca0012_lam",'nlf0416':f"{B}/out_nlf0416_lam"}
alphas=[-2,0,2,4,6,8]
jobs=[('turb',af,a) for af in ('naca0012','nlf0416') for a in alphas]
def stage(model,af,a):
    wd=f"{B}/run10k_{model}_{af}_a{a}"; shutil.rmtree(wd,ignore_errors=True); os.makedirs(wd)
    for f in os.listdir(bases[af]):
        if any(f.endswith(x) for x in SKIP): continue
        s,d=os.path.join(bases[af],f),os.path.join(wd,f)
        (shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    cf=json.load(open(f"{wd}/Flow360.json"))
    cf['timeStepping']['maxPseudoSteps']=10000
    cf['freestream']['alphaAngle']=float(a)
    cf['boundaries']['fluid/farfield']['turbulenceQuantities']['modifiedTurbulentViscosityRatio']=3.0
    json.dump(cf,open(f"{wd}/Flow360.json","w"),indent=1)
    return wd
results={}; sem=threading.Semaphore(8)
def worker(model,af,a,gpu):
    with sem:
        try:
            wd=stage(model,af,a); env,find=make_env()
            run_solver(wd,find,env,gpu=gpu,timeout=2400)
            f=extract_forces(wd); results[(af,a)]=(f["CL"],f["CD"])
            print(f"  done turb {af} a={a}: CL={f['CL']:.4f} CD={f['CD']:.5f}",flush=True)
        except Exception as e:
            results[(af,a)]=("ERR",str(e)[:80]); print(f"  FAIL turb {af} a={a}: {e}",flush=True)
ts=[]
for i,(m,af,a) in enumerate(jobs):
    t=threading.Thread(target=worker,args=(m,af,a,i%8)); t.start(); ts.append(t)
for t in ts: t.join()
with open(f"{B}/run10k_turb_results.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["airfoil","alpha","CL","CD"])
    for (af,a),v in sorted(results.items()): w.writerow([af,a,v[0],v[1]])
print("TURB DONE")
