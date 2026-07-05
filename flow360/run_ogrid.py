import os, sys, json, shutil, threading
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver, extract_forces
B="/home/qiqi/flexcompute/aft-sa/flow360"; SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
base=f"{B}/out_naca0012_ogrid"
def stage(tag,turb):
    wd=f"{B}/ogrid_{tag}"; shutil.rmtree(wd,ignore_errors=True); os.makedirs(wd)
    for f in os.listdir(base):
        if any(f.endswith(x) for x in SKIP): continue
        s,d=os.path.join(base,f),os.path.join(wd,f)
        (shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    if turb:
        cf=json.load(open(f"{wd}/Flow360.json"))
        cf['boundaries']['fluid/farfield']['turbulenceQuantities']['modifiedTurbulentViscosityRatio']=3.0
        json.dump(cf,open(f"{wd}/Flow360.json","w"),indent=1)
    return wd
res={}
def worker(tag,turb,gpu):
    try:
        wd=stage(tag,turb); env,find=make_env()
        if not turb: env["AI_SA"]="1"
        run_solver(wd,find,env,gpu=gpu,timeout=2400)
        f=extract_forces(wd); res[tag]=(f["CL"],f["CD"])
        print("done %s: CL=%.4f CD=%.5f"%(tag,f["CL"],f["CD"]),flush=True)
    except Exception as e: res[tag]=("ERR",str(e)[:90]); print("FAIL %s: %s"%(tag,e),flush=True)
ts=[threading.Thread(target=worker,args=a) for a in [("aftsa",False,0),("turb",True,1)]]
for t in ts: t.start()
for t in ts: t.join()
print("RESULTS:",res)
