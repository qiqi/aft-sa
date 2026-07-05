import os,json,shutil,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env;from rans.solve import run_solver
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
wd=f"{F}/f03_probe";shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
for f in os.listdir(f"{F}/base_nlf_ogrid"):
    if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
    s,d=f"{F}/base_nlf_ogrid/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
c=json.load(open(f"{wd}/Flow360.json"));c['freestream']['alphaAngle']=4.0;c['timeStepping']['maxPseudoSteps']=45000
c['volumeOutput']['outputFields']=["primitiveVars","mut","mutRatio","nuHat","vorticity"]
json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1)
env,find=make_env();env["AI_SA"]="1";env["AI_LAMINAR_SLOWDOWN"]="0.03"
print("running f=0.03 geom, 45000 steps...",flush=True)
try: run_solver(wd,find,env,gpu=0,timeout=10000)
except Exception as e: print("(teardown note)",flush=True)
print("F03 PROBE DONE",flush=True)
