"""Instrument the O-grid NLF a=4 limit cycle: dump surface (Cf,Cp) every 100 steps and
volume (mut,nuHat,vorticity) every 200 steps over steps 13000-15000 (~1 period of ~1900)."""
import os,json,shutil,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
wd=f"{F}/mech_og_a4"
shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
for f in os.listdir(f"{F}/base_nlf_ogrid"):
    if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
    s,d=f"{F}/base_nlf_ogrid/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
c=json.load(open(f"{wd}/Flow360.json"))
c['freestream']['alphaAngle']=4.0;c['timeStepping']['maxPseudoSteps']=15000
so=c['surfaceOutput'][0];so['animationFrequency']=100;so['animationFrequencyOffset']=13000
vo=c['volumeOutput'];vo['animationFrequency']=200;vo['animationFrequencyOffset']=13000
vo['outputFields']=["Mach","primitiveVars","mut","mutRatio","nuHat","vorticity"]
json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1)
env,find=make_env();env["AI_SA"]="1"
print("running instrumented O-grid NLF a4 (15000 steps, periodic output 13000-15000)...",flush=True)
run_solver(wd,find,env,gpu=0,timeout=6000)
print("MECHANISM RUN DONE",flush=True)
# list the periodic outputs produced
import glob
print("surface snaps:",sorted(glob.glob(f"{wd}/**/surface*nlf*.pvtu",recursive=True))[:3],"...")
print("volume snaps:",sorted(glob.glob(f"{wd}/**/volume*.pvtu",recursive=True))[:3],"...")
