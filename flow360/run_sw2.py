"""Now with the FRESH lib: confirm hook (NACA sw=0.5 vs 64 must differ) + test the
gentle-handover damping on NLF O-grid a4 (sw=4,16,64)."""
import os,json,shutil,threading,csv,numpy as np,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env;from rans.solve import run_solver,extract_forces
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
def stage(base,wd,steps,alpha=4):
    shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
    for f in os.listdir(base):
        if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
        s,d=f"{base}/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    c=json.load(open(f"{wd}/Flow360.json"));c['freestream']['alphaAngle']=float(alpha);c['timeStepping']['maxPseudoSteps']=steps
    json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1);return wd
def cdtail(wd):
    rows=list(csv.reader(open(f"{wd}/total_forces_v2.csv")));h=[x.strip() for x in rows[0]];ip,icd=h.index('pseudo_step'),h.index('CD');s,cd=[],[]
    for r in rows[1:]:
        r=[c for c in r if c.strip()!='']
        if len(r)>icd:
            try:s.append(float(r[ip]));cd.append(float(r[icd]))
            except:pass
    s,cd=np.array(s),np.array(cd);m=s>0.7*s.max();return cd[m].mean(),cd[m].std()
res={};lock=threading.Lock();sem=threading.Semaphore(8)
def run(tag,base,sw,steps,gpu):
    with sem:
        try:
            wd=stage(base,f"{F}/sw2_{tag}",steps)
            env,find=make_env();env["AI_SA"]="1";env["AI_SWITCHWIDTH"]=str(sw)
            run_solver(wd,find,env,gpu=gpu,timeout=6000);m,sd=cdtail(wd)
            with lock:res[tag]=dict(CDmean=round(m,5),CDstd_pct=round(100*sd/max(m,1e-9),1))
            print("done %-14s CDmean=%.5f CDstd=%.1f%%"%(tag,m,100*sd/max(m,1e-9)),flush=True)
        except Exception as e:
            with lock:res[tag]=dict(err=str(e)[:70]);print("FAIL %s %s"%(tag,e),flush=True)
NACA=f"{F}/full_naca0012_cavity_aftsa_m2_a4";NLF=f"{F}/base_nlf_ogrid"
jobs=[("naca_sw0.5",NACA,0.5,6000),("naca_sw64",NACA,64,6000),
      ("nlf_sw4",NLF,4,15000),("nlf_sw16",NLF,16,15000),("nlf_sw64",NLF,64,15000)]
ts=[threading.Thread(target=run,args=(t,b,sw,st,i%8)) for i,(t,b,sw,st) in enumerate(jobs)]
for t in ts:t.start()
for t in ts:t.join()
json.dump(res,open(f"{F}/run_sw2_results.json","w"),indent=1)
print("SW2 DONE:",json.dumps(res))
