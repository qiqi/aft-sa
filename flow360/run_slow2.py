"""Clean confirmation: f=0.005, 0.008 (fully steady?), 55000 steps, 2 runs only (avoid teardown-
under-concurrency). Verify CDstd->~0 and xtr->ef50 fixed point (0.286) => fixed-point preserved."""
import os,json,shutil,threading,csv,numpy as np,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env;from rans.solve import run_solver
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
def stage(wd,steps=55000,alpha=4):
    shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
    for f in os.listdir(f"{F}/base_nlf_ogrid"):
        if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
        s,d=f"{F}/base_nlf_ogrid/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    c=json.load(open(f"{wd}/Flow360.json"));c['freestream']['alphaAngle']=float(alpha);c['timeStepping']['maxPseudoSteps']=steps
    json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1);return wd
res={};lock=threading.Lock();sem=threading.Semaphore(2)
def run(tag,fslow,gpu):
    with sem:
        try:
            wd=stage(f"{F}/slow2_{tag}");env,find=make_env();env["AI_SA"]="1";env["AI_LAMINAR_SLOWDOWN"]=str(fslow)
            try: run_solver(wd,find,env,gpu=gpu,timeout=12000)
            except Exception as e: print("(teardown note %s: %s)"%(tag,str(e)[:40]),flush=True)  # data may still be valid
            rows=list(csv.reader(open(f"{wd}/total_forces_v2.csv")));h=[x.strip() for x in rows[0]];ip,icd=h.index('pseudo_step'),h.index('CD');s,cd=[],[]
            for r in rows[1:]:
                r=[c for c in r if c.strip()!='']
                if len(r)>icd:
                    try:s.append(float(r[ip]));cd.append(float(r[icd]))
                    except:pass
            s,cd=np.array(s),np.array(cd);m=s>0.6*s.max()
            rr=list(csv.reader(open(f"{wd}/nonlinear_residual_v2.csv")));rh=[x.strip() for x in rr[0]];rl=[c for c in rr[-1] if c.strip()!='']
            with lock:res[tag]=dict(CDmean=round(cd[m].mean(),5),CDstd_pct=round(100*cd[m].std()/cd[m].mean(),2),cont=float(rl[rh.index('0_cont')]),laststep=int(s.max()))
            print("done f=%-6s laststep=%d CDmean=%.5f CDstd=%.2f%% cont=%.1e"%(tag,int(s.max()),cd[m].mean(),100*cd[m].std()/cd[m].mean(),float(rl[rh.index('0_cont')])),flush=True)
        except Exception as e:
            with lock:res[tag]=dict(err=str(e)[:70]);print("FAIL %s %s"%(tag,e),flush=True)
ts=[threading.Thread(target=run,args=(t,fs,i)) for i,(t,fs) in enumerate([("0.005",0.005),("0.008",0.008)])]
for t in ts:t.start()
for t in ts:t.join()
json.dump(res,open(f"{F}/run_slow2_results.json","w"),indent=1);print("SLOW2 DONE:",json.dumps(res))
