"""Canonical steady O-grid NLF polar at equationEvalFrequency=50 (the timescale fix). Run the
missing alphas (-2,0,2); a4/a6/a8 already done (lim_B_ef50, l2_gen_a6/a8_ef50)."""
import os,json,shutil,threading,csv,numpy as np,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env;from rans.solve import run_solver
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
def stage(wd,alpha,steps=45000):
    shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
    for f in os.listdir(f"{F}/base_nlf_ogrid"):
        if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
        s,d=f"{F}/base_nlf_ogrid/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    c=json.load(open(f"{wd}/Flow360.json"));c['freestream']['alphaAngle']=float(alpha);c['timeStepping']['maxPseudoSteps']=steps
    c['turbulenceModelSolver']['equationEvalFrequency']=50;c['turbulenceModelSolver']['CFLMultiplier']=1.0
    json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1);return wd
def stats(wd):
    rows=list(csv.reader(open(f"{wd}/total_forces_v2.csv")));h=[x.strip() for x in rows[0]];icd=h.index('CD');icl=h.index('CL');ip=h.index('pseudo_step');s,cd,cl=[],[],[]
    for r in rows[1:]:
        r=[c for c in r if c.strip()!='']
        if len(r)>max(icd,icl):
            try:s.append(float(r[ip]));cd.append(float(r[icd]));cl.append(float(r[icl]))
            except:pass
    s,cd,cl=np.array(s),np.array(cd),np.array(cl);m=s>0.6*s.max()
    rr=list(csv.reader(open(f"{wd}/nonlinear_residual_v2.csv")));rh=[x.strip() for x in rr[0]];rl=[c for c in rr[-1] if c.strip()!='']
    return cd[m].mean(),cd[m].std(),cl[m].mean(),float(rl[rh.index('0_cont')])
res={};lock=threading.Lock();sem=threading.Semaphore(8)
def run(alpha,gpu):
    with sem:
        try:
            wd=stage(f"{F}/og50_a{alpha}",alpha);env,find=make_env();env["AI_SA"]="1"
            run_solver(wd,find,env,gpu=gpu,timeout=9000);m,sd,cl,rc=stats(wd)
            with lock:res[alpha]=dict(CD=round(m,5),CDstd_pct=round(100*sd/max(m,1e-9),2),CL=round(cl,4),cont=rc)
            print("done a=%g CD=%.5f CDstd=%.2f%% CL=%.4f cont=%.1e"%(alpha,m,100*sd/max(m,1e-9),cl,rc),flush=True)
        except Exception as e:
            with lock:res[alpha]=dict(err=str(e)[:70]);print("FAIL a=%g %s"%(alpha,e),flush=True)
ts=[threading.Thread(target=run,args=(a,i%8)) for i,a in enumerate([-2,0,2])]
for t in ts:t.start()
for t in ts:t.join()
json.dump(res,open(f"{F}/run_ogrid_ef50_results.json","w"),indent=1);print("OGRID EF50 DONE:",json.dumps(res))
