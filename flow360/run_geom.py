"""Geometric-blend slowdown test: sSlow=f^(1-is_turb). Milder f (0.05,0.1,0.2) should now work
(holds slowdown across chi~1-4). O-grid NLF a4, DEFAULT coupling. 3 runs (avoid teardown-under-load)."""
import os,json,shutil,threading,csv,numpy as np,sys
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env;from rans.solve import run_solver
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
def stage(wd,steps=40000,alpha=4):
    shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
    for f in os.listdir(f"{F}/base_nlf_ogrid"):
        if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
        s,d=f"{F}/base_nlf_ogrid/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    c=json.load(open(f"{wd}/Flow360.json"));c['freestream']['alphaAngle']=float(alpha);c['timeStepping']['maxPseudoSteps']=steps
    json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1);return wd
def xtr(d):
    import vtk;from vtk.util.numpy_support import vtk_to_numpy
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{d}/surface_fluid_nlf0416.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());cf=vtk_to_numpy(g.GetPointData().GetArray('Cf'))
    y=p[:,1];s=np.abs(y+0.0)<1e-3;X,Z,Cf=p[s][:,0],p[s][:,2],cf[s];n=len(X);pts=np.column_stack([X,Z])
    st=int(np.argmin(X));o=[st];u=np.zeros(n,bool);u[st]=True
    for _ in range(n-1):
        c=o[-1];dd=np.sum((pts-pts[c])**2,1);dd[u]=1e9;nx=int(np.argmin(dd));o.append(nx);u[nx]=True
    o=np.array(o);xo,zo,co=X[o],Z[o],Cf[o];te=int(np.argmax(xo));b1,b2=slice(0,te+1),slice(te,n)
    up=b1 if zo[b1].mean()>zo[b2].mean() else b2;xu,cu=xo[up],co[up];oo=np.argsort(xu);xu,cu=xu[oo],cu[oo]
    k=np.ones(7)/7;cm=np.convolve(cu,k,'same');dc=np.diff(cm);dm=.5*(xu[1:]+xu[:-1]);w=(dm>0.1)&(dm<0.8)
    return float(dm[w][np.argmax(dc[w])])
res={};lock=threading.Lock();sem=threading.Semaphore(3)
def run(tag,fslow,gpu):
    with sem:
        try:
            wd=stage(f"{F}/geom_{tag}");env,find=make_env();env["AI_SA"]="1";env["AI_LAMINAR_SLOWDOWN"]=str(fslow)
            try: run_solver(wd,find,env,gpu=gpu,timeout=10000)
            except Exception as e: print("(teardown note %s)"%tag,flush=True)
            rows=list(csv.reader(open(f"{wd}/total_forces_v2.csv")));h=[x.strip() for x in rows[0]];ip,icd=h.index('pseudo_step'),h.index('CD');s,cd=[],[]
            for r in rows[1:]:
                r=[c for c in r if c.strip()!='']
                if len(r)>icd:
                    try:s.append(float(r[ip]));cd.append(float(r[icd]))
                    except:pass
            s,cd=np.array(s),np.array(cd);m=s>0.6*s.max()
            rr=list(csv.reader(open(f"{wd}/nonlinear_residual_v2.csv")));rh=[x.strip() for x in rr[0]];rl=[c for c in rr[-1] if c.strip()!='']
            try:xt=xtr(wd)
            except:xt=float('nan')
            with lock:res[tag]=dict(CDmean=round(cd[m].mean(),5),CDstd_pct=round(100*cd[m].std()/cd[m].mean(),2),xtr=round(xt,3),cont=float(rl[rh.index('0_cont')]),laststep=int(s.max()))
            print("done f=%-5s laststep=%d CDmean=%.5f CDstd=%.2f%% xtr=%.3f cont=%.1e"%(tag,int(s.max()),cd[m].mean(),100*cd[m].std()/cd[m].mean(),xt,float(rl[rh.index('0_cont')])),flush=True)
        except Exception as e:
            with lock:res[tag]=dict(err=str(e)[:70]);print("FAIL %s %s"%(tag,e),flush=True)
ts=[threading.Thread(target=run,args=(t,fs,i)) for i,(t,fs) in enumerate([("0.05",0.05),("0.1",0.1),("0.2",0.2)])]
for t in ts:t.start()
for t in ts:t.join()
json.dump(res,open(f"{F}/run_geom_results.json","w"),indent=1);print("GEOM DONE:",json.dumps(res))
