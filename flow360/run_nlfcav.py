import os,json,shutil,threading,numpy as np,vtk,sys
from vtk.util.numpy_support import vtk_to_numpy
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
from rans.env import make_env
from rans.solve import run_solver,extract_forces
F="/home/qiqi/flexcompute/aft-sa/flow360";SKIP=('.pvtu','.vtu','.csv','.log','.sock','.gltf','.pvd')
def stage(wd,alpha,chi):
    shutil.rmtree(wd,ignore_errors=True);os.makedirs(wd)
    for f in os.listdir(f"{F}/base_nlf_cavity"):
        if any(f.endswith(x) for x in SKIP) or f in('ipc_data','restartOutput'):continue
        s,d=f"{F}/base_nlf_cavity/{f}",f"{wd}/{f}";(shutil.copytree if os.path.isdir(s) else shutil.copy)(s,d)
    c=json.load(open(f"{wd}/Flow360.json"));c['freestream']['alphaAngle']=float(alpha);c['timeStepping']['maxPseudoSteps']=10000
    c['freestream']['turbulenceQuantities']={'modelType':'ModifiedTurbulentViscosityRatio','modifiedTurbulentViscosityRatio':chi}
    fk=next(k for k in c['boundaries'] if 'farfield' in k);c['boundaries'][fk]['turbulenceQuantities']['modifiedTurbulentViscosityRatio']=chi
    json.dump(c,open(f"{wd}/Flow360.json","w"),indent=1);return wd
def xtr(o):
    r=vtk.vtkXMLPUnstructuredGridReader();r.SetFileName(f"{o}/surface_fluid_nlf0416.pvtu");r.Update()
    g=r.GetOutput();p=vtk_to_numpy(g.GetPoints().GetData());pd=g.GetPointData();nm=[pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    a=vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith('cf'))));v=np.linalg.norm(a,axis=1) if a.ndim>1 else a
    x,z=p[:,0],p[:,2];u=z>1e-6;xs,cf=x[u],v[u];o2=np.argsort(xs);xs,cf=xs[o2],cf[o2]
    b=np.linspace(0,1,51);xc=.5*(b[1:]+b[:-1]);cb=np.array([np.median(cf[(xs>=b[i])&(xs<b[i+1])]) if((xs>=b[i])&(xs<b[i+1])).any() else np.nan for i in range(50)])
    ok=np.isfinite(cb);xc,cb=xc[ok],cb[ok];dd=np.diff(cb);dm=.5*(xc[1:]+xc[:-1]);w=(dm>0.04)&(dm<0.95)
    return float(dm[w][np.argmax(dd[w])]) if w.any() else np.nan
res={};lock=threading.Lock();sem=threading.Semaphore(8)
def run(tag,a,chi,aft,gpu):
    with sem:
        try:
            wd=stage(f"{F}/full_nlf0416_cavity_{'aftsa_m2' if aft else 'turb'}_a{a}",a,chi)
            env,find=make_env()
            if aft:env["AI_SA"]="1"
            run_solver(wd,find,env,gpu=gpu,timeout=2400);f=extract_forces(wd)
            with lock:res[tag]=dict(CL=f["CL"],CD=f["CD"],xtr=xtr(wd) if aft else 0.0)
            print("done %-20s CL=%+.4f CD=%.5f xtr=%.3f"%(tag,f["CL"],f["CD"],res[tag]["xtr"]),flush=True)
        except Exception as e:
            with lock:res[tag]=dict(err=str(e)[:70]);print("FAIL %s %s"%(tag,e),flush=True)
jobs=[]
for a in [-2,0,2,4,6,8]:
    jobs.append((f"aftsa_a{a}",a,0.02,True));jobs.append((f"turb_a{a}",a,3.0,False))
ts=[threading.Thread(target=run,args=(t,a,c,af,i%8)) for i,(t,a,c,af) in enumerate(jobs)]
for t in ts:t.start()
for t in ts:t.join()
json.dump(res,open(f"{F}/run_nlfcav_results.json","w"),indent=1)
print("CAVITY NLF 10k DONE:",json.dumps({k:(round(v['CD'],5),v.get('xtr')) if 'CD' in v else 'ERR' for k,v in res.items()}))
