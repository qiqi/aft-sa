import sys, os, numpy as np
sys.path.insert(0,"/home/qiqi/flexcompute/flexfoil/rans")
os.environ.setdefault("FLOW360_SUPPRESS_BETA_WARNING","1")
os.environ["AI_CHI_INF"]="0.02"   # laminar freestream
from rans.config import CaseConfig
from rans import case as _case, mesh as _mesh
from rans.env import make_env

P3D="external/construct2d/naca0012.p3d"
OUT="/home/qiqi/flexcompute/aft-sa/flow360/out_naca0012_ogrid"
os.makedirs(OUT,exist_ok=True)
toks=open(P3D).read().split(); ni,nj=int(toks[0]),int(toks[1])
vals=np.array(toks[2:2+2*ni*nj],float); X=vals[:ni*nj].reshape(nj,ni).T; Y=vals[ni*nj:2*ni*nj].reshape(nj,ni).T
Ni=ni-1                      # drop duplicate last column (periodic O-grid)
def k(i,j): return i*nj+j
P=np.empty((Ni*nj,2))
for i in range(Ni):
    for j in range(nj): P[k(i,j)]=(X[i,j],Y[i,j])
N=Ni*nj
quads=[(k(i,j),k((i+1)%Ni,j),k((i+1)%Ni,j+1),k(i,j+1)) for i in range(Ni) for j in range(nj-1)]
wall=[(k(i,0),k((i+1)%Ni,0)) for i in range(Ni)]
far =[(k(i,nj-1),k((i+1)%Ni,nj-1)) for i in range(Ni)]
span=0.1; nspan=1; NL=nspan+1
def nid(L,kk): return L*N+kk+1
phys=[(2,2,"naca0012"),(2,3,"farfield"),(2,4,"symmetry1"),(2,5,"symmetry2"),(3,1,"fluid")]
elems=[]; eid=1
def emit(s):
    global eid; elems.append(f"{eid} {s}"); eid+=1
msh=OUT+"/mesh.msh"
with open(msh,"w") as f:
    f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n%d\n"%len(phys))
    for d,t,n in phys: f.write('%d %d "%s"\n'%(d,t,n))
    f.write("$EndPhysicalNames\n$Nodes\n%d\n"%(NL*N))
    for L in range(NL):
        ys=-span*L/nspan
        for kk in range(N): f.write("%d %.16g %.16g %.16g\n"%(nid(L,kk),P[kk,0],ys,P[kk,1]))
    f.write("$EndNodes\n")
    for a,b,c,d in quads: emit("3 2 4 4 %d %d %d %d"%(nid(0,a),nid(0,b),nid(0,c),nid(0,d)))      # symmetry1
    for a,b,c,d in quads: emit("3 2 5 5 %d %d %d %d"%(nid(nspan,a),nid(nspan,b),nid(nspan,c),nid(nspan,d)))  # symmetry2
    for a,b in wall:
        for L in range(nspan): emit("3 2 2 2 %d %d %d %d"%(nid(L,a),nid(L,b),nid(L+1,b),nid(L+1,a)))
    for a,b in far:
        for L in range(nspan): emit("3 2 3 3 %d %d %d %d"%(nid(L,a),nid(L,b),nid(L+1,b),nid(L+1,a)))
    for L in range(nspan):
        for a,b,c,d in quads: emit("5 2 1 1 %d %d %d %d %d %d %d %d"%(nid(L,a),nid(L,b),nid(L,c),nid(L,d),nid(L+1,a),nid(L+1,b),nid(L+1,c),nid(L+1,d)))
    f.write("$Elements\n%d\n"%len(elems)); f.write("\n".join(elems)+"\n$EndElements\n")
print("wrote mesh.msh: %d nodes(2D), %d quads, %d hexes"%(N,len(quads),len(quads)*nspan))
env,find=make_env()
_mesh.gmsh_to_cgns(msh, OUT+"/mesh.cgns", find("flow360gmshtocgns"), env)
print("gmsh_to_cgns OK ->", os.path.getsize(OUT+"/mesh.cgns"), "bytes")
cfg=CaseConfig.load("flow360/naca0012_re1m.json"); cfg.solver.max_steps=10000
t={}
_case.preprocess(OUT,"mesh.cgns",find,env,cfg=cfg,wall_names=["fluid/naca0012"],
   boundary_names=["fluid/farfield","fluid/naca0012","fluid/symmetry1","fluid/symmetry2"],
   timings=t,sdk_cache_dir=None,sim_builder=_case.build_simulation_json)
print("preprocess OK; Flow360.json exists:", os.path.exists(OUT+"/Flow360.json"))
