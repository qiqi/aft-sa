"""Convert each Construct2D proper_struct_LX.p3d into a Flow360-ready CGNS case dir,
parallel to the proper_cav_LX cavity layout."""
import os, sys, json, shutil
import numpy as np
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/flow360")
os.environ["AI_CHI_INF"] = "0.02"
from rans.env import make_env
from rans.config import CaseConfig
from rans import case as _case, mesh as _mesh

C2D_DIR = "/home/qiqi/flexcompute/aft-sa/external/construct2d"
OUT_BASE = "/home/qiqi/flexcompute/aft-sa/flow360"
CFG_JSON = f"{OUT_BASE}/naca0012_re1m.json"
LEVELS = ['L0', 'L1', 'L2', 'L3', 'L4']
nspan = 1
span = 0.1
wall = "naca0012"

ncells = {}
import time
t_total = time.time()
for tag in LEVELS:
    p3d = f"{C2D_DIR}/proper_struct_{tag}.p3d"
    if not os.path.exists(p3d):
        print(f"SKIP {tag}: no .p3d"); continue
    OUT = f"{OUT_BASE}/proper_str_{tag}"
    print(f"\n=== building struct case {tag} from {p3d} ===", flush=True)
    t = time.time()
    os.makedirs(OUT, exist_ok=True)
    toks = open(p3d).read().split()
    ni, nj = int(toks[0]), int(toks[1])
    vals = np.array(toks[2:2+2*ni*nj], float)
    X = vals[:ni*nj].reshape(nj, ni).T
    Y = vals[ni*nj:2*ni*nj].reshape(nj, ni).T
    Ni = ni - 1
    k = lambda i, j: i*nj + j
    P = np.empty((Ni*nj, 2))
    for i in range(Ni):
        for j in range(nj):
            P[k(i, j)] = (X[i, j], Y[i, j])
    N = Ni * nj
    quads = [(k(i, j), k((i+1) % Ni, j), k((i+1) % Ni, j+1), k(i, j+1)) for i in range(Ni) for j in range(nj-1)]
    wallE = [(k(i, 0), k((i+1) % Ni, 0)) for i in range(Ni)]
    farE = [(k(i, nj-1), k((i+1) % Ni, nj-1)) for i in range(Ni)]
    NL = nspan + 1
    nid = lambda L, kk: L*N + kk + 1
    phys = [(2, 2, wall), (2, 3, "farfield"), (2, 4, "symmetry1"), (2, 5, "symmetry2"), (3, 1, "fluid")]
    elems = []
    eid = [1]
    def emit(s):
        elems.append(f"{eid[0]} {s}"); eid[0] += 1
    with open(OUT + "/mesh.msh", "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n%d\n" % len(phys))
        for d, t_, n in phys: f.write('%d %d "%s"\n' % (d, t_, n))
        f.write("$EndPhysicalNames\n$Nodes\n%d\n" % (NL*N))
        for L in range(NL):
            ys = -span*L/nspan
            for kk in range(N): f.write("%d %.16g %.16g %.16g\n" % (nid(L, kk), P[kk, 0], ys, P[kk, 1]))
        f.write("$EndNodes\n")
        for a, b, c, d in quads: emit("3 2 4 4 %d %d %d %d" % (nid(0, a), nid(0, b), nid(0, c), nid(0, d)))
        for a, b, c, d in quads: emit("3 2 5 5 %d %d %d %d" % (nid(nspan, a), nid(nspan, b), nid(nspan, c), nid(nspan, d)))
        for a, b in wallE:
            for L in range(nspan): emit("3 2 2 2 %d %d %d %d" % (nid(L, a), nid(L, b), nid(L+1, b), nid(L+1, a)))
        for a, b in farE:
            for L in range(nspan): emit("3 2 3 3 %d %d %d %d" % (nid(L, a), nid(L, b), nid(L+1, b), nid(L+1, a)))
        for L in range(nspan):
            for a, b, c, d in quads: emit("5 2 1 1 %d %d %d %d %d %d %d %d" % (nid(L, a), nid(L, b), nid(L, c), nid(L, d), nid(L+1, a), nid(L+1, b), nid(L+1, c), nid(L+1, d)))
        f.write("$Elements\n%d\n" % len(elems)); f.write("\n".join(elems) + "\n$EndElements\n")
    print(f"  wrote mesh.msh ({len(quads)} quads, t={time.time()-t:.0f}s)", flush=True)
    env, find = make_env()
    _mesh.gmsh_to_cgns(OUT + "/mesh.msh", OUT + "/mesh.cgns", find("flow360gmshtocgns"), env)
    print(f"  wrote mesh.cgns, t={time.time()-t:.0f}s", flush=True)
    cfg = CaseConfig.load(CFG_JSON)
    cfg.solver.max_steps = 25000
    cfg.flow.alpha_deg = 4.0
    _case.preprocess(OUT, "mesh.cgns", find, env, cfg=cfg,
                     wall_names=[f"fluid/{wall}"],
                     boundary_names=[f"fluid/farfield", f"fluid/{wall}", "fluid/symmetry1", "fluid/symmetry2"],
                     timings={}, sdk_cache_dir=None, sim_builder=_case.build_simulation_json)
    ncells[tag] = len(quads)
    print(f"  done struct {tag}: ncells2d={len(quads)}, t={time.time()-t:.0f}s, cum={time.time()-t_total:.0f}s", flush=True)

json.dump(ncells, open(f"{OUT_BASE}/proper_struct_case_ncells.json", 'w'), indent=1)
print(f"\nTotal time: {time.time()-t_total:.0f}s")
print(f"struct case cell counts: {ncells}")
