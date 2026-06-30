"""Convert E387 Construct2D .p3d → Flow360 CGNS cases (L0/L1/L2 × α∈{0,2,5,7}).
Mirrors build_nlf_struct_cases.py at Re=200k, M=0.1.
"""
import os, sys, json, shutil, time
import numpy as np
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/flow360")
os.environ["AFT_CHI_INF"] = "0.02"
from rans.env import make_env
from rans.config import CaseConfig
from rans import case as _case, mesh as _mesh

C2D_DIR = "/home/qiqi/flexcompute/aft-sa/external/construct2d"
OUT_BASE = "/home/qiqi/flexcompute/aft-sa/flow360"
CFG_JSON = f"{OUT_BASE}/naca0012_re1m.json"

LEVELS = ['L0', 'L1', 'L2']
ALPHAS = [0.0, 2.0, 5.0, 7.0]
wall = "eppler387"
nspan = 1
span = 0.1

RES_FIELDS = ['residualTurbulence', 'residualNavierStokes', 'nuHat',
              'wallDistance', 'vorticityMagnitude']
CHI_INPUT = 8.76e-6
MU_REF = 5.0e-7   # M / Re = 0.1 / 2e5

def write_msh_from_p3d(p3d, out_dir):
    toks = open(p3d).read().split()
    ni, nj = int(toks[0]), int(toks[1])
    vals = np.array(toks[2:2+2*ni*nj], float)
    X = vals[:ni*nj].reshape(nj, ni).T
    Y = vals[ni*nj:2*ni*nj].reshape(nj, ni).T
    Ni = ni - 1
    k = lambda i, j: i*nj + j
    N = Ni * nj
    P = np.empty((N, 2))
    for i in range(Ni):
        for j in range(nj):
            P[k(i, j)] = (X[i, j], Y[i, j])
    quads = [(k(i, j), k((i+1) % Ni, j), k((i+1) % Ni, j+1), k(i, j+1))
             for i in range(Ni) for j in range(nj-1)]
    wallE = [(k(i, 0), k((i+1) % Ni, 0)) for i in range(Ni)]
    farE = [(k(i, nj-1), k((i+1) % Ni, nj-1)) for i in range(Ni)]
    NL = nspan + 1
    nid = lambda L, kk: L*N + kk + 1
    phys = [(2, 2, wall), (2, 3, "farfield"), (2, 4, "symmetry1"),
            (2, 5, "symmetry2"), (3, 1, "fluid")]
    elems = []
    eid = [1]
    def emit(s):
        elems.append(f"{eid[0]} {s}"); eid[0] += 1
    with open(out_dir + "/mesh.msh", "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n%d\n" % len(phys))
        for d, t_, n in phys: f.write('%d %d "%s"\n' % (d, t_, n))
        f.write("$EndPhysicalNames\n$Nodes\n%d\n" % (NL*N))
        for L in range(NL):
            ys = -span*L/nspan
            for kk in range(N):
                f.write("%d %.16g %.16g %.16g\n" % (nid(L, kk), P[kk, 0], ys, P[kk, 1]))
        f.write("$EndNodes\n")
        for a, b, c, d in quads:
            emit("3 2 4 4 %d %d %d %d" % (nid(0, a), nid(0, b), nid(0, c), nid(0, d)))
        for a, b, c, d in quads:
            emit("3 2 5 5 %d %d %d %d" % (nid(nspan, a), nid(nspan, b),
                                           nid(nspan, c), nid(nspan, d)))
        for a, b in wallE:
            for L in range(nspan):
                emit("3 2 2 2 %d %d %d %d" % (nid(L, a), nid(L, b),
                                               nid(L+1, b), nid(L+1, a)))
        for a, b in farE:
            for L in range(nspan):
                emit("3 2 3 3 %d %d %d %d" % (nid(L, a), nid(L, b),
                                               nid(L+1, b), nid(L+1, a)))
        for L in range(nspan):
            for a, b, c, d in quads:
                emit("5 2 1 1 %d %d %d %d %d %d %d %d" % (
                    nid(L, a), nid(L, b), nid(L, c), nid(L, d),
                    nid(L+1, a), nid(L+1, b), nid(L+1, c), nid(L+1, d)))
        f.write("$Elements\n%d\n" % len(elems))
        f.write("\n".join(elems) + "\n$EndElements\n")
    return len(quads)

def patch_flow360_for_e387(cfg_path, alpha):
    d = json.load(open(cfg_path))
    d['freestream']['Mach'] = 0.1
    d['freestream']['muRef'] = MU_REF
    d['freestream']['alphaAngle'] = alpha
    tq = d['freestream'].setdefault('turbulenceQuantities', {})
    tq['modelType'] = 'ModifiedTurbulentViscosityRatio'
    tq['modifiedTurbulentViscosityRatio'] = CHI_INPUT
    for bname, bcfg in d.get('boundaries', {}).items():
        if 'farfield' in bname or bcfg.get('type') == 'Freestream':
            btq = bcfg.setdefault('turbulenceQuantities', {})
            btq['modelType'] = 'ModifiedTurbulentViscosityRatio'
            btq['modifiedTurbulentViscosityRatio'] = CHI_INPUT
    vol = d.setdefault('volumeOutput', {}).setdefault('outputFields', [])
    for f in RES_FIELDS:
        if f not in vol: vol.append(f)
    for sn, sc in d.get('sliceOutput', {}).get('slices', {}).items():
        sf = sc.setdefault('outputFields', [])
        for f in RES_FIELDS:
            if f not in sf: sf.append(f)
    d.setdefault('fluidProperties', {})['sutherlandConstantDim'] = 110.4
    d['runControl']['restart'] = False
    d['timeStepping']['maxPseudoSteps'] = 80000
    d['timeStepping']['absoluteTolerance'] = 1e-30
    d.setdefault('turbulenceModelSolver', {})['absoluteTolerance'] = 1e-30
    json.dump(d, open(cfg_path, 'w'), indent=1)

t_total = time.time()
ncells_record = {}
for tag in LEVELS:
    p3d = f"{C2D_DIR}/proper_struct_eppler_{tag}.p3d"
    if not os.path.exists(p3d):
        print(f"SKIP {tag}: no {p3d}"); continue
    for alpha in ALPHAS:
        case_dir = f"{OUT_BASE}/strL{tag[1:]}prop_eppler387_Re200k_a{int(alpha)}"
        print(f"\n=== building struct case {tag} alpha={alpha} → "
              f"{os.path.basename(case_dir)} ===", flush=True)
        t = time.time()
        if os.path.exists(case_dir): shutil.rmtree(case_dir)
        os.makedirs(case_dir, exist_ok=True)
        n_quads = write_msh_from_p3d(p3d, case_dir)
        print(f"  wrote mesh.msh ({n_quads} quads)", flush=True)
        env, find = make_env()
        _mesh.gmsh_to_cgns(case_dir + "/mesh.msh", case_dir + "/mesh.cgns",
                            find("flow360gmshtocgns"), env)
        print(f"  wrote mesh.cgns, t={time.time()-t:.0f}s", flush=True)
        cfg = CaseConfig.load(CFG_JSON)
        cfg.solver.max_steps = 80000
        cfg.flow.alpha_deg = float(alpha)
        cfg.flow.mach = 0.1
        cfg.flow.reynolds = 2.0e5
        cfg.elements[0].name = wall
        _case.preprocess(case_dir, "mesh.cgns", find, env, cfg=cfg,
                         wall_names=[f"fluid/{wall}"],
                         boundary_names=[f"fluid/farfield", f"fluid/{wall}",
                                          "fluid/symmetry1", "fluid/symmetry2"],
                         timings={}, sdk_cache_dir=None,
                         sim_builder=_case.build_simulation_json)
        patch_flow360_for_e387(f"{case_dir}/Flow360.json", alpha)
        ncells_record[f"{tag}_a{int(alpha)}"] = n_quads
        print(f"  done {tag} α={alpha}: {n_quads} quads, t={time.time()-t:.0f}s, "
              f"cum={time.time()-t_total:.0f}s", flush=True)

json.dump(ncells_record, open(f"{OUT_BASE}/eppler_struct_case_ncells.json", 'w'), indent=1)
print(f"\nTotal time: {time.time()-t_total:.0f}s")
print(f"E387 struct case cell counts: {ncells_record}")
