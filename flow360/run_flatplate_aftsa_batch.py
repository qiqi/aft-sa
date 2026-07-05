"""Set up + launch Flow360 flat-plate cases at all 5 Schubauer-Skramstad Tu
levels, in parallel. Each case uses the same mesh (built once and cloned)
and the same chi_inf-via-Mack mapping (Anchor A), compensated for fSlow=0.01.
"""
import os, sys, json, shutil, threading, time, math
import numpy as np
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/flow360")
os.environ["AI_CHI_INF"] = "1e-3"   # arbitrary; overridden per case below
from rans.env import make_env
from rans.config import CaseConfig
from rans import case as _case, mesh as _mesh
from rans.solve import run_solver

B = "/home/qiqi/flexcompute/aft-sa/flow360"
CFG_JSON = f"{B}/naca0012_re1m.json"
F_SLOW = 0.01
C_V1 = 7.1

# Tu->chi_inf via the SA-AI-tuned mapping (Anchor A). Single source of truth
# in scripts/calibrate_kernel.py. chi_input = chi_in_domain * fSlow.
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/scripts")
from calibrate_kernel import chi_inf_from_Tu_pct as chi_in_domain

TU_LIST = [0.026, 0.06, 0.18, 0.30, 0.85]
CASES = [(Tu, chi_in_domain(Tu)*F_SLOW, chi_in_domain(Tu)) for Tu in TU_LIST]

# Plate geometry — extended to L=6 so low-Tu cases can complete transition
# under the slower kernel (a_max=0.125).
L_plate = 6.0; H_dom = 0.5; span = 0.05; nspan = 1
n_x_plate = 320; n_y = 80; ratio_x = 1.015; ratio_y = 1.12

def stretched(n, L, r):
    if abs(r-1) < 1e-6: return np.linspace(0, L, n+1)
    h0 = L*(1-r)/(1-r**n)
    return h0*(1-r**np.arange(n+1))/(1-r)

def build_plate_mesh(out_dir):
    x_grid = stretched(n_x_plate, L_plate, ratio_x)
    y_grid = stretched(n_y, H_dom, ratio_y)
    Nx, Ny = len(x_grid), len(y_grid)
    NL = nspan + 1; N = Nx * Ny
    nid = lambda L, i, j: L*N + i*Ny + j + 1
    phys = [(2, 2, "wall"), (2, 3, "farfield"),
            (2, 4, "symmetry1"), (2, 5, "symmetry2"), (3, 1, "fluid")]
    elems = []; eid = [1]
    def emit(s):
        elems.append(f"{eid[0]} {s}"); eid[0] += 1
    with open(out_dir + "/mesh.msh", "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n%d\n" % len(phys))
        for d,t,n in phys: f.write('%d %d "%s"\n' % (d, t, n))
        f.write("$EndPhysicalNames\n$Nodes\n%d\n" % (NL*N))
        for L in range(NL):
            ys = -span*L/nspan
            for i in range(Nx):
                for j in range(Ny):
                    f.write("%d %.16g %.16g %.16g\n" % (nid(L,i,j), x_grid[i], ys, y_grid[j]))
        f.write("$EndNodes\n")
        for i in range(Nx-1):
            for j in range(Ny-1):
                emit("3 2 5 5 %d %d %d %d" % (nid(0,i,j), nid(0,i+1,j), nid(0,i+1,j+1), nid(0,i,j+1)))
        for i in range(Nx-1):
            for j in range(Ny-1):
                emit("3 2 4 4 %d %d %d %d" % (nid(nspan,i,j), nid(nspan,i+1,j), nid(nspan,i+1,j+1), nid(nspan,i,j+1)))
        for i in range(Nx-1):
            for L in range(nspan):
                emit("3 2 2 2 %d %d %d %d" % (nid(L,i,0), nid(L,i+1,0), nid(L+1,i+1,0), nid(L+1,i,0)))
        for i in range(Nx-1):
            for L in range(nspan):
                emit("3 2 3 3 %d %d %d %d" % (nid(L,i,Ny-1), nid(L,i+1,Ny-1), nid(L+1,i+1,Ny-1), nid(L+1,i,Ny-1)))
        for j in range(Ny-1):
            for L in range(nspan):
                emit("3 2 3 3 %d %d %d %d" % (nid(L,0,j), nid(L,0,j+1), nid(L+1,0,j+1), nid(L+1,0,j)))
        for j in range(Ny-1):
            for L in range(nspan):
                emit("3 2 3 3 %d %d %d %d" % (nid(L,Nx-1,j), nid(L,Nx-1,j+1), nid(L+1,Nx-1,j+1), nid(L+1,Nx-1,j)))
        for i in range(Nx-1):
            for j in range(Ny-1):
                for L in range(nspan):
                    emit("5 2 1 1 %d %d %d %d %d %d %d %d" % (
                        nid(L,i,j), nid(L,i+1,j), nid(L,i+1,j+1), nid(L,i,j+1),
                        nid(L+1,i,j), nid(L+1,i+1,j), nid(L+1,i+1,j+1), nid(L+1,i,j+1)))
        f.write("$Elements\n%d\n" % len(elems))
        f.write("\n".join(elems) + "\n$EndElements\n")
    return Nx, Ny, len(elems)

def patch_flow360(p, chi_input):
    d = json.load(open(p))
    d['freestream']['Mach'] = 0.1
    d['freestream']['muRef'] = 1.0e-7
    d['freestream']['alphaAngle'] = 0.0
    tq = d['freestream'].setdefault('turbulenceQuantities', {})
    tq['modelType'] = 'ModifiedTurbulentViscosityRatio'
    tq['modifiedTurbulentViscosityRatio'] = chi_input
    for bn, bc in d.get('boundaries', {}).items():
        if 'farfield' in bn or 'inlet' in bn or bc.get('type') == 'Freestream':
            btq = bc.setdefault('turbulenceQuantities', {})
            btq['modelType'] = 'ModifiedTurbulentViscosityRatio'
            btq['modifiedTurbulentViscosityRatio'] = chi_input
    d.setdefault('volumeOutput', {})['outputFields'] = [
        'velocity','p','rho','vorticityMagnitude','nuHat','wallDistance','Mach',
        'residualNavierStokes','residualTurbulence']
    d.setdefault('fluidProperties', {})['sutherlandConstantDim'] = 110.4
    d['runControl']['restart'] = False
    d['timeStepping']['maxPseudoSteps'] = 60000
    d['timeStepping']['absoluteTolerance'] = 1e-30
    d.setdefault('turbulenceModelSolver', {})['absoluteTolerance'] = 1e-30
    json.dump(d, open(p, 'w'), indent=1)

def case_dir_for(Tu_pct):
    return f"{B}/flatplate_aftsa_Tu{int(round(Tu_pct*1000)):04d}"   # Tu0026, Tu0060, etc.

def setup_case(Tu, chi_input, chi_domain):
    cd = case_dir_for(Tu)
    if os.path.exists(cd): shutil.rmtree(cd)
    os.makedirs(cd)
    Nx, Ny, ne = build_plate_mesh(cd)
    print(f"[Tu={Tu}%] mesh {Nx}x{Ny}, {ne} elems", flush=True)
    env, find = make_env()
    _mesh.gmsh_to_cgns(cd + "/mesh.msh", cd + "/mesh.cgns", find("flow360gmshtocgns"), env)
    cfg = CaseConfig.load(CFG_JSON)
    cfg.solver.max_steps = 60000
    cfg.flow.alpha_deg = 0.0
    cfg.flow.mach = 0.1
    cfg.flow.reynolds = 1.0e6
    cfg.elements[0].name = 'wall'
    _case.preprocess(cd, "mesh.cgns", find, env, cfg=cfg,
                     wall_names=[f"fluid/wall"],
                     boundary_names=[
                         "fluid/farfield", "fluid/wall",
                         "fluid/symmetry1", "fluid/symmetry2"],
                     timings={}, sdk_cache_dir=None,
                     sim_builder=_case.build_simulation_json)
    patch_flow360(f"{cd}/Flow360.json", chi_input)
    print(f"[Tu={Tu}%] set up @ chi_in_domain={chi_domain:.3e}, chi_input={chi_input:.3e}", flush=True)
    return cd

def run_case(Tu, chi_input, cd, gpu, sem, lock, results):
    with sem:
        env, find = make_env()
        env["AI_SA"] = "1"
        env["AI_LAMINAR_SLOWDOWN"] = str(F_SLOW)
        tag = f"Tu{Tu}"
        t0 = time.time()
        print(f"[{tag}] launching on GPU {gpu}", flush=True)
        try:
            run_solver(cd, find, env, gpu=gpu, timeout=10800)
            try:
                lastF = open(f"{cd}/total_forces_v2.csv").readlines()[-1]
                lastR = open(f"{cd}/nonlinear_residual_v2.csv").readlines()[-1]
                with lock:
                    results[tag] = {
                        'dt_s': time.time()-t0,
                        'res': lastR.strip().split(',')[2],
                        'CD': lastF.strip().split(',')[3],
                    }
                print(f"[{tag}] DONE in {results[tag]['dt_s']:.0f}s  res={results[tag]['res']}  CD={results[tag]['CD']}", flush=True)
            except Exception as e:
                print(f"[{tag}] DONE but couldn't read forces: {e}", flush=True)
        except Exception as e:
            print(f"[{tag}] FAILED: {e}", flush=True)
            with lock:
                results[tag] = {'error': str(e)}

if __name__ == '__main__':
    # Setup all 5 cases sequentially (fast — just meshing + JSON)
    case_dirs = {}
    for Tu, chi_input, chi_dom in CASES:
        case_dirs[Tu] = setup_case(Tu, chi_input, chi_dom)
    # Run all 5 in parallel
    sem = threading.Semaphore(8)
    lock = threading.Lock(); results = {}
    threads = []
    for i, (Tu, chi_input, _) in enumerate(CASES):
        cd = case_dirs[Tu]
        t = threading.Thread(target=run_case, args=(Tu, chi_input, cd, i % 8, sem, lock, results))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    json.dump(results, open(f"{B}/flatplate_batch_results.json", 'w'), indent=1)
    print(f"Done. {len(results)}/{len(CASES)}")
    for k,v in sorted(results.items()):
        print(f"  {k}: {v}")
