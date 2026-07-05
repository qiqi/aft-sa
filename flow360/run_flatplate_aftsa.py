"""Flow360 flat-plate verification of SA-AI against the JAX BL solver and
Schubauer-Skramstad. Quasi-2D structured mesh: x in [-x0, L], y in [0, H],
inlet symmetry strip x in [-x0, 0], wall x in [0, L], freestream top, outflow.

Uses the same rans-pipeline scaffolding as the NLF cases.
"""
import os, sys, json, shutil, time
import numpy as np
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/flow360")
os.environ["AI_CHI_INF"] = "8.4e-3"  # Tu=0.18%
from rans.env import make_env
from rans.config import CaseConfig
from rans import case as _case, mesh as _mesh

B = "/home/qiqi/flexcompute/aft-sa/flow360"
CFG_JSON = f"{B}/naca0012_re1m.json"

# Plate geometry (non-dimensional: length unit = 1, Re/L = 1e6).
# Wall from x=0 to x=L. Inlet at x=0, outlet at x=L, top = freestream.
# All non-wall non-span boundaries lumped into "farfield" since the rans
# pipeline only handles {wall, farfield, symmetry1, symmetry2}.
L_plate = 4.0          # Re_x at trailing edge = 4e6 (covers S-S range)
H_dom   = 0.5          # height of domain
span    = 0.05         # quasi-2D extrusion span
nspan   = 1            # 1 span cell

# Mesh resolution. Want fine near x=0 (LE) and y=0 (wall).
n_x_plate = 280        # cells along plate
n_y       = 80
ratio_x   = 1.015      # gentle growth along plate
ratio_y   = 1.12       # stretch in y, fine near wall

def stretched(n, L, r):
    if abs(r-1) < 1e-6: return np.linspace(0, L, n+1)
    h0 = L*(1-r)/(1-r**n)
    return h0*(1-r**np.arange(n+1))/(1-r)

def build_plate_mesh(out_dir):
    """Write a gmsh-format quasi-2D flat plate mesh, then convert to CGNS."""
    # 1D node distributions
    x_grid = stretched(n_x_plate, L_plate, ratio_x)
    y_grid = stretched(n_y, H_dom, ratio_y)
    Nx, Ny = len(x_grid), len(y_grid)
    print(f"Mesh: Nx={Nx}, Ny={Ny}")

    NL = nspan + 1
    N = Nx * Ny
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
        # Span-end symmetries (y_span=0 -> symmetry1, y_span=-span -> symmetry2)
        for i in range(Nx-1):
            for j in range(Ny-1):
                emit("3 2 5 5 %d %d %d %d" % (nid(0,i,j), nid(0,i+1,j), nid(0,i+1,j+1), nid(0,i,j+1)))
        for i in range(Nx-1):
            for j in range(Ny-1):
                emit("3 2 4 4 %d %d %d %d" % (nid(nspan,i,j), nid(nspan,i+1,j), nid(nspan,i+1,j+1), nid(nspan,i,j+1)))
        # Wall at z=0 (j=0), entire bottom is plate
        for i in range(Nx-1):
            for L in range(nspan):
                emit("3 2 2 2 %d %d %d %d" % (nid(L,i,0), nid(L,i+1,0), nid(L+1,i+1,0), nid(L+1,i,0)))
        # Farfield: top (j=Ny-1), inlet (i=0), outlet (i=Nx-1), all into "farfield"
        for i in range(Nx-1):
            for L in range(nspan):
                emit("3 2 3 3 %d %d %d %d" % (nid(L,i,Ny-1), nid(L,i+1,Ny-1), nid(L+1,i+1,Ny-1), nid(L+1,i,Ny-1)))
        for j in range(Ny-1):
            for L in range(nspan):
                emit("3 2 3 3 %d %d %d %d" % (nid(L,0,j), nid(L,0,j+1), nid(L+1,0,j+1), nid(L+1,0,j)))
        for j in range(Ny-1):
            for L in range(nspan):
                emit("3 2 3 3 %d %d %d %d" % (nid(L,Nx-1,j), nid(L,Nx-1,j+1), nid(L+1,Nx-1,j+1), nid(L+1,Nx-1,j)))
        # Volumes (hex)
        for i in range(Nx-1):
            for j in range(Ny-1):
                for L in range(nspan):
                    emit("5 2 1 1 %d %d %d %d %d %d %d %d" % (
                        nid(L,i,j), nid(L,i+1,j), nid(L,i+1,j+1), nid(L,i,j+1),
                        nid(L+1,i,j), nid(L+1,i+1,j), nid(L+1,i+1,j+1), nid(L+1,i,j+1)))
        f.write("$Elements\n%d\n" % len(elems))
        f.write("\n".join(elems) + "\n$EndElements\n")
    return len(elems)

def patch_flow360(p, alpha=0.0):
    d = json.load(open(p))
    d['freestream']['Mach'] = 0.1
    d['freestream']['muRef'] = 1.0e-7     # gives Re/L = 1e6 at Mach 0.1 (rough)
    d['freestream']['alphaAngle'] = alpha
    # chi_inf_input compensated for AI_LAMINAR_SLOWDOWN=0.01:
    #   in-domain chi_inf = chi_input / fSlow = 8.4e-5 / 0.01 = 8.4e-3 (Tu=0.18%)
    CHI_INPUT = 8.4e-5
    tq = d['freestream'].setdefault('turbulenceQuantities', {})
    tq['modelType'] = 'ModifiedTurbulentViscosityRatio'
    tq['modifiedTurbulentViscosityRatio'] = CHI_INPUT
    for bn, bc in d.get('boundaries', {}).items():
        if 'farfield' in bn or 'inlet' in bn or bc.get('type') == 'Freestream':
            btq = bc.setdefault('turbulenceQuantities', {})
            btq['modelType'] = 'ModifiedTurbulentViscosityRatio'
            btq['modifiedTurbulentViscosityRatio'] = CHI_INPUT
    d.setdefault('volumeOutput', {})['outputFields'] = [
        'velocity','p','rho','vorticityMagnitude','nuHat','wallDistance','Mach',
        'residualNavierStokes','residualTurbulence']
    d.setdefault('fluidProperties', {})['sutherlandConstantDim'] = 110.4
    d['runControl']['restart'] = False
    d['timeStepping']['maxPseudoSteps'] = 60000
    d['timeStepping']['absoluteTolerance'] = 1e-30
    d.setdefault('turbulenceModelSolver', {})['absoluteTolerance'] = 1e-30
    json.dump(d, open(p, 'w'), indent=1)

if __name__ == '__main__':
    case_dir = f"{B}/flatplate_aftsa_Tu018"
    if os.path.exists(case_dir): shutil.rmtree(case_dir)
    os.makedirs(case_dir)
    t = time.time()
    n_elem = build_plate_mesh(case_dir)
    print(f"wrote mesh.msh ({n_elem} elements), t={time.time()-t:.0f}s")
    env, find = make_env()
    _mesh.gmsh_to_cgns(case_dir + "/mesh.msh", case_dir + "/mesh.cgns",
                       find("flow360gmshtocgns"), env)
    print(f"wrote mesh.cgns, t={time.time()-t:.0f}s")

    cfg = CaseConfig.load(CFG_JSON)
    cfg.solver.max_steps = 60000
    cfg.flow.alpha_deg = 0.0
    cfg.flow.mach = 0.1
    cfg.flow.reynolds = 1.0e6   # per unit length
    cfg.elements[0].name = 'wall'
    _case.preprocess(case_dir, "mesh.cgns", find, env, cfg=cfg,
                     wall_names=[f"fluid/wall"],
                     boundary_names=[
                         "fluid/farfield", "fluid/wall",
                         "fluid/symmetry1", "fluid/symmetry2"],
                     timings={}, sdk_cache_dir=None,
                     sim_builder=_case.build_simulation_json)
    patch_flow360(f"{case_dir}/Flow360.json")
    print(f"set up {case_dir}, t={time.time()-t:.0f}s")
