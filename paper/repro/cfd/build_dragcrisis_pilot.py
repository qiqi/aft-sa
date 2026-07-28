"""Build the drag-crisis PILOT case: D=1 circular cylinder, quasi-2D O-grid,
one URANS case at Re_D=1e5, M=0.1 (subcritical; see
agent-paper-review/2026-07-27-1540-dragcrisis-cylinder-feasibility.md, the
"Recommendation" paragraph).

Mesh. Construct2D was tried first (the Eppler/NLF-ladder mesher) but its
airfoil surface processing breaks on a closed circle (the LE/TE split puts
wall points on the far field: max skew 90 deg, eta-growth 6e3 -- unusable).
For a circle the ideal O-grid is ANALYTIC radial extrusion (exactly
orthogonal, exact y1/growth control), so this script generates the grid
directly and writes it in Construct2D's .p3d format; everything downstream
(.p3d -> gmsh .msh quasi-2D one-cell-span -> CGNS -> rans.case.preprocess ->
Flow360.json) is byte-identical machinery to flow360/build_eppler_struct_cases.py.

Sizing (pilot = L1 of the campaign family, 1540 note Sec. 4):
  N_s = 1200 uniform (Delta_s = pi/1200 = 2.6e-3 D), y1 = 8e-6 D
  (y+ ~ 0.06 at Re=1e5, y+ <= 0.7 at the future 2e6 top of the matrix),
  geometric growth solved exactly to the 100 D far-field radius (g ~ 1.10,
  148 layers -> 177.6k quads, Eppler-L1/L2 class).

Case. Unsteady dual-time: dt = 0.1 (solver units; = 0.01 D/U at M=0.1),
maxPseudoSteps 40 with NS relativeTolerance 1e-3. The staged startup +
freestream-seed conventions (chi_BC = chi_inf * fSlow) are applied per stage
by run_dragcrisis_pilot.py -- this builder writes the stage-0 JSON only.

Usage (local, no GPU; needs flexfoil/rans + the compute venv tools):
  python paper/repro/cfd/build_dragcrisis_pilot.py [--out OUT_BASE]
Case data convention: OUT_BASE defaults to /local_data/qiqi/sa-ai/dragcrisis_pilot.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/flow360")

# ---- pilot constants ------------------------------------------------------
# RE is the pilot default; --re overrides (critical-range gate pair = 3e5).
# The mesh is Re-independent (one family serves the matrix, y+ checked in
# mesh_stats); only muRef and the y+ estimates change with Re.
RE = 1.0e5
MACH = 0.1
MU_REF = MACH / RE                     # 1e-6
CHI_INF = 8.76e-4                      # LTPT-class seed, c_v1*e^-9 (campaign convention)
DT = 0.1                               # solver units = 0.01 D/U at M=0.1
N_SURF = 1200                          # uniform theta
Y1 = 8.0e-6                            # first cell height / D
GROWTH_TARGET = 1.10
R_WALL = 0.5                           # D = 1, centered at (0.5, 0) like an airfoil chord
R_OUT = 100.0                          # far-field radius / D
SPAN = 0.1
NSPAN = 1
WALL = "cylinder"

CFG_JSON = "/home/qiqi/flexcompute/sa-ai/flow360/naca0012_re1m.json"
RES_FIELDS = ['residualTurbulence', 'residualNavierStokes', 'nuHat',
              'wallDistance', 'vorticityMagnitude']


# ---- analytic O-grid ------------------------------------------------------
def solve_growth(y1, height, n_cells):
    """Exact geometric growth g with y1*(g^n-1)/(g-1) = height."""
    lo, hi = 1.0001, 2.0
    for _ in range(200):
        g = 0.5 * (lo + hi)
        tot = y1 * (g ** n_cells - 1.0) / (g - 1.0)
        if tot < height:
            lo = g
        else:
            hi = g
    return 0.5 * (lo + hi)


def build_ogrid(n_surf=N_SURF, y1=Y1, g_target=GROWTH_TARGET,
                r_wall=R_WALL, r_out=R_OUT):
    """Radial O-grid; returns X, Y arrays shaped (ni, nj) with the wrap point
    duplicated (ni = n_surf + 1), Construct2D .p3d convention."""
    height = r_out - r_wall
    n_cells = int(np.ceil(np.log1p(height * (g_target - 1.0) / y1)
                          / np.log(g_target)))
    g = solve_growth(y1, height, n_cells)
    j = np.arange(n_cells + 1)
    r = r_wall + y1 * (g ** j - 1.0) / (g - 1.0)
    r[-1] = r_out
    # CLOCKWISE loop (Construct2D's .p3d convention; verified on
    # proper_struct_eppler_L1.p3d, wall-loop signed area < 0). Counterclockwise
    # inverts the extruded hexes and MeshProcessor fails dual-closedness.
    th = np.linspace(0.0, -2.0 * np.pi, n_surf + 1)  # wrap duplicate at end
    X = 0.5 + np.cos(th)[:, None] * r[None, :]
    Y = np.sin(th)[:, None] * r[None, :]
    return X, Y, g, n_cells


def write_p3d(X, Y, path):
    ni, nj = X.shape
    with open(path, "w") as f:
        f.write(f"{ni} {nj}\n")
        for arr in (X, Y):
            flat = arr.reshape(-1, order='F')
            for i in range(0, len(flat), 6):
                f.write(" ".join(f"{v:.16e}" for v in flat[i:i + 6]) + "\n")


def mesh_stats(X, Y, g, n_cells, y1=Y1, r_out=R_OUT):
    """Print + return the mesh statistics the pilot must verify before running."""
    r = np.hypot(X - 0.5, Y)
    ds = np.hypot(np.diff(X[:, 0]), np.diff(Y[:, 0]))
    h1 = r[:, 1] - r[:, 0]
    # y+ estimate: y+ = (y1/D) * Re * sqrt(Cf/2); laminar shoulder Cf ~ 0.01
    # at 1e5, turbulent Cf ~ 3.2e-3 at 2e6 (flat-plate 0.026/Re^(1/7)).
    yp = lambda Re, Cf: y1 * Re * np.sqrt(Cf / 2.0)
    stats = {
        "n_surf": X.shape[0] - 1, "n_layers": n_cells,
        "cells2d": (X.shape[0] - 1) * n_cells,
        "ds_over_D": float(ds.mean()), "ds_uniformity": float(ds.max() / ds.min()),
        "y1": float(h1.mean()), "y1_spread": float(h1.max() - h1.min()),
        "growth": float(g),
        "wall_radius": float(r[:, 0].mean()), "outer_radius": float(r[:, -1].mean()),
        "wall_AR": float(ds.mean() / h1.mean()),
        "yplus_est_Re1e5_lamCf0.01": float(yp(1e5, 0.01)),
        "yplus_est_Re2e6_turbCf3.2e-3": float(yp(2e6, 3.2e-3)),
        "yplus_est_Re1e7_turbCf2.6e-3": float(yp(1e7, 2.6e-3)),
        "yplus_est_Re2e7_turbCf2.4e-3": float(yp(2e7, 2.4e-3)),
    }
    for k, v in stats.items():
        print(f"  {k:32s} {v}")
    assert abs(stats["y1"] - y1) < 1e-9 and stats["y1_spread"] < 1e-12
    assert stats["growth"] <= 1.15 and stats["ds_uniformity"] < 1.0 + 1e-9
    assert abs(stats["outer_radius"] - r_out) < 1e-6 * r_out
    return stats


# ---- .p3d -> quasi-2D gmsh .msh (verbatim machinery from ------------------
# flow360/build_eppler_struct_cases.py::write_msh_from_p3d; O-topology wrap)
def write_msh_from_arrays(X, Y, out_dir, wall=WALL, nspan=NSPAN, span=SPAN):
    ni, nj = X.shape
    Ni = ni - 1
    k = lambda i, j: i * nj + j
    N = Ni * nj
    P = np.empty((N, 2))
    for i in range(Ni):
        for j in range(nj):
            P[k(i, j)] = (X[i, j], Y[i, j])
    quads = [(k(i, j), k((i + 1) % Ni, j), k((i + 1) % Ni, j + 1), k(i, j + 1))
             for i in range(Ni) for j in range(nj - 1)]
    wallE = [(k(i, 0), k((i + 1) % Ni, 0)) for i in range(Ni)]
    farE = [(k(i, nj - 1), k((i + 1) % Ni, nj - 1)) for i in range(Ni)]
    NL = nspan + 1
    nid = lambda L, kk: L * N + kk + 1
    phys = [(2, 2, wall), (2, 3, "farfield"), (2, 4, "symmetry1"),
            (2, 5, "symmetry2"), (3, 1, "fluid")]
    elems = []
    eid = [1]

    def emit(s):
        elems.append(f"{eid[0]} {s}")
        eid[0] += 1

    with open(out_dir + "/mesh.msh", "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n%d\n" % len(phys))
        for d, t_, n in phys:
            f.write('%d %d "%s"\n' % (d, t_, n))
        f.write("$EndPhysicalNames\n$Nodes\n%d\n" % (NL * N))
        for L in range(NL):
            ys = -span * L / nspan
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
                                              nid(L + 1, b), nid(L + 1, a)))
        for a, b in farE:
            for L in range(nspan):
                emit("3 2 3 3 %d %d %d %d" % (nid(L, a), nid(L, b),
                                              nid(L + 1, b), nid(L + 1, a)))
        for L in range(nspan):
            for a, b, c, d in quads:
                emit("5 2 1 1 %d %d %d %d %d %d %d %d" % (
                    nid(L, a), nid(L, b), nid(L, c), nid(L, d),
                    nid(L + 1, a), nid(L + 1, b), nid(L + 1, c), nid(L + 1, d)))
        f.write("$Elements\n%d\n" % len(elems))
        f.write("\n".join(elems) + "\n$EndElements\n")
    return len(quads)


# ---- Flow360 case patch (unsteady dual-time) ------------------------------
def patch_flow360_unsteady(cfg_path, re=RE):
    d = json.load(open(cfg_path))
    d['freestream']['Mach'] = MACH
    d['freestream']['muRef'] = MACH / re
    d['freestream']['alphaAngle'] = 0.0
    # seed: stage-dependent (chi_inf * fSlow); run_dragcrisis_pilot.py patches
    # per stage. Write the physical chi as a placeholder so a bare run is at
    # least seeded (fSlow=1 semantics).
    tq = d['freestream'].setdefault('turbulenceQuantities', {})
    tq['modelType'] = 'ModifiedTurbulentViscosityRatio'
    tq['modifiedTurbulentViscosityRatio'] = CHI_INF
    for bname, bcfg in d.get('boundaries', {}).items():
        if 'farfield' in bname or bcfg.get('type') == 'Freestream':
            btq = bcfg.setdefault('turbulenceQuantities', {})
            btq['modelType'] = 'ModifiedTurbulentViscosityRatio'
            btq['modifiedTurbulentViscosityRatio'] = CHI_INF
    # unsteady dual-time. The base case's ADAPTIVE CFL is a steady pseudo-
    # transient policy: in dual time it resets to CFL~0.1 at the start of
    # EVERY physical step and only crawls to ~4 within the pseudo budget
    # (verified on the first pilot launch; within-step residual fell just
    # 2-3x in 40 pseudo steps at 0.54 s/step). Use the standard unsteady
    # ramp CFL + relative-tolerance early exit instead.
    ts = d['timeStepping']
    ts['timeStepSize'] = DT
    ts['physicalSteps'] = 100          # stage driver overrides per stage
    # 24 sub-iterations give ~2 orders in-step residual drop on this mesh
    # (measured: 40 gave 2.5+ orders and never hit the 1e-2 relative exit
    # before ~37); 24 keeps the 22k-step pilot inside the GPU-hour budget.
    ts['maxPseudoSteps'] = 24
    ts['orderOfAccuracy'] = 2
    ts['CFL'] = {"type": "ramp", "initial": 1.0, "final": 100.0, "rampSteps": 15}
    ts.pop('adaptiveOthers', None)
    d['navierStokesSolver']['relativeTolerance'] = 1e-2
    d['turbulenceModelSolver']['relativeTolerance'] = 1e-2
    d.setdefault('fluidProperties', {})['sutherlandConstantDim'] = 110.4
    d['runControl']['restart'] = False
    d['runControl']['dumpRestartFilesFrequency'] = 1000
    # outputs: slice + surface get the kernel-check fields; time averages are
    # switched on by the stage driver in the production stage.
    vol = d.setdefault('volumeOutput', {}).setdefault('outputFields', [])
    for f in RES_FIELDS:
        if f not in vol:
            vol.append(f)
    # signed separation angles need the Cf VECTOR (the scalar "Cf" written on
    # surface pvtu is a magnitude -- see regen_eppler_v2.airfoil_walk_contour)
    for so in d.get('surfaceOutput', []) if isinstance(d.get('surfaceOutput'), list) else [d.get('surfaceOutput', {})]:
        for sn, sc in so.get('surfaces', {}).items():
            sf = sc.setdefault('outputFields', [])
            if 'CfVec' not in sf:
                sf.append('CfVec')
    for sn, sc in d.get('sliceOutput', {}).get('slices', {}).items():
        sf = sc.setdefault('outputFields', [])
        for f in RES_FIELDS:
            if f not in sf:
                sf.append(f)
    json.dump(d, open(cfg_path, 'w'), indent=1)


def make_steady_twin(unsteady_dir, steady_dir):
    """Clone the built unsteady case into a STEADY pseudo-transient twin with
    the campaign solver settings (adaptive CFL, timeStepSize inf — exactly the
    strL*prop_eppler387 blocks). Run it with run_dragcrisis_pilot.py
    --stage steady1 / steady2 (the staged-fSlow protocol)."""
    if os.path.exists(steady_dir):
        shutil.rmtree(steady_dir)
    shutil.copytree(unsteady_dir, steady_dir,
                    ignore=shutil.ignore_patterns(
                        "*.csv", "*.log", "restart*", "ipc*", "*.pvtu", "*.vtu",
                        "restartOutput", "visualize", "*.gltf"))
    p = os.path.join(steady_dir, "Flow360.json")
    d = json.load(open(p))
    d["timeStepping"] = {
        "CFL": {"convergenceLimitingFactor": 0.25, "max": 10000.0,
                "maxRelativeChange": 1.0, "min": 0.1, "type": "adaptive"},
        "adaptiveOthers": {"convergenceLimitingFactorChangeRate": 1.0,
                           "maxLimitingFactor": 2.0,
                           "nonLinearNormForForceJacobian": 5e-05,
                           "nonLinearNormForLimitingFactorChange": 1e-10},
        "maxPseudoSteps": 30000, "orderOfAccuracy": 2,
        "physicalSteps": 1, "timeStepSize": "inf",
        "absoluteTolerance": 1e-30,
    }
    d["navierStokesSolver"]["relativeTolerance"] = 0.0
    d["turbulenceModelSolver"]["relativeTolerance"] = 0.0
    d["turbulenceModelSolver"]["absoluteTolerance"] = 1e-30
    d["runControl"]["restart"] = False
    d["runControl"]["dumpRestartFilesFrequency"] = -1
    for k in ("timeAverageSliceOutput", "timeAverageSurfaceOutput"):
        d.pop(k, None)
    json.dump(d, open(p, "w"), indent=1)
    print(f"=== steady twin ready: {steady_dir} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/local_data/qiqi/sa-ai/dragcrisis_pilot")
    ap.add_argument("--name", default="cylL1_Re100k")
    ap.add_argument("--steady", action="store_true",
                    help="clone <out>/<name> into <out>/<name>_steady with the "
                         "campaign steady pseudo-transient settings (no re-mesh)")
    ap.add_argument("--re", type=float, default=RE,
                    help="Reynolds number based on D (sets muRef; mesh unchanged)")
    # matrix-extension mesh families (2026-07-28): 'lowre' (creeping/steady
    # arm, R_OUT=1000 for logarithmic low-Re blockage; y1=1e-3, BL is
    # O(D/sqrt(Re))), 'lowre300' (R_OUT=300 far-field-sensitivity twin),
    # 'highre' (transcritical arm to 1-2e7: y1=1e-6 for y+<=1, N_SURF=1600
    # for the shrinking shoulder LSB). Same growth<=1.1 solve as the pilot.
    ap.add_argument("--nsurf", type=int, default=N_SURF)
    ap.add_argument("--y1", type=float, default=Y1)
    ap.add_argument("--rout", type=float, default=R_OUT)
    args = ap.parse_args()
    if args.steady:
        make_steady_twin(os.path.join(args.out, args.name),
                         os.path.join(args.out, args.name + "_steady"))
        return

    case_dir = os.path.join(args.out, args.name)
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(case_dir, exist_ok=True)

    print("=== analytic O-grid (circle D=1) ===")
    X, Y, g, n_cells = build_ogrid(n_surf=args.nsurf, y1=args.y1,
                                   r_out=args.rout)
    stats = mesh_stats(X, Y, g, n_cells, y1=args.y1, r_out=args.rout)
    write_p3d(X, Y, os.path.join(case_dir, "cylinder_ogrid.p3d"))
    n_quads = write_msh_from_arrays(X, Y, case_dir)
    print(f"  wrote mesh.msh ({n_quads} quads, quasi-2D span {SPAN})")

    from rans.env import make_env
    from rans.config import CaseConfig
    from rans import case as _case, mesh as _mesh
    os.environ["AFT_CHI_INF"] = repr(CHI_INF)   # rans.case reads it at build time

    env, find = make_env()
    _mesh.gmsh_to_cgns(case_dir + "/mesh.msh", case_dir + "/mesh.cgns",
                       find("flow360gmshtocgns"), env)
    print("  wrote mesh.cgns")

    cfg = CaseConfig.load(CFG_JSON)
    cfg.flow.alpha_deg = 0.0
    cfg.flow.mach = MACH
    cfg.flow.reynolds = args.re
    cfg.elements[0].name = WALL
    _case.preprocess(case_dir, "mesh.cgns", find, env, cfg=cfg,
                     wall_names=[f"fluid/{WALL}"],
                     boundary_names=[f"fluid/farfield", f"fluid/{WALL}",
                                     "fluid/symmetry1", "fluid/symmetry2"],
                     timings={}, sdk_cache_dir=None,
                     sim_builder=_case.build_simulation_json)
    patch_flow360_unsteady(f"{case_dir}/Flow360.json", re=args.re)
    json.dump(stats, open(f"{case_dir}/mesh_stats.json", 'w'), indent=1)
    print(f"=== case ready: {case_dir} ===")


if __name__ == "__main__":
    main()
