"""Set up a variant case directory from an existing cav_L1 unified case.

Variant A: chi^{1/4} σ_t + a_max=0.2 + chi_∞=1.6e-7  (true unified equivalent)
Variant B: chi-linear σ_t + a_max=0.2 + chi_∞=1.6e-7 (drop the 1/4 power)
Both fix the farfield BC chi_∞ that the solver actually reads.
"""
import json, shutil, sys, os
from pathlib import Path

F = Path("/home/qiqi/flexcompute/aft-sa/flow360")
SRC = F / "cav_L1_uni_a0"           # take the existing unified case as base
CHI_INF = 1.6e-7                    # target

def patch_flow360(j_path: Path, chi_inf: float):
    c = json.load(open(j_path))
    # farfield BC — the one the solver applies at the inflow
    for bk, bv in c.get("boundaries", {}).items():
        tq = bv.get("turbulenceQuantities")
        if tq and tq.get("modelType") == "ModifiedTurbulentViscosityRatio":
            tq["modifiedTurbulentViscosityRatio"] = chi_inf
            print(f"  Flow360.json: boundaries.{bk}.turbQuant = {chi_inf}")
    # freestream initial state — set consistent
    if "freestream" in c and "turbulenceQuantities" in c["freestream"]:
        c["freestream"]["turbulenceQuantities"]["modifiedTurbulentViscosityRatio"] = chi_inf
        print(f"  Flow360.json: freestream.turbQuant = {chi_inf}")
    json.dump(c, open(j_path, "w"), indent=1)

def patch_simulation(j_path: Path, chi_inf: float):
    c = json.load(open(j_path))
    found = False
    def walk(o, path=""):
        nonlocal found
        if isinstance(o, dict):
            if o.get("type_name") == "ModifiedTurbulentViscosityRatio" and "modified_turbulent_viscosity_ratio" in o:
                o["modified_turbulent_viscosity_ratio"] = chi_inf
                print(f"  simulation.json: {path} = {chi_inf}"); found = True
            for k, v in o.items(): walk(v, path+f".{k}")
        elif isinstance(o, list):
            for i, x in enumerate(o): walk(x, path+f"[{i}]")
    walk(c)
    if not found: print(f"  WARNING: no ModifiedTurbulentViscosityRatio found in {j_path}")
    json.dump(c, open(j_path, "w"), indent=1)

def main():
    variant = sys.argv[1]  # 'A' or 'B'
    dst = F / f"cav_L1_var{variant}_a0"
    if dst.exists():
        print(f"removing existing {dst}")
        shutil.rmtree(dst)
    print(f"copying {SRC.name} -> {dst.name}")
    shutil.copytree(SRC, dst)
    # Patch BC chi_inf in both JSONs
    patch_flow360(dst / "Flow360.json", CHI_INF)
    patch_simulation(dst / "simulation.json", CHI_INF)
    # Clean prior outputs
    for f in dst.iterdir():
        if f.suffix in ('.pvtu','.vtu','.pvd','.log','.sock') or f.name == 'ipc_data' or f.name == 'restartOutput':
            if f.is_dir(): shutil.rmtree(f, ignore_errors=True)
            else:
                try: f.unlink()
                except: pass
    print(f"done: {dst}")

if __name__ == "__main__":
    main()
