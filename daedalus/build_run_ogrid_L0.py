"""Rebuild the structured L0 mesh (the canon campaign only ran L1/L2)
and stage/run the three canon L0 cases. The ogrid L0 mesh regenerates
from the tracked ogrid_wing.py ladder; the predecessor campaign's L0
solutions and meshes are no longer on disk. The unstructured (cavity)
3D L0 mesh generator was not recovered -- if it resurfaces, mirror this
script for the cavity family.

  python3 build_run_ogrid_L0.py mesh          # build wing_ogrid_L0 -> mesh.cgns (CPU)
  python3 build_run_ogrid_L0.py stage         # stage case dirs from the L1 siblings
  python3 build_run_ogrid_L0.py run <gpu>     # run a4,a5,a6 sequentially on <gpu>
"""
import os
import sys
import json
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
MESH_DIR = f"{HERE}/_ogrid_L0_mesh"


def build_mesh():
    os.makedirs(MESH_DIR, exist_ok=True)
    subprocess.run([sys.executable, f"{HERE}/ogrid_wing.py", "0",
                    f"{MESH_DIR}/wing_ogrid_L0"], check=True)
    from rans import mesh as _mesh
    from rans.env import make_env
    env, find = make_env()
    _mesh.gmsh_to_cgns(f"{MESH_DIR}/wing_ogrid_L0.msh", f"{MESH_DIR}/mesh.cgns",
                       find("flow360gmshtocgns"), env)
    print("mesh.cgns:", os.path.getsize(f"{MESH_DIR}/mesh.cgns"), "bytes")


def stage():
    for a in (4, 5, 6):
        src, dst = f"{HERE}/case_ogrid_L1_saai_a{a}", f"{HERE}/case_ogrid_L0_saai_a{a}"
        os.makedirs(dst, exist_ok=True)
        for f in ("Flow360.json", "Flow360Mesh.json", "gpubind.sh"):
            p = f"{src}/{f}"
            if os.path.exists(p):
                shutil.copy(p, f"{dst}/{f}")
        mdst = f"{dst}/mesh.cgns"
        os.path.exists(mdst) and os.remove(mdst)
        os.link(f"{MESH_DIR}/mesh.cgns", mdst)
        print("staged", dst)


def run(gpu):
    for a in (4, 5, 6):
        subprocess.run([sys.executable, f"{HERE}/run_solution.py",
                        f"case_ogrid_L0_saai_a{a}", str(gpu), "saai"],
                       cwd=HERE, check=False)
        print(f"L0 a{a} done", flush=True)
    print("OGRID-L0-DONE", flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == "mesh":
        build_mesh()
    elif mode == "stage":
        stage()
    elif mode == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
