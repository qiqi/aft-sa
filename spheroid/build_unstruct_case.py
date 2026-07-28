"""Stage Flow360 cases for the UNSTRUCTURED (Flynn360 cavity-family)
spheroid meshes -- the alpha = 0, Re_L = 7.2e6 discriminating condition.

The case is the O-grid campaign case verbatim (Flow360.json copied from
spheroid_fv1/case_ogrid_<level>_saai_re72a0: same freestream Mach 0.1 /
muRef -> Re_L 7.2e6 / T 288.15, same chi_inf seed 8.76e-6 (fSlow = 1
campaign convention), same solver blocks, adaptive CFL max 200,
maxPseudoSteps 20000) with only the mesh-topology-driven edits:
  - boundaries: fluid/wall+fluid/symmetry+fluid/farfield (half O-grid)
    -> farfield/body (NoSlipWall) + farfield/farfield (Freestream);
    NO symmetry boundary (full-circumference body);
  - refArea DOUBLED (0.0109083 -> 0.0218166 = pi*B^2, full frontal
    area), so CL/CD print on the same scale as the half-model cases;
  - surfaceOutput points at farfield/body (Cp, Cf, CfVec, yPlus);
  - Flow360Mesh.json: noSlipWalls = ["farfield/body"] (wing cavity
    convention).

Usage:
    python3 build_unstruct_case.py L1 [--root /local_data/qiqi/sa-ai/spheroid_fv1]
Then:  python3 daedalus/run_solution.py <case_dir> <gpu> saai
"""
import argparse
import json
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESHD = '/local_data/qiqi/sa-ai/spheroid_meshes'


def build(level, root):
    tmpl = os.path.join(REPO, 'spheroid_fv1',
                        f'case_ogrid_{level}_saai_re72a0', 'Flow360.json')
    case = os.path.join(root, f'case_unstr_{level}_saai_re72a0')
    os.makedirs(case, exist_ok=True)

    d = json.load(open(tmpl))
    seed = d['boundaries']['fluid/farfield']['turbulenceQuantities']
    d['boundaries'] = {
        'farfield/farfield': {'type': 'Freestream',
                              'turbulenceQuantities': seed},
        'farfield/body': {'type': 'NoSlipWall', 'heatFlux': 0.0,
                          'roughnessHeight': 0.0},
    }
    d['geometry']['refArea'] = 2.0 * d['geometry']['refArea']
    for so in d.get('surfaceOutput', []):
        so['surfaces'] = {'farfield/body': {
            'outputFields': ['Cp', 'Cf', 'CfVec', 'yPlus']}}
    with open(os.path.join(case, 'Flow360.json'), 'w') as f:
        json.dump(d, f, indent=1, sort_keys=True)

    # full structural shape required (MeshProcessor asserts on missing keys;
    # the wing README's "strip nothing" lesson) -- the daedalus cavity
    # Flow360Mesh.json verbatim, boundary names identical by construction
    with open(os.path.join(case, 'Flow360Mesh.json'), 'w') as f:
        json.dump({
            'slidingInterfaces': [],
            'elementBasedGradientZones': [],
            'rotatingVolumeZones': [],
            'boundaries': {'noSlipWalls': ['farfield/body'],
                           'periodicBoundaries': [],
                           'SymmetryPlane': []},
            'numHalos': 1,
            'zones': {'farfield': {
                'boundaryNames': ['farfield/body', 'farfield/farfield'],
                'donorInterfaceNames': [], 'donorZoneNames': [],
                'receiverInterfaceNames': []}},
        }, f, indent=1)

    mesh = os.path.join(MESHD, f'mesh_unstr_{level}.cgns')
    dst = os.path.join(case, 'mesh.cgns')
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(mesh, dst)

    link = os.path.join(REPO, 'spheroid_fv1', os.path.basename(case))
    if not os.path.lexists(link):
        os.symlink(case, link)
    print('staged', case, f'(refArea {d["geometry"]["refArea"]}, seed '
          f'{seed["modifiedTurbulentViscosityRatio"]}, alpha '
          f'{d["freestream"]["alphaAngle"]}, muRef '
          f'{d["freestream"]["muRef"]:.4e})')
    return case


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('level', help='L0 | L1')
    ap.add_argument('--root', default='/local_data/qiqi/sa-ai/spheroid_fv1')
    a = ap.parse_args()
    lev = 'L' + a.level.lstrip('Ll')
    build(lev, a.root)
