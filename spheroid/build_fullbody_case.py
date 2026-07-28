"""Stage the two FULL-BODY spheroid discriminator cases (alpha = 0,
Re_L = 7.2e6) on the full-circumference O-grid (ogrid_spheroid_full.py).

Arm A (case_ogridfull_L1_saai_re72a0): the half-model campaign case
spheroid_fv1/case_ogrid_L1_saai_re72a0 BYTE-COMPARABLE -- Flow360.json is
that case's file with ONLY the mesh-topology/output edits:
  - boundaries: fluid/symmetry removed (full body has none); fluid/wall +
    fluid/farfield unchanged (same freestream block, same chi_inf seed
    8.76e-6 = physical, fSlow = 1 campaign convention);
  - refArea DOUBLED (0.0109083 -> 0.0218166 = pi*B^2 full frontal area) so
    CL/CD print on the half-model scale;
  - volumeOutput REMOVED (disk discipline); instead sliceOutput gets the
    meridian plane z = 0 (both phi = +-90 meridians, for the BL/H/chi
    harvest) and constant-x planes at x/L = 0.20/0.42/0.70 (mesh
    x = -0.30/-0.08/+0.20; the azimuthal-staggering rings), fields
    primitiveVars + nuHat;
  - surfaceOutput points at fluid/wall (the half-model file carried a
    stale "fluid/wing" key that matched nothing).
Everything else -- solver blocks, Roe, lowMachPreconditioner false, adaptive
CFL max 200, maxPseudoSteps 20000, muRef, Mach 0.1 -- is byte-identical.

Arm B (case_ogridfull_L1_saai_re72a0_lowmach): identical to arm A except
navierStokesSolver.lowMachPreconditioner = true (the low-Mach-dissipation
arm of agent-paper-review/2026-07-28-0105 Sec 7).

Usage:
    python3 build_fullbody_case.py L1 [--root /local_data/qiqi/sa-ai/spheroid_fv1]
Then:
    python3 daedalus/run_solution.py spheroid_fv1/<case> <gpu> saai
"""
import argparse
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESHD = '/local_data/qiqi/sa-ai/spheroid_meshes'
A_ELL = 0.5                        # body centered at origin, x/L = mesh x + 0.5
SLICE_X = (0.20, 0.42, 0.70)       # x/L stations from the nose


def build(level, root, arm):
    tmpl = os.path.join(REPO, 'spheroid_fv1',
                        f'case_ogrid_{level}_saai_re72a0', 'Flow360.json')
    suffix = '' if arm == 'A' else '_lowmach'
    case = os.path.join(root, f'case_ogridfull_{level}_saai_re72a0{suffix}')
    os.makedirs(case, exist_ok=True)

    d = json.load(open(tmpl))
    del d['boundaries']['fluid/symmetry']
    d['geometry']['refArea'] = 2.0 * d['geometry']['refArea']
    d.pop('volumeOutput', None)
    # autoVisOutput must STAY (the solver dereferences it unconditionally --
    # removing it aborts with json type_error.305 at output-writer init);
    # only retarget its stale fluid/wing surface.
    d['autoVisOutput']['surfaceOutput']['surfaces'] = {
        'fluid/wall': {'outputFields': ['Cp']}}
    for so in d.get('surfaceOutput', []):
        so['surfaces'] = {'fluid/wall': {
            'outputFields': ['Cp', 'Cf', 'CfVec', 'yPlus']}}
    slices = {'meridian': {
        'outputFields': ['primitiveVars', 'nuHat'],
        'sliceNormal': [0.0, 0.0, 1.0], 'sliceOrigin': [0.0, 0.0, 0.0]}}
    for xq in SLICE_X:
        slices[f'x{int(round(100 * xq)):03d}'] = {
            'outputFields': ['primitiveVars', 'nuHat'],
            'sliceNormal': [1.0, 0.0, 0.0],
            'sliceOrigin': [xq - A_ELL, 0.0, 0.0]}
    d['sliceOutput']['slices'] = slices
    if arm == 'B':
        d['navierStokesSolver']['lowMachPreconditioner'] = True
        # REQUIRED companion key: the solver reads it unconditionally when
        # the preconditioner is on (NavierStokesSolver.cpp
        # initLowMachPreconditioner; missing -> nlohmann abort).  Convention
        # (flow360translator test refs): threshold = freestream Mach.
        d['navierStokesSolver']['lowMachPreconditionerThreshold'] = \
            d['freestream']['Mach']
    with open(os.path.join(case, 'Flow360.json'), 'w') as f:
        json.dump(d, f, indent=4, sort_keys=True)

    with open(os.path.join(case, 'Flow360Mesh.json'), 'w') as f:
        json.dump({
            'slidingInterfaces': [],
            'elementBasedGradientZones': [],
            'rotatingVolumeZones': [],
            'boundaries': {'noSlipWalls': ['fluid/wall'],
                           'periodicBoundaries': [],
                           'SymmetryPlane': []},
            'numHalos': 1,
            'zones': {'fluid': {
                'boundaryNames': ['fluid/farfield', 'fluid/wall'],
                'donorInterfaceNames': [], 'donorZoneNames': [],
                'receiverInterfaceNames': []}},
        }, f, indent=4)

    mesh = os.path.join(MESHD, f'mesh_re65full_{level}.cgns')
    dst = os.path.join(case, 'mesh.cgns')
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(mesh, dst)

    link = os.path.join(REPO, 'spheroid_fv1', os.path.basename(case))
    if not os.path.lexists(link):
        os.symlink(case, link)
    print(f'staged {case}\n  arm {arm}: lowMachPreconditioner = '
          f'{d["navierStokesSolver"]["lowMachPreconditioner"]}, refArea '
          f'{d["geometry"]["refArea"]}, seed '
          f'{d["freestream"]["turbulenceQuantities"]["modifiedTurbulentViscosityRatio"]}, '
          f'alpha {d["freestream"]["alphaAngle"]}, muRef '
          f'{d["freestream"]["muRef"]:.4e}, slices {sorted(slices)}')
    return case


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('level', help='L1 (ladder generic but L1 is the arm)')
    ap.add_argument('--root', default='/local_data/qiqi/sa-ai/spheroid_fv1')
    ap.add_argument('--arms', default='AB', help='which arms to stage')
    a = ap.parse_args()
    lev = 'L' + a.level.lstrip('Ll')
    for arm in a.arms:
        build(lev, a.root, arm)
