"""Stage the spheroid RESEED cases: the freestream-turbulence
mis-specification test (user directive 2026-07-28).

The committed campaign ran every spheroid case at the flight-quiet seed
chi_inf = 8.76e-6 (N_crit = ln(c_v1/chi_inf) = 13.6, Mack Tu = 0.0103%),
while the measurements are DFVLR 3x3m tunnel data.  The tunnel's documented
disturbance level is Stock's limiting N_TS = 8.0 (AIAA J 44(1) 2006,
Fig. 11a -- calibrated on THESE measurements; no explicit Tu% is published
anywhere in the chain, see the 2026-07-28 reseed record).  Center seed =
that documented N_crit = 8.0 through the paper's Mack map (eq:tumap):
chi_inf = c_v1 * exp(-N_crit) = 2.382e-3 (Tu 0.1064%).  The re72a0 bracket
brackets it by Tu x/geo 2 (N_crit -+ 2.4*ln 2):

  tag      N_crit   Tu[%]    chi_inf      budget N(chi=1)=ln(1/chi_inf)
  n9p66    9.6636   0.0532   4.5126e-4    7.704
  n8       8.0000   0.1064   2.3818e-3    6.040
  n6p34    6.3364   0.2128   1.2571e-2    4.376

Campaign convention (verified against the committed case JSONs + the
ai_constants echo): AI_LAMINAR_SLOWDOWN = 1.0, seed is PHYSICAL, no
pre-compensation; the seed carries in freestream.turbulenceQuantities AND
the fluid/farfield Freestream BC; IC = cold-start freestream (so the IC
chi is the same new seed).

Cases staged (all half-model L1 O-grid, mesh_re65_L1.cgns -- topology and
level exonerated by 0413/0350; converged-protocol maxPseudoSteps = 45000
per 0350 Sec 6: 20k is a cold-start artifact, settle needs ~40-45k):

  case_ogrid_L1_saai_re72a0_reseed_{n9p66,n8,n6p34}  Re 7.2e6, alpha 0
  case_ogrid_L1_saai_re65a0_reseed_n8                Re 6.5e6, alpha 0
                             (muRef from the campaign's re65 family JSONs)
  case_ogrid_L1_saai_re72a2p5_reseed_n8              Re 7.2e6, alpha 2.5

Template = the matching committed campaign case Flow360.json with ONLY:
  - the seed edits above;
  - maxPseudoSteps 20000 -> 45000 (single converged leg);
  - volumeOutput REMOVED (disk discipline); sliceOutput gets the meridian
    plane z = 0 (phi = 90 flank), a near-symmetry plane y = +1e-5 (both
    phi = 0 leeward and phi = 180 windward meridians of the half model --
    needed for the alpha = 2.5 crossflow check; exactly y = 0 is the
    SlipWall itself) and constant-x planes at x/L = 0.20/0.42/0.70,
    fields primitiveVars + nuHat;
  - surfaceOutput/autoVisOutput retargeted at fluid/wall (the committed
    half-model files carry a stale "fluid/wing" key that matched nothing,
    same fix as build_fullbody_case.py).
Flow360Mesh.json is the template case's file verbatim (keeps the
half-model fluid/symmetry zone list).

Usage:
    python3 spheroid/build_reseed_cases.py [--root /local_data/qiqi/sa-ai/spheroid_fv1]
Then per case (on the run host, GPUs per the standing allocation):
    python3 daedalus/run_solution.py <case_dir> <gpu> saai
"""
import argparse
import json
import math
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'paper', 'repro'))
from lib.calibrate_kernel import A_TU, B_TU, C_V1, Tu_pct_from_chi_inf  # noqa: E402

MESHD = '/local_data/qiqi/sa-ai/spheroid_meshes'
A_ELL = 0.5                        # body centered at origin, x/L = mesh x + 0.5
SLICE_X = (0.20, 0.42, 0.70)
Y_SYMM_OFF = 1e-5                  # near-symmetry slice offset (mesh units)

LN2 = math.log(2.0)
SEEDS = {                          # tag -> N_crit (see module docstring)
    'n9p66': 8.0 + B_TU * LN2,     # Tu/2
    'n8':    8.0,                  # Stock's documented DFVLR N_TS
    'n6p34': 8.0 - B_TU * LN2,     # Tu*2
}
CASES = [
    # (template campaign case,            tag,     muRef override)
    ('case_ogrid_L1_saai_re72a0',        'n9p66',  None),
    ('case_ogrid_L1_saai_re72a0',        'n8',     None),
    ('case_ogrid_L1_saai_re72a0',        'n6p34',  None),
    ('case_ogrid_L1_saai_re72a0',        'n8',     'RE65'),   # re65a0: new condition
    ('case_ogrid_L1_saai_re72a2p5',      'n8',     None),
]


def chi_of(tag):
    return C_V1 * math.exp(-SEEDS[tag])


def build(root, tmpl_case, tag, mu_override):
    tmpl_dir = os.path.join(root, tmpl_case)
    d = json.load(open(os.path.join(tmpl_dir, 'Flow360.json')))

    name = tmpl_case
    if mu_override == 'RE65':
        # Re_L = 6.5e6: muRef from the campaign's re65 family (Mach/muRef)
        d['freestream']['muRef'] = 0.1 / 6.5e6
        name = name.replace('re72a0', 're65a0')
    case = os.path.join(root, f'{name}_reseed_{tag}')
    os.makedirs(case, exist_ok=True)

    chi = chi_of(tag)
    d['freestream']['turbulenceQuantities'][
        'modifiedTurbulentViscosityRatio'] = chi
    nfs = 0
    for bc in d['boundaries'].values():
        if bc.get('type') == 'Freestream':
            bc['turbulenceQuantities']['modifiedTurbulentViscosityRatio'] = chi
            nfs += 1
    assert nfs == 1 and d['initialCondition'] == {'type': 'freestream'}

    d['timeStepping']['maxPseudoSteps'] = 45000
    d.pop('volumeOutput', None)
    d['autoVisOutput']['surfaceOutput']['surfaces'] = {
        'fluid/wall': {'outputFields': ['Cp']}}
    for so in d.get('surfaceOutput', []):
        so['surfaces'] = {'fluid/wall': {
            'outputFields': ['Cp', 'Cf', 'CfVec', 'yPlus']}}
    slices = {
        'meridian': {'outputFields': ['primitiveVars', 'nuHat'],
                     'sliceNormal': [0.0, 0.0, 1.0],
                     'sliceOrigin': [0.0, 0.0, 0.0]},
        'symm': {'outputFields': ['primitiveVars', 'nuHat'],
                 'sliceNormal': [0.0, 1.0, 0.0],
                 'sliceOrigin': [0.0, Y_SYMM_OFF, 0.0]},
    }
    for xq in SLICE_X:
        slices[f'x{int(round(100 * xq)):03d}'] = {
            'outputFields': ['primitiveVars', 'nuHat'],
            'sliceNormal': [1.0, 0.0, 0.0],
            'sliceOrigin': [xq - A_ELL, 0.0, 0.0]}
    d['sliceOutput']['slices'] = slices

    with open(os.path.join(case, 'Flow360.json'), 'w') as f:
        json.dump(d, f, indent=4, sort_keys=True)
    shutil.copyfile(os.path.join(tmpl_dir, 'Flow360Mesh.json'),
                    os.path.join(case, 'Flow360Mesh.json'))

    dst = os.path.join(case, 'mesh.cgns')
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.join(MESHD, 'mesh_re65_L1.cgns'), dst)

    link = os.path.join(REPO, 'spheroid_fv1', os.path.basename(case))
    if not os.path.lexists(link):
        os.symlink(case, link)
    fs = d['freestream']
    print(f"staged {case}\n  N_crit {SEEDS[tag]:.4f}  chi_inf {chi:.4e}  "
          f"Tu {Tu_pct_from_chi_inf(chi):.4f}%  budget N(chi=1) "
          f"{math.log(1.0 / chi):.3f}\n  alpha {fs['alphaAngle']}  Re_L "
          f"{fs['Mach'] / fs['muRef']:.3e}  maxPseudoSteps "
          f"{d['timeStepping']['maxPseudoSteps']}  slices {sorted(slices)}")
    return case


def stage_restart_leg(case, steps):
    """Convert a finished case dir into a restart leg (0350 Sec 9
    mechanics): promote restartOutput/* to the case root, flip the IC to
    restart, set the per-invocation maxPseudoSteps, and archive the
    previous leg's forces CSV as total_forces_leg<n>.csv (the solver
    renumbers pseudo-steps from 0 and overwrites the CSV).  The original
    partitioner dmps stay -- restart dmps are partition-locked."""
    import glob
    import shutil as sh
    for f in glob.glob(os.path.join(case, 'restartOutput', 'restart*')):
        sh.copy2(f, case)
    n = 1
    while os.path.exists(os.path.join(case, f'total_forces_leg{n}.csv')):
        n += 1
    os.rename(os.path.join(case, 'total_forces_v2.csv'),
              os.path.join(case, f'total_forces_leg{n}.csv'))
    fp = os.path.join(case, 'Flow360.json')
    d = json.load(open(fp))
    d['initialCondition'] = {'type': 'restart'}
    d['runControl']['restart'] = True
    d['timeStepping']['maxPseudoSteps'] = int(steps)
    tmp = fp + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(d, f, indent=4, sort_keys=True)
    os.replace(tmp, fp)
    print(f'restart leg staged: {case} (+{steps} steps, previous forces '
          f'-> total_forces_leg{n}.csv)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/local_data/qiqi/sa-ai/spheroid_fv1')
    ap.add_argument('--restart-leg', metavar='CASE_DIR',
                    help='stage a restart leg on an existing case instead '
                         'of building the reseed set')
    ap.add_argument('--steps', type=int, default=10000)
    a = ap.parse_args()
    if a.restart_leg:
        stage_restart_leg(a.restart_leg, a.steps)
    else:
        assert abs(A_TU + 8.43) < 1e-12 and abs(B_TU - 2.4) < 1e-12
        for tmpl, tag, mu in CASES:
            build(a.root, tmpl, tag, mu)
