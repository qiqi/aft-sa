"""Stage the spheroid TUNNEL-CONDITION cases: matched Mach + facility seed.

Supersedes the M=0.1 / flight-quiet convention for the spheroid comparisons.
Two changes from `build_reseed_cases.py`, both user directives (2026-08-02):

1. **Mach = the tunnel's Mach**, not 0.1.  Inviscid compressibility at the
   ONERA F1 condition (M ~ 0.22) is a larger error than the density factor
   the transition sensors omit.  `muRef = Mach/Re_L`, so Mach cannot be
   changed without recomputing muRef -- otherwise Re silently moves.

2. **Every condition is run at BOTH the facility-calibrated seed and the
   facility-MEASURED seed**, because those two disagree by a factor 3-24 and
   the disagreement is the point (see the README of
   paper/data/ercoftac_case074 and paper/expert_feedback.md).

## Tunnel Mach

The DFVLR 3x3 m Goettingen tunnel is atmospheric, so Re_L fixes U_inf and
hence Mach.  Calibrated on the two conditions where ERCOFTAC Case 074
publishes both Re and U_inf:

    nwg10  Re 7.70e6, U 55.0 m/s -> nu = U L/Re = 1.714e-5, M = 0.157
    nwg30  Re 6.54e6, U 45.0 m/s -> nu = 1.651e-5,          M = 0.128

giving M ~ 1.994e-8 * Re_L (L = 2.4 m, a ~ 351 m/s at the tunnel's warm
running temperature -- it has no cooler).  That rule reproduces both anchors
to 2%.

The ONERA F1 tunnel is PRESSURISED (up to 4 bar), so Re_L does NOT fix
U_inf and the tunnel Mach of Stock's Fig. 17a conditions is NOT recoverable
from anything we hold -- it needs report IB 222-84 A 34.  For the
cross-facility pair we therefore run F1 at the SAME Mach as its matched
Goettingen condition, which is the correct choice anyway: it isolates the
seed as the only difference.  (For reference, f1_30 at Re 43.54e6 ran at
U 75.8 m/s, p0 3.8 bar -> M 0.216.)

## Seeds

chi_inf = c_v1 * exp(-N_crit), c_v1 = 7.1; Mack map N_crit = -8.43 -
2.4 ln(Tu_frac).  AI_LAMINAR_SLOWDOWN is 1.0 for every spheroid case, so the
seed is PHYSICAL -- no pre-compensation (unlike the airfoil campaign JSONs).

    tag        basis                                   N_crit   Tu%    chi_inf
    gcal       Stock Goettingen limiting N_TS = 8.0      8.00   0.106  2.382e-3
    gmeas      Goettingen hot wire, low end              5.28   0.330  3.604e-2
    gmeas_hi   Goettingen hot wire, high end             4.82   0.400  5.719e-2
    fcal       Stock CERT/ONERA F1 limiting N_TS = 7.0    7.00   0.161  6.474e-3
    fmeas      ONERA F1 hot wire (<0.1% -> upper bound)   8.15   0.100  2.053e-3
    wshop      1st AIAA TMW / gamma-Re_theta inflow       7.18   0.150  5.432e-3

The seed carries in `freestream.turbulenceQuantities` AND every Freestream
boundary, and the IC is asserted cold-start freestream: the far field is at
R_FF = 30 L, so a warm restart across seeds would need chi to advect 30 body
lengths.  A cold start puts the new seed everywhere at step 0 instead.  Cost
is the cold-start budget: maxPseudoSteps = 45000 (20k is a cold-start
artifact; settling needs ~40-45k -- record 0350 Sec 6).

`lowMachPreconditioner` is enabled: at the Re 1.52e6 conditions the tunnel
Mach is only 0.030, which is stiff without it.  Record 0413 established that
it leaves the converged laminar mean flow alone (H within 0.003, front
-0.0026) and converges ~2.5x faster, at the cost of +3.4e-4 in CD -- which
is immaterial here because no force measurements exist at these conditions.

Run from anywhere:  python3 spheroid/build_tunnel_cases.py --list
                    python3 spheroid/build_tunnel_cases.py --stage a0_gcal a0_gmeas
"""
import argparse
import json
import math
import os
import shutil

C_V1 = 7.1
L_MODEL = 2.4                      # m, ERCOFTAC Case 074
NU_TUNNEL = 1.68e-5                # m^2/s, mean of the two NWG anchors
A_SOUND = 351.0                    # m/s at the tunnel running temperature
STEPS = 45000

chi_of_N = lambda N: C_V1 * math.exp(-N)
N_of_Tu = lambda tu_pct: -8.43 - 2.4 * math.log(tu_pct / 100.0)
mach_goettingen = lambda Re: Re * NU_TUNNEL / (L_MODEL * A_SOUND)

SEEDS = {                          # tag -> chi_inf
    'gcal':     chi_of_N(8.00),
    'gmeas':    chi_of_N(N_of_Tu(0.33)),
    'gmeas_hi': chi_of_N(N_of_Tu(0.40)),
    'fcal':     chi_of_N(7.00),
    'fmeas':    chi_of_N(N_of_Tu(0.10)),
    'wshop':    chi_of_N(N_of_Tu(0.15)),
}

# tag -> (template case, alpha, Re_L, facility of the measurement)
CONDITIONS = {
    'a0':     ('case_ogrid_L1_saai_re72a0',   0.0,  7.20e6, 'NWG'),
    'a2p5':   ('case_ogrid_L1_saai_re72a2p5', 2.5,  7.20e6, 'NWG'),
    'a5lo':   ('case_ogrid_L1_saai_a5',       5.0,  1.52e6, 'NWG'),
    'a10lo':  ('case_ogrid_L1_saai_a10',     10.0,  1.52e6, 'NWG'),
    'a5hi':   ('case_ogrid_L1_saai_re65a5',   5.0,  6.49e6, 'NWG'),
    'a10hi':  ('case_ogrid_L1_saai_re65a10', 10.0,  6.56e6, 'NWG'),
    # ONERA F1 cross-facility twin of a10hi: same mesh, same Re, same Mach,
    # seed is the ONLY difference.  Measured front = Stock Fig. 17a squares.
    'a10f1':  ('case_ogrid_L1_saai_re65a10', 10.0,  6.56e6, 'F1'),
}


def build(root, cond_tag, seed_tag, dry=False):
    tmpl, alpha, Re, facility = CONDITIONS[cond_tag]
    chi = SEEDS[seed_tag]
    mach = mach_goettingen(Re)               # F1 twin deliberately matched
    case = os.path.join(root, f'case_ogrid_L1_tun_{cond_tag}_{seed_tag}')
    src = os.path.join(root, tmpl, 'Flow360.json')
    d = json.load(open(src))

    fs = d['freestream']
    fs['Mach'] = round(mach, 6)
    fs['muRef'] = mach / Re                  # MUST track Mach, else Re moves
    fs['alphaAngle'] = alpha
    fs['turbulenceQuantities'] = {
        'modelType': 'ModifiedTurbulentViscosityRatio',
        'modifiedTurbulentViscosityRatio': chi}

    nfs = 0
    for bc in d['boundaries'].values():
        if bc.get('type') == 'Freestream':
            bc.setdefault('turbulenceQuantities', {})
            bc['turbulenceQuantities'] = {
                'modelType': 'ModifiedTurbulentViscosityRatio',
                'modifiedTurbulentViscosityRatio': chi}
            nfs += 1
    assert nfs == 1, f'expected exactly one Freestream BC, found {nfs}'
    assert d['initialCondition'] == {'type': 'freestream'}, \
        'IC is not cold-start freestream -- chi would have to advect 30 L'

    # lowMachPreconditioner ALONE aborts: NavierStokesSolver does a const
    # json operator[] on lowMachPreconditionerThreshold, which the campaign
    # templates never carried because the preconditioner was off.  0.1 is the
    # value of the validated arm (case_ogridfull_L1_saai_re72a0_lowmach,
    # record 0413).
    d['navierStokesSolver']['lowMachPreconditioner'] = True
    d['navierStokesSolver']['lowMachPreconditionerThreshold'] = 0.1
    d['timeStepping']['maxPseudoSteps'] = STEPS

    print(f'{cond_tag:7s} {seed_tag:9s} alpha={alpha:5.1f} Re={Re:.3g} '
          f'M={mach:.4f} muRef={fs["muRef"]:.5g} chi_inf={chi:.4g} '
          f'({facility})  -> {os.path.basename(case)}')
    if dry:
        return case
    os.makedirs(case, exist_ok=True)
    tmp = os.path.join(case, 'Flow360.json.tmp')
    json.dump(d, open(tmp, 'w'), indent=1)
    os.replace(tmp, os.path.join(case, 'Flow360.json'))
    # Mesh inputs: reuse the template's.  The partition/native-dump artifacts
    # depend only on the mesh and Flow360Mesh.json, NOT on Flow360.json, and
    # every case here reuses its template's mesh -- so linking them skips
    # MeshPartitioner + MeshProcessor entirely (npart=2, i.e. 2 ranks/2 GPUs).
    for name in ('mesh.cgns', 'mesh.cgns.json',
                 'mesh.cgns.partitionerData.npart.2.json',
                 'mesh.cgns.partitionerData.npart.2_rank_1_of_1.dmp'):
        s = os.path.join(root, tmpl, name)
        t = os.path.join(case, name)
        if os.path.exists(s) and not os.path.exists(t):
            os.symlink(os.path.realpath(s), t)
    for name in ('Flow360Mesh.json', 'gpubind.sh'):
        s, t = os.path.join(root, tmpl, name), os.path.join(case, name)
        if os.path.exists(s) and not os.path.exists(t):
            shutil.copy2(s, t)
    os.chmod(os.path.join(case, 'gpubind.sh'), 0o755)
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/local_data/qiqi/sa-ai/spheroid_fv1')
    ap.add_argument('--stage', nargs='*', metavar='COND_SEED',
                    help='e.g. a0_gcal a0_gmeas a10hi_gcal a10f1_fcal')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    if a.list or not a.stage:
        print('conditions:')
        for k, (t, al, Re, fac) in CONDITIONS.items():
            print(f'  {k:7s} alpha={al:5.1f} Re={Re:9.3g} '
                  f'M_tunnel={mach_goettingen(Re):.4f} ({fac})  tmpl={t}')
        print('\nseeds:')
        for k, c in SEEDS.items():
            N = -math.log(c / C_V1)
            print(f'  {k:9s} N_crit={N:5.2f} Tu={100*math.exp(-(N+8.43)/2.4):6.3f}%'
                  f'  chi_inf={c:.4g}')
        return

    for spec in a.stage:
        cond, seed = spec.rsplit('_', 1)
        assert cond in CONDITIONS, f'unknown condition {cond}'
        assert seed in SEEDS, f'unknown seed {seed}'
        build(a.root, cond, seed, dry=a.dry_run)


if __name__ == '__main__':
    main()
