"""Diagnose the alpha=-8 (and -4) NLF transition locations of fig:nlfaft.

Question (user): at c_l ~ -0.5 the plotted fronts look completely wrong --
upper (pressure side at negative lift) trips at 0.14-0.22 c while lower
(suction side, -Cp ~ 4.7 peak) stays 'laminar' to 0.89 c. Physically the
suction side should trip near the LE and the pressure side should run
laminar long. Determine whether this is (a) an extraction artifact
(surface mislabeling / probe leakage in converge_by_xtr's chi criterion),
or (b) the genuine solution.

Forensics per case: the signed-Cf walk per GEOMETRIC surface (transition
shows as the laminar->turbulent Cf jump) against the chi-probe fronts
(regen-style max-chi crossing), plus Cp to identify the suction side.

Run from paper/: python3 diag_negalpha_fronts.py [old|new]
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'repro', 'cfd'))
root = ('/home/qiqi/flexcompute/sa-ai/flow360_fv1'
        if (len(sys.argv) > 1 and sys.argv[1] == 'new')
        else '/home/qiqi/flexcompute/sa-ai/flow360_fr')
os.environ['SAAI_CFD_ROOT'] = root
import regen_nlf_v2 as R

NU = 2.5e-8




for tag in ('am4', 'am8'):
    for fam in ('str', 'cav'):
        d = f"{root}/{fam}L2prop_nlf0416_Re4M_{tag}"
        if not os.path.isdir(d):
            print(f"{fam} {tag}: missing"); continue
        (xu, cfu, cpu), (xl, cfl, cpl) = R.airfoil_walk_contour(d)
        xc, mu, ml = R.max_chi_vs_x(d)
        chi_u = mu / NU
        chi_l = ml / NU
        cross_u = xc[np.nanargmax(chi_u > 1.0)] if np.any(chi_u > 1.0) else None
        cross_l = xc[np.nanargmax(chi_l > 1.0)] if np.any(chi_l > 1.0) else None
        # suction side = surface with the more negative Cp minimum
        print(f"== {fam} L2 {tag}  ({root.split('/')[-1]})")
        print(f"   Cp min:  upper {np.min(cpu):+.2f}   lower {np.min(cpl):+.2f}"
              f"   -> suction side = {'upper' if np.min(cpu) < np.min(cpl) else 'lower'}")
        print(f"   chi>1 front (slice probe):  upper {cross_u}   lower {cross_l}")
        # Cf character at stations
        for side, x, cf in (('upper', xu, cfu), ('lower', xl, cfl)):
            o = np.argsort(x); x, cf = np.asarray(x)[o], np.asarray(cf)[o]
            samp = [(s, float(np.interp(s, x, cf))) for s in
                    (0.05, 0.1, 0.2, 0.4, 0.6, 0.85)]
            print(f"   Cf({side}): " + "  ".join(f"{s}:{c:+.2e}" for s, c in samp))
