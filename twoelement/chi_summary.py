"""Per-surface amplification and flap loading vs alpha.

max chi over each surface is the decisive transition indicator here -- the shape
factor is not, because these layers amplify past c_v1 without H collapsing.
The flap's own suction peak is reported alongside so a bubble that vanishes
because the flap UNLOADED is not mistaken for one quenched by the wake.
"""
import glob, os, re, sys
import numpy as np
import plot_solution_mesh as PM, plot_paper_style as PS
from measure_l1_spacing import read_contours, surface_frame

C_V1 = 7.1
rows = []
pat = sys.argv[1] if len(sys.argv) > 1 else 'case_L*_v2_yp0.5*'
for d in sorted(glob.glob(pat)):
    if not os.path.isdir(d):
        continue
    m = re.match(r'case_(L\d)_v2_yp0\.5(?:_a([+-][\d.]+))?(?:_s[\d.]+)?$', d)
    if not m:
        continue
    lvl, al = m.group(1), float(m.group(2) or -1.0)
    try:
        hdr, pts, curves = read_contours('%s/contours_L1.txt' % d)
        walls = [n for w, n in curves if w]
        fld = PS.Field(d); nu = PS.nu_of(d)
    except Exception as e:
        print('%s %s: skip (%s)' % (lvl, al, str(e)[:50])); continue
    rec = {'lvl': lvl, 'al': al}
    for k, nm in enumerate(('fore', 'flap')):
        cont, seg, sarc, ile, u2 = surface_frame(pts, walls[k])
        pm = PS.probe_maxima(fld, cont, ile, u2, nu)
        for side in ('upper', 'lower'):
            if side in pm:
                rec['%s_%s' % (nm, side)] = float(np.nanmax(pm[side][3]))
    # flap suction peak
    g, p2, arr = PM.read_vtu('%s/surface_fluid_flap_proc0.vtu' % d)
    rec['flap_peak'] = float(np.nanmax(-arr['Cp']))
    rows.append(rec)

print()
print('%-4s %-6s | %-31s | %-31s | %8s'
      % ('lvl', 'alpha', 'max chi FORE (up / lo)', 'max chi FLAP (up / lo)',
         'flap -Cp'))
for r in sorted(rows, key=lambda z: (z['lvl'], z['al'])):
    f = lambda k: ('%9.2f' % r[k]) if k in r else '        -'
    mark = lambda k: ('T' if r.get(k, 0) > C_V1 else '.')
    print('%-4s %+6.1f | %s %s  %s %s | %s %s  %s %s | %8.3f'
          % (r['lvl'], r['al'], f('fore_upper'), mark('fore_upper'),
             f('fore_lower'), mark('fore_lower'), f('flap_upper'),
             mark('flap_upper'), f('flap_lower'), mark('flap_lower'),
             r['flap_peak']))
print("\n  T = max chi exceeds c_v1 = 7.1 somewhere on that surface")
