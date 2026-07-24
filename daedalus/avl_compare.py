"""AVL + XFOIL cross-check of the Flow360 wing solutions.

AVL (vortex lattice, Trefftz-plane induced drag) on the same approximate
planform + blended DAE sections; XFOIL strip profile drag at the AVL local
cl and local Re, free transition at N_crit (matching the RANS receptivity)
and tripped (matching fully-turbulent RANS). Prints CL / CDi / CDp / CD
against the Flow360 numbers.

Usage: python3 avl_compare.py [alpha] [ncrit]
"""
import os
import re
import subprocess
import sys
import numpy as np
from wing_geometry import SectionFamily, chord, HALF_SPAN, C_ROOT, XQC

AVL_BIN = ('/tmp/claude-1006/-home-qiqi-flexcompute/'
           '15845519-8cb3-4677-8f3c-47bcc8951d95/scratchpad/Avl/bin/avl')
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'avl')
S_REF, C_REF, B_REF = 30.84, 0.903, 34.14
MACH = 0.1
RE_ROOT = 5.0e5           # matches the Flow360 cases (muRef = M / (Re/c_root))

ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
NCRIT = float(sys.argv[2]) if len(sys.argv) > 2 else 13.6


def write_section(fam, eta, path):
    """Unit-chord Selig loop of the blended section at eta."""
    zu, zl = fam._blend(eta)
    xs = fam.xs
    xw = np.concatenate([xs[::-1], xs[1:-1]])
    zw = np.concatenate([zu[::-1], zl[1:-1]])
    with open(path, 'w') as f:
        f.write(f'DAE blend eta={eta:.2f}\n')
        for x, z in zip(xw, zw):
            f.write(f' {x:.6f} {z:.6f}\n')


def build_model():
    os.makedirs(WORK, exist_ok=True)
    fam = SectionFamily(64)
    etas = [0.0, 0.30, 0.60, 0.88, 1.0]
    for e in etas:
        write_section(fam, e, f'{WORK}/sec_{int(e*100):03d}.dat')
    with open(f'{WORK}/daedalus.avl', 'w') as f:
        f.write(f"""Daedalus approximate wing
{MACH}
0  0  0
{S_REF}  {C_REF}  {B_REF}
{XQC}  0.0  0.0
SURFACE
Wing
12  1.0  40  -1.0
YDUPLICATE
0.0
""")
        for e in etas:
            c = chord(e)
            y = e * HALF_SPAN
            f.write(f"""SECTION
{XQC - 0.25 * c}  {y}  0.0  {c}  0.0
AFILE
sec_{int(e*100):03d}.dat
""")
    return etas


def run_avl():
    cmds = f"""LOAD daedalus.avl
OPER
A A {ALPHA}
X
FT
ft.txt
FS
fs.txt

QUIT
"""
    p = subprocess.run([AVL_BIN], input=cmds, capture_output=True, text=True,
                       cwd=WORK, timeout=300)
    ft = open(f'{WORK}/ft.txt').read()
    cl = float(re.search(r'CLtot\s*=\s*([\d.eE+-]+)', ft).group(1))
    # Trefftz CDff, not near-field CDind (single-precision CDind is a
    # small-difference-of-large-terms casualty at this AR; see Appendix D)
    cdi = float(re.search(r'CDff\s*=\s*([\d.eE+-]+)', ft).group(1))
    e_osw = re.search(r'e =\s*([\d.eE+-]+)', ft)
    # strip data: columns j Yle Chord Area c_cl ai cl_norm cl cd ...
    strips = []
    for ln in open(f'{WORK}/fs.txt'):
        t = ln.split()
        if len(t) >= 8:
            try:
                j = int(t[0])
                yle, ch = float(t[1]), float(t[2])
                cl_s = float(t[7])
                strips.append((yle, ch, cl_s))
            except ValueError:
                continue
    return cl, cdi, (float(e_osw.group(1)) if e_osw else np.nan), np.array(strips)


def xfoil_cd(secfile, re_c, cl, ncrit, tripped):
    cmds = [f'LOAD {secfile}', 'PANE', 'OPER', f'MACH {MACH}', f'VISC {re_c:.0f}',
            'VPAR', f'N {ncrit}']
    if tripped:
        cmds += ['XTR 0.02 0.02']
    cmds += ['', 'ITER 300', f'CL {cl:.4f}', f'CL {cl:.4f}', '', 'QUIT', '']
    p = subprocess.run(['xvfb-run', '-a', 'xfoil'], input='\n'.join(cmds),
                       capture_output=True, text=True, cwd=WORK, timeout=240)
    cd = None
    for ln in p.stdout.splitlines():
        m = re.search(r'CD\s*=\s*([\d.eE+-]+)', ln)
        if m:
            cd = float(m.group(1))      # keep the last (converged) value
    return cd


def profile_drag(strips, ncrit, tripped):
    """XFOIL cd at ~10 stations, then integrate over the strip distribution."""
    fam = SectionFamily(64)
    ys = strips[:, 0]; chords = strips[:, 1]; cls = strips[:, 2]
    pos = ys > 0
    ys, chords, cls = ys[pos], chords[pos], cls[pos]
    order = np.argsort(ys)
    ys, chords, cls = ys[order], chords[order], cls[order]
    etas_st = np.array([0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.85, 0.92, 0.97])
    cds = []
    for e in etas_st:
        y = e * HALF_SPAN
        c = chord(e)
        cl_loc = np.interp(y, ys, cls)
        re_c = RE_ROOT * c / C_ROOT
        sec = f'{WORK}/sec_st_{int(e*100):03d}.dat'
        if not os.path.exists(sec):
            write_section(fam, e, sec)
        cd = xfoil_cd(os.path.basename(sec), re_c, cl_loc, ncrit, tripped)
        cds.append(cd if cd is not None else np.nan)
        print(f'   eta {e:.2f}: c {c:.3f} m, Re {re_c:.2e}, cl {cl_loc:.3f} '
              f'-> cd {cds[-1] if cds[-1] is not None else float("nan"):.5f}',
              flush=True)
    cds = np.array(cds, dtype=float)
    good = np.isfinite(cds)
    # integrate cd*c over the semispan (fill gaps by interpolation)
    ee = np.linspace(0.005, 0.995, 200)
    cd_e = np.interp(ee, etas_st[good], cds[good])
    c_e = chord(ee)
    cdp = 2.0 * np.trapz(cd_e * c_e, ee * HALF_SPAN) / S_REF
    return cdp


if __name__ == '__main__':
    build_model()
    cl, cdi, e_osw, strips = run_avl()
    print(f'AVL @ alpha={ALPHA}: CLtot {cl:.4f}, CDff {cdi:.5f}, e {e_osw:.3f}, '
          f'{len(strips)} strips')
    print(f'-- XFOIL strips, free transition N={NCRIT}:')
    cdp_free = profile_drag(strips, NCRIT, tripped=False)
    print(f'-- XFOIL strips, tripped (xtr=0.02):')
    cdp_trip = profile_drag(strips, NCRIT, tripped=True)
    print()
    print(f'{"":22} {"CL":>8} {"CDi":>9} {"CDp":>9} {"CD":>9}')
    print(f'{"AVL+XFOIL free N13.6":22} {cl:8.4f} {cdi:9.5f} {cdp_free:9.5f} '
          f'{cdi + cdp_free:9.5f}')
    print(f'{"AVL+XFOIL tripped":22} {cl:8.4f} {cdi:9.5f} {cdp_trip:9.5f} '
          f'{cdi + cdp_trip:9.5f}')
    print(f'{"Flow360 O-grid SA-AI":22} {"1.0253":>8} {"":>9} {"":>9} {"0.01915":>9}')
    print(f'{"Flow360 cavity SA-AI":22} {"1.0177":>8} {"":>9} {"":>9} {"0.02098":>9}')
    print(f'{"Flow360 O-grid turb":22} {"0.9861":>8} {"":>9} {"":>9} {"0.02620":>9}')
    print(f'{"Flow360 cavity turb":22} {"0.9853":>8} {"":>9} {"":>9} {"0.02799":>9}')
