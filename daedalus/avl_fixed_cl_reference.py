"""AVL+XFOIL reference at MATCHED TOTAL LIFT (fixed-CL trim), not matched
alpha: AVL is trimmed with the 'A C <CL>' constraint to the canon RANS CL
(finest completed level per incidence), the induced drag is the Trefftz
CDff at that trim, and XFOIL runs the blended sections at the trimmed
strip loading. This makes the drag comparison lift-consistent -- at
matched alpha the camber-only lattice carries ~5% less lift (thin-airfoil
truncation; see the eta=0.31 decomposition in Sec. daedalus) and hence
~10% less induced drag.

Requires the double-precision AVL rebuild (tools/avl; the single-precision
near-field totals are roundoff casualties -- Appendix D).

  python3 avl_fixed_cl_reference.py
Prints one reference row per incidence and writes avl/ftcl_a{4,5,6}.txt,
avl/fscl_a{4,5,6}.txt (cached; delete to re-run AVL).
"""
import os
import re
import subprocess
import numpy as np
import sectional_compare as SC
from wing_geometry import chord, HALF_SPAN, C_ROOT

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = f'{HERE}/avl'
AVL_DP = '/home/qiqi/flexcompute/sa-ai/tools/avl/Avl/bin/avl'
MACH, RE_ROOT, NCRIT, S_REF = 0.1, 5.0e5, 13.6, 30.84
ETAS_X = [0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.85, 0.92, 0.97]


def canon_cl(a):
    """Target CL: finest completed canon case at this incidence."""
    for lv in ('L2', 'L1'):
        fn = f'{HERE}/case_ogrid_{lv}_saai_a{a}/total_forces_v2.csv'
        if not os.path.exists(fn) or sum(1 for _ in open(fn)) < 2001:
            continue
        hdr = open(fn).readline().split(',')
        iCL = [i for i, h in enumerate(hdr) if h.strip() == 'CL'][0]
        f = np.genfromtxt(fn, delimiter=',', skip_header=1)
        m = f[:, 1] >= f[-1, 1] - 500
        return float(f[m, iCL].mean()), lv
    raise RuntimeError(f'no completed canon case at a{a}')


def run_avl_cl(a, cl_tgt):
    ft, fs = f'ftcl_a{a}.txt', f'fscl_a{a}.txt'
    if not (os.path.exists(f'{WORK}/{ft}') and os.path.exists(f'{WORK}/{fs}')):
        cmds = (f'LOAD daedalus.avl\nOPER\nA C {cl_tgt:.4f}\nX\nFT\n{ft}\n'
                f'FS\n{fs}\n\nQUIT\n')
        subprocess.run([AVL_DP], input=cmds, capture_output=True, text=True,
                       cwd=WORK, timeout=1200)
    t = open(f'{WORK}/{ft}').read()
    g = lambda k: float(re.search(k + r'\s*=\s*([-\d.eE+]+)', t).group(1))
    rows = []
    for ln in open(f'{WORK}/{fs}'):
        w = ln.split()
        if len(w) >= 10:
            try:
                int(w[0]); rows.append((float(w[1]), float(w[2]), float(w[7])))
            except ValueError:
                continue
    r = np.array([x for x in rows if x[0] > 0])
    return g('Alpha'), g('CLtot'), g('CDff'), r[np.argsort(r[:, 0])]


def xfoil_cd(secfile, re_c, cl, ncrit):
    cmds = [f'LOAD {secfile}', 'PANE', 'OPER', f'MACH {MACH}',
            f'VISC {re_c:.0f}', 'VPAR', f'N {ncrit}', '', 'ITER 300',
            f'CL {cl:.4f}', f'CL {cl:.4f}', '', 'QUIT', '']
    p = subprocess.run(['xfoil'], input='\n'.join(cmds), capture_output=True,
                       text=True, cwd=WORK, timeout=240)
    cd = None
    for ln in p.stdout.splitlines():
        m = re.search(r'CD\s*=\s*([\d.eE+-]+)', ln)
        if m:
            cd = float(m.group(1))
    return cd


def profile_cd(strips):
    cds = []
    for e in ETAS_X:
        cl_loc = float(np.interp(e * HALF_SPAN, strips[:, 0], strips[:, 2]))
        cd = xfoil_cd(f'sec_st_{int(e*100):03d}.dat',
                      RE_ROOT * chord(e) / C_ROOT, cl_loc, NCRIT)
        cds.append(cd if cd else np.nan)
    cds = np.array(cds)
    good = np.isfinite(cds)
    ee = np.linspace(0.005, 0.995, 200)
    return 2.0 * np.trapezoid(np.interp(ee, np.array(ETAS_X)[good], cds[good])
                              * chord(ee), ee * HALF_SPAN) / S_REF


if __name__ == '__main__':
    print(f"{'a':>2} {'CL_tgt(src)':>14} {'trim_a':>7} {'CDff':>9} "
          f"{'CDp':>9} {'CD_ref':>9}")
    for a in (4, 5, 6):
        cl_tgt, lv = canon_cl(a)
        al, cl, cdff, strips = run_avl_cl(a, cl_tgt)
        cdp = profile_cd(strips)
        print(f"{a:>2} {cl_tgt:>9.4f}({lv}) {al:>7.3f} {cdff:>9.5f} "
              f"{cdp:>9.5f} {cdff + cdp:>9.5f}", flush=True)
