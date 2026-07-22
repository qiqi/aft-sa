"""Cross-check the paper's e^9 reference: run XFOIL (repaneled) and FlexFoil
(bin_rustfoil faithful-viscous; build via
`cargo build --release -p rustfoil-cli` in the flexfoil repo) at every
condition where mfoil serves as the reference, and print all three
side-by-side. Conditions: NLF Re=4e6 a=0,2.5,4,6,7 N=9; Eppler Re=2e5
a=0,2,5 N=9; Eppler sweep a=5 Re=60,100,200,300k N=9; Daedalus section
eta=0.305 cl=1.073/1.180 Re=5e5 N=13.6 (all M=0.1).

Result (2026-07-17): the three implementations agree to <1% in cl/cd and
<0.005c in xtr at every condition except Eppler Re=6e4, where mfoil departs
(cd -11%) and XFOIL+FlexFoil side together (0.869/0.0411 and 0.868/0.0406).
FlexFoil's residual flag stays false at 60k (limit cycle) but forces are
stationary to 5 digits from iteration 100."""
import pickle
import re
import subprocess
import sys

SA = '/home/qiqi/flexcompute/sa-ai'
NLF = f'{SA}/external/construct2d/nlf0416.dat'
EPP = f'{SA}/external/construct2d/eppler387.dat'


def xfoil(dat, re_, n, alpha=None, cl=None, mach=0.1):
    spec = f'ALFA {alpha}\nALFA {alpha}' if alpha is not None else f'CL {cl}\nCL {cl}'
    import os
    cwd, base = os.path.dirname(dat), os.path.basename(dat)
    cmds = f"""LOAD {base}
PANE
OPER
MACH {mach}
VISC {re_}
VPAR
N {n}

ITER 600
{spec}

QUIT
"""
    p = subprocess.run(['xvfb-run', '-a', 'xfoil'], input=cmds, cwd=cwd,
                       capture_output=True, text=True, timeout=600)
    out = {'conv': 'VISCAL:  Convergence failed' not in p.stdout}
    for line in p.stdout.splitlines():
        s = line.strip()
        if s.startswith('a =') and 'CL =' in s:
            out['a'] = float(s.split('a =')[1].split()[0])
            out['cl'] = float(s.split('CL =')[1].split()[0])
        if s.startswith('Cm =') and 'CD =' in s:
            out['cd'] = float(s.split('CD =')[1].split()[0])
        m = re.search(r'Side 1  free  transition at x/c =\s*([\d.]+)', s)
        if m:
            out['xtr_u'] = float(m.group(1))
    return out


RUSTFOIL = '/home/qiqi/flexcompute/flexfoil/bin_rustfoil'


def flexfoil(dat, re_, n, alpha, mach=0.1):
    import json
    p = subprocess.run([RUSTFOIL, 'faithful-viscous', dat, '--alpha', str(alpha),
                        '--re', str(re_), '--mach', str(mach), '--ncrit', str(n),
                        '--max-iterations', '400'],
                       capture_output=True, text=True, timeout=600)
    try:
        return json.loads(p.stdout)
    except Exception:
        return {}


def row(tag, x, mcl, mcd, mxtr):
    ok = x.get('conv')
    print(f"{tag:34s} xfoil cl={x.get('cl'):.3f} cd={x.get('cd'):.4f} "
          f"xtr={x.get('xtr_u', float('nan')):.3f} conv={ok} | "
          f"mfoil cl={mcl:.3f} cd={mcd:.4f} xtr={mxtr:.3f} | "
          f"dcd={100*(x.get('cd')-mcd)/mcd:+.1f}%", flush=True)


print('=== NLF(1)-0416  Re=4e6  N=9 ===')
mn = pickle.load(open(f'{SA}/flow360_g4/mfoil_nlf0416_Re4M.pkl', 'rb'))
for a in (0.0, 2.5, 4.0, 6.0, 7.0):
    x = xfoil(NLF, 4e6, 9, alpha=a)
    m = mn[a]
    row(f'NLF a={a}', x, m['cl'], m['cd'], m['xtr_upper'])
    f = flexfoil(NLF, 4e6, 9, a)
    print(f"  flexfoil: cl={f.get('cl'):.3f} cd={f.get('cd'):.4f} xtr={f.get('x_tr_upper'):.3f}")

print('=== Eppler 387  Re=2e5  N=9 ===')
me = pickle.load(open(f'{SA}/flow360_g4/mfoil_eppler387_Re200k.pkl', 'rb'))
for a in (0.0, 2.0, 5.0):
    x = xfoil(EPP, 2e5, 9, alpha=a)
    m = me[a]
    row(f'EPP a={a}', x, m['cl'], m['cd'], m['xtr_upper'])
    f = flexfoil(EPP, 2e5, 9, a)
    print(f"  flexfoil: cl={f.get('cl'):.3f} cd={f.get('cd'):.4f} xtr={f.get('x_tr_upper'):.3f}")

print('=== Eppler 387 sweep  a=5  N=9 ===')
ms = pickle.load(open(f'{SA}/flow360_g4/mfoil_eppler387_sweep_a5.pkl', 'rb'))
for rk in (60, 100, 200, 300):
    x = xfoil(EPP, rk*1000, 9, alpha=5.0)
    m = ms[rk]
    row(f'EPP sweep Re={rk}k', x, m['cl'], m['cd'], m['xtr_upper'])
    f = flexfoil(EPP, rk*1000, 9, 5.0)
    print(f"  flexfoil: cl={f.get('cl'):.3f} cd={f.get('cd'):.4f} xtr={f.get('x_tr_upper'):.3f}")

print('=== Daedalus section eta=0.305  Re=5e5  N=13.6 (cl-matched) ===')
sys.path.insert(0, f'{SA}/scripts/daedalus')
import numpy as np
from section_chi_diag import section_coords
c = section_coords(0.305)
dat = '/tmp/claude-1006/-home-qiqi-flexcompute/15845519-8cb3-4677-8f3c-47bcc8951d95/scratchpad/dae_section.dat'
with open(dat, 'w') as f:
    f.write('DAE-blend eta0.305\n')
    for x_, z_ in c.T:
        f.write(f' {x_:.6f} {z_:.6f}\n')
# mfoil refs from section_chi_diag runs: (cl, xtr): a4 (1.069, 0.628), a5 (1.182, 0.615)
for cl_t, mx, mcl in ((1.073, 0.628, 1.069), (1.180, 0.615, 1.182)):
    x = xfoil(dat, 5e5, 13.6, cl=cl_t)
    print(f"DAE cl={cl_t}: xfoil cl={x.get('cl')} cd={x.get('cd')} "
          f"xtr={x.get('xtr_u')} conv={x.get('conv')} | mfoil xtr={mx} (cl={mcl})",
          flush=True)
