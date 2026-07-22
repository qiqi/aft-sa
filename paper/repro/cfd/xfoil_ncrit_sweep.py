"""XFOIL N_crit sweep on the Eppler 387 at Re=2e5, alpha in {5, 7}:
does any single N_crit reconcile both the alpha=5 match and the alpha=7
oil-flow reattachment (0.48)? Reports xtr_upper and the Cf-based
reattachment (last upward crossing of Cf=+1e-3 in the LSB window, the
paper's measure) for N_crit in {7, 9, 11, 13}."""
import subprocess, os
import numpy as np

DAT = '/home/qiqi/flexcompute/sa-ai/external/construct2d/eppler387.dat'
MACH, RE = 0.1, 200000

def run_xfoil(alpha, ncrit):
    dp_f = f'/tmp/xf_dump_n{ncrit}_a{alpha}.dat'
    if os.path.exists(dp_f):
        os.remove(dp_f)
    cmds = f"""LOAD {DAT}
OPER
MACH {MACH}
VISC {RE}
VPAR
N {ncrit}

ITER 400
ALFA {alpha}
ALFA {alpha}
DUMP {dp_f}

QUIT
"""
    p = subprocess.run(['xvfb-run', '-a', 'xfoil'], input=cmds,
                       capture_output=True, text=True, timeout=300)
    cl, conv, xtr = None, False, None
    for line in p.stdout.splitlines():
        s = line.strip()
        if s.startswith('a =') and 'CL =' in s:
            try:
                cl = float(s.split('CL =')[1].split()[0]); conv = True
            except Exception:
                pass
        if 'Point added at' in s and xtr is None:
            pass
    # transition location from the stdout 'side 1 free transition at' echoes
    for line in p.stdout.splitlines():
        if 'side 1  free  transition at x/c' in line.lower() or \
           ('side 1' in line and 'transition at x/c' in line):
            try:
                xtr = float(line.split('=')[-1].split()[0])
            except Exception:
                pass
    # Cf from DUMP: columns s, x, y, Ue, Dstar, Theta, Cf, H
    xs, cfs = [], []
    if os.path.exists(dp_f):
        for ln in open(dp_f):
            t = ln.split()
            if len(t) >= 8:
                try:
                    xs.append(float(t[1])); cfs.append(float(t[6]))
                except Exception:
                    continue
    xs = np.array(xs); cfs = np.array(cfs)
    # upper surface = from LE (min x) backwards to first point (TE upper);
    # dump runs TE-upper -> LE -> TE-lower -> wake. Take the first segment.
    ile = int(np.argmin(xs[:len(xs)//2 + 1])) if len(xs) else 0
    xu, cfu = xs[:ile+1][::-1], cfs[:ile+1][::-1]
    reatt = None
    m = (xu > 0.05) & (xu < 0.9)
    xw, cw = xu[m], cfu[m]
    for i in range(1, len(xw)):
        if cw[i] >= 1e-3 and cw[i-1] < 1e-3:
            f = (1e-3 - cw[i-1])/(cw[i] - cw[i-1])
            reatt = float(xw[i-1] + f*(xw[i] - xw[i-1]))
    return cl, conv, xtr, reatt

print(f"{'Ncrit':>6} {'alpha':>6} {'CL':>8} {'xtr_up':>8} {'reattach':>9}")
for ncrit in [7.0, 9.0, 11.0, 13.0]:
    for alpha in [5.0, 7.0]:
        try:
            cl, conv, xtr, re_ = run_xfoil(alpha, ncrit)
            print(f"{ncrit:6.0f} {alpha:6.0f} {cl if cl else float('nan'):8.3f} "
                  f"{xtr if xtr else float('nan'):8.3f} {re_ if re_ else float('nan'):9.3f}"
                  f"{'' if conv else '  NOT CONV'}", flush=True)
        except Exception as e:
            print(f"{ncrit:6.0f} {alpha:6.0f}  ERR {e}", flush=True)
