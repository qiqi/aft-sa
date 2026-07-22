"""Section-characteristics table for the Eppler 387 alpha=5 Reynolds sweep
(Sec. VI.A): c_l, c_d, c_m(quarter-chord) from the structured and unstructured
L1 CFD, the e^9 panel reference (mfoil; XFOIL at 460k), and the LTPT experiment
(NASA TM-4062 Table B1, hand-read from the scanned listing at the tabulated
angle nearest alpha = 5 deg).

CFD: median over the last 20% of total_forces_v2.csv. Flow360's CMy is about
the leading edge (momentCenter=[0,0,0], momentLength=1), so the quarter-chord
moment is  c_m = CMy + 0.25*(c_l*cos(a) + c_d*sin(a))  (verified: str L1 at
Re=200k gives -0.0785 vs the experimental -0.0785..-0.0809).

mfoil is re-run here (alpha=5, Ma=0.1, ncrit=9, same eppler387.dat) to obtain
c_m, which the cached sweep pickle lacks; the fresh c_l/c_d are printed next to
the pickle values as a consistency check. XFOIL supplies the 460k row (mfoil
diverges there); its OPER echo carries Cm directly.

Experimental values (Table B1; corrected data; c_m about quarter chord):
  R= 60k RUNS 27,28 (M=0.05): a=4.99 -> .838/.0439/-.1139
  R=100k RUNS 15,16 (M=0.03): a=5.01 -> .873/.0237/-.0889 (repeat cd .0240)
  R=200k RUNS 9,10,13 (M=0.06): a=5.00 -> .891/.0138/-.0809
         (repeats a=5.03/5.06/5.06: .895/.894/.897, cd .0137-.0139)
  R=300k RUNS 3,4,5 (M=0.09): a=5.01 -> .901/.0114/-.0799
  R=460k RUN 20 (M=0.13): a=5.02 -> .914/.0093/-.0807
60k repeat-run scatter (RUNS 27,28, quoted in the text): at a=4.00 three
consecutive points give c_l = .643/.697/.721 (cd .0431/.0386/.0400); past
a~5 the measured lift collapses from .838 (a=4.99) to .639 (a=5.51).
"""
import os, sys, csv, pickle, subprocess
import numpy as np

B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fr")
DAT = '/home/qiqi/flexcompute/sa-ai/external/construct2d/eppler387.dat'
ALPHA = 5.0
RES = [60, 100, 200, 300, 460]

EXP = {  # Re_k: (alpha_deg, cl, cd, cm) -- TM-4062 Table B1, nearest alpha
    60:  (4.99, 0.838, 0.0439, -0.1139),
    100: (5.01, 0.873, 0.0237, -0.0889),
    200: (5.00, 0.891, 0.0138, -0.0809),
    300: (5.01, 0.901, 0.0114, -0.0799),
    460: (5.02, 0.914, 0.0093, -0.0807),
}

def cav_dir(Rk): return f"{B}/cavL1prop_eppler387_Re200k_a5" if Rk == 200 else f"{B}/sweep_Re{Rk}k_a5"
def str_dir(Rk): return f"{B}/strL1prop_eppler387_Re200k_a5" if Rk == 200 else f"{B}/sweep_str_Re{Rk}k_a5"

def cfd_forces(d):
    rows = [r for r in list(csv.reader(open(f"{d}/total_forces_v2.csv")))[1:] if len(r) > 10]
    t = rows[int(0.8 * len(rows)):]
    cl = float(np.median([float(r[2]) for r in t]))
    cd = float(np.median([float(r[3]) for r in t]))
    cmy = float(np.median([float(r[8]) for r in t]))
    a = np.radians(ALPHA)
    cm = cmy + 0.25 * (cl * np.cos(a) + cd * np.sin(a))
    return cl, cd, cm

def mfoil_point(Re):
    sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai/src/validation')
    import matplotlib as mpl; mpl.use('Agg'); mpl.rcParams['text.usetex'] = False
    from contextlib import redirect_stdout, redirect_stderr
    import mfoil as M
    coords = []
    for line in open(DAT):
        p = line.split()
        if len(p) >= 2:
            try: coords.append((float(p[0]), float(p[1])))
            except ValueError: pass
    m = M.mfoil(coords=np.array(coords).T)
    m.setoper(alpha=ALPHA, Re=Re * 1000.0, Ma=0.1)
    m.param.ncrit = 9.0
    with open(os.devnull, 'w') as nf:
        with redirect_stdout(nf), redirect_stderr(nf):
            m.solve()
    return float(m.post.cl), float(m.post.cd), float(m.post.cm), bool(m.glob.conv)

def xfoil_point(Re_k):
    cmds = f"""LOAD {DAT}
OPER
MACH 0.1
VISC {Re_k * 1000}
VPAR
N 9

ITER 400
ALFA {ALPHA}
ALFA {ALPHA}

QUIT
"""
    p = subprocess.run(['xvfb-run', '-a', 'xfoil'], input=cmds,
                       capture_output=True, text=True, timeout=240)
    cl = cd = cm = None
    for line in p.stdout.splitlines():
        s = line.strip()
        if s.startswith('a =') and 'CL =' in s:
            cl = float(s.split('CL =')[1].split()[0])
        if s.startswith('Cm =') and 'CD =' in s:
            cm = float(s.split('Cm =')[1].split()[0])
            cd = float(s.split('CD =')[1].split()[0])
    return cl, cd, cm

def main():
    mn = pickle.load(open(f"{B}/mfoil_eppler387_sweep_a5.pkl", 'rb'))
    rows = {}
    for Rk in RES:
        s = cfd_forces(str_dir(Rk))
        c = cfd_forces(cav_dir(Rk))
        if Rk in mn:
            cl, cd, cm, conv = mfoil_point(Rk)
            print(f"  mfoil Re={Rk}k fresh cl={cl:.3f} cd={cd:.4f} cm={cm:.4f} conv={conv} "
                  f"(pkl cl={mn[Rk]['cl']:.3f} cd={mn[Rk]['cd']:.4f})")
            e9 = (cl, cd, cm)
        else:
            cl, cd, cm = xfoil_point(Rk)
            print(f"  xfoil Re={Rk}k cl={cl} cd={cd} cm={cm}")
            e9 = (cl, cd, cm)
        rows[Rk] = (s, c, e9, EXP[Rk])

    print("\n% ---- LaTeX rows: Re | str L1 (cl cd cm) | cav L1 | e9 | Exp ----")
    for Rk in RES:
        s, c, e9, e = rows[Rk]
        def f3(v): return f"{v:.3f}"
        def f4(v): return f"{v:.4f}"
        print(f"    ${Rk}$k & {f3(s[0])} & {f4(s[1])} & {f4(s[2])} "
              f"& {f3(c[0])} & {f4(c[1])} & {f4(c[2])} "
              f"& {f3(e9[0])} & {f4(e9[1])} & {f4(e9[2])} "
              f"& {f3(e[1])} & {f4(e[2])} & {f4(e[3])} \\\\")

if __name__ == '__main__':
    main()
