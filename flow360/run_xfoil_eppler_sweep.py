"""Run XFOIL on the Eppler 387 at alpha=5, M=0.1, Ncrit=9 for the Reynolds
numbers where mfoil failed to converge (the thin high-Re bubble). Writes into
xfoil_eppler387_sweep_a5.pkl keyed by Re (in thousands), same per-surface shape
as mfoil_eppler387_sweep_a5.pkl: {Re_k: {'upper':{'x','cp','cf'},'lower':{...},
'cl','cd','conv'}}. Uses the SAME eppler387.dat contour as the mfoil reference
and the CFD meshes.

XFOIL's CPWR/DUMP writers call the plot library, so they need a display: we run
under `xvfb-run` (virtual framebuffer) and drive XFOIL through a stdin pipe
(a `< file` redirect trips an FPE on EOF in this build; the pipe does not).
"""
import subprocess, os, pickle, numpy as np

DAT = '/home/qiqi/flexcompute/aft-sa/external/construct2d/eppler387.dat'
PKL = '/home/qiqi/flexcompute/aft-sa/flow360/xfoil_eppler387_sweep_a5.pkl'
ALPHA = 5.0
MACH = 0.1
NCRIT = 9.0

def run_xfoil(Re):
    cp_f = f'/tmp/xf_cp_{int(Re)}.dat'
    dp_f = f'/tmp/xf_dump_{int(Re)}.dat'
    for f in (cp_f, dp_f):
        if os.path.exists(f): os.remove(f)
    cmds = f"""LOAD {DAT}
OPER
MACH {MACH}
VISC {int(Re)}
VPAR
N {NCRIT}

ITER 400
ALFA {ALPHA}
ALFA {ALPHA}
CPWR {cp_f}
DUMP {dp_f}

QUIT
"""
    p = subprocess.run(['xvfb-run', '-a', 'xfoil'], input=cmds,
                       capture_output=True, text=True, timeout=240)
    # CL/CD and convergence from the last converged OPER echo
    cl = cd = None; conv = False
    for line in p.stdout.splitlines():
        s = line.strip()
        if s.startswith('a =') and 'CL =' in s:
            try: cl = float(s.split('CL =')[1].split()[0]); conv = True
            except: pass
        if 'CD =' in s and '=>' in s:
            try: cd = float(s.split('CD =')[1].split()[0])
            except: pass
    return cp_f, dp_f, cl, cd, conv, p.stdout

def parse_surface(cp_f, dp_f):
    # CPWR: '#  x  Cp' -- surface nodes only, TE(upper)->LE->TE(lower)
    cp = np.array([[float(t) for t in l.split()]
                   for l in open(cp_f) if l.strip() and not l.startswith('#')])
    xcp, cpv = cp[:, 0], cp[:, 1]
    # DUMP: 's x y Ue/Vinf Dstar Theta Cf H ...' surface rows have >=8 cols; the
    # trailing wake has x>1 -- cut it off after the airfoil loop returns to TE.
    drows = []
    for l in open(dp_f):
        if l.strip().startswith('#') or not l.strip(): continue
        p = l.split()
        try: drows.append((float(p[1]), float(p[6])))   # x, Cf
        except: pass
    xd = np.array([r[0] for r in drows]); cfd = np.array([r[1] for r in drows])
    le = int(np.argmin(xd)); end = len(xd)
    for i in range(le + 1, len(xd)):
        if xd[i] > 1.0011: end = i; break               # wake begins
    xd, cfd = xd[:end], cfd[:end]
    def split(x, v):
        le = int(np.argmin(x))
        ux, uv = x[:le + 1][::-1], v[:le + 1][::-1]
        lx, lv = x[le:], v[le:]
        return (ux[np.argsort(ux)], uv[np.argsort(ux)]), (lx[np.argsort(lx)], lv[np.argsort(lx)])
    (uxc, ucp), (lxc, lcp) = split(xcp, cpv)
    (uxd, ucf), (lxd, lcf) = split(xd, cfd)
    out = {'upper': {'x': uxc, 'cp': ucp, 'cf': np.interp(uxc, uxd, ucf)},
           'lower': {'x': lxc, 'cp': lcp, 'cf': np.interp(lxc, lxd, lcf)}}
    return out

def build(Re):
    cp_f, dp_f, cl, cd, conv, log = run_xfoil(Re)
    if not (os.path.exists(cp_f) and os.path.exists(dp_f)):
        print(f"Re={Re}: XFOIL produced no output"); return None
    d = parse_surface(cp_f, dp_f)
    d.update(cl=cl, cd=cd, conv=conv)
    print(f"Re={Re/1000:.0f}k: CL={cl} CD={cd} conv={conv} "
          f"nUp={len(d['upper']['x'])} nLo={len(d['lower']['x'])} "
          f"Cp[TE_u]={d['upper']['cp'][-1]:.3f}")
    return d

if __name__ == '__main__':
    data = pickle.load(open(PKL, 'rb')) if os.path.exists(PKL) else {}
    for Re_k in [460]:
        d = build(Re_k * 1000)
        if d is not None: data[Re_k] = d
    pickle.dump(data, open(PKL, 'wb'))
    print("wrote", PKL, "keys=", sorted(data.keys()))
