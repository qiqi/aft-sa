"""Regenerate the mfoil/XFOIL e^9 reference pickles the CFD figures overlay.

The original pickles were built ad hoc on another machine (flow360_g4 tree)
and never committed; this script reconstructs them into SAAI_CFD_ROOT with
the structure the consumers (regen_nlf_polar, regen_{nlf,eppler}_v2,
regen_epp_resweep_suite, regen_resweep_table) expect:

  mfoil_<case>.pkl:  {alpha_or_Rek: {'cl','cd','cm','conv',
                       'xtr_upper','xtr_lower',
                       'upper': {'x','cp','cf','n','H'}, 'lower': {...}}}
  xfoil_<case>.pkl:  {alpha_or_Rek: {'cl','cd','cm',
                       'upper': {'x','cp','cf'}, 'lower': {...}}}

mfoil: the in-repo solver (src/validation/mfoil.py, Fidkowski's port);
per-side splits and amplification from M.vsol.Is / M.post.sa; transition
x from M.vsol.Xt[side,1]. XFOIL: /usr/bin/xfoil via xvfb-run, CL/CD from
OPER echo, Cp from CPWR, x/Cf from the BL DUMP file.

Conditions (M=0.1, ncrit=9): NLF(1)-0416 Re=4e6 (mfoil a=0,2.5,4,6,7;
xfoil a=0,4,9,15); E387 Re=2e5 (both a=0,2,5,7); E387 sweep a=5
(mfoil Re=60,100,200,300,460k; xfoil sweep pickle already exists).

Run: python3 repro/cfd/regen_reference_pickles.py
"""
import os
import sys
import pickle
import subprocess
import tempfile
import numpy as np

SA = '/home/qiqi/flexcompute/sa-ai'
B = os.environ.get('SAAI_CFD_ROOT', f'{SA}/flow360_fr')
NLF_DAT = f'{SA}/external/construct2d/nlf0416.dat'
EPP_DAT = f'{SA}/external/construct2d/eppler387.dat'
sys.path.insert(0, SA)


def _coords(dat):
    pts = []
    for line in open(dat):
        p = line.split()
        if len(p) >= 2:
            try:
                pts.append((float(p[0]), float(p[1])))
            except ValueError:
                pass
    return np.array(pts).T


def mfoil_entry(dat, Re, alpha, ncrit=9.0, Ma=0.1):
    from contextlib import redirect_stdout, redirect_stderr
    import matplotlib as mpl
    mpl.use('Agg'); mpl.rcParams['text.usetex'] = False
    sys.path.insert(0, f'{SA}/src/validation')
    import mfoil as MF
    m = MF.mfoil(coords=_coords(dat))
    m.param.ncrit = ncrit
    m.param.doplot = False
    m.param.verb = 0
    m.setoper(alpha=float(alpha), Re=float(Re), Ma=Ma)
    try:
        with open(os.devnull, 'w') as nf, \
                redirect_stdout(nf), redirect_stderr(nf):
            m.solve()
        conv = bool(m.glob.conv)
    except Exception as e:
        print(f"    mfoil FAILED: {e}", flush=True)
        return dict(cl=np.nan, cd=np.nan, cm=np.nan, conv=False)
    x = m.foil.x[0, :]
    N = m.foil.N
    e = dict(cl=float(m.post.cl), cd=float(m.post.cd), cm=float(m.post.cm),
             conv=conv,
             xtr_lower=float(m.vsol.Xt[0, 1]), xtr_upper=float(m.vsol.Xt[1, 1]))
    for si, key in ((0, 'lower'), (1, 'upper')):
        Is = np.asarray(m.vsol.Is[si])
        Ia = Is[Is < N]                      # airfoil stations only
        e[key] = dict(x=np.asarray(x[Ia], float),
                      cp=np.asarray(m.post.cp[Ia], float),
                      cf=np.asarray(m.post.cf[Ia], float),
                      n=np.asarray(m.post.sa[Ia], float),
                      H=np.asarray(m.post.Hk[Ia], float))
    return e


def xfoil_entry(dat, Re, alpha, ncrit=9.0, Ma=0.1):
    with tempfile.TemporaryDirectory() as td:
        cpf, dpf = f'{td}/cp.out', f'{td}/bl.out'
        cmds = f"""PLOP
G F

LOAD {dat}
PANE
OPER
MACH {Ma}
VISC {Re}
VPAR
N {ncrit}

ITER 400
ALFA {alpha}
ALFA {alpha}
CPWR {cpf}
DUMP {dpf}

QUIT
"""
        p = subprocess.run(['xfoil'], input=cmds,
                           capture_output=True, text=True, timeout=600)
        cl = cd = cm = None
        for line in p.stdout.splitlines():
            s = line.strip()
            if s.startswith('a =') and 'CL =' in s:
                cl = float(s.split('CL =')[1].split()[0])
            if s.startswith('Cm =') and 'CD =' in s:
                cm = float(s.split('Cm =')[1].split()[0])
                cd = float(s.split('CD =')[1].split()[0])
        if cl is None:
            print("    xfoil FAILED (no forces echo)", flush=True)
            return None
        e = dict(cl=cl, cd=cd, cm=cm, conv=True)
        # Cp file: x y cp (airfoil nodes, TE->LE->TE)
        try:
            cpd = np.loadtxt(cpf, skiprows=1)
            xs, cps = cpd[:, 0], cpd[:, -1]
        except Exception:
            xs = cps = None
        # BL dump: s x y ue ds th cf  (airfoil+wake; wake rows have cf=0 tail)
        try:
            bld = np.loadtxt(dpf, skiprows=1)
            xb, cfb = bld[:, 1], bld[:, 6]
        except Exception:
            xb = cfb = None
        if xs is not None:
            # split at the LE (min x)
            i0 = int(np.argmin(xs))
            e['upper'] = dict(x=xs[:i0+1][::-1], cp=cps[:i0+1][::-1])
            e['lower'] = dict(x=xs[i0:], cp=cps[i0:])
        if xb is not None and 'upper' in e:
            na = len(xs)
            xba, cfa = xb[:na], cfb[:na]
            i0 = int(np.argmin(xba))
            e['upper']['cf'] = cfa[:i0+1][::-1]
            e['upper']['x_cf'] = xba[:i0+1][::-1]
            e['lower']['cf'] = cfa[i0:]
            e['lower']['x_cf'] = xba[i0:]
            # consumers index cf against 'x'; keep lengths consistent
            for k in ('upper', 'lower'):
                if len(e[k].get('cf', [])) == len(e[k]['x']):
                    e[k].pop('x_cf', None)
                else:
                    e[k]['cf'] = np.interp(e[k]['x'], e[k]['x_cf'], e[k]['cf'])
                    e[k].pop('x_cf')
        return e


def main():
    os.makedirs(B, exist_ok=True)
    jobs = [
        ('mfoil_nlf0416_Re4M.pkl', 'mfoil', NLF_DAT, 4.0e6,
         [0.0, 2.5, 4.0, 6.0, 7.0], 9.0),
        ('xfoil_nlf0416_Re4M.pkl', 'xfoil', NLF_DAT, 4.0e6,
         [0.0, 4.0, 9.0, 15.0], 9.0),
        ('mfoil_eppler387_Re200k.pkl', 'mfoil', EPP_DAT, 2.0e5,
         [0.0, 2.0, 5.0, 7.0], 9.0),
        ('xfoil_eppler387_Re200k.pkl', 'xfoil', EPP_DAT, 2.0e5,
         [0.0, 2.0, 5.0, 7.0], 9.0),
    ]
    for fname, tool, dat, Re, alphas, nc in jobs:
        out = {}
        print(f"== {fname} ==", flush=True)
        for a in alphas:
            e = (mfoil_entry if tool == 'mfoil' else xfoil_entry)(dat, Re, a, nc)
            if e is None:
                continue
            out[float(a)] = e
            print(f"  a={a}: cl={e.get('cl')} cd={e.get('cd')} "
                  f"conv={e.get('conv')} xtr_up={e.get('xtr_upper')}", flush=True)
        pickle.dump(out, open(f'{B}/{fname}', 'wb'))
        print(f"wrote {B}/{fname}", flush=True)

    out = {}
    print("== mfoil_eppler387_sweep_a5.pkl ==", flush=True)
    for Rk in (60, 100, 200, 300, 460):
        e = mfoil_entry(EPP_DAT, Rk*1000.0, 5.0, 9.0)
        out[int(Rk)] = e
        print(f"  Re={Rk}k: cl={e.get('cl')} cd={e.get('cd')} "
              f"conv={e.get('conv')} xtr_up={e.get('xtr_upper')}", flush=True)
    pickle.dump(out, open(f'{B}/mfoil_eppler387_sweep_a5.pkl', 'wb'))
    print(f"wrote {B}/mfoil_eppler387_sweep_a5.pkl", flush=True)
    # the sweep xfoil + flexfoil pickles already exist in flow360_ai; link them
    for f in ('xfoil_eppler387_sweep_a5.pkl', 'flexfoil_eppler387_sweep_a5.pkl',
              'flexfoil_nlf0416_Re4M.pkl'):
        src, dst = f'{SA}/flow360_ai/{f}', f'{B}/{f}'
        if os.path.exists(src) and not os.path.exists(dst):
            os.link(src, dst)
            print(f"linked {dst}", flush=True)
    print("REFERENCE-PICKLES-DONE", flush=True)


if __name__ == '__main__':
    main()
