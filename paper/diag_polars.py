"""Diagnostic drag polars + force-vs-alpha for both airfoils (SA-AI vs mfoil).
NOT a paper figure -- written to /tmp for examination.

For each airfoil, each (mesh, level, alpha) the converged CL/CD/CDp/CDf are the
median over the last 20% of the pseudo-step history in total_forces_v2.csv
(robust to any residual limit-cycle oscillation). Overlaid against the mfoil
e^N reference from the pickle.
"""
import os, csv, pickle, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = "/home/qiqi/flexcompute/aft-sa/flow360"

CONFIGS = {
    'nlf0416_Re4M':  dict(alphas=[0,4,9,15], mfoil='mfoil_nlf0416_Re4M.pkl',
                          title='NLF(1)-0416, Re=4e6'),
    'eppler387_Re200k': dict(alphas=[0,2,5,7], mfoil='mfoil_eppler387_Re200k.pkl',
                             title='Eppler 387, Re=2e5'),
}
# CSV columns (0-indexed): 1 pseudo_step, 2 CL, 3 CD, 11 CDPressure, 19 CDSkinFriction
COL = dict(step=1, CL=2, CD=3, CDp=11, CDf=19)

def converged_forces(cd):
    """Median of last 20% of history -> (CL, CD, CDp, CDf) or None."""
    p = f"{cd}/total_forces_v2.csv"
    if not os.path.exists(p): return None
    rows = list(csv.reader(open(p)))[1:]
    vals = {k: [] for k in COL}
    for r in rows:
        if len(r) <= COL['CDf']:
            continue
        try:
            for k, c in COL.items():
                vals[k].append(float(r[c]))
        except ValueError:
            pass
    if len(vals['CL']) < 3: return None
    n = len(vals['CL']); tail = slice(max(0, int(0.8*n)), n)
    return {k: float(np.median(np.array(v)[tail])) for k, v in vals.items()}

def gather(family, alphas):
    out = {}  # (mesh,level) -> dict(alpha->forces)  + 'mfoil'
    for mesh in ['cav', 'str']:
        for level in ['L0', 'L1', 'L2']:
            key = f"{mesh}{level}"
            series = {}
            for a in alphas:
                f = converged_forces(f"{B}/{mesh}{level}prop_{family}_a{a}")
                if f: series[a] = f
            if series: out[key] = series
    return out

def main():
    for family, cfg in CONFIGS.items():
        alphas = cfg['alphas']
        data = gather(family, alphas)
        mf = pickle.load(open(f"{B}/{cfg['mfoil']}", 'rb'))
        mf_a = [a for a in alphas if float(a) in mf]
        mf_cl = [mf[float(a)]['cl'] for a in mf_a]
        mf_cd = [mf[float(a)]['cd'] for a in mf_a]

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        style = {'cav': '--', 'str': '-'}
        lw = {'L0': 0.9, 'L1': 1.6, 'L2': 2.6}
        col = {'L0': 'C2', 'L1': 'C0', 'L2': 'C1'}
        for key, series in sorted(data.items()):
            mesh, level = key[:3], key[3:]
            aa = sorted(series)
            cl = [series[a]['CL'] for a in aa]
            cd = [series[a]['CD'] for a in aa]
            cdp = [series[a]['CDp'] for a in aa]
            cdf = [series[a]['CDf'] for a in aa]
            kw = dict(ls=style[mesh], lw=lw[level], color=col[level], marker='o', ms=4)
            ax[0].plot(aa, cl, **kw)
            ax[1].plot(aa, cd, **kw)
            ax[2].plot(cd, cl, **kw)
        # mfoil reference
        for a_ax, xx, yy in [(ax[0], mf_a, mf_cl), (ax[1], mf_a, mf_cd), (ax[2], mf_cd, mf_cl)]:
            a_ax.plot(xx, yy, 'k:', lw=1.6, marker='s', ms=4, label='mfoil $e^9$')
        ax[0].set_xlabel(r'$\alpha$ (deg)'); ax[0].set_ylabel('$C_L$'); ax[0].set_title('lift vs incidence')
        ax[1].set_xlabel(r'$\alpha$ (deg)'); ax[1].set_ylabel('$C_D$'); ax[1].set_title('drag vs incidence')
        ax[2].set_xlabel('$C_D$'); ax[2].set_ylabel('$C_L$'); ax[2].set_title('drag polar')
        for a_ax in ax: a_ax.grid(alpha=0.3)
        # legend
        handles = [Line2D([],[],color=col['L0'],lw=0.9,label='L0'),
                   Line2D([],[],color=col['L1'],lw=1.6,label='L1'),
                   Line2D([],[],color=col['L2'],lw=2.6,label='L2'),
                   Line2D([],[],color='0.3',ls='-',lw=1.6,label='str (O-grid)'),
                   Line2D([],[],color='0.3',ls='--',lw=1.6,label='cav (unstructured)'),
                   Line2D([],[],color='k',ls=':',lw=1.6,marker='s',ms=4,label='mfoil $e^9$')]
        ax[2].legend(handles=handles, fontsize=8, frameon=False, loc='best')
        fig.suptitle(f"{cfg['title']} — SA-AI converged forces vs mfoil (DIAGNOSTIC, not in paper)",
                     fontsize=12)
        plt.tight_layout(rect=(0,0,1,0.95))
        out = f"/tmp/polar_{family}.png"
        plt.savefig(out, dpi=130, bbox_inches='tight')
        plt.savefig(f"/tmp/polar_{family}.pdf", bbox_inches='tight')
        plt.close()
        print(f"wrote {out}")
        # also print the numbers
        print(f"  {cfg['title']}: converged forces")
        for key, series in sorted(data.items()):
            for a in sorted(series):
                s = series[a]
                print(f"    {key} a={a}: CL={s['CL']:+.4f} CD={s['CD']:.5f} "
                      f"(CDp={s['CDp']:.5f} CDf={s['CDf']:.5f})")
        for a in mf_a:
            print(f"    mfoil a={a}: CL={mf[float(a)]['cl']:+.4f} CD={mf[float(a)]['cd']:.5f}")

if __name__ == '__main__':
    main()
