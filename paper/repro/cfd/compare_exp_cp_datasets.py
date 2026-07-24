"""Compare the multiple TM-4062 App. D pressure datasets near alpha=5 deg
(data/exp_cp_tables.json 'datasets_a5'): R=60k up-sweep vs down-sweep
(hysteresis runs) and the three R=100k tunnel conditions (Pt=5/M=0.08,
Pt=10/M=0.04, Pt=15/M=0.03+turbulator). The question: do the repeats show
branch differences at alpha~5?

Prints per-pair upper-surface RMS differences, the aft-recovery onset
(last station with dCp/dx < 0.5 before the trailing edge, i.e. where the
pressure recovery of turbulent reattachment begins), and the x/c=0.95
recovery level. -> figs_explore/exp_cp_datasets_a5.png (preview)
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_H = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(_H, '..', '..', 'data', 'exp_cp_tables.json')))
D = T['datasets_a5']
PREV = os.path.abspath(os.path.join(_H, '..', 'analytic', 'figs_explore'))


def rec_onset(xc, cp):
    """x/c where aft pressure recovery begins: the station after which Cp
    rises monotonically-ish to the TE (slope threshold on the aft half)."""
    xc, cp = np.asarray(xc), np.asarray(cp)
    m = xc >= 0.4
    x, c = xc[m], cp[m]
    dc = np.diff(c)/np.diff(x)
    steep = np.where(dc > 2.0)[0]          # reattachment recovery: ~0.15+/0.05c
    return float(x[steep[0]]) if len(steep) else np.nan


def stats(a, b):
    ua, ub = D[a]['upper'], D[b]['upper']
    d = np.asarray(ua['cp']) - np.asarray(ub['cp'])
    return np.sqrt(np.mean(d**2)), np.max(np.abs(d))


print("== aft-recovery onset x/c and Cp(0.95), upper surface ==")
for k, v in D.items():
    u = v['upper']
    print(f"  {k:16s} a={v['alpha']:.2f}  x_rec={rec_onset(u['xc'], u['cp']):.3f}"
          f"  Cp(.95)={u['cp'][-1]:+.4f}   [{v['label']}]")

print("== upper-surface pairwise differences (RMS, max |dCp|) ==")
for a, b in [("60k_up", "60k_dn"), ("60k_up_a551", "60k_dn_a550"),
             ("100k_pt5", "100k_pt5_a502"), ("100k_pt5", "100k_pt10"),
             ("100k_pt15", "100k_pt5"), ("100k_pt15", "100k_pt10"),
             ("100k_pt5", "100k_pt15_tape"), ("100k_pt10", "100k_pt15_tape")]:
    r, m = stats(a, b)
    print(f"  {a:16s} vs {b:16s}: rms={r:.4f}  max={m:.4f}")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
SETS = {0: [("60k_up", "C0", "-", "o", None),
            ("60k_dn", "C3", "--", "s", None),
            ("60k_up_a551", "C0", "-", "o", 0.35),
            ("60k_dn_a550", "C3", "--", "s", 0.35)],
        1: [("100k_pt15", "C4", "-.", "D", None),
            ("100k_pt5", "C0", "-", "o", None),
            ("100k_pt5_a502", "C0", "-", ".", 0.45),
            ("100k_pt10", "C2", "--", "s", None),
            ("100k_pt15_tape", "C1", ":", "^", None)]}
TIT = {0: "$R=6\\times10^4$: up-sweep vs down-sweep (hysteresis runs)",
       1: "$R=10^5$: three tunnel conditions"}
for i, ax in enumerate(axs):
    for k, c, ls, mk, al in SETS[i]:
        v = D[k]
        lab = f"$\\alpha={v['alpha']:.2f}^\\circ$ " + v['label'].split(', ', 1)[1]
        kw = dict(color=c, alpha=al if al else 1.0)
        ax.plot(v['upper']['xc'], -np.asarray(v['upper']['cp']), ls=ls, marker=mk,
                ms=3.5, lw=1.2, label=lab, **kw)
        ax.plot(v['lower']['xc'], -np.asarray(v['lower']['cp']), ls=ls, marker=mk,
                ms=2.0, lw=0.7, **kw)
    ax.set_title(TIT[i], fontsize=10)
    ax.set_xlabel('$x/c$'); ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc='upper right')
axs[0].set_ylabel('$-C_p$')
plt.tight_layout()
os.makedirs(PREV, exist_ok=True)
out = f"{PREV}/exp_cp_datasets_a5.png"
plt.savefig(out, dpi=140)
print("wrote", out)
