"""Diagnosis of the L0 (coarsest-grid) drag over-prediction, NLF vs Eppler
(RESPONSES.md 2026-07-25 ~11:40 UTC): overlays -Cp and upper-surface signed
Cf for L0 (dashed) against L2 (solid) on both mesh families, and prints the
pressure/friction drag split (CD_p from the closed surface-pressure integral,
friction as the remainder from the canon force totals).

Findings: NLF's L0 excess is an attached-BL FORM-DRAG error (fronts within
0.06c, Cp overlays; cavity L0's turbulent Cf is half the resolved level) --
model-independent.  Eppler's structured-L0 excess is a TRANSITION-ZONE
failure: the bubble comes out short and shallow with a soft early recovery
(the O-grid's cosine clustering is coarsest exactly at midchord where the
bubble lives, Delta_s 1.7-1.9%c vs the cavity's 1.1-1.4%c).

  python3 diag_l0_failure_modes.py     -> /tmp/l0_failure_modes.png
"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'repro', 'cfd'))
import regen_eppler_v2 as R

FR = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fr")

fig, axs = plt.subplots(2, 2, figsize=(13, 8))
for col, (af, zte, tag, tit) in enumerate(
        (('eppler387', 0.000833, 'eppler387_Re200k_a5', 'Eppler a5 Re=2e5'),
         ('nlf0416', 0.001665, 'nlf0416_Re4M_a4', 'NLF a4 Re=4e6'))):
    axp, axf = axs[0, col], axs[1, col]
    for fam, colr in (('str', 'C0'), ('cav', 'C1')):
        for lv, ls, lw in (('L0', '--', 1.0), ('L2', '-', 1.5)):
            (xu, cfu, cpu), (xl, cfl, cpl) = R.airfoil_walk_contour(
                f'{FR}/{fam}{lv}prop_{tag}', af=af, z_te_half=zte)
            o = np.argsort(xu)
            axp.plot(xu[o], -cpu[o], ls, color=colr, lw=lw, label=f'{fam} {lv}')
            axf.plot(xu[o], cfu[o], ls, color=colr, lw=lw)
            ol = np.argsort(xl)
            axp.plot(xl[ol], -cpl[ol], ls, color=colr, lw=lw * 0.7, alpha=0.5)
    axp.set_title(tit); axp.set_ylabel('$-C_p$ (upper bold, lower faint)')
    axp.grid(alpha=0.3); axp.legend(fontsize=7)
    axf.set_ylabel('$C_f$ upper'); axf.set_xlabel('$x/c$'); axf.grid(alpha=0.3)
    axf.axhline(0, color='gray', lw=0.5)
    axf.set_ylim(*((-0.006, 0.02) if af == 'eppler387' else (-0.002, 0.008)))
plt.tight_layout()
plt.savefig('/tmp/l0_failure_modes.png', dpi=120)
print('wrote /tmp/l0_failure_modes.png')


def _canon_cd(case):
    fn = f'{FR}/{case}/total_forces_v2.csv'
    hdr = open(fn).readline().split(',')
    iCD = [i for i, h in enumerate(hdr) if h.strip() == 'CD'][0]
    f = np.genfromtxt(fn, delimiter=',', skip_header=1)
    m = f[:, 1] >= f[-1, 1] - 500
    return float(f[m, iCD].mean())


def _cd_pressure(case, wall, alpha):
    """Wind-axis pressure drag from the closed surface-Cp integral;
    friction is the remainder against the canon force total."""
    import vtk as _vtk
    from vtk.util.numpy_support import vtk_to_numpy as _v2n
    r = _vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f'{FR}/{case}/surface_fluid_{wall}.pvtu'); r.Update()
    g = r.GetOutput()
    cp = _v2n(g.GetPointData().GetArray('Cp'))
    pts = _v2n(g.GetPoints().GetData())
    ids = _vtk.vtkIdList(); F = np.zeros(3)
    for i in range(g.GetNumberOfCells()):
        g.GetCellPoints(i, ids)
        idx = [ids.GetId(j) for j in range(ids.GetNumberOfIds())]
        if len(idx) < 3:
            continue
        P = pts[idx]; n = np.zeros(3)
        for j in range(1, len(idx) - 1):
            n += np.cross(P[j] - P[0], P[j + 1] - P[0]) / 2
        F += np.mean(cp[idx], axis=0) * n
    a = np.deg2rad(alpha); wind = np.array([np.cos(a), 0, np.sin(a)])
    span = pts[:, 1].max() - pts[:, 1].min()
    return abs(float(F @ wind) / span)


print(f"{'case':34s} {'CD':>7} {'CD_p':>7} {'CD_f':>7}   (x1e4)")
for _case, _wall, _a in (
        ('strL0prop_nlf0416_Re4M_a4', 'nlf0416', 4),
        ('strL2prop_nlf0416_Re4M_a4', 'nlf0416', 4),
        ('cavL0prop_nlf0416_Re4M_a4', 'nlf0416', 4),
        ('cavL2prop_nlf0416_Re4M_a4', 'nlf0416', 4),
        ('strL0prop_eppler387_Re200k_a5', 'eppler387', 5),
        ('strL2prop_eppler387_Re200k_a5', 'eppler387', 5),
        ('cavL0prop_eppler387_Re200k_a5', 'eppler387', 5),
        ('cavL2prop_eppler387_Re200k_a5', 'eppler387', 5)):
    _cd = _canon_cd(_case); _cdp = _cd_pressure(_case, _wall, _a)
    print(f'{_case:34s} {_cd*1e4:7.1f} {_cdp*1e4:7.1f} {(_cd-_cdp)*1e4:7.1f}')
