"""Reconcile the Daedalus SECTIONAL drag comparison with the FORCE-TOTAL
comparison: chord-weighted integration of the exact sectional curves that
Fig. daepolar(b) plots, for both accountings --

  RANS:      solver strip cd from Y_slicing_forceDistribution.csv
             (sectional_compare.native_strips, wind axes),
  reference: AVL strip induced cl*a_i Trefftz-rescaled to CDff, plus the
             FlexFoil/XFOIL profile cd at the strip loading (the figure's
             own cd_ref recipe, on the FlexFoil eta stations),

each integrated as CD = 2 * trapz(cd_sec * c dy) / S_ref and compared with
its own ledger total (canon force means / matched-lift AVL+XFOIL row).
Prints each accounting's tiling error and the resulting offset between
Delta(sectional) and Delta(force totals).

  python3 daedalus_strip_accounting.py
"""
import os, sys, pickle
import numpy as np

sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai/daedalus')
sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai/paper')
import sectional_compare as SC
from polar_compare import run_avl
from wing_geometry import chord, HALF_SPAN

S_REF = 30.84
ALPHAS = [4, 5, 6]
# canon force totals (tab:daetotals; structured L2 vs matched-lift reference)
RANS_TOTAL = {4: 0.01997, 5: 0.02243, 6: 0.02509}
REF_TOTAL = {4: 0.02125, 5: 0.02315, 6: 0.02530}
RANS_CASE = {a: f'case_ogrid_L2_saai_a{a}' for a in ALPHAS}
FF = pickle.load(open('/home/qiqi/flexcompute/sa-ai/flow360_ai/'
                      'flexfoil_daedalus_strips.pkl', 'rb'))

print('alpha |   RANS strip integral    |    ref strip integral    | '
      'D_sec-D_tot')
for a in ALPHAS:
    SC.ALPHA = np.deg2rad(a)
    e, cl, cd = SC.native_strips(RANS_CASE[a])
    cd_rans = 2.0 * np.trapezoid(cd * chord(e), e * HALF_SPAN) / S_REF

    cl_t, cdi_t, strips = run_avl(a)
    eta_s = strips[:, 0] / HALF_SPAN
    ci = strips[:, 2] * strips[:, 3]                     # cl * a_i
    integ = 2.0 * np.trapezoid(ci * strips[:, 1], strips[:, 0]) / S_REF
    k = cdi_t / integ if integ > 0 else 1.0
    ff = FF[a]
    eta_ff = np.asarray(ff['eta'])
    cdp = np.array([st['cd'] if (st and st.get('cd') is not None) else np.nan
                    for st in ff['stations']], float)
    cd_ref_sec = np.interp(eta_ff, eta_s, ci * k) + cdp
    m = np.isfinite(cd_ref_sec)
    cd_ref = (2.0 * np.trapezoid(cd_ref_sec[m] * chord(eta_ff[m]),
                                 eta_ff[m] * HALF_SPAN) / S_REF)

    d_sec = (cd_rans - cd_ref) * 1e4
    d_tot = (RANS_TOTAL[a] - REF_TOTAL[a]) * 1e4
    print(f'{a}     | {cd_rans:.5f} ({(cd_rans-RANS_TOTAL[a])*1e4:+5.1f} ct '
          f'vs total) | {cd_ref:.5f} ({(cd_ref-REF_TOTAL[a])*1e4:+5.1f} ct '
          f'vs total) | {d_sec-d_tot:+5.1f} ct')
