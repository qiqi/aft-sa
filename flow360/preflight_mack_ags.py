"""PREFLIGHT (no reruns): does adopting Mack (A=-8.43, B=2.4) still reproduce AGS
on the flat plate?

The amplification envelope N(Re_theta) = ln(chi/chi_inf) is seed-independent in the
pre-transition region (dChi/ds ~ a*omega*chi with a independent of chi), so a single
existing run defines N(Re_theta). Transition for ANY seed chi_inf is where the envelope
crosses N_crit = ln(c_v1/chi_inf). We read the envelope from the existing runs and
predict transition Re_theta under the Mack seeds, comparing to AGS.
"""
import os, sys, numpy as np
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/paper")
sys.path.insert(0, "/home/qiqi/flexcompute/aft-sa/scripts")
from regen_flatplate_flow360 import cf_and_retheta, case_dir, AGS_Reth, TU_LIST
import math

C_V1 = 7.1
A_OLD, B_OLD = -9.088, 2.705
A_MACK, B_MACK = -8.43, 2.4

def chi_inf(Tu_pct, A, B):
    return C_V1 * math.exp(-(A - B*math.log(Tu_pct/100.0)))

# --- Build the model envelope N(Re_theta) from each existing run. The seed used in
#     each existing run was the OLD map value (times fSlow, but fSlow cancels: chi_inf
#     in-domain relaxes to the BC value; we recover it from the run's own freestream). ---
env = {}   # Tu -> (Re_theta_array, N_array)  where N = ln(chi/chi_inf_run)
retr_old = {}  # measured transition Re_theta (chi crosses c_v1) in each existing run
for tu in TU_LIST:
    cd = case_dir(tu)
    xc, Re_th, cf_vol, chi_mx, _, _ = cf_and_retheta(cd)
    m = np.isfinite(Re_th) & np.isfinite(chi_mx) & (Re_th > 1)
    Re_th, chi_mx, xc = Re_th[m], chi_mx[m], xc[m]
    # freestream chi in-domain = min chi near inlet (laminar seed before growth)
    chi_seed = np.nanmin(chi_mx[xc < 0.5]) if (xc < 0.5).any() else np.nanmin(chi_mx)
    N = np.log(chi_mx / chi_seed)
    env[tu] = (Re_th, N, chi_seed)
    # transition Re_theta: first Re_theta where chi_mx >= c_v1
    idx = np.where(chi_mx >= C_V1)[0]
    retr_old[tu] = Re_th[idx[0]] if len(idx) else np.nan
    print(f"Tu={tu:.2f}% run: chi_seed(in-domain)={chi_seed:.3e}  N_max={N.max():.2f}  "
          f"Re_theta_tr(old,measured)={retr_old[tu]:.0f}")

# --- Use the LOWEST-Tu run (longest laminar runway, largest N range) as the master
#     envelope, since its N spans 0..~12 and covers every Mack N_crit. ---
master_tu = TU_LIST[0]
Re_env, N_env, _ = env[master_tu]
# monotonize the envelope in the laminar/amplifying region (N increasing with Re_theta)
o = np.argsort(Re_env); Re_env, N_env = Re_env[o], N_env[o]
# keep only up to the first time N reaches its running max near transition
def retheta_at_N(Ncrit):
    """Interpolate Re_theta where the master envelope crosses Ncrit."""
    # restrict to the monotone-rising part (before transition blow-up)
    kmax = np.argmax(N_env)
    Re_r, N_r = Re_env[:kmax+1], N_env[:kmax+1]
    if Ncrit <= N_r.min(): return Re_r[0]
    if Ncrit >= N_r.max(): return np.nan
    return float(np.interp(Ncrit, N_r, Re_r))

print(f"\nMaster envelope from Tu={master_tu}%: N spans [{N_env.min():.2f}, {N_env.max():.2f}]")
print(f"\n{'Tu%':>6} {'AGS Reth':>9} {'oldMap Ncrit':>12} {'Reth(old,pred)':>14} "
      f"{'Mack Ncrit':>11} {'Reth(Mack,pred)':>15} {'Mack err%':>9}")
for tu in TU_LIST:
    Nold = A_OLD - B_OLD*math.log(tu/100.0)
    Nmack = A_MACK - B_MACK*math.log(tu/100.0)
    re_old_pred = retheta_at_N(Nold)
    re_mack = retheta_at_N(Nmack)
    ags = AGS_Reth(tu)
    err = 100*(re_mack-ags)/ags if np.isfinite(re_mack) else float('nan')
    print(f"{tu:>6.2f} {ags:>9.0f} {Nold:>12.2f} {re_old_pred:>14.0f} "
          f"{Nmack:>11.2f} {re_mack:>15.0f} {err:>9.1f}")

print("\nSeeds (in-domain chi_inf) old vs Mack:")
for tu in TU_LIST:
    print(f"  Tu={tu:.2f}%: old={chi_inf(tu,A_OLD,B_OLD):.3e}  Mack={chi_inf(tu,A_MACK,B_MACK):.3e}")
