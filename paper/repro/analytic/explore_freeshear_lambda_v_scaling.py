"""Why a thin-Lambda_v-band gate would NOT generalize, and what does.

Lambda_v = -u'u'' d^3 / ((u'd)^2 + u^2) carries a d^3 wall-DISTANCE scaling.
For a wall-bounded BL the critical layer sits at d ~ delta, giving the
Lambda_v ~ +0.06 wavepacket locus. But for a FREE shear layer at wall
distance h >> thickness delta (backward-facing step, mixing layer), the
same KH critical layer maps to |Lambda_v| ~ h/delta, growing without bound:
a thin band at +0.06 would never amplify it -> no transition.

The flip side: |Lambda_v| at the critical layer IS a local, generalizable
DETACHMENT indicator (~ d/delta). So the generalizable fix is NOT to open
only on a band, but to keep Q open everywhere (free-shear-generalizable)
and GRADE the ceiling a_max by |Lambda_v|: full a_max for large |Lambda_v|
(detached free shear -> BFS transitions correctly), reduced for small
|Lambda_v| (wall-proximate just-separated layer -> the over-amplification
fix; Hammond-Redekopp wall stabilization).
"""
import numpy as np


def lambda_v_crit(h, delta=1.0, n=4000):
    """Peak |Lambda_v| in the KH core of a tanh layer centered at wall
    distance h (thickness delta), with d = wall distance."""
    y = np.linspace(0.02, h + 8*delta, n)
    u = 0.5*(1 + np.tanh((y - h)/delta))
    up = np.gradient(u, y)
    upp = np.gradient(up, y)
    Lv = -up*upp*y**3/((up*y)**2 + u**2 + 1e-30)
    core = np.abs(y - h) < 1.5*delta
    j = np.argmax(np.abs(Lv[core]))
    return Lv[core][j]


if __name__ == "__main__":
    print("free tanh shear layer (delta=1) at wall distance h:")
    print(f"{'h/delta':>8} {'Lambda_v (KH core)':>20}")
    for h in (1.5, 3, 6, 12, 25):
        print(f"{h:8.1f} {lambda_v_crit(h):20.2f}")
    print("\n|Lambda_v|_crit ~ h/delta: wall-bounded (h~delta) -> O(1);")
    print("free/detached (h>>delta) -> large. So |Lambda_v| = detachment,")
    print("and a_max should be graded by it, NOT gated to a thin band.")
