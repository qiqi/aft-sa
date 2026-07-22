# sigma-d-tie: pointwise linearity tie — verification log

Proposal: replace the calibrated destruction width tau_D by the algebraic tie
sigma_D = 1 - R*(1 - sigma_P), R = c_b1/(kappa^2 c_w1) = 0.2489
(floor 0.7511 at sigma_P = 0; floor + slope = 1 by the definition of c_w1),
so that nuHat = kappa*u_tau*y is an EXACT solution of the gated SA-AI wall
layer at every height, for any tau. Motivation: recover standard SA's linear
nuHat (destroyed by the chi-gates below chi ~ 20; only the intercept was
repaired by tau_D = 1.36).

## Test 1 — wall-layer exactness (test_wall_layer_tie.py)  PASS
Constant-stress solve, standard SA vs SA-AI with the tie:
max relative deviation from kappa*y+ = 3e-16 (both); dB = 0.0 exactly;
dnut identically 0. tau_D eliminated.

## Test 2 — laminar-instrument destruction floor (test_destruction_floor.py)
Destruction (2012-limited S-tilde, f_w, implicit) added to the eq:transport
march at the PHYSICAL amplitude chi = chi_inf * nuhat, chi_inf = c_v1 e^-9
(Mack N_crit = 9 seed). Envelope measures, floor on vs off:

| profile           | Rt1     | Rt9     | mean    | late    |
|-------------------|---------|---------|---------|---------|
| Blasius (anchor)  | +0.41%  | +0.37%  | -0.35%  | -0.44%  |
| sep-limit (anchor)| +0.84%  | +0.57%  | -0.37%  | +0.88%  |
| beta=0.35 (K_r)   | +0.04%  | +0.85%  | -2.57%  | -5.96%  |

Anchors: within the triple-solve residuals (2.2%) -> calibration survives;
re-solve would move digits sub-1%. K_r fit point: mean -2.6% -> K_r would
re-fit slightly lower (~5.3-5.4) on adoption. D/P at the band peak confirms
the scaling: 2.5e-7 (N=2), 6.7e-5 (N=5.7), 1.1e-2 (N=8.3, chi=3.5).

CAUTION (protocol): with destruction the pure-amplification envelope
saturates near N ~ 12; the x_max adaptation must target N <= 10.5 or the
under-resolved march fakes huge shifts (first attempt showed +110% Rt9,
entirely protocol pollution). Same lesson as feedback-reproduce-match-protocols.

## Test 3 — CFD A/B (Flow360, AI_SIGMAD_TIE=1, constants held fixed)
compute branch sigma-d-tie: env flag AI_SIGMAD_TIE (ModelConstants.h,
SATurbulenceSolver.cpp, SAAiTransition.h, SpalartAllmaras.h). See below.

### Test 3 results (2026-07-15)

Flat plate Tu=0.04% (plain protocol, strict A/B: same rebuilt binary,
same driver, AI_SIGMAD_TIE=1 vs 0):
- transition front (max Cf jump): x = 3.187 (tie) vs 3.139 (control)
  -> +1.5% in x = +0.76% in Re_theta, aftward (conservative), consistent
  with the instrument's Blasius mean-rate shift (-0.35%).
- turbulent Cf @ x=5: +0.22% (virtual-origin shift of the later front);
  laminar Cf identical to 6 digits.
- NOTE the shipped flow360_g4 baseline sits at x = 3.045: the fresh
  control differs from it by +3.1% in x (build/protocol drift, nothing
  to do with the tie). Only same-binary A/B is meaningful.

Eppler 387 a5 strL2 (ladder protocol, CONVERGED 39992 steps both, strict
A/B): CL 0.9316 vs 0.9315, CD 0.01229 vs 0.01229, front 0.580 vs 0.580;
pointwise max|dCf| = 2e-5 (0.02% of the Cf peak, at x/c = 0.55). The
adverse/bubble regime is unchanged by the tie, as predicted (bubble at
high Gamma, destruction floor invisible below chi = 1, turbulent side
exact by the identity).

NLF(1)-0416 a4 strL2 (ladder, CONVERGED 49990 steps both, strict A/B):
fronts unchanged to extraction precision (upper 0.2884c both, lower
identical), CL 0.9556 / CD 0.00717 both, pointwise max|dCf| = 3.3e-5 at
the front. The instrument's K_r-point sensitivity (-2.6% mean at
beta=0.35) does NOT appear on the airfoil: over the rooftop the cliff
forbids growth outright and the amplifying stretch runs at chi << 1,
where the destruction floor is invisible -- the same structure that made
the fronts K_r-insensitive (<= 0.003c) in the paper.

## Verdict (2026-07-15)

End-to-end PASS. The tie recovers Spalart's exactly linear nuHat at
every height (machine precision), eliminates tau_D as a constant,
leaves both airfoil regimes unchanged (max|dCf| ~ 2-3e-5), and shifts
the flat-plate onset +0.76% in Re_theta (aftward) at the most
sensitive Tu. Adoption checklist: re-solve the triple with the
destruction floor in the instrument (expected sub-1% digit shifts),
re-fit K_r (~5.4), re-derive the tau bracket statement (tau_D gone),
rewrite paper III.E around the identity, run the full fleet.

## ADOPTED (2026-07-15)

Constants kept (the instrument stays destruction-free, rationale stated
in III.A; back-reaction quantified in III.E). tau_D eliminated
everywhere: paper (eq:blend + eq:sigmadtie, III.E identity derivation,
Sec VII, conclusion), whitepaper (mirror), repro (wall_layer.R_TIE,
fig07 exactness assertion replaces derive_tau_d, tab02, saai_env
exports AI_SIGMAD_TIE=1, consistency-test canon sigmaDTie=1.0), C++
default ai_sigmaDTie=1.0. Flat-plate fleet (5 Tu) re-run with the tie;
Fig 8 + AGS digits regenerated (+5/-3..-11/-6%, within ~11%; chi=c_v1
+13..+19%, +36%). Airfoil figures retained (A/B: unchanged to 2-3e-5).
K_r NOT re-fit: the fit condition is stated on the destruction-free
instrument, where it still holds exactly; the floor's -2.6% at the fit
point is disclosed in III.E as part of the instrument-omission error.

## Full-fleet closure (2026-07-15, later)

54/54 cases re-run with the tie (flow360_tie; 4 sweep disk-full
casualties retried). Every airfoil/sweep figure and table regenerated
from the tie tree and reconciled: tab:nlftrans digit-identical;
tab:eppxtr third decimals updated (L2 family agreement 0.002c at a7);
tab:eppresweep only the bistable 6e4 row and 1e5 hundredths moved; all
caption conventions verified (recovery 0.713 vs ~0.71 etc.). Paper
appendix: 18 wall-anchored contour sheets, all from converged tie runs.
