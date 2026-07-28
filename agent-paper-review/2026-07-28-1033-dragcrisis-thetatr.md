# Drag-crisis transition angle theta_tr(Re, seed, branch) — radial-ray convention, separation angles, FPG kernel readout

*2026-07-28 10:33, CFD post-processing thread. Executes the 2026-07-28 user
directive ("shoot lines from the surface radially ... to a large distance
(or to infinity) and search maximum xi along that line", xi = chi =
nuHat/muRef) plus the same-day coordinator extension (separation angles
from the wall-Cf sequence; per-ray max Re_Omega and max P = Ohat*Ihat; the
high-Re favorable-gradient gate-vs-rate question). Pure CPU
post-processing of all 129 slice-bearing cases of the completed steady
campaign (records 2026-07-28-0010 matrix, -0412 extension); no new runs,
no tex edits. Companion: -0117 (literature ground truth).*

Script: `paper/repro/cfd/dragcrisis_transition_angle.py` (rerunnable;
`--plot-only` re-plots from the JSON). The kernel tail and onset gate are
IMPORTED from `spheroid_flank_kernel_audit.py` (canon __aiRateFromXYZ,
saai_env constants) — reused, not reimplemented.

## Conventions (stated once)

- theta from the FORWARD stagnation point (theta = 0 upstream), cylinder
  center (0.5, 0), R = 0.5 D (verified rmin = 0.5 on all three mesh
  families). Rays: 480 samples log-spaced in wall distance from 2e-7 D to
  the slice edge — reach 99.5 D (pilot/highre), 299.5 / 999.4 D
  (lowre300 / lowre), all beyond the >= 20 D target. theta step 0.5 deg.
- chi = nuHat/muRef (the fig:dragcrisisfields convention). Both sides
  probed; every reported profile is the two-side MEAN (symmetry verified
  below).
- theta_tr = smallest grid theta with maxchi(theta) >= 1 (solver front
  level); companion threshold chi >= c_v1 = 7.1 (f_v1 = 1/2,
  fully-active).
- Near-wall chi=1 front and separation angles CROSS-REFERENCED from
  matrix_summary.jsonl, not recomputed (near-wall: max chi in band
  d < 2e-3 D per 1-deg bin, min of sides; separations: tangential-Cf sign
  crossings, two-side mean, the spurious endpoint crossing at theta = 180
  dropped). Note the near-wall front lives on a 1-deg grid vs 0.5 here:
  comparisons carry +-1 deg.
- Kernel coordinates per ray point: X = |u|, Y = |omega| d (solver
  vorticityMagnitude), Re_Omega = d^2 |omega| / nu (nu = muRef/rho),
  P = Shat*Ihat; Re_Omega_c = softmin_2(1851.2, 124.6 + 1.424/P^2),
  rate = 0.19 clip(P,0,1) gate (tanh ramp 0.35). Z BOTH ways, both in the
  JSON: Z_i = +1/2 d^2 (lap u).u_hat via the audit's chained-VTK-gradient
  convention on the slice (quasi-2D, so in-plane = full Laplacian), and
  Z_ray = 1/2 d^2 (d2u/ds2).u_hat (radial rays ARE wall-normal on a
  cylinder; the audit's kzray variant). "_bl" = restricted to the
  attached layer s <= y_e (ray speed-max height within 0.05 D, the audit
  edge convention).

## Symmetry / robustness

- Side symmetry: max |dlog10 chi| between sides over ALL cases and rays =
  0.0175 (median 0.0054); per-side theta_tr never differs by more than
  one grid step (0.5 deg). The two-side mean is therefore safe.
- Threshold robustness: theta_tr(c_v1) sits +1.0..+7.0 deg aft of
  theta_tr(chi=1) at Re >= 6e4 (n=108) and +7.5..+25 deg aft on the slow
  aloft fronts at Re < 6e4 (n=14) — same shape, no crossing reordering.
- Attached-layer restriction (s <= y_e) does not move theta_tr at
  Re >= 3e3 (the shear-layer max sits inside y_e); it matters only for
  the far-wake fronts at Re <= 1e3.
- Seam checks: theta_tr at 2e6 Tu 0.2 cold: pilot 98.0 vs highre 97.5
  (one grid step); 4e6 Tu 0.2 up: pilot 96.5 = highre 96.5. The 2e6
  dn_highre value 101.0 vs pilot dn 98.0 is BRANCH (descending from the
  2e7 state), not mesh — same decomposition as 0412 seam finding.

## theta_tr table, radial-ray chi >= 1 (authoritative family per 0412:
## pilot <= 2e6, highre >= 4e6; cold == up to 0.5 deg where run)

  Re      0.05up 0.05dn  0.2up  0.2dn  0.7up  0.7dn
  1e4        -    96.5     -    96.5     -    96.5
  2e4        -    91.0     -    91.0     -    91.0
  4e4        -    86.5     -    86.5     -    86.5
  6e4      84.0   84.0   84.0   84.0   84.0   84.0
  1e5      81.5   81.5   81.5   81.5   80.5   80.5
  1.5e5    79.5   80.0   80.0   80.0   78.5   79.0
  2e5      79.0   79.0   79.0   79.0   78.0   78.5
  2.5e5    79.0   79.0   79.0   79.0   78.5   78.5
  3e5      78.5   79.0   79.0   79.0   80.0   79.5
  3.5e5    79.0   79.5   79.5   79.5   82.0   80.5
  4e5      80.0   80.5   80.5   80.5   86.5   87.0
  5e5      84.5   82.0   86.0   82.0   90.5   90.0
  7e5      94.5   92.5   96.0   95.5   92.0   92.0
  1e6      99.0   98.5   97.5   97.5   93.5   93.5
  2e6     100.5  100.5   98.0   98.0   93.0   93.0
  4e6      99.0  104.0   96.5  100.5   89.0   95.0
  7e6      94.5  101.0   91.0   98.5   88.5   93.0
  1e7      92.5   99.0   88.5   96.5   85.0   90.5
  2e7      87.0    -     83.0    -     77.0    -

Low arm below 6e4 is seed-independent to the grid step (0010 finding 5
carries over). Wake-transition angles on the creeping/lowre arm (radial
rays cross the wake far aloft): Re 100 / 300 / 1e3 / 3e3 -> 155 / 139.5 /
123.5 / 108.5 deg with the max at s = 0.74 / 0.32 / 0.13 / 0.056 D — the
convention picks up wake turbulence onset there, not a boundary-layer
front (excluded from the figure's crisis story; in the JSON).

No-transition cases (largest maxchi, all far aloft): Re=1: 0.0109 (lowre
and lowre300), Re=3: 0.0123, Re=10: 0.0144, Re=30: 0.0222/0.0221 — chi
never approaches 1 exactly where the physical flow is steady laminar
(Re < 47); the first crossing appears at Re=100 (maxchi 3.03 at 155 deg,
s = 0.74 D, wake shear layer).

## Separation angles (same table axes; first / final)

  Re      0.05up      0.05dn      0.2up       0.2dn       0.7up       0.7dn
  1e4        -        80.6/134.5     -        80.6/134.5     -        80.6/134.3
  4e4        -        75.8/132.8     -        75.8/132.8     -        75.9/132.5
  6e4     74.8/131.1  74.8/130.6  74.8/131.1  74.8/130.6  74.9/130.6  74.9/130.4
  1e5     73.7/73.7   73.8/73.8   73.7/73.7   73.8/73.8   74.0/74.0   74.0/74.0
  2e5     73.1/73.1   73.3/73.3   73.1/73.1   73.4/73.4   74.4/74.4   74.3/74.3
  3.5e5   74.4/74.4   74.8/74.8   74.6/74.6   74.9/74.9   78.2/78.2   79.0/79.0
  5e5     79.8/79.8   77.6/77.6   82.7/82.7   77.8/77.8   90.8/90.8   90.2/90.2
  7e5     89.1/89.1   88.0/88.0   91.8/107.2  91.2/105.7  92.9/105.6  93.2/105.8
  1e6     93.4/112.5  93.5/111.7  95.1/112.5  94.1/111.1 111.7/111.7  95.0/110.7
  2e6     97.5/118.3  97.5/118.4  98.3/118.1  98.3/118.1 117.2/117.2 117.2/117.2
  4e6     99.2/121.0 101.9/122.3 120.2/120.2 122.1/122.1 115.0/142.4 120.4/120.4
  7e6    121.2/121.2 124.9/124.9 120.3/120.3 124.3/124.3 119.0/119.0 123.2/123.2
  1e7    121.4/121.4 123.6/123.6 120.4/120.4 123.5/123.5 119.4/119.4 122.5/122.5
  2e7    121.6/121.6    -        120.5/120.5    -        119.4/119.4    -

(first = first Cf sign change; final = last, both < 180. At Re <= 6e4 the
sequence is separation ~75-81, base-region reattachment ~91-122, final
re-separation ~130-135 — the steady symmetric base-closure structure of
the 0010 record. Where first == final and ~120: NO laminar separation
remains, the wall sequence is a single turbulent separation.)

## Convention comparison (radial-ray vs near-wall chi=1 front)

- Re >= 1.5e5: the two conventions agree to +-1.5 deg everywhere (radial
  0.5-deg grid vs near-wall 1-deg bins; 108 cases). The radial-ray map is
  a strict superset of the near-wall convention there.
- Re = 1e5: gap +4.5..+5.5 deg (near-wall 85-87, radial 80.5-81.5; the
  chi maximum at theta_tr sits at s ~ 0.006-0.010 D, above the 2e-3
  near-wall band).
- Re = 6e4: gap +53 deg (137 near-wall vs 84 radial, s@tr = 0.008 D).
- Re = 4e4: gap +64.5 deg (151 vs 86.5, s@tr = 0.011 D).
- Re = 1e4-2e4: near-wall front DOES NOT EXIST (band max chi < 1);
  radial-ray gives 96.5 / 91 deg with the max at s = 0.026 / 0.017 D.
- VERDICT (the directive's expectation, quantified): the radial-max
  convention catches the SUBCRITICAL lifted-shear-layer transition that
  the near-wall convention misses; at 4e4-6e4 the near-wall number is a
  base-region echo 53-65 deg aft of where the separated shear layer
  actually transitions, and below 4e4 the near-wall convention reports
  "no transition" while the shear layer aloft is already turbulent.
  The 0010 supercritical front narrative (93-100 deg class) is unchanged.

## Physics readout

1. **theta_tr(Re) is V-shaped, not monotonic.** Down the subcritical arm
   the aloft front creeps forward: 96.5 (1e4) -> 84 (6e4) -> 79 (2e5),
   saturating ~4-6 deg AFT of the laminar separation angle (73-75) — the
   separated shear layer transitions almost immediately aloft, the 0010
   "immediate shear-layer transition" caveat now measured as an angle.
   Through the crisis band the front moves AFT (79 -> 86-98) as the
   transitioned wake retreats and an LSB forms at the shoulder; the
   crisis minimum of Cd corresponds to the front's local MAXIMUM
   retreat, ~98-101 deg (Tu 0.05/0.2) / 93 (0.7) at 1e6-2e6.
2. **The transcritical march IS monotonic from 2e6 to 2e7** (up branch,
   authoritative families): 100.5 -> 87.0 (Tu 0.05), 98.0 -> 83.0
   (Tu 0.2), 93.0 -> 77.0 (Tu 0.7), i.e. 13-17.5 deg forward per decade,
   Tu-ordered the right way. Consistent with the 0412 near-wall spot
   checks (101 -> 82 at Tu 0.2). At 2e7 the front reaches the suction
   peak (phi_Cp_min ~ 83-87) but goes no further — vs the ~25-35 deg
   attachment-region class the transcritical experiments report [mem;
   figures carry no experimental numbers pending the gated Achenbach
   digitization]. The dn branch (deeper-recovery second wake state, 0412
   finding 3) sits 4-8 deg AFT of up at the same Re.
3. **Separation vs transition converging/crossing = the macroscopic
   story** (both on one axes in the figure). Subcritical: transition
   trails separation aloft by 4-11 deg (order: sep THEN transition).
   They converge through the crisis (at 5e5 Tu 0.2 up: sep 82.7,
   theta_tr 86.0) and lock together supercritically as the LSB: sep_first
   98.3 / theta_tr 98.0 at 2e6 Tu 0.2 (transition AT the bubble). Then
   the LSB VANISHES — sep_first jumps 98 -> 120 with no reattachment
   pair — between 2e6 and 4e6 at Tu 0.2 (up AND dn), between 4e6 and 7e6
   at Tu 0.05, by 1e6 at Tu 0.7 (up): transition now happens on the
   ATTACHED wall ahead of a single turbulent separation, the classic
   transcritical wall sequence — while Cd stays at the 0.18-0.22 floor
   because the turbulent separation stays at ~120 deg (the 0412 verdict:
   the missing transcritical Cd rise is a turbulent-separation/wake
   deficiency, not a transition-location failure; now sharpened to "the
   wall SEQUENCE is already transcritical").
4. **FPG gate-vs-rate (the user hypothesis test; second figure).** In the
   nose FPG region (theta 10 deg .. phi_Cp_min ~ 84):
   - The onset gate OPENS well ahead of the front: max_bl
     Re_Omega/Re_Omega_c first reaches 1 at theta = 82.5 / 81 / 72.5 /
     65 / 46.5 deg for Re = 2e6 / 4e6 / 7e6 / 1e7 / 2e7 (Tu 0.2 up),
     while the front sits at 98 / 96.5 / 91 / 88.5 / 83 — i.e. the front
     trails the opened gate by 15-36 deg. At the front's location the
     model is NOT onset-blocked at any high-arm Re.
   - The rate coordinate stays abysmally low there: max_bl P (ray
     operator) = 0.08-0.12 at 2e6-1e7, 0.21 at 2e7, and only 0.01-0.05
     over most of theta 10-70; max_bl rate = 0.19 clip(P) gate <= 0.008
     (2e6-4e6), 0.016 (1e7), 0.022 (2e7) — a factor 10-20 below the
     fully-inflected a_max = 0.19. The forward march is RATE-limited at
     the front: the FPG profile is uninflected, P ~ Ihat stays small, and
     chi cannot e-fold from chi_inf to 1 in the short arc available.
   - The gate ceiling (1851.2, since P << 1 makes the soft-min saturate)
     DOES seal the deep nose: ratio < 1 for theta < 46.5 even at 2e7
     (< 82.5 at 2e6). So the user hypothesis splits: "rate abysmally low"
     = confirmed as the binding constraint at the observed fronts;
     "onset too high" = binding only deeper into the nose (and at 2e6,
     where gate opening and Cp-minimum nearly coincide, both bind
     simultaneously — consistent with the front being pinned AT the LSB
     there). Extrapolating the ratio growth (~Re), the gate alone would
     not open at 30 deg until Re ~ 1e8; the rate would still be ~0.01.
     Reaching the experimental 25-35 deg class inside this model needs
     the amplifying-zone gate revision or a bypass-class mechanism, not
     constant tuning [K-gate memos; follow-on work, not this paper].

## Findings ledger (honest-report items)

1. First compute pass (chi-only) completed but its JSON dump crashed on a
   numpy float32 (not JSON-serializable); recomputed after the fix — the
   published table numbers from the second pass match the first pass's
   console table exactly (all 129 rows diffed: th1/thv1/nw identical). A dict-tie
   sort crash in the first plot call (seam duplicates share Re) was fixed
   with an explicit sort key; plots regenerated via --plot-only.
2. Operator honesty: the solver-as-implemented Z (chained VTK nested
   gradient on the slice) is NOISE-DOMINATED inside the thin FPG boundary
   layer at high Re (P_i flips sign point-to-point; the slice's cut
   triangles at y1 = 1e-6 amplify float32 noise in third derivatives).
   The ray operator (exact wall-normal second derivative on the probe
   line) is smooth. BOTH are stored per case; the FPG figure draws the
   ray operator solid and the VTK operator thin-dashed; quantitative FPG
   P values above are the ray operator; the two agree in the separated
   shear layer and LSB (where the layer is thick) and on the verdict.
3. The stagnation ray (theta < ~3 deg) shows P ~ 0.5 — a U -> 0
   coordinate artifact (Shat -> 1 and Ihat finite as X vanishes), gated
   off by Re_Omega ~ 0 there; excluded from the FPG table via
   theta >= 10.
4. Tu 0.7 4e6 up shows an anomalous aft bubble (sep 115, reattach 136.8,
   final 142.4) — that case carries the campaign's strongest limit-cycle
   flag (0412: Cd 0.2974*); treat its separation row as the snapshot of a
   breather, not a converged steady state (the theta_tr value, 89.0, is
   in family with its neighbors).
5. The base-region "final separation" at Re <= 6e4 (~130-135 deg) is the
   steady symmetric base-closure structure, not a physical mean-flow
   feature under shedding (0010 disclosure carries over).
6. y+ caveat carries over from 0412: the 2e7 points run at y+max = 1.4
   (stretch); their theta_tr values are the campaign's, no re-mesh here.

## Cost

129 cases, ~26 s/case single-process CPU (VTK gradient filters dominate),
~56 min wall; peak RSS < 2 GB (slices processed serially and released).

## Artifacts (full paths)

- Script: `/home/qiqi/flexcompute/sa-ai/paper/repro/cfd/dragcrisis_transition_angle.py`
- Per-case JSON (theta grid, compressed maxchi/Re_Omega/P/gate/rate
  profiles, theta_tr at both thresholds, separations, reach):
  `/home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/data/dragcrisis_theta_tr.json` (3.5 MB, committed)
- Figures (exploratory, figs_explore-first per directive):
  `/home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/dragcrisis_theta_tr.png`
  (+ `.pdf` on disk; *.pdf is repo-gitignored outside paper/figs),
  `/home/qiqi/flexcompute/sa-ai/paper/repro/cfd/figs_explore/dragcrisis_thetatr_fpg.png`
- Data root (inputs, unchanged): `/local_data/qiqi/sa-ai/dragcrisis_matrix/`
- Cross-referenced, not recomputed: `matrix_summary.jsonl`
  (chi1_front_*, crossings_*, phi_Cp_shoulder_*)
