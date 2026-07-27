# Drag-crisis STEADY campaign matrix — COMPLETE (84/84 cases, 5.8 GPU-h)

*2026-07-28 ~00:10, CFD campaign thread. Executes the steady-only campaign
directive (supersedes the URANS half of the gate pair; the cancelled URANS
Re=3e5 run was stopped cleanly, its steady twin retained). Companion
records: 2026-07-27-2006 (pilot), 2026-07-27-1540 (design),
2026-07-27-1130 (literature). Driver/figs committed:
paper/repro/cfd/run_dragcrisis_matrix.py, dragcrisis_matrix_janitor.py,
regen_dragcrisis_matrix_figs.py. Data: /local_data/qiqi/sa-ai/
dragcrisis_matrix/ (per-case summary.json + matrix_summary.jsonl; surface +
slice outputs kept, volume disabled, mesh files hardlinked, solver logs
truncated post-hoc — disk order). Figure:
paper/repro/cfd/figs_explore/dragcrisis_matrix.png.*

## What ran

12 Re {6e4..2e6} x 3 seeds (Mack map: Tu 0.05% -> chi_inf 3.89e-4, 0.2% ->
1.08e-2, 0.7% -> 2.19e-1) x {up, dn} continuation ladders (previous-Re
restart as init; first point of each ladder cold) + 12 cold two-stage
references at Tu 0.2%. Steady pseudo-transient only, campaign settings
(adaptive CFL, staged fSlow 0.1 -> 0.01 with seed re-compensation, canon
fv1 env — ai_constants echo preserved per case in ai_constants_echo.log,
84/84 identical to the campaign canon). Pilot L1 O-grid (1200x148, y1
8e-6 D; y+ <= 0.64 at 2e6). Per-case convergence: live CD-flatness monitor
(|dCD| < 1e-3 per 1000 pseudo steps -> graceful stop.json), limit-cycle
detection at the cap. GPUs 0-3 local under the shared team cap, occupancy-
checked per launch.

## Headline: the model produces the drag crisis, as a Tu-ordered family

Cd(Re) along the up-ladders (medians; * = a stage limit-cycled):

  Re      6e4    1e5    2e5    3e5    3.5e5  4e5    5e5     7e5     1e6     2e6
  Tu0.05  0.845  0.840  0.814  0.776  0.753  0.725  0.644*  0.409*  0.286*  0.219*
  Tu0.2   0.845  0.839  0.813  0.774  0.749  0.719  0.620*  0.344*  0.284*  0.221*
  Tu0.7   0.842  0.831  0.781  0.677* 0.616* 0.495* 0.430*  0.365*  0.294*  0.233*

- Crisis center (steepest Cd drop): ~4e5 at Tu 0.7%, ~5.5-6.5e5 at Tu
  0.05-0.2% — Re_crit DECREASES with Tu, the Achenbach-1971-style family
  the campaign targets, produced by chi_inf alone (no correlation, no
  extra equation).
- Supercritical floor 0.217-0.233 — the experimental two-bubble minimum
  class (0.20-0.25); and it RISES slightly with Tu (0.219 -> 0.233 at
  2e6), the correct roughness/turbulence trend.
- Post-crisis wall structure is a genuine LSB + turbulent separation:
  e.g. Tu 0.2% at 7e5: laminar sep 91.8 deg -> turbulent REATTACHMENT
  101.9 -> final turbulent separation 107.2; at 2e6 the bubble shrinks to
  ~1.5 deg (98.3 -> 99.8) with final separation 118.1 — bubble shortening
  with Re is physical. Final turbulent separation (107-118 deg) remains
  well short of the experimental ~147 deg [mem: Achenbach] — expected
  class of error (SA turbulent Cf on curved walls + steady symmetric wake
  dead-air), flagged in the 1540 design as the Achenbach-Cf-traverse item.
- Shoulder suction deepens through the crisis: Cp_min -0.91 (subcritical)
  -> -2.4 at 2e6 (experimental supercritical shoulder ~ -2.4..-2.6 [mem]);
  base Cp recovers -0.62 -> -0.26.
- chi=1 near-wall front: 137 deg (6e4, base-region only) -> 78-79 deg
  through the crisis onset (transition reaching the separation point) ->
  93-100 deg supercritical (the LSB region). The subcritical caveat from
  the pilot kernel check (immediate shear-layer transition at all Re)
  shows up as the WEAK seed dependence below 2e5.

## Hysteresis / branch structure (up vs dn vs cold)

- OUTSIDE the crisis band, up/dn/cold agree to <=0.003 in Cd (ladder
  continuation and cold protocol are equivalent — protocol-neutrality
  verified at all 12 cold points).
- INSIDE the crisis band there is a reproducible branch spread, ~0.05 in
  Cd, localized exactly where stages limit-cycle: Tu 0.7%: up 0.495 vs dn
  0.547 (4e5), up 0.616 vs dn 0.671 (3.5e5); Tu 0.2%: up 0.620 vs dn
  0.668 vs cold 0.650 (5e5). NOTE the sign: the dn branch (initialized
  from the more-transitioned high-Re state) lands at HIGHER Cd — opposite
  to the experimental hysteresis direction (attached state persisting to
  lower Re). With 27/84 cases carrying a limit-cycle flag concentrated in
  this band (Cd tail p2p median 0.011, max 0.199 — mostly weak breathers,
  a few genuine cycles), the honest statement is: the steady branch is
  NON-UNIQUE/WEAKLY UNSTEADY through the crisis, the ~0.05 spread
  quantifies it, and branch selection there is protocol-dependent. Firm
  hysteresis claims at these points belong to URANS continuation (the
  Schewe physics), not the steady solver.
- CL = 0 to 1e-7 in ALL 84 cases: the steady pseudo-transient never
  found the one-bubble asymmetric state — with a symmetric grid and
  symmetric protocol it converges to the symmetric branch even where the
  physical flow is one-bubble asymmetric. FINDING: the one-bubble state
  (mean CL up to ~1, Schewe) is invisible to this protocol; it needs
  URANS (the pilot's alpha-kick protocol) or an asymmetric perturbation
  study.

## Findings ledger (honest-report items)

1. Steady limit cycles through the crisis band: 27/84 cases, quantified
   per case (Cd_tail_p2p in matrix_summary.jsonl), medians reported,
   nothing tuned. Verdict criterion: |dCD|<1e-3/1000 pseudo steps, else
   cycle flag at cap.
2. Subcritical absolute Cd ~0.84 on the steady branch (2D symmetric, no
   shedding) vs exp 1.2: wake-dynamics bias, conceded by design (pilot
   record Sec. 4); BL-side quantities are the campaign product there.
3. dn-branch-higher-Cd anomaly (above) — needs URANS continuation to
   interpret; do not read as physical hysteresis direction yet.
4. Final turbulent separation 107-118 vs ~147 deg experimental; LSB
   present with physically-shrinking length. The Cf(theta) traverse
   comparison against Achenbach 1968 is the natural next validation step
   (surface outputs retained for all 84 cases).
5. retag collision bug (1.5e5/2.5e5/3.5e5 -> "2e5"/"4e5" under :.0e) hit
   on the first launch, clobbered one ladder dir, caught within minutes
   via duplicate ticker rows; fixed (lossless integer tags), full restart
   from scratch — all published data is from the clean run.
6. Disk order executed: volume output disabled matrix-wide, mesh files
   hardlinked (one 350 MB copy total instead of 84), solver logs truncated
   to 200 KB after extracting the constants echo, superseded-restart
   cleanup between ladder steps (janitor script, committed). Matrix
   footprint 5.7 GB.

## Cost

84 cases in 5.76 GPU-h total (V100; median warm case ~140 s, cold ~340 s,
crisis-band cases up to ~520 s) — well inside the 10-20 GPU-h estimate,
~30x cheaper than the URANS design for the Re x Tu x branch content. Wall
clock ~1.6 h on 4 GPUs (shared-cap discipline: occupancy check per launch;
GPUs released on completion).

## Suggested next steps (not started)

- Cf(theta)/Cp(theta) overlays vs Achenbach 1968 at matched Re (surface
  data on disk; digitization would follow the paper/tools README pattern).
- 2-4 URANS spot checks (pilot protocol) for St ladder + one-bubble +
  true hysteresis at the crisis center of each seed level.
- The 6e4 early-transition caveat: a lower-Re extension (1e4-4e4) would
  show whether Cd rises toward the subcritical plateau as the shear-layer
  transition point physically retreats — cheap (steady).
