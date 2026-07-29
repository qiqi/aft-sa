# vg-kernel solver implementation + budgeted validation (2026-07-29)

*Solver+CFD agent. Implemented the two-branch "vg" SA-AI rate+gate in the
Flow360 solver (env-gated, default OFF = bit-identical canon), verified it, and
ran a budgeted overnight validation on the cases the low-H FPG modification most
affects. Reference form: agent-paper-review/2026-07-28-1218 Parts V/VIII and
2026-07-29-0121 Part VIII; analytic reference `paper/repro/analytic/
fpg_recalibration_study.py` form 'vg'. Solver branch `explore-lambda-v`,
sa-ai branch `paper-composite-snapshot`.*

## Phase A — implementation + build + verification

### Implementation (env-gated, default OFF)
`compute/src/Flow360Core/Applications/Solver/SpalartAllmaras/SAAiTransition.h`
`__aiRateFromXYZ` (serves the magnitude kernel `__aiRate` and the invariant
kernel `__aiRateInvariant`; NOT the Gram/pair variant AI_INVARIANT_KERNEL>=2):

- RATE (env `AI_A_VISC` = a_visc > 0):
  `a = clip( softmax2(a_inv*Shat*<g>+, a_visc*Shat*<-Z/R>+), 0, a_inv )`,
  a_inv = rateScale (0.19), softmax2(x,y)=sqrt(x^2+y^2).
- GATE (env `AI_REOMC_BC` = B_c > 0):
  `Re_Omega_c = softmin2( reOmA + reOmB/P_I^2 , reOmA + B_c/P_curv^2 )`,
  P_I = Shat*<g>+, P_curv = Shat*<-Z/R>+, A_c = reOmA shared, ceiling DROPPED.
- Both env default 0 -> the UNCHANGED canon expressions run (exact canon
  rate a_max*clip<Shat*g> and ceiling gate softmin2(reOmCeil, reOmA+reOmB/P^2)).

New constants `aVisc`/`reOmBc` in `aiSaConstants` (SAAiTransition.h) +
`ai_aVisc`/`ai_reOmBc` in `modelConstants` (ModelConstants.h); wired in
SpalartAllmaras.h; env read (`AI_A_VISC`, `AI_REOMC_BC`) + log echo in
SATurbulenceSolver.cpp. Paper ON values: a_visc=0.0276, B_c=130.
Commit: `29726cb5f9` (compute, explore-lambda-v).

### Build
CLEAN rebuild per BUILD_CONSISTENCY.md
(`cmake --build build/release/Flow360Core --target install --clean-first -j24`,
exit 0). SpalartAllmaras objects uniform 2026-07-29; install binary + libs
fresh; `AI_A_VISC`/`AI_REOMC_BC` + `ai_aVisc`/`ai_reOmBc` echo compiled into
libflow360sasolverlib.so.

### Verification gates — ALL PASS
- (a) canon bit-identical: kernel unit test (real solver `__aiRate`, g++ host
  build vs the analytic canon) = **0.000e+00** relative error over 1921 input
  rows; CFD short run (canon env, 2e6 highre restart) echoes
  ai_aVisc=0.000000, ai_reOmBc=0.000000 and Cd=0.204764 (= canon 2e6 dn).
- (b) solver-vs-analytic vg: same unit test with vg constants matches the
  analytic 'vg' (`_P_and_thresh` + `sphere_rate_eps`) to **8.9e-16** over 252
  vg-active rows.
- (c) vg echo: CFD run with AI_A_VISC=0.0276 AI_REOMC_BC=130 echoes
  ai_aVisc=0.027600, ai_reOmBc=130.000000.
Artifacts: scratchpad test_vg_kernel.cpp / check_vg_kernel.py (kernel test);
verify_vg_gates.py (CFD gates).

## Phase B — cylinder campaign (vg ON; tagged _vg; canon JSONs UNTOUCHED)

Results in `/local_data/qiqi/sa-ai/dragcrisis_matrix/vg_summary.jsonl`
(+ per-case summary.json). theta_tr (radial-ray chi=1) from
`repro/cfd/vg_theta_tr_extract.py` ->
`repro/cfd/figs_explore/data/vg_theta_tr.json`. Driver:
`repro/cfd/run_dragcrisis_vg.py`. Canon reference = matrix_summary.jsonl (up
branch, highre mesh; dn branch noted).

### PRIMARY — high-Re drag crisis, Tu=0.2%, highre mesh (2e6..2e7)

vg up-ladder warm-started from the canon 2e6 dn state, warm-chained upward.

| Re_D | canon theta_tr(chi1) | vg theta_tr | canon Cd (up / dn) | vg Cd | verdict |
|------|------|------|------|------|------|
| 2e6  | 98.0 | 99.0 | 0.2214 / 0.2049 | 0.2088 | conv |
| 4e6  | 96.5 | 94.0 | 0.2035 / 0.1742 | 0.2159 | cap  |
| 7e6  | 91.0 | 90.0 | 0.1970 / 0.1539 | 0.2090 | conv |
| 1e7  | 88.5 | 88.0 | 0.1974 / 0.1602 | 0.2055 | conv |
| 2e7  | 83.0 | 82.5 | 0.1992 / -      | 0.2048 | conv |

Reading: at practical Re (2e6-2e7) the vg front is within +-2.5 deg of canon and
Cd within +-0.013 (vg ~+3-6% over canon-up). **No transcritical front advance
(front stays ~82-99 deg, not 25-35 deg) and no Cd rise toward 0.5-0.7** at these
Re -- exactly the analytic prediction (2026-07-28-1218 Part III/V: clean noses
at Re_D <= 2e7, no spurious bypass). This is the intended regression-clean
behavior, not the headline.

### PRIMARY headline — ultra arm (Re_D 2e7..1e10), ultra mesh

Protocol-matched vg warm up-ladder (2e7 cold, warm-chained up) vs the canon
ultra up-ladder (matrix_summary.jsonl). theta_tr = radial-ray chi=1
(vg_theta_tr_extract.py); Cd from vg_summary.jsonl.

| Re_D | canon theta_tr | vg theta_tr | canon Cd | vg Cd | vg verdict |
|------|------|------|------|------|------|
| 2e7  | 89.0 | 89.0 | 0.1917 | 0.1916 | conv |
| 5e7  | ~81  | ~78  | 0.1867 | 0.1888 | conv |
| 1e8  | 63.0 | 61.0 | 0.1971 | 0.2018 | conv |
| 3e8  | 39.5 | 33.5 | 0.2057 | 0.2107 | conv |
| 1e9  | 16.5 | 15.0 | 0.2056 | 0.2091 | conv |
| 1e10 | 3.0  | 2.5  | 0.1889 | 0.1989 | limit_cycle |

Reading (THE headline test):
- **The transition front DOES advance forward at ultra Re -- into the
  transcritical ~25-35 deg angular class by Re_D ~ 3e8 (theta_tr 33.5 deg) and
  to ~2.5 deg at 1e10 -- but this advance is ALREADY present in the canon
  kernel (canon 39.5/16.5/3.0 deg); vg tracks it within ~6 deg and does NOT
  advance the front beyond canon.**
- **Cd does NOT rise toward the experimental transcritical 0.5-0.7.** Both
  canon and vg stay at Cd ~ 0.19-0.21 across the whole ultra arm (vg is
  +0.005..+0.010 over canon, i.e. ~+3-5%, same order as the highre offset).
  The RANS solution (canon AND vg) stays in the low-Cd supercritical-like
  branch; the transcritical rise is NOT recovered.

Cold-start bistability note (separate cold ladder, vg_summary warm_src=None):
the vg 1e9 COLD start hit a limit cycle (Cd 0.225, front 91 deg) with a
transient excursion to Cd ~ 0.7 during the fSlow=0.1 stage -- a hint that a
high-Cd separated branch is accessible to the vg kernel -- but the steady
protocol-matched warm-ladder solution settles on the canon-like low-Cd branch.
Not a converged transcritical state.

## REGRESSION (vg ON; warm-restart clones of canon converged cases;
same mesh/config/seed, kernel-only delta). Driver `repro/cfd/run_vg_regression.py`.

- **Flat plate ZPG** (anchor; near-wall chi=1 crossing Re_x, same method both):
  | Tu% | canon Re_x_tr | vg Re_x_tr | delta |
  |-----|------|------|------|
  | 0.16 | 1.812e6 | 1.677e6 | -7.5% |
  | 0.08 | 2.519e6 | 2.293e6 | -9.0% |
  Small forward shift, CONSISTENT with the memo's documented Blasius
  Re_x(N9) -8.2% (the a_visc floor lifting the ZPG rate). Not a break.
- **Eppler 387 Re=200k a=2** (favorable rooftop; upper-surface Cf-jump xtr,
  0.02c bins): canon xtr 0.58 -> vg xtr 0.58 (unchanged at bin resolution;
  any shift < 0.02c). CL 0.606, Cd 0.01152. Clean.

## Budget
GPUs 0-3 only (0 carried a foreign job throughout -- untouched; 4/5/7 vishal's
-- untouched). All work on GPUs 1/2/3. Aggregate solver wall ~2.5 GPU-h across
the three GPUs, ~1.5 h elapsed: clean rebuild ~7 min; highre ladder 5 cases
~15 min (GPU1); flat-plate x2 + Eppler regression ~30 min (GPU2/3); ultra cold
1e9+1e10 ~20 min + protocol-matched warm ladder 6 cases ~45 min (GPU1). Well
under the 6 h budget. Disk: /local_data stayed ~364 GB free.

## Artifacts (full paths)
- Solver: compute commit 29726cb5f9 (explore-lambda-v), files under
  compute/src/Flow360Core/Applications/Solver/SpalartAllmaras/
- Scripts (sa-ai, paper-composite-snapshot):
  paper/repro/cfd/run_dragcrisis_vg.py (cylinder driver, highre+ultra),
  paper/repro/cfd/run_vg_regression.py (flat-plate/Eppler regression),
  paper/repro/cfd/vg_theta_tr_extract.py (radial-ray theta_tr),
  paper/repro/cfd/verify_vg_gates.py (Phase-A CFD gate check)
- Data: /local_data/qiqi/sa-ai/dragcrisis_matrix/vg_summary.jsonl
  (highre 2e6..2e7 + ultra 2e7..1e10; cold 1e9 row has warm_src=null),
  per-case summary.json + solver.log (constants echo) in each
  cyl_Re*_up_{highre,ultra}_vg case dir;
  paper/repro/cfd/figs_explore/data/vg_theta_tr.json (highre) +
  the ultra rows re-run into the same file;
  flat-plate/Eppler: /local_data/qiqi/sa-ai/flow360_fv1/
  {flatplate_sphere_Tu0160,flatplate_sphere_Tu0080,
  strL2prop_eppler387_Re200k_a2}_vg/vg_regression.json
- canon references: /local_data/qiqi/sa-ai/dragcrisis_matrix/matrix_summary.jsonl
  (UNTOUCHED)

## Verdict (plain)
The vg kernel is correctly implemented and verified (canon bit-identical
default; ON path matches the analytic vg to round-off). Scientifically:

**The vg kernel does NOT recover the transcritical front-advance/Cd-rise.**
- At practical Re (2e6-2e7) vg is regression-clean: front within +-2.5 deg and
  Cd within +-6% of canon -- no spurious nose bypass, exactly as the analytic
  study predicted.
- At ultra Re the transition front advances into the transcritical angular
  class (theta_tr ~ 33 deg at 3e8, ~2.5 deg at 1e10), but that advance is
  ALREADY produced by the canon kernel; vg tracks it within ~6 deg and adds
  nothing. Crucially, Cd does NOT rise -- both canon and vg sit at ~0.19-0.21
  across the whole ultra arm, far below the experimental transcritical 0.5-0.7.
  The missing Cd rise of Sec VII is NOT closed by the vg modification.
- A cold-start bistability hint (transient Cd ~ 0.7 at 1e9, then a limit cycle
  at 0.225) suggests a high-Cd separated branch exists but is not the settled
  steady solution under the canon warm-ladder protocol.
- Regression anchors hold: flat-plate ZPG front shifts forward only ~8%
  (matching the documented Blasius Re_x(N9) -8.2% cost), Eppler rooftop front
  unchanged at bin resolution.

Bottom line: the port is sound and safe to keep env-gated (default off); the
low-H FPG amplification behaves as designed on calibrated cases, but it does
not, on its own, produce the transcritical drag rise in the RANS drag-crisis
cylinder.
