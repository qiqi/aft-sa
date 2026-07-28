# Drag-crisis ULTRA arm to Re=1e10 (ten decades) + fully-turbulent-SA control

*2026-07-28 16:42, CFD campaign thread. Executes two same-day user
directives: (phase 3) extend the single-model Cd(Re) traverse to Re_D=1e10
— ten decades, the crisis mid-span — testing whether the three freestream
seeds COLLAPSE as transition saturates toward the nose; and (FT-SA control)
run classical fully-turbulent SA (transition OFF) on the high-Re cylinder to
attribute the missing transcritical rise. ALL GPU work relocated to
017-v100-dev GPUs 0-3 (vishal held every local GPU) — per-case occupancy +
lockfile as before; results synced back to the unified local tree. Same
model/constants/env; new mesh family 'ultra'; NO tex edits. Companion
records: -0010 (matrix), -0412 (Re 1..2e7 extension), -0117 (litrange),
-1033 (theta_tr radial-ray fronts), -1041 (FPG rate audit), -0317 (figure
restyle). Driver reused: run_dragcrisis_extension.py (+ ultra/FT chains),
build_dragcrisis_pilot.py (--nsurf/--y1/--rout), harvest, janitor, figs.*

## Remote relocation + canon proof (standing rule)

- 017-v100-dev: compute repo explore-lambda-v @ 05d05a123b (canon commit,
  = local), solver binary present; /local_data 732 G free. GPUs 0-3 idle at
  handoff; 4-7 left alone. Scripts + mesh templates rsynced to
  `.../dragcrisis_matrix/scripts` and the tree root.
- ai_constants echo: 83/83 SA-AI ultra+seam cases line-identical to campaign
  canon (timestamps stripped). FT-SA cases (AI_SA=0): the SA-AI echo block
  is ABSENT — the janitor's echo extractor produced a 0-byte
  ai_constants_echo.log for all 6 FT cases (it greps the FULL log at
  truncation time; empty ⇒ no "SA-AI transition constants" block was ever
  emitted ⇒ classical SA ran). That 0-vs-18-line distinction, captured
  before log truncation, is the positive AI-off proof.

## New mesh family 'ultra'

Analytic O-grid, growth 1.0998, R_out 100 D, **y1 = 1e-9 D**, N_SURF 1600,
242 layers, 387,200 quads. y1 sized from the measured turbulent Cf: the 2e7
highre field gave Cf_max≈1.0e-2; y+ = y1·Re·√(Cf/2) ⇒ at 1e10 (Cf_max
measured 3.9e-3) **y+max = 0.44-0.47** (target ≤0.7, PASS). Shoulder BL at
1e10 δ≈3.7e-3 D holds ~135 radial cells; azimuthal ds/δ≈0.53 (2 cells across
δ) — N_SURF kept at 1600 (bumping buys little; the wall-normal stack is where
the resolution must live and it is generous). Also FIXES the 2e7 y+ debt: the
0412 highre 2e7 points ran at y+=1.4; ultra covers 2e7 at y+≈0.004, so the
ultra 2e7 points REPLACE the caveated highre ones as authoritative.

## Solver sanity at 1e10 (gate BEFORE laddering — muRef = 1e-11)

ONE cold two-stage case (1e10, Tu 0.2) first: converged clean (stage-1
limit-cycle breather, stage-2 converged), NS residual 1.2e-9→6.0e-10 flat
over 10,780 pseudo steps, min_rho 0.983 / min_p 0.70 (no float-underflow
pathology at muRef=1e-11), y+max 0.47, echo canon-identical. **Gate PASSED**
— ladders launched. Notably this cold point found a near-ATTACHED state (see
branch split below), which the sanity check did not veto (it is a physical
solver state, not a failure).

## Extended Cd table (ultra arm; * = a stage limit-cycled)

Up-ladders (cold two-stage at 2e7, warm upward):

  Re        2e7     5e7     1e8     3e8     1e9     3e9     1e10
  Tu0.05  0.2114* 0.1842  0.1878  0.1984  0.1999  0.1957  0.1856*
  Tu0.2   0.1917* 0.1867  0.1971  0.2057  0.2056  0.2001  0.1889*
  Tu0.7   0.2025* 0.2046  0.2132  0.2191  0.2161  0.2057* 0.1737*

Dn-ladders (from 3e9 down to 5e7):

  Re        5e7     1e8     3e8     1e9     3e9
  Tu0.05  0.2231* 0.2073* 0.1711  0.1685  0.1714
  Tu0.2   0.2333* 0.1859* 0.1721  0.1700  0.1732
  Tu0.7   0.2380* 0.2083* 0.1685* 0.1618  0.1633

1e10 special points (Tu0.2): up-ladder 0.1889 (sep 121°); COLD-sanity
0.1079 (sep ~180°, attached); FT-SA cold 0.1249 (sep ~180°).

**Total span: Re = 1 → 1e10 = 10.0 decades**, one closure, one constant set.
The crisis (5e5-class) sits mid-span. Composite-curve datasets (litrange
Sec. 1) top out at Re≈1e7 (Roshko/Shih); Re>1e7 is beyond ANY experiment or
scale-resolving computation — this arm is a pure model extrapolation, and is
reported as such (no validation target exists above 1e7).

## Seed-collapse verdict: PARTIAL, then top-decade branch scatter

The prediction (seeds collapse as transition saturates at the nose) is
BORNE OUT in mechanism but NOT to a single clean curve:

- The chi=1 front marches monotonically to the nose with Re (up Tu0.2:
  89°@2e7 → 62°@1e8 → 16°@1e9 → 7°@3e9 → 2°@1e10) — transition IS
  saturating forward, exactly the predicted driver.
- Up-ladder seed spread narrows accordingly: 9.8%@2e7 → 12.7%@1e8 (a bump)
  → 7.8%@1e9 → **4.96%@3e9** (tightest), i.e. the seeds DO converge over
  3e8-3e9 as fronts crowd the nose.
- BUT at 1e10 the spread re-widens to 8.4% (Tu0.7 up drops to 0.1737 while
  Tu0.05/0.2 hold ~0.186-0.189) and the seed ORDER scrambles — the top
  decade is contaminated by the branch multiplicity below, not a clean
  asymptote. Dn-ladders collapse tighter (2.1%@3e8).
- Honest statement: the seeds converge toward a common supercritical
  band ~0.19-0.21 as transition saturates at the nose (mechanism confirmed),
  but a single-curve collapse is prevented by (a) residual ~5% seed scatter
  and (b) the up/cold branch split at the very top. NOT "the seeds collapse
  onto one curve" without those two caveats.

## Branch split at ultra-high Re (major finding)

At 1e10 the steady solver supports (at least) TWO wake states:
- SEPARATED (warm up-ladder): Cd 0.174-0.189, separation ~116-121°, base
  Cp -0.21, shoulder Cp -2.52 — the continuation of the supercritical
  branch carried up from 2e7.
- ATTACHED (cold two-stage): Cd 0.108, separation pushed to ~180° (no
  crossing before the base), base Cp ≈ -0.02 (near-zero ⇒ pressure recovers
  almost fully), knee 162°. The cold start, seeded uniformly, relaminarizes
  into a nearly-attached low-drag state.
The gap is ~0.08 in Cd (0.108 vs 0.189), FAR larger than the crisis-band
spread (0.05, record -0010). It first appears near 1e8 (dn-ladder Cd already
undershoots up by ~0.02-0.05, e.g. 1e9: up 0.206 vs dn 0.170) and widens
with Re. As at the crisis: branch selection is protocol-dependent, CL=0 to
~1e-7 throughout (symmetric branch only), firm asymptote-level claims belong
to URANS. The Cd "asymptote" is therefore a BAND 0.11-0.21, not a level.

## FT-SA control (AI OFF) — attribution of the missing transcritical rise

Classical fully-turbulent SA (AI_SA=0, freestream chi_inf=3.0, the
SA-recommended fully-turbulent level; no fSlow seed scaling — that is an
AI-term device; cold two-stage each). Verified AI-off per the echo test.

| Re  | FT-SA Cd | FT sep(knee) | FT base Cp | SA-AI up Cd (Tu0.2) | SA-AI sep |
|-----|----------|--------------|------------|---------------------|-----------|
| 2e6 | 0.371    | 105°         | -0.42      | 0.227 (cold)        | 94°       |
| 4e6 | 0.299    | 108°         | -0.38      | 0.203               | 95°       |
| 7e6 | 0.280    | 109°         | -0.36      | 0.197               | 119°      |
| 1e7 | 0.270    | 110°         | -0.35      | 0.197               | 120°      |
| 2e7 | 0.246    | 112°         | -0.32      | 0.199               | 120°      |
| 1e10| 0.125    | 123°         | -0.08      | 0.189 up / 0.108 cold | 121°/180° |

**Discriminating readout, stated plainly:** fully-turbulent SA does NOT sit
at the SA-AI operating point. It separates EARLIER (knee 105-112° vs SA-AI
119-120° in the supercritical range) and carries HIGHER drag (0.25-0.37 vs
0.19-0.23) with lower base pressure — the classical fully-turbulent
signature (eddy viscosity thickens the attached BL, promoting earlier
turbulent separation). So the transition model is NOT merely reproducing the
fully-turbulent baseline: SA-AI's laminar run-up keeps the BL thinner and
separation LATER than FT-SA.

BUT neither reaches the experimental transcritical recovery: FT-SA
0.25-0.37 and SA-AI 0.19-0.23 both fall well short of Roshko/Schewe
Cd≈0.55-0.7 at 3.5e6-1e7, and BOTH separate at 105-123° vs the experimental
~147°. Therefore the missing transcritical rise is **NOT a transition-model
fault** (SA-AI's front is already e^N-consistent, record -1041; moving it
does not rebuild pressure drag — the front marches to 2° at 1e10 with Cd
unchanged) and **NOT curable by turning transition off** (FT-SA is closer
but still ~2x under experiment). It is a BASELINE-SA turbulent-separation +
steady-symmetric-wake deficiency: SA turbulent Cf on the strongly convex aft
surface separates too early, and the steady symmetric dead-air wake cannot
reproduce the pressure recovery of the real (unsteady, narrow) supercritical
wake. This exonerates the transition closure for the transcritical gap and
localizes the deficiency in the SA turbulent branch + steady-wake content —
the fault attribution the directive sought. (Consistent quasi-evidence: at
2e7 the three SA-AI seeds' fronts span 77-89° yet Cd spans only 0.185-0.215
— front location is nearly decoupled from Cd once supercritical.)

## Seam deltas

- 2e7 highre↔ultra (per seed, up): Tu0.2 0.1992 vs 0.1917 → 3.8%; Tu0.7
  0.2146 vs 0.2025 → 5.8%; Tu0.05 0.1850 vs 0.2114 → 13.3% (the ultra 2e7
  point is a limit-cycled COLD ladder start vs the highre warm up-ladder
  top — branch+protocol, not pure mesh; ultra is the in-y+-validity value
  and is authoritative). Both meshes now agree the supercritical floor to a
  few % where protocol matches; ultra REPLACES the y+=1.4 highre 2e7 points.
- 5e7 highre(cold)↔ultra: 0.1725 vs 0.1867 up (highre y+≈3.5 there — a
  continuity check only, caveated as designed).
- Earlier seams unchanged (0412): 300/1e3 <0.6%, 2e6 cold-cold 2.3%,
  4e6 2.4%.

## y+ ladder (measured, y1·Re·√(Cf_max/2), ultra final fields)

  Re    2e7    5e7    1e8    3e8    1e9    3e9    1e10
  y+   0.004  0.006  0.01   0.02   0.05   0.14   0.44-0.47

All ultra points in validity (≤0.7); the mesh is over-resolved below ~3e9
(fine — no cost concern at these cell counts) and lands at ~0.45 at 1e10.

## Findings ledger

1. Ten-decade single-model traverse achieved (Re 1→1e10); crisis mid-span;
   Re>1e7 is model extrapolation (no experiment/LES exists there — stated).
2. Branch multiplicity at ultra-high Re: attached (Cd~0.11, cold) vs
   separated (Cd~0.19, warm) states, gap ~0.08, from ~1e8 up. Asymptote is
   a protocol-dependent BAND, not a level. Needs URANS to select.
3. Seed collapse is partial: fronts saturate at the nose (89°→2°), spread
   narrows to ~5% at 3e9, but re-scatters at 1e10 via (2). Not a clean
   single curve.
4. FT-SA control: fully-turbulent SA separates earlier (105-112°) and drags
   higher (0.25-0.37) than SA-AI; BOTH ~2x under the experimental
   transcritical recovery. Missing rise = baseline-SA turbulent separation +
   steady-wake, NOT the transition model. AI-off proven by empty echo.
5. Remote relocation clean: canon binary verified (same commit), 83/83
   SA-AI echoes canon, 6/6 FT echoes empty (AI-off). No cross-machine model
   drift.
6. Syncback caveat (process hygiene): the first syncback anchor logic was
   faulty and merged jsonl rows without pulling all case dirs; caught at
   harvest (missing summaries), fixed by an explicit per-case rsync of all
   38 ultra + 6 FT dirs. No data lost (remote is source of truth); the
   unified local tree is now complete (173 cases in the figure).
7. Steady limit cycles: pervasive but mild through the ultra arm (many
   stage-1 breathers; a few genuine, flagged per case in jsonl).

## Cost

Ultra + FT: 43 new cases. Extension-campaign cumulative (this harvest):
89 cases, 9.89 GPU-h. Ultra/FT increment ≈ 6.8 GPU-h (V100; warm ladder
steps 210-350 s, cold two-stage starts 850-1900 s, the 2e7 cold starts
dominate). Two remote GPUs, occupancy-checked, released on completion.
Campaign grand total (matrix + all extensions): 173 cases.

## Artifacts (full paths)

- Data root: `/local_data/qiqi/sa-ai/dragcrisis_matrix/` (local, unified) and
  the same tree on 017-v100-dev. Ultra cases
  `cyl_Re<int>_Tu{0.05,0.2,0.7}_{up,dn}_ultra`, sanity
  `cyl_Re10000000000_Tu0.2_cold_ultra`, FT `cyl_Re<int>_Tuft_cold_{highre,ultra}`,
  5e7 seam `cyl_Re50000000_Tu0.2_cold_highre`. `matrix_summary.jsonl`
  (173 rows), templates `template_ultra` (+ prior families), remote worker
  logs `chain_ext_{ultraA,ultraB,sanity}.log`.
- Repo copies: `paper/data/dragcrisis_matrix_summary.jsonl`,
  `paper/data/dragcrisis_cd_re_computed.json`.
- Paper figure (Re 1→1e10, log-log, lit overlay + FT-SA line):
  `paper/figs/dragcrisis_cd_re.pdf`; preview
  `paper/repro/cfd/figs_explore/dragcrisis_cd_re.png`. Regen:
  `regen_dragcrisis_cd_re.py --re-window full --include-ultra --logy`.
- Exploratory: `paper/repro/cfd/figs_explore/dragcrisis_fullspan.png`,
  `dragcrisis_matrix.png`.
- Scripts (committed with this record): `run_dragcrisis_extension.py`
  (ultra + FT chains, ft_env AI_SA=0), `run_dragcrisis_matrix.py` (env_fn
  hook), `dragcrisis_extension_harvest.py` (seed-collapse + FT sections),
  `regen_dragcrisis_cd_re.py` (FT line), `build_dragcrisis_pilot.py`,
  `regen_dragcrisis_matrix_figs.py`, `dragcrisis_matrix_janitor.py`.
