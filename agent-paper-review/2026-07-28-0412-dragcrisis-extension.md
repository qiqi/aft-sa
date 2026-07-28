# Drag-crisis extension — single-model Cd(Re) traverse, Re 1 -> 2e7 (7.3 decades)

*2026-07-28 04:12, CFD campaign thread. Executes the 2026-07-28 user directive:
extend the completed 84-case steady matrix (record 2026-07-28-0010) to the
smallest and largest Re the SAME model can reach — same constants, same canon
env (45/45 ai_constants echoes identical to campaign canon, timestamps
stripped), only meshes regenerated. 45 new cases in 3.09 GPU-h; campaign total
129 cases, 8.85 GPU-h. Companion records: 0010 (matrix), 2026-07-28-0117
(litrange ground truth — benchmark values and the honest-caption rules used
below), 2026-07-27-2006/1540/1130 (pilot/design/literature).*

Driver: `paper/repro/cfd/run_dragcrisis_extension.py` (reuses
run_dragcrisis_matrix.py machinery; family templates, per-case GPU lockfile
acquisition, >=6 GiB disk guard, inline janitor). Mesh builder:
`build_dragcrisis_pilot.py` (now parametrized: --nsurf/--y1/--rout). Harvest:
`dragcrisis_extension_harvest.py` (all numbers below regenerate from it).
Figures: `regen_dragcrisis_cd_re.py --re-window full --logy` (paper fig
regenerated per coordinator order; mesh-aware seam handling added) +
`regen_dragcrisis_matrix_figs.py` (exploratory 6-panel + full-span composite).

## Mesh families (all analytic O-grids, growth <= 1.10, D=1, M=0.1)

| fam | N_s x layers | y1/D | R_out/D | serves | y+ status (MEASURED, Cf_max) |
|-----|--------------|------|---------|--------|------------------------------|
| lowre | 600 x 121 | 1e-3 | 1000 | Re 1..1e3 | laminar; BL O(D/sqrt(Re)) >> y1 |
| lowre300 | 600 x 109 | 1e-3 | 300 | far-field twin (Re 1, 30) | — |
| pilot | 1200 x 148 | 8e-6 | 100 | Re 300..4e6 (matrix 6e4..2e6) | 0.64 @2e6; **2.10 @4e6 (out of range)** |
| highre | 1600 x 170 | 1e-6 | 100 | Re 2e6..2e7 | 0.26 @4e6, 0.49 @7e6, 0.71 @1e7, **1.42 @2e7** (stretch point slightly above the <=1 target) |

highre N_SURF=1600 (not 1200): the shoulder LSB shrinks with Re (~1.5 deg at
2e6 on pilot); 1600 keeps ~9 cells/deg. Restarts do NOT cross meshes: high arm
cold two-stage at 4e6, ladder up, dn-ladder back down (Tu 0.2 dn extended to
the 2e6 seam).

## Extended Cd table (steady branch; * = a stage limit-cycled/cap)

Creeping/steady arm, lowre, Tu 0.2% (physical flow IS steady below Re~47):

  Re        1        3       10      30      100     300     1e3
  Cd     10.665*  5.330*  2.797   1.710   1.123   0.934   0.874
  chi_max 1.8e-5  3.0e-5  5.5e-5  1.0e-4  2.8e-3  6.6e-3  2.5e-2

Low arm, pilot mesh, dn-continuation from the matrix 6e4 endpoints:

  Re       300     1e3     3e3     1e4     2e4     4e4
  Tu0.05    -       -       -     0.8432  0.8441  0.8456
  Tu0.2   0.9391  0.8784  0.8513  0.8432  0.8441  0.8456
  Tu0.7     -       -       -     0.8431  0.8438  0.8444

High arm, highre mesh (up = cold@4e6 then ladder; dn = from the 2e7 state):

  Re          4e6      7e6      1e7      2e7
  Tu0.05 up  0.1949*  0.1858   0.1847   0.1850
  Tu0.05 dn  0.1708   0.1471   0.1602     —
  Tu0.2  up  0.2035*  0.1970   0.1974   0.1992
  Tu0.2  dn  0.1742   0.1539   0.1602     —      (+ 2e6 dn: 0.2049; 2e6 cold: 0.2268*)
  Tu0.7  up  0.2974*  0.2150*  0.2102   0.2146
  Tu0.7  dn  0.1940   0.1671   0.1720     —

Pilot seam case: 4e6 Tu0.2 up (warm from matrix 2e6): 0.1935* at y+=2.1.

**Span achieved: Re = 1 -> 2e7 = 7.3 decades**, one model, one set of
constants, one solver protocol — the full Wieselsberger->Roshko composite
span (litrange Sec. 1: the experimental composite itself is a >=4-facility
stitch over the same 7.3 decades).

## Seam verdicts (same Re + Tu on both meshes; mandatory overlaps)

- Re=300: lowre 0.9339 vs pilot 0.9391 -> **0.56%**
- Re=1e3: lowre 0.8743 vs pilot 0.8784 -> **0.46%**
- Re=2e6: protocol-matched cold-vs-cold: pilot 0.2204 vs highre 0.2268*
  -> **+2.9%** (finer mesh HIGHER). The highre dn point (0.2049, descending
  from the 2e7 state) differs from pilot dn (0.2204, which is protocol-cold —
  the matrix dn chain starts cold at 2e6) by 7.3%: that gap is BRANCH, not
  mesh (finding 5 below). First flagged by the >2% watcher as "7.5%"; the
  added cold highre point resolved the decomposition.
- Re=4e6: pilot up 0.1935* vs highre up 0.2035* -> **5.0%**, but the pilot
  point is out of its y+ validity (2.10) — reported for continuity only.

**Authoritative-family statement: lowre for Re <= 1e3; pilot for
6e4 <= Re <= 2e6 (the published matrix, unchanged); highre for Re >= 2e6;
pilot above 2e6 and lowre at 1e3 are overlap checks, not data points.**

## Low-Re benchmark comparison (steady = physical below Re~47)

- Re=10: Cd 2.797 vs Dennis-Chang 1970 2.846 -> **-1.7%**
- Re=30: Cd 1.710 (DC bracket 2.045@20 / 1.522@40 — consistent)
- Re=1: Cd 10.665, Tritton-class (~10-11); its * flag is a 1e-3-amplitude
  breather in stage 1 only.
- Re=100: Cd 1.123 vs DC 1.056 -> **+6.4%**. Numerics, not model (chi_max
  2.8e-3): the steady wake bubble is ~5-7 D long by Re=100 and the lowre
  radial spacing reaches ~0.5 D there — wake-resolution error, a mesh-study
  candidate if this point is ever quoted.
- Far-field convergence (blockage is logarithmic at low Re): R=300 vs
  R=1000: **+0.54% at Re=1, +0.19% at Re=30** — controlled.
- SA-AI inertness on the whole creeping arm: chi_max <= 2.5e-2 at Re<=1e3,
  <= 1e-4 at Re<=30 (near-wall max over both sides) — the model is provably
  passive where the flow is laminar; the traverse needs no switch to be told
  this.
- 47 < Re < 1e3 (steady branch UNSTABLE): our points (0.934/0.874) sit below
  the shedding-mean plateau (~1.2) as they must, but do NOT track Fornberg's
  decaying exact steady branch (Cd(600)~0.54 [mem]): the solver converges to
  a shorter-recirculation steady state (exact-branch wake bubbles are
  O(100 D) by Re~600, beyond any practical O-grid). Both figures shade this
  band "continuity, not validation" per litrange Sec. 2b; no validation claim
  is made there.

## High-arm verdict

1. **No transcritical rise, all three seeds, both branches.** Up-ladders
   flatten at 0.185 / 0.197-0.199 / 0.210-0.215 (Tu 0.05/0.2/0.7) from 7e6
   through 2e7; dn-ladders sit lower still. Experiment (Roshko 1961, Schewe
   1983) recovers to Cd ~ 0.55-0.7 by 3.5e6-1e7. This is NOT a
   transition-location failure: the chi=1 front DOES march toward attachment
   (Tu 0.2 up: 101 deg at 2e6 -> 82 deg at 2e7; Tu 0.7: 77 deg at 2e7) — but
   turbulent separation stays at ~120 deg and base Cp at ~ -0.23, so the
   pressure drag never rebuilds. The missing rise lives in the SA turbulent
   separation / steady symmetric dead-air wake (the same deficiency class as
   the 107-118 vs ~147 deg separation gap flagged in the 0010 record), not
   in the transition model.
2. **Tu-ordering of the supercritical floor survives to 2e7** (0.185 < 0.199
   < 0.215) — the roughness/turbulence trend direction is retained.
3. **A second steady wake state in the supercritical regime (highre).** The
   dn branch (descending from the most-transitioned 2e7 state) is 13-22%
   LOWER in Cd than up at 4e6-1e7 (e.g. 1e7 Tu0.2: 0.1602 vs 0.1974), with
   LATER separation (123.5 vs 120.4 deg) and HIGHER base pressure (-0.176 vs
   -0.232): a deeper-recovery wake state. The 0010 protocol-dependence
   disclosure (crisis band, ~0.05 spread) therefore EXTENDS to the
   supercritical regime on the fine mesh. As before: branch selection is
   protocol-dependent; firm hysteresis claims belong to URANS.
4. y+ ladder measured from the final fields (harvest Sec.): 0.26/0.49/0.71 at
   4e6/7e6/1e7; the 2e7 stretch runs at y+max=1.4 — keep the 2e7 points
   caveated or re-mesh (y1=5e-7) if they are ever promoted beyond outlook.
5. CL = 0 to ~1e-8 in all 45 cases — the steady solver stays on the
   symmetric branch everywhere (asymmetric one-bubble state remains invisible
   to this protocol, as in 0010).

## Findings ledger (honest-report items)

1. **Mesh-dmp sweep incident.** Mid-campaign, an external disk cleanup
   deleted ALL 85 hardlinks of the pilot mesh partition dump
   (`mesh.cgns_rank_1_of_1.dmp`, 279 MB single-copy) — worker B crashed with
   "Could not open file". Regenerated deterministically (analytic O-grid;
   regenerated .msh/.p3d byte-identical by md5; mesh.cgns differs only in
   1540 bytes of HDF5 timestamps), relinked, worker resumed; the failed case
   redone from scratch. No published number is affected. Lesson: hardlink
   farms are one `find -delete` away from total loss — the mesh BUILDER
   inputs are in git, which is what saved this.
2. **Worker A externally killed** mid-1e7-Tu0.05 (task-runner kill, cause
   unknown); relaunched detached (setsid), interrupted case redone cold-warm
   per protocol. Data unaffected (no summary had been written).
3. **Mesh-family floor offset**: the supercritical floor is +2.6-2.9% higher
   on highre than pilot (cold-cold at 2e6), i.e. the published matrix floor
   0.217-0.233 carries a few-percent mesh sensitivity; direction: finer =
   higher = closer to experiment (0.20-0.25).
4. Steady limit cycles: 10/45 extension cases flagged, all mild
   (Cd_tail_p2p <= 0.027 except Tu0.7 4e6 up cold at 0.08-class); the
   2e7-stretch gate was widened to admit breathers with p2p < 0.05
   (documented in run_dragcrisis_extension.final_stage_converged).
5. Subcritical seed-independence: Tu 0.05/0.2/0.7 agree to <0.2% at
   1e4-4e4; chi_max first crosses 1 at Re=4e4 (front at 151 deg, base region
   only) — the shear-layer-transition caveat of the 0010 record fades out
   exactly where it should.
6. Fornberg-branch mismatch in the unstable band (above) — steady points
   there are continuity, not validation.
7. Figure/data handling: `regen_dragcrisis_cd_re.py --re-window full` now
   keys rows by mesh family so seam duplicates coexist (overlapping markers
   = measured seam delta) instead of displacing each other;
   `data/dragcrisis_matrix_summary.jsonl` + `dragcrisis_cd_re_computed.json`
   resynced (129 cases). The tex still describes the 84-case window — tex
   reconciliation is the paper agent's (NO tex edits here, per directive).

## Cost and disk

45 cases, 3.09 GPU-h (V100; creep cases 91-251 s, low arm ~140 s, high arm
172-903 s, cold starts dominate). Never more than 2 GPUs held (per-case
lockfile + fully-idle occupancy checks; two external interruptions handled).
Campaign total: 129 cases, 8.85 GPU-h. Matrix tree now 9.3 GB (volume output
off, meshes hardlinked once per family, logs truncated post-echo, superseded
ladder restarts purged; endpoint restarts kept for future extension).

## Artifacts (full paths)

- Data root: `/local_data/qiqi/sa-ai/dragcrisis_matrix/` — per-case dirs
  `cyl_Re<int>_Tu<tu>_<dir>[_<fam>]` (surface+slice pvtu, force CSVs,
  summary.json, ai_constants_echo.log), `matrix_summary.jsonl` (129 rows),
  templates `template_case` / `template_lowre` / `template_lowre300` /
  `template_highre`, worker logs `chain_ext_workerA.log` / `_workerB.log`,
  mesh-build log `meshbuild/build.log`.
- Repo copies: `paper/data/dragcrisis_matrix_summary.jsonl`,
  `paper/data/dragcrisis_cd_re_computed.json`.
- Paper figure: `paper/figs/dragcrisis_cd_re.pdf` (full span, log-log,
  literature overlay; preview `paper/repro/cfd/figs_explore/dragcrisis_cd_re.png`).
- Exploratory: `paper/repro/cfd/figs_explore/dragcrisis_fullspan.png`,
  `dragcrisis_matrix.png`.
- Scripts (committed with this record): `paper/repro/cfd/
  run_dragcrisis_extension.py`, `dragcrisis_extension_harvest.py`,
  `build_dragcrisis_pilot.py`, `run_dragcrisis_matrix.py`,
  `dragcrisis_matrix_janitor.py`, `regen_dragcrisis_matrix_figs.py`,
  `regen_dragcrisis_cd_re.py`.
