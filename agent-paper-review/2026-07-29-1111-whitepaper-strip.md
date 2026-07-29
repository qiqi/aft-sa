# Whitepaper strip + enrichment pass (2026-07-29 1111)

Target: `paper/whitepaper.tex` only. Two folded operations in one commit:
1. STRIP: sentence-by-sentence removal of purely-qualitative /
   interpretive prose and of body prose that merely restates numbers
   already carried by an adjacent figure/table/equation (constants
   table is the single source for constant determinations).
2. ENRICH: added identifying quantitative labels from `sa-ai.tex`
   (verbatim) to figure captions so each figure is self-contained.

Build: pdflatex x2, 0 errors, 0 undefined refs, 0 missing figures.
Page count: 62 (before) -> 60 (after strip) -> 61 (after enrichment).

## Sentences removed, by section (full sentences; clause-trims noted separately)

- Preamble: 5 (entire opening paragraph, incl. the "6:1 spheroid
  omitted" note and the reproducibility/repro-path prose).
- Sec 1.1 Transport eq: 3 full + 2 lead-ins ("The model is ... gives
  rise to its own transition"; "identical in form ... four places";
  "with omega the vorticity magnitude and d the wall distance").
- Sec 1.2 Local indicators: 5 full + 3 clause-trims ("a direction
  defined only up to sign ..."; "Two odd functions carry the model:";
  "the accumulated curvature ... Rayleigh's theorem ..."; "homogeneous
  of degree zero ..."; "Because the rate depends on the frozen mean
  flow ..."). Fig sphere pointer reduced to minimal.
- Sec 1.3 Rate/onset: 5 full ("Two functions ... carry the whole
  model."; "A favorable gradient ..."; "The ceiling a_max=0.19 is an
  eigenvalue, not a fit ..." [redundant w/ constants table]; "The ramp
  width w=0.35 ..." [table]; "The single scale k=0.712 ..." [table])
  + 2 lead-ins. Unique validation numbers (Re_theta=1181/1108, 6% rms,
  +-4%, 11-12%, beta=+0.35/0.45 rates) kept verbatim.
- Sec 1.4 Reduced diffusion: 2 lead-ins ("At full molecular diffusion
  the drain overwhelms ..."; "Pressing further costs mesh:").
- Sec 1.5 Handover/tie: 3 full ("The max in Eq. hands production ...";
  "The production width tau=4 ..." [table]; "The linear profile has
  zero curvature ...") + 1 lead-in.
- Sec 1.6 Eddy-viscosity: 2 full ("equals unity identically ..."; "The
  s ramp is deliberately faster ...") + 2 clause-trims.
- Sec 1.7 Freestream seed: 3 full ("The map is Mack's ... unchanged:";
  "with Tu_frac the rms turbulence intensity ..."; "The correspondence
  is not exact ...") + 2 lead-ins.
- Sec 1.9 Running the model: 4 full ("Every laminar solution may
  coexist ..."; "An implementer needs a branch-selection protocol
  ..."; "The scaling multiplies the residual ..."; "Equivalent
  protocols for solvers without residual scaling ...") + 2
  clause-trims. Load-bearing seed/slowdown convention (chi_inf f,
  f=1e-2, 1/f mis-seed) kept verbatim.
- Sec 2 intro: 2 clause-trims.
- Sec 2.1-2.4 (flat plate / NLF / Eppler / resweep): ~11 qualitative
  clause-trims off figure/table pointers ("the Flow360 sweep---", "the
  OpenFOAM replication---", "consistently slightly early", "the
  untuned N=9-class fronts land ...", "the finest grids track the
  measured bucket ...", "separation reproduced to a few percent ...",
  "the computed state ... is unique among protocol-admitted
  initializations", ...). All numeric clauses kept verbatim.
- Sec 2.5 Drag crisis: ~7 clause-trims + concession-list tails cut
  ("---opposite to the physical hysteresis sense---so no hysteresis
  claim is made"; "is a statement about the symmetric steady protocol,
  not the physics"; "sits well below measurement, as any steady
  symmetric wake solution must"; "remain early against measurement";
  "There is no transcritical recovery:"; "not the transition
  content"). All numbers (27/84, 0.011, 0.199, 0.055, 0.003, floors
  0.185/0.199/0.215, angles, etc.) kept verbatim.
- Sec 2.6 Daedalus: 3 clause-trims.

Approx total: ~40 full qualitative/redundant sentences deleted, plus
~30 qualitative clause-trims off numeric sentences (numeric clauses
preserved verbatim). New final "Proposed two-branch kernel" section
verified already numbers/equations only -- left as-is.

## Enrichment added to captions (numbers copied verbatim from sa-ai.tex)

- f:sphere (fig:model): FS betas +0.30/0/-0.10/-0.16/-0.19/-0.1988,
  Stewartson beta=-0.19 H~4.9; Re_theta=500 (attached) / 200
  (reversed); OmegaHat*IHat contours {0.025,0.2,0.4,0.6,0.8,1.0}.
- f:graze (fig:onsetgraze): attached family beta list
  +0.15..-0.1988 at its Re_theta0(H); Stewartson H~4.9 Re_theta0~26;
  max OmegaHat*IHat 0.162/0.078/0.037 at beta -0.10/0/+0.10; envelope
  k=1, model threshold k=0.712.
- f:nuhat (fig:nuhat): per-wedge max OmegaHat*IHat 0.162/0.078/0.037;
  nu_hat_inf=1, a_max=0.19, (c_nu,ai,k)=(1/6,0.712); N=9 7% past
  Drela; favorable N=1 ~11% early.
- f:calibrate (fig:calibrate): H stagnation~2.2/Blasius~2.59/sep
  3.98/Stewartson~10.6; a_max=0.19, c_nu,ai=1/6; onset within ~+-12%
  from H=2.44 (beta=+0.15) to 3.98.
- f:stagbistab: chi_inf=0, laminar for L<=685, Re_r band, maxchi~0.025L,
  L=3000 (Re_r=9e6) maxchi=76, contour levels 1/c_v1/30, maxchi~24 at
  Re_r=1e6.
- f:flatplate: five Tu 0.04-0.60%, chi=1 & chi=c_v1=7.1, laminar/
  turbulent Cf correlations, AGS within 10%.
- f:flatplateof: five Tu 0.04-0.60%.
- f:nlfaft: Re=4e6, chi=1 convention, SA-AI alpha set, workshop sweep.
- f:nlfpolar: fully-turbulent SA chi_inf=3, 1.6-2.5x, +144/+150%
  (0/4deg), +67/+74% (9/15deg).
- f:nlfcflow/cfhigh/negalpha: added Re=4e6.
- f:epppolar: fully-turb SA chi_inf=3 (77-97% bucket, +151% at 7deg),
  mesh agreement dCd<=2.4e-4 / 1.0e-3, bypass -6..-10 counts.
- f:eppbubble: Re=2e5, XFOIL Cl=1.17 vs mfoil 0.928 at 7deg.
- f:eppcflow/cfhigh: added Re=2e5.
- f:eppresweepforces: alpha=5deg, Re range, LTPT run counts 1/2/4/1/1.
- f:dragcrisiscd: Re_D=1-1e10, middle seed Tu=0.2%.
- f:dragcrisisangles: three seeds Tu=0.05/0.2/0.7%.
- f:dragcrisisfpg: nose Re_D 2e6/7e6/2e7, N=ln(1/chi_inf)=4.53.
- f:daepolar: L1 agreement 0.46-0.53% / 0.4-1.6 counts, L0 0.74-0.97%
  / 18-21 counts, taper break eta=0.88.
- f:daesurf: Re_root=5e5.

## Example removed sentences

- "a_max=0.19 is an eigenvalue, not a fit" (author's explicit example;
  number already in constants table).
- "The scaling multiplies the residual ... a branch selector, not an
  accelerator."
- "physically the stagnation zone relaminarizes, so the branch is
  spurious, and any transition model that leaves SA's fully turbulent
  behavior unmodified is bistable at high Reynolds number."
- "The correspondence is not exact: e^N methods declare transition at
  a sharp N_crit ..."
