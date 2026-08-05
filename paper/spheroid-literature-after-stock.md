# Who has computed transition on the inclined 6:1 prolate spheroid after Stock (2006)

Compiled 2026-08-05.  Purpose: find out whether Stock's near-symmetric transition
front at the windward and leeward symmetry planes (phi = 0 and 180) is an artefact
of his own calibration, or an independently reproducible result.

## Where the PDFs are

All in `references/` at the repo root.  Four of the five were already there before
this search; only the case description was added.

| what you want | file | pages |
|---|---|---|
| **all-participant comparison** at alpha=5, split by meridian phi=0/60/180, AND **Coder's AFT2017b at 5 deg** | `TransitionMPW1_summary_Coder_NAS2021.pdf` | Case 3 = pp. 23-33.  alpha=5: phi=0 on **24** (LST+e^N), **25** (transport w/o CF -- AFT2017b is the blue curve), **26** (transport w/ CF); phi=60 on 27-28; phi=180 on **29** (LST+e^N), **30** (transport w/o CF -- AFT2017b blue), **31**.  alpha=15 on 32-33 |
| **Langtry-Menter** (+ Langtry CF extension, OVERFLOW) | `overflow_tmw.pdf` | 15 pp; spheroid = Sec. C, Figs. 12-13.  NB the text layer has a broken font encoding -- read the rendered page, not extracted text |
| **SA-BCM** one-equation transitional + mesh adaptation | `tarsia2025.pdf` | 20 pp; spheroid = Sec. C, Figs. 12-17 |
| DLR automatic two-N-factor e^N, alpha=10/15 | `krimmelbein-krumbein-2010-automatic-transition-prediction.pdf` | 16 pp; spheroid = Sec. A, Figs. 4-5 |
| workshop Case 3 flow conditions, gridding, Tu | `TransitionMPW1_case_descriptions.pdf` | 4 pp; Case 3 on p. 3 |
| workshop agenda (who presented what) | `TransitionMPW1_proceedings.pdf` | 1 p |

**Stock (2006) itself is at `paper/spheroid.pdf`** -- 11 pp, AIAA J. 44(1),
doi 10.2514/1.16026, MIT Libraries copy stamped 26 July 2026.  It is
GITIGNORED (`.gitignore:58: *.pdf`), which is why it is untracked and why a
search for `*stock*` does not find it.  See section 2b.

Two wanted papers could not be downloaded: hal-03871832 (HAL serves an Anubis
JS challenge to non-browsers) and Plasseraud & Mahesh JFM 960 A3 (2023)
(Cambridge paywall, no preprint; the same group's open arXiv:2507.03187 is a
different paper on lee-side vortex topology).

**It is independently reproducible.**  See "The answer to the symmetry question"
below.  This retires the open question raised in
`repro/cfd/diag_spheroid_symmetry_planes.py`, but only its EMPIRICAL half: what
makes e^N symmetric there is still not derived.

---

## 1. The 1st AIAA CFD Transition Modeling and Prediction Workshop (TMPW-1)

Held virtually 21-22 January 2021 with the AIAA SciTech Forum.  **Case 3 is the
DLR inclined 6:1 prolate spheroid**, and its specified conditions are ours:

| quantity | workshop value | ours |
|---|---|---|
| Mach | 0.13 | 0.1294 (tunnel Mach at Re_L = 6.49e6) |
| alpha | **5, 10, 15 deg** | 0, 2.5, 5, 10 |
| Re_L | 6.5e6 | 6.49e6 (Goettingen), 6.56e6 (both facilities at a=10) |
| reference temperature | 540 R | -- |
| freestream Tu | **0.15 %** | 0.0724 % calibrated / 0.33-0.40 % measured |
| free-stream eddy-viscosity ratio | left to the participant | -- |

Verbatim from the case description: "Freestream turb. Intensity 0.15% (or,
critical Tollmien-Schlichting and crossflow amplification factors calibrated
based on wind-tunnel data)".  So the workshop's 0.15 % is a THIRD value, sitting
between Stock's calibrated seed and the measured hot-wire Tu, and the workshop
explicitly permits the calibrated-N alternative instead.

Quantities of interest: surface c_f, surface c_p, and transition location.
**11 data submittals for Case 3** (5 North America, 3 Europe, 3 Asia; 3
government, 5 academia, 1 industry, 2 industry/academia).

Coder's summary presentation organises the alpha = 5 comparison **by meridian --
one slide each at phi = 0, phi = 60, phi = 180** -- which is exactly the cut our
symmetry question needs, and splits the submittals into three families:

  * **LST + e^N** (loosely coupled linear stability), over SST and SSG/LRR base flows
  * **transport models without CF transition**: SA + AFT2017b, SA + B-C
    (algebraic), Lag Elliptic Blending, SST + gamma-Re_theta (LM2009),
    SST + gamma (MS2015)
  * **transport models with CF transition**: SA + MB+CF,
    SSG/LRR + gamma-Re_theta(LM2009)-CF(He), SST + gamma-Re_theta(LM2009 or
    2015)-CF(He), SST + gamma-Re_theta(LM2009)-CF+, SST + gamma(MS2015)-CF(He)

Sources
  * case description: https://transitionmodeling.larc.nasa.gov/wp-content/uploads/sites/109/2020/02/TransitionMPW_CaseDescriptions.pdf
  * Coder summary slides: https://www.nas.nasa.gov/assets/nas/pdf/ams/2021/AMS_20210304_Coder.pdf (Case 3 on pp. 23-33; alpha=5 phi=0 on p. 24, phi=180 on p. 29)
  * workshop site: https://transitionmodeling.larc.nasa.gov/workshop_i/

## 2. The answer to the symmetry question

Read off the c_f(x/L) curves on Coder's slides at alpha = 5, Re = 6.5e6.
Transition = the c_f jump.  Measured hot-film points are the green squares.

Curves are identified by the legend colour (blue = SA + AFT2017b, green =
SA + B-C, purple = Lag Elliptic Blending, **red = BOTH** SST + gamma-Re_theta
LM2009 and SST + gamma MS2015, so the red curves cannot be separated from each
other); the boxed letters are unpublished participant codes.  Slide 30 carries an
erratum stating that "k" is really SST-2003 + LM2009.mod + CF(HE).mod and belongs
on the with-CF page, so ignore "k" here.

| family | phi = 0 (windward) | phi = 180 (leeward) | windward - leeward |
|---|---|---|---|
| **measured** (green squares) | 0.52-0.58 | 0.50-0.55 | ~0.00 |
| **LST + e^N** (2 submittals) | 0.50, 0.55 | 0.54, 0.56 | ~0.00 |
| **SA + AFT2017b** (blue "l") | **never transitions** | **0.19** | **> 0.8** |
| SA + B-C (green "h") | 0.78 rise / 0.85 complete | 0.40 | -0.42 |
| Lag Elliptic Blending (purple "m") | 0.05 | 0.055 | ~0.00 (both bypassed at nose) |
| red "f" | 0.55 | 0.315 | -0.24 |
| red "m" | 0.27 | 0.87 | +0.60 (the one inverted case) |
| red "l" | -- | 0.31 | -- |
| red "j" | never transitions | 0.49 | > 0.5 |

The AFT2017b entry is worth restating exactly, because it is the most extreme in
the set and it is the model closest to ours in construction: at phi = 0 the blue
curve stays on the laminar c_f branch for the whole body, rising only in the tail
closure past x/L = 0.95, while at phi = 180 it jumps sharply at x/L = 0.19 and
peaks at c_f = 0.0045 by 0.23.  There is no windward front to subtract, so the
asymmetry is not merely large, it is qualitative.

Two things follow, and both matter to us.

**(a) The near-symmetry is a property of e^N, not of Stock.**  An independent
loosely-coupled linear-stability calculation on a modern RANS base flow puts the
front at 0.50-0.56 at BOTH symmetry planes and matches the measurement at both.
Stock's result is reproducible; there is nothing to explain away.

**(b) Our leeward-early bias is the signature failure of the local
transport-equation family, not something idiosyncratic to SA-AI.**  Every
transport model in the workshop that transitions at all in the interior
transitions EARLIER at phi = 180 than at phi = 0, by 0.24 to 0.73 x/L.  Ours is
in that band.  The single most damning entry is **SA + AFT2017b**: Coder's own
amplification-factor transport model, the published construction closest in
spirit to SA-AI's use of chi as a transported e^N surrogate, is the WORST of the
set -- laminar to 0.93 windward and 0.20 leeward.

That localises the cause.  What the LST submittals have and the transport family
does not is a boundary-layer march with real streamwise history along the actual
3-D surface streamline (including the symmetry-plane divergence/convergence terms)
before any amplitude is assigned.  A local model reads elevated near-wall strain
on the converging leeward meridian and fires, without the streamwise run a TS
wave would need to reach the same amplitude.  This is a much sharper statement
than "correlation models are poorly calibrated here."

Caveat: these are numbers read off raster slide images, good to roughly +/-0.02
x/L, and the participant identities behind the plot letters are not published.
Use them as the shape of the field, not as digitized data.

## 2b. What Stock himself says (`paper/spheroid.pdf`)

**He states the answer, and it reframes the question.**  From the discussion of
his Fig. 18, which overlays the computed front for every incidence at comparable
Reynolds number (Re = 7.2e6 for alpha = 0 and 2.5; 6.40e6-6.54e6 for alpha =
5-29.7):

> "Increasing the angle of attack moves transition continously downstream on the
> windward symmetry plane and continously upstream on the leeward symmetry plane
> for alpha > 5 deg.  There exists a distinct zone in the lower-half of the
> prolate spheroid around phi = 50 deg, which separates the downstream and
> upstream motion of the transition line."

So the windward/leeward split **opens above alpha = 5 and is essentially absent
at and below it**.  The near-agreement at alpha = 2.5 and 5 is not a coincidence
needing its own mechanism; it is the small-incidence limit of a split that grows
monotonically with alpha.  Read directly off Fig. 18 (x/L = (X/a + 1)/2):

| alpha | front vs azimuth |
|---|---|
| 0 | dead vertical, X/a = -0.15, **x/L = 0.425** at every phi |
| 2.5 | dead vertical, X/a = -0.09, **x/L = 0.455** at every phi |
| 5 | first real azimuthal structure: a U with both ends near x/L = 0.50, trough x/L ~ 0.345 at phi = 110-120, and a bump back to x/L ~ 0.565 at phi ~ 163 |
| 10-29.7 | the U deepens and the ends separate, windward marching aft, leeward marching forward |

This is a stronger indictment of our alpha = 2.5 result than the two-plane
comparison was.  Stock's alpha = 2.5 front is **azimuthally flat to 0.030 x/L
over the whole body**, and ours sweeps by an order of magnitude more -- so the
`diag_spheroid_fpg_sensitivity.py` finding (dx_front/dbeta ~ 4.2 against a
measured ~1.4) is the primary defect, and the windward/leeward gap is just its
most visible slice.

**His method carries exactly the term our march omits.**  Inviscid field from
potential theory; viscous layer from a finite-difference **three-dimensional**
laminar boundary-layer method (the CERT/ONERA code -- he thanks Cousteix, Arnal
and Houdeville for it) in xi',phi' coordinates at Delta xi' = Delta phi' = 1 deg
with 121 wall-normal points, ~4 M surface-mesh points, N integrated along 21
inviscid streamlines.  He calls out the reason the geometry is interesting in the
same breath: the spheroid exhibits "highly divergent and convergent
three-dimensional viscous flows", and "because of the high convergence and
divergence of the flow ... the computational mesh has to be rearranged
continuously".  Our Mangler-weighted axisymmetric Thwaites march has no such
term, which is the leading suspect for the +0.146.

**Does he actually resolve the effect of lateral squeezing on the profile?**
Asked because the whole diagnosis in `repro/analytic/spheroid_small_alpha_bl.py`
turns on it.  Honest answer, in three parts.

*Structurally, yes, necessarily.*  He solves the boundary layer by finite
differences in the SURFACE coordinate system (xi', phi'), not along streamlines,
with 121 wall-normal points and Delta xi' = Delta phi' = 1 deg.  A 3-D
boundary-layer solution on a two-dimensional surface mesh carries the
d/d(phi') terms exactly -- that is what makes it 3-D -- so the lateral
convergence and divergence act on the full profile, not through a
momentum-thickness width model.  And he gives the reason for that choice himself:
"Basically, it is possible to execute the finite difference computation of the
boundary layer in streamline coordinates.  Because of the high convergence and
divergence of the flow, however, the computational mesh has to be rearranged
continuously.  Hence, it is much easier to calculate the boundary-layer
development in the xi', phi' coordinate system."

*But there is NO passage where he writes the equations or discusses what the
lateral terms do to the profile shape.*  The method is cited to his Ref. 46 and
the CERT/ONERA code is acknowledged.  So this part is inference from what the
method must contain, not a quotation.  Do not claim otherwise in the paper.

*The evidence that his GROWTH is near-symmetric is much stronger, and it is
direct.*  Two figures:

  * **Fig. 8a** (alpha = 10, Re = 6.56e6) plots envelope N_TS along streamlines
    1-21, with 1 the windward symmetry plane and 21 the leeward.  His own text:
    "the N_TS factors achieve large values on the windward side, streamlines
    1-6, and on the leeward side, streamlines 16-21.  The remaining streamlines
    exhibit moderate N_TS factors up to separation."  So in his solution the
    WINDWARD meridian amplifies as strongly as the leeward one.  A
    momentum-integral treatment says the opposite -- windward is thinner and
    accelerating, so it should amplify far less.
  * **Fig. 11a** plots N_TS at every measured transition location across the
    whole Goettingen campaign.  The points lie between about 6.5 and 9.0 about
    the fitted N_TS = 8.0 line, a total spread of ~2.5.  Our own threshold
    sensitivity is dN = 2 <-> dx ~ 0.12 x/L, so that spread is worth about
    +/-0.07 in front position.  If his boundary layer contained the asymmetry a
    momentum integral predicts (+0.287 x/L, i.e. dN ~ 4.8 between the two
    planes), the windward and leeward points in Fig. 11a would separate into two
    clouds ~5 apart in N.  They do not.

So the answer to "did he model the profile response to squeezing" is: he solved
equations that contain it, and the outcome -- large N_TS on both symmetry planes,
small N scatter at measured transition -- is inconsistent with the large
asymmetry any width-based estimate produces.  That is as far as the printed
paper can take it.

**The threshold is fitted to this dataset**, as suspected -- stated in the
abstract: "First, the values of both N factors at the measured transition
locations are calculated, which deliver the stability limit of the prolate
spheroid in the considered wind tunnel.  Second, based on the knowledge of the
stability limit, the transition locations are evaluated."

**He drops curvature deliberately**, which we had not registered: "the
computation with curvature effects included produces results for which a
comprehensive stability limit cannot be found.  To the contrary, the values of
the N factors without curvature effects exhibit ... such a small scatter that
the limiting N factors for both types of waves can be determined with
confidence."

Mechanism statements worth having in the paper:
  * alpha = 5, Re = 6.49e6: "transition is triggered by TS waves near the
    windward and leeward symmetry planes and for the remaining part of the body
    surface simultaneously by TS and CF waves" -- so **no pure-CF band at all at
    alpha = 5**.
  * His Fig. 8 N_TS envelopes along streamlines 1-21 (1 = windward plane,
    21 = leeward): at alpha = 10, N_TS reaches large values at BOTH ends
    (streamlines 1-6 and 16-21) and only moderate values between -- that is the
    U.  At alpha = 20 "the streamlines on the windward side show drastically
    reduced N_TS factors" while 19-21 stay large; at 29.7 only the leeward plane
    has remarkable N_TS.  So the windward collapse happens between 10 and 20.

**Two corrections to our own bookkeeping, found by this check.**  Our diagnostic
hardcoded Stock's alpha = 2.5 pair as (windward 0.460, leeward 0.430).  The
digitized `computed_ts_front` runs 0.430 at phi = 1.8 to 0.460 at phi = 178.4,
i.e. **the pair was swapped**: Stock is leeward-LATE by 0.030, not windward-late.
Same at alpha = 5: (0.432 at phi = 2.4, 0.507 at phi = 175.9), leeward-late by
0.075.  So the three-way ordering at both incidences is

| | alpha = 2.5 | alpha = 5 |
|---|---|---|
| Stock (windward - leeward) | **-0.030** | **-0.075** |
| measured | +0.018 | +0.002 |
| ours (2-D envelope on our c_p) | +0.067 | +0.146 |

Stock's asymmetry is the OPPOSITE SIGN to ours, so the discrepancy at the
leeward plane is larger than previously stated, not smaller.

And a data-quality flag: Fig. 18 puts alpha = 5 at x/L ~ 0.50 on BOTH planes,
against the 0.432 the fig14c trace gives at phi = 2.4, and the two also disagree
at phi = 60 (~0.45 vs 0.474).  The leeward half and the trough agree well.  The
**windward half of the fig14c computed-front trace should be re-checked** against
the figure before it is trusted to better than 0.07.

## 2c. Can the low-Re (1.5e6) branch be pushed to higher incidence?

Asked because a crossflow-free branch at high incidence would be worth a great
deal.  The answer from Stock is **crossflow never contaminates it at any
incidence he ran -- but neither does anything else, because nothing transitions.**

At $Re_L \approx 1.5\times10^6$ the layer runs laminar to the open
free-vortex-layer separation at every measured incidence:

| alpha | Re_L | Stock's finding, close to verbatim |
|---|---|---|
| 5 | 1.52e6 | "predicted to remain laminar up to separation for the small-Reynolds-number case, except for streamline 12, where transition is provoked by TS waves" -- and he adds that at streamline 12 the layer is "drastically thickened ... by the action of the circumferential pressure gradient" |
| 10 | 1.52e6 | "predicted to stay laminar up to separation ... Indeed, the measured transition locations show the same tendency" |
| 29.7 | 1.53e6 | "the boundary-layer flow is predicted to remain laminar up to the free vortex separation line for the low Reynolds numbers Re = 1.52e6 ... and 1.53e6" |

So extending alpha on the low-Re branch buys **separation-line** comparisons,
not transition fronts.  The two knobs are not independent: at fixed Re, raising
alpha trades TS for CF; at fixed alpha, lowering Re trades transition for
laminar separation.  There is no "high alpha, TS-only" window on this body.

Where transition reappears at high incidence is already CF-dominated.  At
alpha = 29.5 Stock runs Re = 3.01e6, 4.48e6 and 8.52e6: "for the smallest
Reynolds number, the flow remains laminar up to separation", and for the other
two "transition is triggered by CF waves, except close to the symmetry planes".
So the first Reynolds number at which a high-alpha case transitions at all is
one where crossflow does the triggering.

Rough consistency check, MY estimate and not Stock's: his limiting factors for
Goettingen are N_TS = 8.0 and N_CF = 5.5 (ONERA F1: 7.0 and 6.0), and his Fig. 9
has N_CF reaching about 6 near separation at alpha = 10, Re = 6.56e6.
Amplification factors scale roughly as sqrt(Re) at fixed geometry, so dropping
to 1.52e6 (a factor 2.08 in sqrt) puts N_CF near 3 and N_TS near 4 -- both
comfortably under their limits, which is exactly why nothing fires.

What measured low-Re incidences actually exist: **5, 10 and 29.7 only**.  The
DFVLR record's 15-29.5 deg cases are all at 3-8.5e6.  We already hold the
alpha = 29.7 / 1.53e6 hot films digitized (`stock2006_fig16c_digitized.json`,
12 stations), and an alpha = 29.7 solution exists in the older M = 0.1 campaign,
so the low-Re branch could be closed out at 29.7 within the tunnel campaign
without new experimental data.

Consequence for the write-up, and it is worth stating in the papers: the low-Re
rows carry by far the best residuals in `tab:sphtunnel` (rms 0.027-0.082 against
0.18-0.49 at high Re), but that agreement is **partly structural**.  When both
the measurement and the model are pinned by separation rather than by
instability growth, landing on each other is a weaker result than it looks.

## 3. Krimmelbein & Krumbein (DLR), 2009-2011

TAU + automatic two-N-factor e^N along line-in-flight cuts (LILO/COAST3),
31 streamlines, 2.8 M points, 128 wall-normal points (60-100 in the laminar
layer), low-Mach preconditioning, SA turbulence.  Cases: **alpha = 10 and 15
only**, Re_L = 1.5e6 (M = 0.03) and 6.5e6 (M = 0.13).

They **adopt Stock's N_TS-N_CF interaction diagram wholesale** -- their Fig. 4 is
taken from Stock's AIAA J. paper -- so their agreement is not an independent test
of that threshold.  Findings: at Re = 1.5e6, alpha = 10 transition is purely TS
and very well predicted.  At Re = 6.5e6 both alpha = 10 and 15 come out slightly
too far upstream but still in good agreement, with a large pure-crossflow region
appearing at 15.

Directly relevant quotation:

> "Generally for all cases, transition is caused by Tollmien-Schlichting waves
> near the windward and leeward symmetry lines of the prolate spheroid, where the
> flow is more two-dimensional."

So DLR states the mechanism at both planes is TS -- consistent with our reading
of Stock -- but never asks why the two planes agree with each other.

Source: https://elib.dlr.de/67581/1/Krimmelbein-Krumbein_2010.pdf
Also: "Transition Prediction for Three-Dimensional Configurations", Springer
(2009), doi 10.1007/978-3-642-04093-1_7.

## 4. Denison et al., OVERFLOW for TMPW-1, AIAA 2022-0908

SST and Langtry-Menter (LM2009) with Langtry's crossflow extension, rms
roughness 3.3 um.  Notes that the workshop's baseline Tu of 0.15 % is slightly
above the 0.1 % Langtry used, and that both lie within the range reported for the
test.  Their phi convention is ours: phi = 0 at the bottom (windward).  Finding,
paraphrased closely:

  * baseline settings predict transition **downstream** of the measurements on
    the **lower (windward) part**;
  * raising the rms roughness by up to 6x -- an unrealistic 20 um -- pulls the
    front upstream on the **upper (leeward)** side, close to Langtry's published
    alpha = 15 locus, but **still does not capture the front on the lower side**;
  * a cited earlier study got somewhat better results at low phi with a LOWER
    eddy-viscosity ratio nu_t_inf/nu_inf = 1.2 plus sustaining terms; raising the
    ratio to 100 here did not help.

**CORRECTION.**  An earlier revision of this note said their bias is "opposite
in sign to ours at phi = 0".  That is WRONG, and reading their Fig. 13a settles
it.  At alpha = 5 (their x axis is X/a, so x/L = (X/a+1)/2):

| | phi = 0 windward | phi = 180 leeward |
|---|---|---|
| experiment (black) | X/a ~ +0.02, x/L ~ 0.51 | X/a ~ -0.03, x/L ~ 0.49 |
| LM baseline (red) | X/a ~ +0.45, **x/L ~ 0.72** | X/a ~ +0.05, x/L ~ 0.53 |
| LM modified roughness (blue) | X/a ~ +0.45, **x/L ~ 0.72** | X/a ~ -0.02, x/L ~ 0.49 |

Langtry-Menter is windward-LATE by about 0.20 x/L, which is the SAME direction
and a comparable magnitude to our +0.246 at the calibrated seed.  Every model in
this literature fails the same way: too late windward, too early leeward.  The
mistake came from reading our own measured-seed row (windward +0.007, leeward
-0.292), where the windward residual happens to look fine -- but that is the same
shape error expressed at a different seed, not a different sign.

Which is what the physics demands, and it is worth stating plainly: at the
leeward plane the boundary layer is thickened both by the adverse gradient and by
the lateral convergence, so Re_theta reaches any local threshold sooner there.
Any model whose trigger is a local thickness or Reynolds number must fire early
on the leeward side and late on the windward side.  That is a property of the
model class, not of a particular calibration.

Source: https://ntrs.nasa.gov/api/citations/20210025711/downloads/scitech22_MDenison.pdf

## 5. Tarsia Morisco & Alauzet (Inria/ONERA), AIAA SciTech 2025-0146

SA-BCM one-equation transitional model with anisotropic metric-based mesh
adaptation, up to 5.2 M vertices, on the TMPW-1 spheroid at **alpha = 5,
Re = 6.5e6, M = 0.13, Tu = 0.15 %**.  This is the closest published relative of
SA-AI in form (one extra-equation-free SA variant), and it fails outright: the
transition ring sits at **x/L ~ 0.06**, "almost uniformly along the azimuth
direction", against a measured ~0.55, and is **still moving downstream at the
finest mesh** -- not mesh-converged for the transition point even where the drag
is.  They also note plain SA-neg produces a small spurious transition region at
the nose.

Worth citing as the calibration failure mode SA-AI has to beat: a Tu-correlation
one-equation model at Tu = 0.15 % fires immediately and loses all azimuthal
structure.

Source: https://pages.saclay.inria.fr/cosimo.tarsia-morisco/papers/AIAAScitech_2025_0146.pdf

## 6. Other post-Stock work, not yet read in full

  * "Laminar-to-Turbulence Transition Modeling around the 6:1 Prolate Spheroid at
    Different Angles of Attack" (2022), hal-03871832.  gamma and gamma-Re_theta
    on k-omega SST (2003), three AoA at Re = 6.5e6; DLR helicity criterion vs a
    recalibrated T_c1, with a LOCAL sweep-angle approximation added "in order to
    achieve better results on non-wing-like geometries".  Full text behind an
    Anubis challenge; abstract only so far.
  * "Numerical Prediction of Laminar-to-Turbulent Transition Around the Prolate
    Spheroid", J. Marine Sci. Appl. (2020), doi 10.1007/s11804-020-00184-w.  SA,
    SST k-omega, SST-Trans.
  * "Transport Modeling for the Prediction of Crossflow Transition", AIAA J.
    (2018), doi 10.2514/1.J056200.
  * "Extension of a Reynolds-Stress-Based Transition Transport Model for Crossflow
    Transition", J. Aircraft, doi 10.2514/1.C034586.
  * "Comparative Study of First and Second-Order Closure RANS Transition
    Turbulence Models for the 6:1 Prolate Spheroid" (2022).
  * "Estimation of Discretization Uncertainty Using the gamma-Re_theta Transition
    Model for Transitional Flows on 6:1 Spheroid" (2022).
  * "Smooth transitional RANS model and applications with crossflow effects",
    Adv. Aerodyn. (2025), doi 10.1186/s42774-025-00208-5.
  * Plasseraud & Mahesh, "Large-eddy simulation of tripping effects on the flow
    over a 6:1 prolate spheroid at angle of attack", JFM (2023).  Re_L = 4.2e6,
    alpha = 20 -- not our conditions, but it enumerates three natural mechanisms
    on this body: streamwise TS on the windward side, crossflow from the
    secondary-flow inflection, and **centrifugal instability from the
    wall-parallel inflection of the streamlines**.  We have not considered the
    third at all.

## 7. Gaps this search leaves open

  * **Nobody after Stock has published a stability calculation below alpha = 5.**
    TMPW-1 starts at 5; DLR ran 10 and 15.  Our alpha = 0 and 2.5 cases have no
    post-2006 computational company at all, and alpha = 2.5 is where our
    azimuthal over-sweep is worst.
  * **Nobody uses the ONERA/CERT F1 data.**  Every post-Stock computation cites
    Kreplin/Vollmers/Meier at Goettingen (Stock's Ref. 49).  Our
    `a10f1` comparison appears to be the only recent one against Ref. 50
    (IB 222-84 A 34), which is still not in our bibliography.
  * **The windward/leeward agreement is never discussed as a phenomenon.**  DLR
    notes both planes are TS-driven; nobody asks why the two agree when their
    pressure-gradient histories differ.  The mechanism is still ours to derive.
  * TMPW-1 participant submittal files, if they can be obtained, would replace
    the slide-read numbers in section 2 with real curves -- and give an
    independent e^N front at alpha = 5 to plot beside Stock's in
    `f:sphtunnelfronts`.
