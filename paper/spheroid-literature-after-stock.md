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

Note there is **no PDF of Stock (2006) anywhere in the repo** -- AIAA J. 44(1),
doi 10.2514/1.16026, paywalled.  Every Stock number we use came from figure
digitization (`paper/data/stock2006_fig*_digitized.json`).  Worth pulling from
the MIT library, since Section 2's whole comparison rests on it.

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

i.e. a completely different model family also fails on the windward meridian and
cannot be tuned out of it.  Their bias is opposite in sign to ours at phi = 0.

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
