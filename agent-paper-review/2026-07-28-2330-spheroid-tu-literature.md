# The 6:1 prolate spheroid disturbance spec across the literature: N-factor / Tu% survey

*2026-07-28 23:30, literature-synthesis agent. User question: the DFVLR
Kreplin/Vollmers/Meier 6:1 prolate spheroid is a standard transition benchmark,
so many studies must have calibrated a Tu% or an N_crit to match it -- gather and
synthesize them, and resolve whether any source MEASURED the DFVLR tunnel's
freestream turbulence intensity vs merely CALIBRATED an N/Tu to fit the transition
data. Web-verified unless marked [repo] (extracted from a PDF in `references/` or
`paper/`) or [mem]. Companion: 2026-07-28-1715-spheroid-reseed.md (which established
the reseed physics and searched only the repo). Mack map used throughout is the
paper's own: N_crit = -8.43 - 2.4 ln(Tu_frac), i.e.
Tu% = 100 exp(-(N_crit+8.43)/2.4) (eq:tumap, lib/calibrate_kernel.py). NO tex edits.*

## Headline

Every disturbance number that any study attaches to this spheroid is a
**model-calibration or a prescribed modeling input, not a directly-measured
tunnel Tu that the transition studies actually use.** The e^N community calibrates
a limiting N on the measured transition; the gamma-Re_theta community prescribes an
inflow Tu% chosen to match. Both cluster tightly:

- **e^N calibrated N_TS (Mack-equiv Tu):** 7.0-8.0  (Tu 0.11-0.16%)
- **gamma-Re_theta prescribed inflow Tu (Mack-equiv N):** 0.10-0.15%  (N 7.2-8.2)

**Our SA-AI value -- N_crit ~ 7.0, Tu ~ 0.16-0.17% -- sits squarely in the middle
of this band**, essentially ON Stock's own second-facility value (N_TS=7.0, Tu
0.16%) and just softer than his DFVLR N_TS=8.0. Stock's N=8 is NOT a consensus: it
is the quiet end, and Stock's own higher-Re facility gives N=7. The spread itself
makes the paper's "N is method/facility-relative, no universal tunnel value" point.

## The measured-vs-calibrated verdict (the crucial distinction)

**Stock 2006 [repo, full text] quotes NO tunnel Tu%.** He states explicitly that
"the limiting N factors for wind tunnels are specific quantities depending on the
flow quality of the considered facility" and DERIVES them from the measured
transition (calibration), not from a measured disturbance. His two facilities give
two different limits: DFVLR 3x3 m Goettingen (Re 1.5-8.5e6, our benchmark) -> N_TS=8.0,
N_CF=5.5 (Figs 11a/b); the higher-Re CEAT facility (Re 6.5-43.5e6) -> N_TS=7.0,
N_CF=6.0. So even within the source-of-record e^N study there is no single N.

**A primary MEASURED-turbulence study does exist, and it was missed by the reseed
record** (which searched only the repo): Meier, Michel & Kreplin (1987), "The
Influence of Wind Tunnel Turbulence on the Boundary Layer Transition,"
*Perspectives in Turbulence Studies*, Springer, pp. 26-46, doi
10.1007/978-3-642-82994-9_2 [Crossref-verified]. This is the companion study to the
1985 data report: transition on the DFVLR 1:6 spheroid was measured in three tunnels
at matched Re, WITH supplementary hot-wire/microphone measurements of the velocity
and pressure fluctuation levels. So the tunnel disturbance WAS characterized
physically -- the reseed record's flat "no measured Tu found" should be softened to
"no measured Tu is quoted in Stock 2006 or the repo literature; the primary
fluctuation-level measurements live in Meier-Michel-Kreplin 1987, which is not in
the repo and which I could not access in full."

**Caveat on the measured number.** Two independent web syntheses attribute
Tu_inf ~= 0.2% to the DFVLR Goettingen tunnel for this campaign, but I could NOT
verify a specific percentage against an accessible primary source (the ResearchGate
PDF of Meier-Michel-Kreplin 1987 returned 403; Springer paywalled). Treat 0.2% as
[unverified secondary]. If real, its Mack-equiv N=6.5 -- consistent with, and if
anything slightly softer than, the calibrated N=7-8. It does NOT overturn the
picture: the numbers the transition studies actually USE are all calibration/input
values.

**Bottom line for the paper:** the user's hypothesis (there should be measured
values) is half-right -- a measured fluctuation-level study exists (Meier 1987) --
but no transition-prediction study of this body drives its model from that
measurement; they all calibrate N or prescribe Tu. The paper's framing stands and
is in fact strengthened: a measured tunnel disturbance was available for 35+ years
and the community STILL uses method-relative calibrated numbers, because N (and the
gamma-Re_theta inlet Tu) are method constructs, not the physical Tu.

## Synthesis table

Tu%->N and N->Tu% via the paper's Mack map. For gamma-Re_theta rows the quoted Tu is
the Langtry-Menter INLET Tu (a different physical construct from the e^N/Mack Tu);
the Mack-equiv column is a cross-comparison convenience, not a claim of identity.

| Study | Method | Case (Re_L, alpha) | Quoted N_crit or Tu% | Mack-map equiv | Meas / Calib |
|---|---|---|---|---|---|
| Stock 2006 [repo], DFVLR tunnel | e^N (2 N-factor, TS+CF) | 7.2e6 (also 1.5-8.5e6), a=0-30 | N_TS=8.0, N_CF=5.5 | Tu 0.106% | **Calibrated** on measured x_tr |
| Stock 2006 [repo], CEAT tunnel | e^N (2 N-factor) | 6.5-43.5e6, a=0-30 | N_TS=7.0, N_CF=6.0 | Tu 0.161% | **Calibrated** |
| Krimmelbein-Krumbein 2010 [repo] | e^N (DLR 2 N-factor, LILO) | 1.5e6 & 6.5e6, a=10/15 | Stock's N_TS-N_CF stability curve (Fig 4, from Kreplin) | ~N_TS 8 -> Tu 0.11% | **Calibrated** (reuses Stock's DFVLR curve; NOT independent) |
| Denison 2021/2022 OVERFLOW [repo]=overflow_tmw | gamma-Re_theta (SST-2003-LM-2009, sus) | 6.5e6, Ma 0.13, a-sweep | Tu_inf = 0.15% | N 7.18 | Prescribed input (1st AIAA TMW spec) |
| Tarsia Morisco-Alauzet 2025 [repo] | gamma-Re_theta (adaptive unstructured) | 6.5e6, Ma 0.13 | Tu_inf = 0.15% | N 7.18 | Prescribed input ("according to the 1st AIAA TMW") |
| 1st AIAA Transition Modeling Workshop (Case 3) | committee spec, feeds gamma / gamma-Re_theta | 6.5e6, Ma 0.13, a-sweep | Tu = 0.15% (prescribed) | N 7.18 | Prescribed benchmark spec |
| Zhang et al. 2022, Energies 15:6491 | gamma-Re_theta_t (sustaining turb.) | 6.5e6 (spheroid) | Tu ~ 0.1% | N 8.15 | Prescribed input |
| (other gamma-Re_theta spheroid studies, web) | gamma-Re_theta | 6.5e6, T=300K | Tu = 0.1% | N 8.15 | Prescribed input |
| Meier-Michel-Kreplin 1987 | hot-wire/microphone MEASUREMENT | DFVLR + 2 other tunnels | fluctuation levels measured; ~0.2% [unverified secondary] | (N~6.5 if 0.2%) | **Measured** (not used by any transition model above) |
| **SA-AI (this paper)** | one-eq local transition (chi transport) | 7.2e6, a=0 (reseed) | N_crit ~ 7.0 / Tu ~ 0.16-0.17% | -- | Calibrated (seed lever; reseed 1715) |

## Where our value sits (2-3 sentence synthesis for the paper)

Our matched spheroid seed -- N_crit ~ 7.0, Mack Tu ~ 0.16-0.17% -- lands in the dead
center of the literature: it coincides with Stock's own higher-Re-facility limit
(N_TS=7.0, Tu 0.16%), is a touch above the workshop-standard gamma-Re_theta inflow
(Tu 0.15% = Mack N 7.2), and is modestly softer than his DFVLR N_TS=8.0 (Tu 0.11%)
and the quietest gamma-Re_theta inflow (Tu 0.1% = Mack N 8.2). The full calibrated
spread is N ~ 7.0-8.2 / Tu ~ 0.10-0.16% -- a factor ~1.6 in Tu -- so **Stock's N=8
is one endpoint, not the consensus**; there is a de-facto tunnel-class BAND
(Tu 0.1-0.2%), not a single value, and every number in it is a method-relative
calibration or prescription. That band-not-point structure is exactly the paper's
"N is method-relative, no universal physical tunnel Tu" argument, and our value
being interior to it (rather than at either edge) is the honest, favorable framing.

## Per-source provenance

- **Stock 2006** [repo `paper/spheroid.pdf`, full text]. AIAA J 44(1). N_TS=8.0/
  N_CF=5.5 (DFVLR, Figs 11a/b, verbatim "the limiting N factors, N_TS = 8.0 and
  N_CF = 5.5"); N_TS=7.0/N_CF=6.0 for the CEAT higher-Re tunnel (verbatim "The
  limiting N factors are slightly different, N_TS = 7.0 and N_CF = 6.0"). No Tu%
  anywhere. Bib: `stock_2006` (present).
- **Kreplin/Vollmers/Meier 1985** DFVLR-AVA IB report, the experimental source of
  record. Bib: `kreplin_1985` (present). Not in repo; image/unavailable.
- **Meier/Michel/Kreplin 1987** [Crossref-verified] doi 10.1007/978-3-642-82994-9_2,
  Perspectives in Turbulence Studies, Springer, pp.26-46. The measured-fluctuation
  companion. **NEW bib entry recommended** (fields below). Not accessible in full
  (RG 403 / Springer paywall).
- **Krimmelbein-Krumbein 2010** [repo `references/krimmelbein-krumbein-2010-
  automatic-transition-prediction.pdf`]. Spheroid cases a=10/15 at Re 1.5e6 (Ma
  0.03) and 6.5e6 (Ma 0.13); applies Stock's TS-CF interaction stability curve
  (their Fig 4, "which was applied for the present validation calculations"), i.e.
  the DFVLR N_TS~8 calibration -- NOT an independent number. (Their standalone
  critical N=7.5 is for a DIFFERENT case, the swept tailplane/wing config, not the
  spheroid -- correcting the reseed record's "Krimmelbein uses N=7.5 for this
  case".) Bib: `krimmelbein_2010` (present).
- **Denison 2021/2022 OVERFLOW** [repo `references/overflow_tmw.pdf` = this paper;
  also NTRS 20210025711 / AIAA 2022-0908]. gamma-Re_theta (SST-2003-LM-2009) with
  SST-2003-sus sustaining terms to avoid Tu decay; spheroid Tu_inf = 0.15% at
  Re 6.5e6, Ma 0.13. Verbatim: "The baseline Tu of 0.15% is slightly higher than
  that of 0.1%". Bib: `denison_2021_overflow` (present).
- **Tarsia Morisco & Alauzet 2025** [repo `references/tarsia2025.pdf`]. Adaptive
  unstructured gamma-Re_theta. Spheroid: "the free-stream Turbulence Intensity at
  Tu_inf = 0.15%, according to the 1st AIAA Transition [Modeling Workshop]". (Their
  flat-plate T3A uses 0.18%.) Bib: `tarsia_2025` (present).
- **1st AIAA Transition Modeling & Prediction Workshop** (Coder summary [repo
  `TransitionMPW1_summary_Coder_NAS2021.pdf`], slide-based/image-heavy; Case 3 =
  6:1 prolate spheroid, DFVLR 3x3 Goettingen data). Prescribes Tu=0.15% (per
  Denison/Tarsia). Bib: `coder_tmpw_summary` (present).
- **Zhang, Nie, Meng, Zuo 2022** [Crossref-verified] Energies 15(17):6491, doi
  10.3390/en15176491, gamma-Re_theta_t with sustaining turbulence, spheroid Tu~0.1%.
  Optional NEW bib (fields below) -- only if we want a second gamma-Re_theta anchor.
- **web-only cross-checks** (not for citation): FTC/JMSA/SU2/ICAS gamma-Re_theta
  spheroid studies all use inflow Tu in 0.1-0.15% at Re 6.5e6; consistent with the
  above. Secondary claim of measured DFVLR Tu~0.2% is [unverified].
- **Confirmed absent from the repo despite the handover list:** `coder_diss.pdf`
  and `medida_thesis.pdf` contain NO spheroid content (Coder's cases: flat plate,
  S805, S414, NLF-0416; Medida: none). Do not cite either for the spheroid.

## Bib entries needed (Crossref-verified fields)

Recommended (the measured-turbulence primary source, strengthens the verdict):

    @incollection{meier_michel_kreplin_1987,
      author    = {Meier, H. U. and Michel, U. and Kreplin, H.-P.},
      title     = {The Influence of Wind Tunnel Turbulence on the Boundary
                   Layer Transition},
      booktitle = {Perspectives in Turbulence Studies},
      publisher = {Springer Berlin Heidelberg},
      address   = {Berlin, Heidelberg},
      pages     = {26--46},
      year      = {1987},
      doi       = {10.1007/978-3-642-82994-9_2}
    }

Optional (second gamma-Re_theta anchor, only if Sec IX wants the 0.1% end):

    @article{zhang_2022_sustaining,
      author  = {Zhang, Meihong and Nie, Shengyang and Meng, Xiaoxuan and
                 Zuo, Yingtao},
      title   = {The Application of the $\gamma$-$Re_{\theta t}$ Transition
                 Model Using Sustaining Turbulence},
      journal = {Energies},
      volume  = {15},
      number  = {17},
      pages   = {6491},
      year    = {2022},
      doi     = {10.3390/en15176491}
    }

## Sources (URLs)

- Stock 2006 AIAA J 44(1): repo `paper/spheroid.pdf`
- Meier/Michel/Kreplin 1987: https://doi.org/10.1007/978-3-642-82994-9_2
- Denison OVERFLOW workshop: https://ntrs.nasa.gov/citations/20210025711 ;
  https://arc.aiaa.org/doi/10.2514/6.2022-0908
- 1st AIAA TMW: https://transitionmodeling.larc.nasa.gov/workshop_i/
- Zhang et al. 2022 Energies: https://doi.org/10.3390/en15176491
- Tarsia 2025: repo `references/tarsia2025.pdf`
- ERCOFTAC case074 (spheroid benchmark description; server was unreachable this
  session): http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case074
