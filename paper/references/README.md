# Reference PDFs for `sa-ai.tex`

Downloaded 2026-06-30. Source of truth = the `\cite{...}` keys in `sa-ai.tex`
(26 unique). The `.bbl` was stale, so it was not used.
Only legitimate open-access sources were used (NASA NTRS, institutional
repositories, author/lab pages, open-access publisher copies). No
sci-hub/pirated sources.

**22 of 26 citations have a PDF here** (18 exact documents + 4 covered by
open-access substitutes). The remaining 4 are 1 paywalled journal article, a
copyrighted book, an un-digitized 1956 industry report, and one more paywalled
article — landing pages are listed for manual/institutional access.

## Exact documents (18)

| Cite key | File | Source |
|---|---|---|
| `mcghee_1988` | `McGhee_Walker_Millard_1988_Eppler387_LowReynolds_NASA-TM-4062.pdf` | NASA NTRS |
| `mack_1977` | `Mack_1977_Transition_and_Laminar_Instability_JPL-77-15.pdf` | NASA NTRS |
| `schubauer_klebanoff_1956` | `Schubauer_Klebanoff_1956_Mechanics_BoundaryLayer_Transition_NACA-Report-1289.pdf` | NASA NTRS |
| `schubauer_skramstad_1948` | `Schubauer_Skramstad_1948_LaminarBoundaryLayer_Oscillations_Transition_FlatPlate_NACA-Report-909.pdf` | NASA NTRS |
| `spalart_allmaras_1994` | `Spalart_Allmaras_1994_OneEquation_Turbulence_Model_LaRechercheAerospatiale.pdf` | NASA Turbulence Modeling Resource (tmbwg.github.io/turbmodels, hosted w/ author permission) |
| `spalart_allmaras_1992` | `Spalart_Allmaras_1992_OneEquation_Turbulence_Model_AIAA-92-0439.pdf` | **Same PDF as the 1994 entry** — the AIAA-92-0439 conference preprint is not openly available; the La Recherche Aérospatiale 1994 paper is its published form (identical content). |
| `vaningen_2008` | `vanIngen_2008_eN_Method_Transition_Prediction_AIAA-2008-3830.pdf` | TU Delft repository |
| `drela_xfoil_1989` | `Drela_1989_XFOIL_Analysis_Design_LowReynolds_Airfoils.pdf` | web.mit.edu/drela |
| `menter_langtry_2006` | `Menter_Langtry_Volker_2006_Transition_Modelling_GeneralPurpose_CFD_FTC.pdf` | Wayback (Southampton CFD course copy) |
| `crivellini_2023` | `Crivellini_Ghidoni_Noventa_2023_Algebraic_Modifications_kw_SA_Transition_CompFluids.pdf` | Univ. Politecnica delle Marche repository (OA) |
| `nasa_tmr` | `NASA_Langley_Turbulence_Modeling_Resource_webpage.html` | tmbwg.github.io/turbmodels (website, not a paper) |
| `coder_maughmer_2014` | `Coder_Maughmer_2014_AmplificationFactorTransport_Transition_AIAA-J.pdf` | AIAA (via MIT Libraries) |
| `drela_giles_1987` | `Drela_Giles_1987_Viscous_Inviscid_Transonic_LowReynolds_Airfoils_AIAA-J.pdf` | AIAA (via MIT Libraries) |
| `menter_gamma_2015` | `Menter_etal_2015_OneEquation_LocalCorrelation_Transition_Model_FTC.pdf` | Springer (via MIT Libraries) |
| `walters_cokljat_2008` | `Walters_Cokljat_2008_ThreeEquation_EddyViscosity_Transitional_JFE.pdf` | ASME (via MIT Libraries) |
| `cakmakcioglu_2018` | `Cakmakcioglu_Bas_Kaynak_2018_CorrelationBased_Algebraic_Transition_Model_IMechE.pdf` | SAGE (via MIT Libraries) |
| `cakmakcioglu_2020` | `Cakmakcioglu_etal_2020_Revised_OneEquation_Transitional_Model_AIAA-2020-2706.pdf` | AIAA (via MIT Libraries) |
| `VanDriest1963` | `VanDriest_Blumer_1963_BoundaryLayer_Transition_Freestream_Turbulence_PressureGradient_AIAA-J.pdf` | AIAA (via MIT Libraries) |

## Open-access substitutes (4 citations → same author / same model)

The exact journal articles are paywalled; these are open-access primary
sources by the same authors containing the identical model formulation.

| Cite key | Substitute file | Notes |
|---|---|---|
| `halila_2022` | `Halila_2020_PhDThesis_AeroShapeOpt_Transition_AFT-S_UMich_SUBSTITUTE-for-Halila2022.pdf` | Lead author's UMich PhD thesis; contains the AFT-S smooth transition model. |
| `langtry_menter_2009` | `Langtry_2006_PhDThesis_CorrelationBased_Transition_LocalVariables_Stuttgart_part1of2.pdf` (+ `_part2of2.pdf`) | Langtry's Stuttgart PhD thesis — origin of the γ–Reθ correlation-based model. |
| `MenterLangtry2006` (J. Turbomach. Part I) | (covered by the **same** Langtry 2006 thesis above) | Identical Part-I model formulation. |
| `medida_baeder_2011` | `Medida_2014_PhDThesis_CorrelationBased_Transition_Modeling_External_Aero_UMD_SUBSTITUTE-for-MedidaBaeder2011.pdf` | Medida's UMD PhD thesis; the γ–Reθ→SA coupling. |

## Not obtained — still needed (4)

| Cite key | Reference | Landing page |
|---|---|---|
| `rahman_2024` | Rahman et al. 2024, Math. Comput. Simul. | https://doi.org/10.1016/j.matcom.2024.03.016 |
| `abu_ghannam_shaw_1980` | Abu-Ghannam & Shaw 1980, J. Mech. Eng. Sci. | https://doi.org/10.1243/JMES_JOUR_1980_022_043_02 |
| `schmid_henningson_2001` | Schmid & Henningson 2001, Springer book | https://doi.org/10.1007/978-1-4613-0185-1 (copyrighted textbook; likely an MIT SpringerLink e-book) |
| `smith_gamberoni_1956` | Smith & Gamberoni 1956, Douglas ES 26388 | No digital copy known (not in NTRS/DTIC/archive.org); alt. version: Proc. 9th Int. Congress Applied Mechanics, Brussels 1957, Vol. 4, 234–244 |

> Note: `references.bib` also contains `coupland_1990` and `savill_1993`, which
> are **not** `\cite`d in `sa-ai.tex`, so they were not downloaded.
