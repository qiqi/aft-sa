# Literature sources for the drag-crisis Cd(Re) overlay (fig:dragcrisiscd)

Acquired 2026-07-28 by the digitization agent. Every dataset was
independently digitized/extracted from the PDFs in this directory by
`repro/cfd/digitize_dragcrisis_lit.py` (per-dataset check PNGs in
`checks/`; extracted working rasters in `src/`; JSONs here). Never
trust the JSONs without their check PNG. PDFs/`src/`/`checks/` stay
on disk (not committed); re-acquire with the URLs below.

## Acquired PDFs (provenance)

| file | source | URL | access |
|---|---|---|---|
| delany_sorensen_tn3038.pdf | Delany & Sorensen, NACA TN 3038 (1953) | https://ntrs.nasa.gov/api/citations/19930083675/downloads/19930083675.pdf (436-byte NTRS prefix stripped) | open |
| wieselsberger_tn84.pdf | Wieselsberger 1921, NACA TN-84 translation (1922) | https://ntrs.nasa.gov/api/citations/19930080855/downloads/19930080855.pdf | open |
| roshko1961_jfm.pdf | Roshko, JFM 10:345-356 (1961) | https://authors.library.caltech.edu/records/m8vtc-33e74/files/ROSjfm61.pdf | open (Caltech CODA) |
| henderson1995_pof.pdf | Henderson, Phys. Fluids 7:2102 (1995) | https://authors.library.caltech.edu/records/bqdev-z6q09/files/1_2E868459.pdf | open (Caltech CODA) |
| catalano_wang_ctr2001.pdf | Wang, Catalano & Iaccarino, CTR Annual Research Briefs 2001, 45-50 | https://web.stanford.edu/group/ctr/ResBriefs01/wang.pdf | open |
| rodriguez2015_ijhff.pdf | Rodriguez, Lehmkuhl, Chiva, Borrell & Oliva, IJHFF 55:91-103 (2015), accepted ms. | https://upcommons.upc.edu/bitstream/2117/85914/1/paper_submission.pdf | green OA |
| stringer2014_oceaneng.pdf | Stringer, Zang & Hillis, Ocean Eng. 87:1-9 (2014), author-accepted ms. | https://core.ac.uk/download/161912606.pdf (Bath portal copy is Cloudflare-gated) | green OA |
| iop2020_transition_ddes.pdf | Stabnikov & Garbaruk, J. Phys.: Conf. Ser. 1697:012224 (2020) | https://iopscience.iop.org/article/10.1088/1742-6596/1697/1/012224/pdf | open access |
| wieselsberger_replot_physics0609138.pdf | Veysey & Goldenfeld, Rev. Mod. Phys. 79:883 (2007) | https://arxiv.org/pdf/physics/0609138 | open (arXiv) |
| qu2013_jfs.pdf | Qu, Norberg, Davidson, Peng & Wang, J. Fluids Struct. 39:347-370 (2013), author ms., doi 10.1016/j.jfluidstructs.2013.02.007 | https://publications.lib.chalmers.se/records/fulltext/180053/local_180053.pdf | green OA (Chalmers CPL) |
| dong_karniadakis2005_jfs.pdf | Dong & Karniadakis, J. Fluids Struct. 20(4):519-531 (2005), doi 10.1016/j.jfluidstructs.2005.02.004 | https://www.math.purdue.edu/~sdong/PDF/DNS10k_JFS05.pdf | open (author's page) |

## Datasets (JSON), class and provenance grade

| JSON | contents | class | grade |
|---|---|---|---|
| tn84_wieselsberger.json | faired curve Re 4.2-3e3 + symbol points 3e3-9.5e5 | experiment | primary |
| tn3038_delany_sorensen.json | 219 circle symbols, Re 1.1e4-2.3e6 | experiment | primary |
| roshko1961.json | 12 plain points 1.8e6-8.7e6 (+6 splitter-plate, do not overlay) | experiment | primary |
| rodriguez2015_fig4_exp.json | Schewe 1983 (51 pts) + Achenbach&Heinecke 1981 (31) [+D&S, Wieselsberger cross-checks] | experiment | secondary (via Rodriguez fig. 4, vector-exact) |
| catalano2001_wmles.json | 3 WMLES points 5e5/1e6/2e6 + Achenbach-1968 curve 6e4-5e6 | WMLES / experiment | primary / secondary |
| rodriguez2015_les.json | WRLES Table 2, 6 pts 2.5e5-8.5e5 | LES | primary (tabulated) |
| stringer2014_urans.json | CFX + OpenFOAM SST URANS, Re 40-1e6 | RANS fully turbulent | primary |
| iop2020_models.json | SST (fully turb.), SST gamma-Re_theta, SST KD sweeps 5e4-1.3e6 + Schewe replot | RANS FT + transition | primary |
| henderson1995.json | 2-D spectral totals, steady + shedding branches Re 25-1000 | 2-D numerical | primary |
| qu2013_dns.json | 2-D unsteady DNS-class sweep Re 50-200 (Table 3, transcribed + text-asserted; Re=150 domain pair kept) | 2-D numerical | primary (tabulated) |
| dong_karniadakis2005_dns3d.json | 3-D spectral DNS at Re=1e4, Table 2 resolution study (transcribed + text-asserted; colon-decimal PDF font) | 3-D DNS | primary (tabulated) |
| veysey_fig7_lowre.json | Tritton/Finn/Jayaweera drag, Re 0.05-6 | experiment | secondary (vector-exact replot) |

Unreachable primaries, noted: Achenbach 1968 (JFM 34, paywalled;
represented by the Catalano-reproduced curve [secondary] and by
Achenbach & Heinecke 1981 via Rodriguez fig. 4 [secondary]);
Schewe 1983 (JFM 133, paywalled; two independent secondary replots
agree - see the record); Cheng/Pullin/Samtaney 2017 WRLES (JFM 820:
Caltech record is metadata-only, no arXiv/KAUST copy found - WRLES
class represented by Rodriguez et al. 2015 instead); Zheng & Lei 2016
FTC gamma-Re_theta (paywalled; class covered by Stabnikov-Garbaruk's
own gamma-Re_theta sweep).
