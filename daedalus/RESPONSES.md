
## 2026-07-25 ~04:40 — CFD status + LM/BCM source matrix (agent verified)
CFD: Daedalus cavity-L2 a4 DONE 03:31 (CL 1.0208 / CD 0.02004 — L2
family agreement: dCL 0.7%, dCD 0.7 counts/0.35%, tighter than L1!);
cavity a5 running (ETA ~09:00), a6 queued (~14:30) = the only
unfinished Daedalus set. GPU queue behind: invariant-kernel 2D verify,
negative-AoA NLF pair, spheroid runs.

LM/BCM digitizable sources (PDFs in references/):
EPPLER Re=2e5: Shahjahan ICAS 2024 (icas2024_0327.pdf) = VECTOR
figures, gamma-Retheta AND SA-BC, polar (Fig 6) + bubble stations
(Fig 9), Tu=0.1% — the single best source, covers both models.
Backups: aiac2017_205.pdf (original BC, raster), dalessandro_2025.pdf
(gamma-SA polar, arXiv).
NLF Re=4e6: Langtry-Menter YES — Denison OVERFLOW workshop paper
(overflow_tmw.pdf, NTRS): Fig 11 = transition x/c vs cl for BOTH
surfaces (drops into fig:nlfaft beside AFT) + lift curve; no cd-cl
polar. BCM: NO usable open data (Tarsia Morisco 2025 is mesh-adaptation
only, single condition; Cakmakcioglu 2020 SA-BCM papers paywalled) —
same honest verdict as AFT-on-Eppler.
Corrections to working assumptions from the agent: D'Alessandro 2025
has NO NLF(1)-0416 (its high-Re airfoil is DU00-W-212); Medida's thesis
"NLF" is NLF(2)-0415 (swept crossflow) — do not use.
PROPOSED: digitize Shahjahan Fig 6 -> LM+BCM polars on Eppler Fig 10;
Denison Fig 11 -> LM transition curves on NLF Fig 6; state BCM-on-NLF
absence in text. Awaiting go-ahead (few hours of figure-reading).
