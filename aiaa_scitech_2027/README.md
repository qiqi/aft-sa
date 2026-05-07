# AIAA SciTech 2027 Abstract — AFT–SA Model

This directory contains the extended abstract submission for AIAA SciTech 2027
(Orlando, FL, January 11–15, 2027) on the AFT–SA closure: a single transport
equation that carries *both* boundary-layer transition (an
Amplification-Factor-Transport envelope) and fully turbulent mixing (the
Spalart–Allmaras model), without adding any second transported variable.

## Target session

**Technical Discipline:** Fluid Dynamics Technical Committee (FDTC)

**Topic area:** Turbulence Modeling — *Boundary-Layer Transition Modeling and
Applications*

Rationale: the FDTC explicitly solicits work on RANS turbulence closures and
boundary-layer transition modeling for aerospace applications. The AFT–SA model
is a transition-aware closure designed to be a drop-in replacement for SA in
production aerospace RANS solvers, which is squarely within the FDTC's
"turbulence modeling" and "boundary-layer transition modeling and applications"
solicitations. The Fluid Instability and Transition (FIT) TC was considered but
is more strongly oriented toward instability physics, hypersonic transition,
and shock/boundary-layer interaction — a less natural fit for a one-equation
engineering closure.

## Submission timeline (AIAA SciTech 2027)

| Milestone | Date |
|-----------|------|
| Call for content opens | March 24, 2026 |
| **Abstract deadline** | **May 21, 2026, 20:00 ET** |
| Author notifications | August 24, 2026 |
| Manuscript deadline | December 1, 2026, 20:00 ET |
| Forum dates | January 11–15, 2027 (Orlando, FL) |

## Format compliance

- Extended abstract / draft manuscript, ≥1,000 words.
- Includes purpose, technical foundation, preliminary results, expected results,
  with key figures, equations, and references.
- Uses the AIAA-style two-column template (`abstract.tex`).

## Files

- `abstract.tex` — the extended abstract source.
- `abstract.pdf` — compiled PDF (build with `pdflatex abstract && pdflatex abstract`).
- `fig/` — figures imported from `paper/fig/`.

## Building

```bash
cd aiaa_scitech_2027
pdflatex abstract.tex
pdflatex abstract.tex
```
