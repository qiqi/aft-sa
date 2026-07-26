# SA-AI in OpenFOAM (shallow port)

Replicates SA-AI paper results with a **shallow** OpenFOAM implementation:
the model is assembled entirely from existing OpenFOAM operators
(`fvm::ddt/div/laplacian`, `fvc::grad/curl/laplacian`, `fvm::Sp`) inside a
standard runtime-loadable RAS model — **no custom numerics**.

## Layout

```
openfoam/
├── src/                      SpalartAllmarasAI RAS model (templated, GPL
│                             boilerplate from stock v2412 SpalartAllmaras)
├── scripts/
│   ├── build_flatplate_cases.py   paper Sec. IV flat-plate grid + 5 Tu cases
│   └── extract_transition.py      chi/Cf vs Re_theta + AGS / S-S comparison
├── cases -> data/cases       (via data symlink)
└── data -> /local_data/qiqi/openfoam-sa-ai   OpenFOAM source+build, case data
```

Large data lives in `/local_data/qiqi/openfoam-sa-ai` (OpenFOAM-v2412 tree,
build log, run directories); this directory holds only code.

## Model: SpalartAllmarasAI

Ported from the Flow360 reference (compute monorepo, `SAAiTransition.h` /
`ModelConstants.h` / `SpalartAllmaras.h`), sphere kernel (model v3),
magnitude-triple form. Three modifications to stock SA (ft2 off):

1. **Production** `P = max[(1-σ_t)·a·|ω|·ν̃, σ_t·Cb1·S̃·ν̃]` with the
   sphere-kernel rate `a = a_max·clip⟨Ω̂Î⟩₀¹·onset(Re_Ω/Re_Ω^c)`,
   `Re_Ω^c = softmin₂(1851.2, 124.6 + 1.424/P²)`, indicators
   `X=|u|, Y=|ω|d, Z=½d²(∇²U)·û` (u″ via `fvc::laplacian(U)` — the shallow
   analog of Flow360's ring-averaged compact Laplacian pre-pass).
2. **Destruction** scaled by the sigma-d tie `σ_D = 1 − R(1−σ_t)`,
   `R = Cb1/(κ²Cw1) ≈ 0.2489`.
3. **ν̃-diffusion** sees `ν/6` (`nuLamScale`); momentum untouched.

`σ_t = 1 − exp(−max(χ−1,0)/4)`. Freestream seed `χ_∞ = Cv1·e^(−N_crit)`,
`N_crit = −8.43 − 2.4·ln(Tu)` (Mack) — set `nuTilda_∞ = χ_∞·ν` on the inlet.

Deliberately not ported (convergence aids / experimental, not model content):
`AI_LAMINAR_SLOWDOWN` (pseudo-time trick; SIMPLE under-relaxation is the
analog here — same fixed point), `AI_FV1BYPASS`, `AI_INVARIANT_KERNEL`,
DES/RC/low-Re-Cw2 options.

Differences vs Flow360 to keep in mind when comparing:
- S̃ uses OpenFOAM's `max(Ω + fv2·ν̃/(κd)², 0.3Ω)` floor instead of Spalart's
  c2/c3 continuation (only differs where S̃ would go negative — deep
  recirculation; irrelevant on flat plate, small on LSB cases).
- u″ is a plain compact FV Laplacian, not the dual-volume ring average;
  on smooth structured grids these agree.
- Incompressible (simpleFoam) vs Flow360's compressible M=0.1.

## Build

```bash
cd /local_data/qiqi/openfoam-sa-ai/OpenFOAM-v2412 && source etc/bashrc
cd ~/flexcompute/sa-ai/openfoam/src && wmake libso
```

## Flat-plate replication (paper Sec. IV)

```bash
python3 scripts/build_flatplate_cases.py       # writes data/cases/flatplate_Tu*
cd data/cases/flatplate_TuXXXX && blockMesh && simpleFoam
python3 scripts/extract_transition.py          # summary + figure
```

Grid mirrors `sa-ai/flow360/build_flatplate_cases.py`: x∈[0,6] 320 cells
(dx₀=8e-4, geometric), z 80 cells (dz₀=7e-6, r=1.12, H≈0.505), quasi-2D,
unit Re = 1e6 (U=1, ν=1e-6). Targets: χ-crossing Re_θ vs the AGS correlation
and the Schubauer–Skramstad band, as in `flat_plate_batch_flow360`.
