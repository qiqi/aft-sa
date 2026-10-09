# SA-AI: Spalart–Allmaras with Autogenous Inception

SA-AI is a single-equation laminar–turbulent transition model built on
Spalart–Allmaras. It adds no new transport variable: in the laminar region the
SA working variable $\tilde\nu$ grows by a local amplification source, and once
it reaches $\chi = \tilde\nu/\nu \approx 1$ it hands over to standard SA.
Setting $\sigma_P \equiv 1$, $c_{\nu,\mathrm{ai}} = 1$ and
$\sigma_{v1} \equiv 0$ recovers baseline SA exactly.

The model is defined in the technical summary,
[`paper/whitepaper.tex`](paper/whitepaper.tex), and the full paper,
[`paper/sa-ai.tex`](paper/sa-ai.tex).

## Model in brief

- **Amplification source.**
  $P_\mathrm{AI} = a\,\sigma_{Re}\,\omega\,\tilde\nu$, with rate
  $a = a_{\max}\,\mathrm{clip}\langle\hat\Omega\hat I\rangle_0^1$.
  The two indicators are read off the local velocity, shear and curvature
  scales $(u,\ d\,u',\ \tfrac12 d^2 u'')$: $\hat\Omega$ is the vorticity
  fraction and $\hat I$ the accumulated inflection.
- **Onset gate.** $\sigma_{Re}$ is a tanh ramp of width $w$ on
  $Re_\Omega / Re_\Omega^c$, where $Re_\Omega = d^2\omega/\nu$ and
  $Re_\Omega^c = k\,\mathrm{softmin}_2\big(C,\ A + B/(\hat\Omega\hat I)^2\big)$.
- **Handover.** The production is
  $\max[(1-\sigma_P)P_\mathrm{AI},\ \sigma_P P_\mathrm{SA}]$, with
  $\sigma_P = \max[1-e^{-(\chi-1)/\tau}, 0]$. The destruction is tied to it:
  $\sigma_D = 1 - \frac{c_{b1}}{\kappa^2 c_{w1}}(1-\sigma_P)$.
- **Reduced laminar diffusion.** The molecular coefficient in the
  $\tilde\nu$ diffusion term is scaled by $c_{\nu,\mathrm{ai}}$.
- **$f_{v1}$ bypass.** In lifted transitional shear layers, $\mu_t$ blends
  toward $\rho\tilde\nu$. This is detected by a parameter-free
  law-of-the-wall check.
- **Freestream seed.** Mack's $e^N$ map:
  $N_\mathrm{crit} = -8.43 - 2.4\ln Tu$ and
  $\chi_\infty = c_{v1}e^{-N_\mathrm{crit}}$.

| constant | value |
|---|---|
| $a_{\max}$ | 0.19 |
| $(C, A, B)$ | (2600, 175, 2) |
| $k$ | 0.712 |
| $w$ | 0.35 |
| $c_{\nu,\mathrm{ai}}$ | 1/6 |
| $\tau$ | 4 |

The baseline SA constants are unchanged.

## Running it

The production implementation is in Flow360, on the
[`sa-ai` branch of flexcompute/compute](https://github.com/flexcompute/compute/tree/sa-ai)
(`src/Flow360Core/Applications/Solver/SpalartAllmaras/`).

- SA-AI is **on by default** in that build. Set `AI_SA=0` for classical SA.
- The compiled defaults are the constants above, and the $f_{v1}$ bypass is on.
- Set the freestream seed through the case JSON:
  `freestream.turbulenceQuantities.modifiedTurbulentViscosityRatio`
  $=\chi_\infty$.
- Transition cases converge slowly, because the front migrates long after the
  residual has flattened. The paper's runs use
  `AI_LAMINAR_SLOWDOWN=0.01` with a staged convergence protocol; see
  [`paper/repro/`](paper/repro/).

## Repository layout

| path | contents |
|---|---|
| `paper/` | paper and whitepaper sources and figures |
| `paper/repro/` | reproduction package: analytic scripts, case driver, the canonical env (`driver/saai_env.py`) |
| `tests/` | includes `test_constants_consistency.py`, which pins paper, scripts and solver to one constant set |
| `src/` | legacy JAX 2D RANS solver used in early development; not the reference implementation |

## License

GPL-3.0; see [LICENSE.md](LICENSE.md).
