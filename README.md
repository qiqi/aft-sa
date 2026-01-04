# AFT-SA: GPU-Accelerated RANS Solver

A 2D structured-grid RANS solver for incompressible airfoil flow using the Artificial Compressibility Method. Fully GPU-accelerated with JAX.

## Features

- **Artificial Compressibility Method** for incompressible flow
- **JST Scheme** (Jameson-Schmidt-Turkel) with 4th-order artificial dissipation
- **Spalart-Allmaras** one-equation turbulence model
- **5-stage Runge-Kutta** time integration with explicit smoothing
- **GPU acceleration** via JAX (runs on CPU if no GPU available)
- **C-grid topology** with proper wake cut handling
- **Batch processing** for parametric sweeps (angle of attack, Reynolds number)

## Installation

```bash
# Clone the repository
git clone https://github.com/qiqi/aft-sa.git
cd aft-sa

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional, but recommended)
pip install jax[cuda12]
```

## Quick Start

Run a NACA 0012 simulation:

```bash
python scripts/solver/run_airfoil.py data/naca0012.dat \
    --alpha 5.0 --reynolds 6e6 --max-iter 500
```

---

## Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Interface                                  │
│   scripts/solver/run_airfoil.py  │  scripts/solver/run_batch.py             │
│         (single case)            │        (parametric sweep)                 │
└─────────────────────────────────┬┴──────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼──────────────────────────────────────────┐
│                         Configuration Layer                                 │
│                    src/config/schema.py + loader.py                        │
│   • YAML config files → dataclass conversion                               │
│   • CLI argument parsing and override merging                              │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼──────────────────────────────────────────┐
│                           Solver Layer                                      │
│                      src/solvers/rans_solver.py                            │
│   • RANSSolver: main solver class                                          │
│   • run_steady_state(): time-marching loop                                 │
│   • Manages state Q = [p, u, v, ν̃] on GPU                                  │
└──────────┬───────────────────────┬───────────────────────┬─────────────────┘
           │                       │                       │
     ┌─────▼─────┐          ┌──────▼──────┐         ┌──────▼──────┐
     │   Grid    │          │  Numerics   │         │   Physics   │
     │ src/grid/ │          │src/numerics/│         │src/physics/ │
     ├───────────┤          ├─────────────┤         ├─────────────┤
     │• mesher   │          │• fluxes     │         │• SA model   │
     │• metrics  │          │• gradients  │         │• laminar    │
     │• plot3d   │          │• viscous    │         │  (AFT)      │
     │• loader   │          │• forces     │         │• boundary   │
     └───────────┘          │• sa_sources │         │  layer      │
                            │• smoothing  │         └─────────────┘
                            └─────────────┘
```

### Module Structure

```
src/
├── constants.py              # NGHOST=2, N_VARS=4
│
├── config/                   # Configuration management
│   ├── schema.py             # Dataclass schemas (SimulationConfig, FlowConfig, etc.)
│   └── loader.py             # YAML loading + CLI override merging
│
├── grid/                     # Grid generation and metrics
│   ├── mesher.py             # Construct2D wrapper for C-grid generation
│   ├── metrics.py            # FVM metrics: face normals, volumes, wall distance
│   ├── plot3d.py             # Plot3D format I/O
│   └── loader.py             # Grid file loading utilities
│
├── numerics/                 # Numerical methods (all JAX-accelerated)
│   ├── fluxes.py             # JST convective flux with artificial dissipation
│   ├── gradients.py          # Green-Gauss gradient reconstruction
│   ├── viscous_fluxes.py     # Viscous stress tensor discretization
│   ├── forces.py             # Lift, drag, Cp, Cf computation
│   ├── sa_sources.py         # SA turbulence source terms
│   ├── smoothing.py          # Implicit residual smoothing (legacy)
│   ├── explicit_smoothing.py # Explicit residual smoothing (current)
│   └── diagnostics.py        # Solution monitoring utilities
│
├── solvers/
│   ├── rans_solver.py        # Main RANSSolver class
│   ├── batch.py              # BatchRANSSolver for parametric sweeps
│   ├── boundary_conditions.py# Wall, farfield, wake cut BCs
│   ├── time_stepping.py      # Local time-stepping + RK integration
│   ├── boundary_layer_solvers.py # 1D BL solvers for SA-AFT development
│   └── factory.py            # Solver creation utilities
│
├── physics/                  # Physical models
│   ├── spalart_allmaras.py   # SA turbulence model with analytical gradients
│   ├── laminar.py            # AFT amplification model for transition
│   ├── boundary_layer.py     # Blasius/Falkner-Skan exact solutions
│   ├── correlations.py       # Drela's N-factor correlations
│   └── jax_config.py         # JAX device selection and configuration
│
├── io/
│   └── plotter.py            # Plotly HTML animation dashboard
│
└── validation/
    └── mfoil.py              # MFOIL validation reference
```

### Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Airfoil    │     │  C-Grid     │     │  FVM        │     │  Initial    │
│  .dat file  │────▶│  Generation │────▶│  Metrics    │────▶│  State Q    │
└─────────────┘     │  (mesher)   │     │  (metrics)  │     │  [p,u,v,ν̃]  │
                    └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                   │
         ┌─────────────────────────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Time-Marching Loop                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│  │Gradients│───▶│Convective│───▶│ Viscous │───▶│   SA    │───▶│  Smooth │   │
│  │Green-   │    │  Flux   │    │  Flux   │    │ Source  │    │Residual │   │
│  │Gauss    │    │  (JST)  │    │         │    │         │    │         │   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └────┬────┘   │
│                                                                    │        │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                        │        │
│  │ Forces  │◀───│   BCs   │◀───│  RK     │◀───────────────────────┘        │
│  │ Output  │    │  Apply  │    │  Step   │                                  │
│  └─────────┘    └─────────┘    └─────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### State Vector

The solution array `Q` has shape `(NI + 2*NGHOST, NJ + 2*NGHOST, 4)` with 2 ghost cell layers:

| Index | Variable | Description | Scaling |
|-------|----------|-------------|---------|
| 0 | p | Pseudo-pressure | β (artificial compressibility) |
| 1 | u | x-velocity | V∞ |
| 2 | v | y-velocity | V∞ |
| 3 | ν̃ | SA turbulent viscosity | ν_laminar (1/Re) |

### Key Design Decisions

1. **JAX-First Implementation**: All numerical kernels are pure JAX functions with `@jax.jit` decorators. This enables:
   - GPU acceleration with zero code changes
   - Automatic differentiation (for sensitivity analysis)
   - Vectorization via `jax.vmap` for batch processing

2. **Dual Implementation Pattern**: Many functions have both NumPy and JAX versions:
   - NumPy: For visualization, I/O, and debugging
   - JAX: For the hot path (time-marching)
   - Example: `compute_aerodynamic_forces()` vs `compute_aerodynamic_forces_jax_pure()`

3. **Explicit Residual Smoothing**: Chosen over implicit smoothing for better GPU parallelization (no tridiagonal solves).

4. **Mesh-Invariant Residual Metric**: `get_residual_l1_scaled()` provides convergence monitoring that is independent of mesh refinement.

---

## Areas for Improvement

### 1. Code Organization

| Issue | Current State | Recommendation |
|-------|--------------|----------------|
| **Dual implementations** | NumPy/JAX versions scattered | Create clear `_numpy.py` / `_jax.py` module split |
| **Large solver file** | `rans_solver.py` is 900+ lines | Extract into: `iteration.py`, `io_methods.py`, `state_management.py` |
| **Physics module exports** | `src/physics/__init__.py` is empty | Proper `__all__` exports for public API |
| **Test fixtures** | Some duplicated across files | Consolidate in `conftest.py` |

### 2. Architecture Improvements

| Issue | Impact | Suggested Fix |
|-------|--------|---------------|
| **Tight coupling** | Hard to test components in isolation | Introduce interfaces/protocols for `FluxComputer`, `SourceComputer` |
| **Configuration sprawl** | Many parameters in `SolverConfig` | Group into nested configs: `NumericsConfig`, `OutputConfig`, etc. (partially done) |
| **Magic numbers** | SA constants scattered in multiple files | Centralize in `src/physics/constants.py` |
| **Boundary condition logic** | Complex conditionals in BC functions | Strategy pattern with `BoundaryCondition` base class |

### 3. Performance Opportunities

| Area | Current | Potential Improvement |
|------|---------|----------------------|
| **Memory allocation** | Some intermediate arrays created each step | Pre-allocate in solver `__init__` |
| **Residual computation** | Full L1 sum every iteration | Compute every N iterations, estimate in between |
| **Force computation** | Two paths (JAX pure / NumPy) | Unify to single JAX path with optional CPU transfer |
| **Grid metrics** | Recomputed on solver init | Cache to disk for repeated runs |

### 4. Testing Gaps

| Area | Current Coverage | Needed |
|------|-----------------|--------|
| **Boundary conditions** | Good | Add wake-cut symmetry tests |
| **Convergence** | Limited | Add manufactured solution tests |
| **Batch solver** | Basic | Stress tests with many simultaneous cases |
| **Transition model** | BL solvers tested | Need 2D airfoil integration tests |

### 5. Documentation

| Gap | Action |
|-----|--------|
| **API docs** | Add docstrings to all public functions |
| **Theory** | Add `docs/numerics.md` explaining JST, SA discretization |
| **Examples** | Add Jupyter notebooks for common workflows |

---

## Future Plan: AFT Model Integration

### Background

The **Amplification Factor Transport (AFT)** model enables natural transition prediction by solving a transport equation for the amplification factor `N`. Currently implemented in:

- `src/physics/laminar.py`: Core amplification rate function
- `src/solvers/boundary_layer_solvers.py`: 1D boundary layer solvers (Blasius, Falkner-Skan, flat plate)

### Integration Strategy

The goal is to add AFT-based transition to the 2D airfoil RANS solver, similar to how `run_flat_plate.py` blends laminar (AFT) and turbulent (SA) regimes.

#### Phase 1: SA-AFT Source Term Blending

**Concept**: Smoothly blend between AFT amplification and SA turbulence based on local `ν̃` value.

```python
# In boundary_layer_solvers.py (current implementation):
is_turb = jnp.clip(1 - jnp.exp(-(nuHat - 1) / 4), min=0.0)  # 0=laminar, 1=turbulent

# Source term blending:
source_aft = a_aft * dudy * nuHat  # Laminar amplification
source_sa = P - D + cb2_term       # SA turbulence

source_blended = (1 - is_turb) * source_aft + is_turb * source_sa
```

**Implementation steps**:

1. Add `compute_aft_amplification_2d()` in `src/physics/laminar.py`:
   ```python
   @jax.jit
   def compute_aft_amplification_2d(u, v, grad, wall_dist):
       """Compute AFT amplification rate for 2D field."""
       omega = jnp.sqrt(grad[:,:,1,1]**2 + grad[:,:,2,0]**2)  # |∂v/∂x - ∂u/∂y|
       Re_omega = wall_dist**2 * omega
       # Approximate Gamma from velocity profile shape
       vel_mag = jnp.sqrt(u**2 + v**2)
       Gamma = 2 * (omega * wall_dist)**2 / (vel_mag**2 + (omega * wall_dist)**2 + 1e-10)
       return compute_nondimensional_amplification_rate(Re_omega, Gamma)
   ```

2. Modify `_update_nu_tilde()` in `rans_solver.py`:
   ```python
   # Compute blending factor
   nuHat_int = Q_int[:, :, 3]
   is_turb = jnp.clip(1 - jnp.exp(-(nuHat_int - 1) / 4), min=0.0)
   
   # AFT source (laminar regime)
   a_aft = compute_aft_amplification_2d(u, v, grad, self.wall_dist)
   source_aft = a_aft * omega * nuHat_int
   
   # SA source (turbulent regime)  
   P, D, cb2_term = compute_sa_source_jax(nuHat_int, grad, self.wall_dist, nu)
   source_sa = P - D + cb2_term
   
   # Blend
   source = (1 - is_turb) * source_aft + is_turb * source_sa
   ```

3. Add transition monitoring to output:
   - Track `is_turb` field in snapshots
   - Visualize transition location (where `is_turb` crosses 0.5)

#### Phase 2: Critical N-Factor Transition

**Concept**: Trigger transition when accumulated amplification reaches critical value.

1. Add N-factor transport equation as 5th variable:
   ```python
   # State vector becomes Q = [p, u, v, ν̃, N]
   # N equation: ∂N/∂t + u·∇N = dn/dRe_θ * dRe_θ/dx
   ```

2. Add `N_crit` parameter (typically 9 for low-turbulence, 4 for high):
   ```yaml
   physics:
     transition:
       model: "aft"       # or "sa-only"
       n_crit: 9.0        # Critical N-factor
       tu_freestream: 0.1 # Freestream turbulence intensity (%)
   ```

3. Transition trigger:
   ```python
   is_transitional = N > N_crit
   is_turb = jnp.where(is_transitional, 1.0, 0.0)
   ```

#### Phase 3: Validation

| Test Case | Expected Result |
|-----------|-----------------|
| Flat plate, low Tu | Transition at Re_θ ~ 1000-2000 |
| Flat plate, high Tu | Bypass transition, earlier |
| NACA 0012, α=0° | Laminar near LE, transition mid-chord |
| NLF 0416 | Significant laminar run on suction side |

### New Files Required

```
src/
├── physics/
│   ├── transition.py         # AFT-SA blending logic
│   └── correlations.py       # (extend) N-factor correlations
│
├── numerics/
│   └── aft_sources.py        # 2D AFT source term computation
│
└── solvers/
    └── rans_aft_solver.py    # RANSSolver subclass with transition
```

### Configuration Extension

```yaml
# config/examples/transition.yaml
physics:
  turbulence:
    model: "sa-aft"         # Options: "sa", "sa-aft", "aft-only"
    chi_inf: 0.1            # Low for transition studies
  
  transition:
    enabled: true
    n_crit: 9.0             # Critical amplification factor
    tu_freestream: 0.1      # Freestream turbulence (%)
    blend_width: 4.0        # nuHat units for smooth blending
```

---

## Command Line Options

```bash
python scripts/solver/run_airfoil.py <airfoil.dat> [options]

Options:
  --alpha ANGLE       Angle of attack in degrees (default: 0)
  --reynolds RE       Reynolds number (default: 6e6)
  --mach MACH         Mach number for compressibility (default: 0.15)
  --cfl CFL           Target CFL number (default: 5.0)
  --max-iter N        Maximum iterations (default: 500)
  --n-surface N       Surface grid points (default: 129)
  --n-normal N        Normal direction points (default: 65)
  --n-wake N          Wake cut points (default: 32)
  --chi-inf CHI       Initial/farfield turbulent viscosity ratio (default: 3.0)
  --config FILE       YAML configuration file
```

## Running Tests

```bash
# Run all tests (excluding slow ones)
pytest tests/ -m "not slow"

# Run with verbose output
pytest tests/ -v

# Run specific test module
pytest tests/numerics/test_sa_sources.py -v
```

## References

- Jameson, Schmidt & Turkel (1981). "Numerical Solutions of the Euler Equations by Finite Volume Methods Using Runge-Kutta Time-Stepping Schemes". AIAA Paper 81-1259.
- Spalart & Allmaras (1992). "A One-Equation Turbulence Model for Aerodynamic Flows". AIAA Paper 92-0439.
- Coder & Maughmer (2014). "Computational Fluid Dynamics Compatible Transition Modeling Using an Amplification Factor Transport Equation". AIAA Journal.
- Turkel (1987). "Preconditioned Methods for Solving the Incompressible and Low Speed Compressible Equations". JCP 72.
- Drela (2003). "Implicit Implementation of the Full eN Transition Criterion". AIAA Paper 2003-4066.

## License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0).

See [LICENSE.md](LICENSE.md) for the full license text.

This license ensures that:
- You can use, modify, and distribute this software freely
- Any derivative works must also be open source under GPL-3.0
- The source code must be made available when distributing

Note: This project includes [Construct2D](external/construct2d/) which is also GPL-3.0 licensed.
