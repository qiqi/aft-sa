# AFT-SA: GPU-Accelerated RANS Solver

A 2D structured-grid RANS solver for incompressible airfoil flow using the Artificial Compressibility Method. Fully GPU-accelerated with JAX.

## Features

- **Artificial Compressibility Method** for incompressible flow
- **JST Scheme** (Jameson-Schmidt-Turkel) with 4th-order artificial dissipation
- **Spalart-Allmaras** one-equation turbulence model
- **5-stage Runge-Kutta** time integration with explicit smoothing
- **GPU acceleration** via JAX (runs on CPU if no GPU available)
- **C-grid topology** with proper wake cut handling

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

## Project Structure

```
aft-sa/
├── src/
│   ├── constants.py              # Global constants (NGHOST=2, N_VARS=4)
│   ├── grid/                     # Grid generation and metrics
│   │   ├── loader.py             # Load/generate C-grids
│   │   ├── metrics.py            # Face normals, volumes
│   │   └── plot3d.py             # Plot3D format I/O
│   ├── numerics/                 # Numerical methods (all JAX)
│   │   ├── fluxes.py             # JST convective flux
│   │   ├── gradients.py          # Green-Gauss gradients
│   │   ├── viscous_fluxes.py     # Viscous stress tensor
│   │   ├── forces.py             # Lift, drag, Cp, Cf
│   │   ├── smoothing.py          # Implicit residual smoothing
│   │   └── explicit_smoothing.py # Explicit residual smoothing
│   ├── solvers/
│   │   ├── rans_solver.py        # Main RANSSolver class
│   │   ├── boundary_conditions.py # Wall, farfield, wake BCs
│   │   └── time_stepping.py      # Local time-stepping
│   └── io/
│       ├── output.py             # VTK output
│       └── plotting.py           # Visualization
├── scripts/
│   ├── solver/
│   │   └── run_airfoil.py        # Main simulation script
│   └── models/                   # Physics model scripts
├── tests/                        # Pytest test suite
├── data/                         # Airfoil geometry files
└── external/
    └── construct2d/              # Grid generator
```

## State Vector

The solution array `Q` has shape `(NI + 4, NJ + 4, 4)` with 2 ghost layers:

| Index | Variable | Description |
|-------|----------|-------------|
| 0 | p | Pseudo-pressure (artificial compressibility) |
| 1 | u | x-velocity |
| 2 | v | y-velocity |
| 3 | ν̃ | Spalart-Allmaras turbulent viscosity |

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
```

## Running Tests

```bash
# Run all tests (excluding slow ones)
pytest tests/ -m "not slow"

# Run with verbose output
pytest tests/ -v
```

## References

- Jameson, Schmidt & Turkel (1981). "Numerical Solutions of the Euler Equations by Finite Volume Methods Using Runge-Kutta Time-Stepping Schemes". AIAA Paper 81-1259.
- Spalart & Allmaras (1992). "A One-Equation Turbulence Model for Aerodynamic Flows". AIAA Paper 92-0439.
- Turkel (1987). "Preconditioned Methods for Solving the Incompressible and Low Speed Compressible Equations". JCP 72.

## License

MIT License
