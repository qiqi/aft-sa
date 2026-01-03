# Batch Solving & YAML Config - Design Plan

## Problem Analysis

**Current GPU utilization: ~6%**
- Grid size: 128×32 = 4,096 cells
- V100 has 5,120 CUDA cores
- GPU is massively underutilized - kernels are memory-bound and launch-overhead dominated

**Solution: Batch solving**
- Process N airfoils simultaneously
- Array shape: `(NI, NJ, 4)` → `(N_batch, NI, NJ, 4)`
- With batch=50, we'd have 200k elements - much better GPU saturation

---

## Part 1: YAML Configuration Design

### Proposed Structure

```yaml
# config/naca0012_aoa_sweep.yaml

grid:
  airfoil: data/naca0012.dat
  n_surface: 129
  n_normal: 65
  n_wake: 64
  y_plus: 1.0

flow:
  reynolds: 6.0e6
  mach: 0.0  # incompressible for now
  
  # Single value OR batch specification
  alpha:
    sweep: [-5, 15, 21]  # start, end, count → 21 cases

solver:
  max_iter: 5000
  tol: 1.0e-10
  print_freq: 100
  
  cfl:
    initial: 0.5
    final: 2.0
    ramp_iter: 500

numerics:
  jst_k4: 0.02
  beta: 10.0
  smoothing:
    type: explicit  # or implicit
    epsilon: 0.2
    passes: 2

output:
  directory: output/aoa_sweep
  html_animation: true
  save_snapshots: false
```

### Batch Parameter Specification Options

```yaml
# Option 1: Linear sweep
alpha:
  sweep: [-5, 15, 21]  # start, end, count

# Option 2: Explicit list
alpha:
  values: [-2, 0, 2, 4, 6, 8, 10, 12]

# Option 3: Single value (no batching on this param)
alpha: 4.0

# Option 4: Multiple sweeps (advanced)
batch:
  - alpha: {sweep: [0, 10, 5]}
    reynolds: 6.0e6
  - alpha: {sweep: [0, 10, 5]}
    reynolds: 3.0e6
```

---

## Part 2: Batch Solver Architecture

### Array Layout Decision

| Option | Layout | Pros | Cons |
|--------|--------|------|------|
| A | `(batch, NI, NJ, vars)` | Standard ML convention, easy vmap | - |
| B | `(NI, NJ, batch, vars)` | Better memory locality for stencils | Non-standard |

**Recommendation: Option A** - JAX's `vmap` works naturally with batch-first layout.

### What Can Vary Across Batch?

| Component | Same Grid | Different Geometry |
|-----------|-----------|-------------------|
| X, Y coordinates | ✓ shared | ✗ per-batch |
| Grid metrics (Si, Sj, vol) | ✓ shared | ✗ per-batch |
| Freestream (α, Re) | ✗ per-batch | ✗ per-batch |
| Q state | ✗ per-batch | ✗ per-batch |
| BCs (wall, farfield) | mostly shared | per-batch normals |

### Implementation Strategy: `jax.vmap`

```python
# Current single-case step
def step_single(Q, dt, metrics, freestream):
    Q = apply_bc(Q, freestream)
    R = compute_fluxes(Q, metrics)
    Q_new = Q + dt * R
    return Q_new

# Batched version via vmap
step_batch = jax.vmap(step_single, in_axes=(0, 0, None, 0))
# Q: (batch, NI, NJ, 4) - vectorize over batch
# dt: (batch, NI, NJ) - per-batch timestep (different CFL stability)
# metrics: shared (None means don't vectorize)
# freestream: (batch,) - different per case
```

### Memory Estimate

For 50 cases on 256×64 grid:
- Q: 50 × 256 × 64 × 4 × 8 bytes = **26 MB**
- Metrics: ~50 MB (shared)
- Working arrays: ~100 MB
- **Total: ~200 MB** (V100 has 16-32 GB - plenty of room)

Could easily do 500+ cases.

### Performance Scaling (measured on V100)

**Small grid (128×32 = 4,096 cells/case):**
| Batch | Total Cells | ms/iter | Speedup |
|-------|-------------|---------|---------|
| 1     | 4,096       | 0.16    | 1.0x    |
| 2     | 8,192       | 0.18    | 1.8x    |
| 4     | 16,384      | 0.26    | 2.5x    |
| 8     | 32,768      | 0.41    | **3.1x** |
| 16    | 65,536      | 0.81    | **3.2x** |
| 32    | 131,072     | 1.71    | 3.0x    |

**Large grid (256×64 = 16,384 cells/case):**
| Batch | Total Cells | ms/iter | Speedup |
|-------|-------------|---------|---------|
| 1     | 16,384      | 0.30    | 1.0x    |
| 4     | 65,536      | 0.83    | 1.4x    |
| 16    | 262,144     | 4.29    | 1.1x    |

**Key findings:**
- Small grids benefit most from batching (3x speedup at batch=8-16)
- Larger grids already saturate GPU, less benefit from batching
- Sweet spot: batch=8-16 for small grids
- Memory is not a bottleneck (plenty of headroom)

---

## Part 3: Implementation Phases

### Phase 1: YAML Config (no batch yet) ✅ COMPLETE
1. ✅ Create `src/config/schema.py` with dataclass models
2. ✅ YAML loader with validation (`src/config/loader.py`)
3. ✅ Update `run_airfoil.py`: `--config config.yaml`
4. ✅ Keep CLI options as overrides: `--config base.yaml --alpha 5`
5. ✅ Example configs in `config/examples/`

**Completed: 2025-01-02**

### Phase 2: Batch Data Structures ✅ COMPLETE
1. ✅ `BatchFlowConditions` - per-case freestream (α, Re, etc.)
2. ✅ Sweep expansion from YAML specs (sweep/values → array)
3. ✅ `BatchState` class holding `Q_batch: (N, NI, NJ, 4)`
4. ✅ Batch initialization from single grid + multiple conditions
5. ✅ Update config schema to parse sweep specifications

**Completed: 2025-01-02**

### Phase 3: Batch Kernels
1. Wrap flux computation with `vmap`
2. Wrap BC application with `vmap`
3. Wrap timestep computation with `vmap`
4. Test: verify `step_batch(Q[None,...])` matches `step_single(Q)`

**Effort: ~3 hours**

### Phase 4: Batch Integration
1. `RANSSolver` batch mode
2. Per-case residual tracking
3. Per-case force computation
4. Batch-aware output (CL-α curve, etc.)

**Effort: ~3 hours**

### Phase 5: Advanced Features (optional)
1. Different geometries (NACA 4-digit family)
2. Early stopping for converged cases
3. Checkpointing for long runs

---

## Part 4: Key Design Decisions

1. **Geometry variation**: Start with same-grid (AoA sweep), support different airfoils later.

2. **Convergence handling**: Run all cases to max_iter (simple first).

3. **Output format**: 
   - Summary CSV with CL, CD vs. parameters
   - Optional per-case VTK files
   - Single HTML with all cases

4. **CLI vs YAML priority**: CLI overrides YAML values.

---

## File Structure

```
src/
├── config/
│   ├── __init__.py
│   ├── schema.py          # Dataclass models
│   └── loader.py          # YAML parsing
├── solvers/
│   ├── rans_solver.py     # Keep single-case (legacy)
│   └── batch_solver.py    # New batch implementation (Phase 2+)
scripts/
└── solver/
    ├── run_airfoil.py     # Updated for YAML
    └── run_batch.py       # New batch entry point (Phase 2+)
config/
├── examples/
│   ├── single_case.yaml
│   ├── aoa_sweep.yaml
│   └── naca0012_production.yaml
```
