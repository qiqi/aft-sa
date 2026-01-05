# Multi-Agent Coordination Board

This file coordinates work between multiple AI agents and human contributors on this repository. **All agents must check this file before making changes.**

---

## Locking Rules

1. **Before editing any file**, check the "Active Agents" table below.
2. **If a file/folder is listed as active**, do NOT modify it. Wait or work on something else.
3. **Before starting work**, add your entry to the "Active Agents" table.
4. **When finished**, remove your entry and move the file to "Ready for Cleanup" if applicable.
5. **Conflicts**: If you need a locked file urgently, coordinate with the human user.

---

## Active Agents

| Agent Name/Role | Target File/Folder | Task Description | Start Time |
|-----------------|-------------------|------------------|------------|
| Claude | `src/numerics/preconditioner.py` | Phase 1: Block-Jacobi Preconditioner | 2026-01-05 |

---

## Ready for Cleanup

Files that have been modified and are ready for the Janitor agent to review:

| File/Folder | Last Modified By | Notes |
|-------------|------------------|-------|
| *None currently* | — | — |

---

## Recently Completed

Archive of recently completed work (for reference):

| Date | Agent | Files | Summary |
|------|-------|-------|---------|
| 2026-01-05 | Claude | `src/numerics/sa_sources.py` | Removed unnecessary `jnp.where` guards in `compute_aft_sa_source_jax` - cleaner code, better AD compatibility |
| 2026-01-05 | Janitor | `src/io/plotter.py`, `src/solvers/rans_solver.py` | Removed unused import `compute_sa_source_jax`, removed unused var `xaxis_key` |
| 2026-01-05 | Claude | `scripts/analysis/` | Cleanup: removed debug scripts for BL analysis, momentum balance diagnostics |
| 2026-01-05 | Claude | `src/solvers/boundary_conditions.py`, `src/solvers/batch.py`, `tests/` | Investigated wall BC - confirmed u_ghost=-u_int is CORRECT (Q stores physical velocity, not perturbation). Fixed test file to use from_alpha(). |
| 2026-01-05 | Claude | `src/io/_layout.py`, `src/io/plotter.py`, `scripts/` | Fixed HTML layout: subplot titles, colorbar y-alignment from actual subplot domains, fixed scaleanchor yaxis references |
| 2026-01-05 | Claude | `scripts/` | Created debug_layout.py - Visual Layout Validator using Playwright |
| 2026-01-05 | Claude | `src/io/` | Fixed animation: all contour/line plots now update per iteration (AFT, y+, Cp, Cf) |
| 2026-01-05 | Claude | `scripts/analysis/` | Created BL profile extraction & Blasius comparison tools for Cf deviation analysis |
| 2026-01-05 | Claude | `src/io/` | Fixed HTML plotter: isotropic contour axes, colorbar positioning, is_turb single-column |
| 2026-01-05 | Janitor | `src/`, `tests/`, `config/` | Removed `mach` and `aft_enabled` configs, changed `divergence_history` default to 1 |
| 2026-01-05 | Janitor | `src/io/`, `src/solvers/`, `src/numerics/`, `src/grid/`, `src/physics/` | Removed 4 unused imports, prefixed 18 intentionally-unused local variables with `_` |
| 2026-01-05 | Claude | `src/io/_layout.py`, `src/io/_plot_registry.py`, `src/io/plotter.py` | Refactored HTML plotter with declarative DashboardLayout system |
| 2026-01-05 | Claude | `src/solvers/boundary_conditions.py`, `src/solvers/batch.py` | Removed wake cut interior averaging - pure periodic BC now |
| 2026-01-04 | Janitor | `src/solvers/boundary_conditions.py`, `src/solvers/time_stepping.py` | Removed JAX_AVAILABLE flags - always assume JAX available |
| 2026-01-04 | Janitor | `src/physics/laminar.py` | Consolidated AFT functions: laminar.py now imports from aft_sources.py |
| 2026-01-03 | Janitor | `src/` (multiple), `tests/` | Removed 20+ unused imports, fixed f-strings, fixed test API calls |
| 2026-01-03 | Claude | `src/io/plotter.py` | Added gzip compression, y+ visualization |
| 2026-01-03 | Claude | `README.md`, `LICENSE.md` | Architecture docs, GPL-3.0 license |
| 2026-01-03 | Claude | `src/solvers/factory.py` | Removed unused `create_solver_quiet()` |

---

## Communication Log

Use this section for agent-to-agent messages or notes:

```
[2026-01-03] System initialized. Coordination board created.
```
