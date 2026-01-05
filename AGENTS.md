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
| *None currently* | — | — | — |

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
