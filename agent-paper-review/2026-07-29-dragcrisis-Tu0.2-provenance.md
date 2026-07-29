# Drag-crisis Tu = 0.2% simulation provenance — every case and its fork

*2026-07-29, paper agent. Answers "what did you run and what is each forked
from?" for the middle seed only (Tu = 0.2%, chi_inf from Mack's map). This is
the complete inventory behind the Tu=0.2 curves in Fig.~18 (fig:dragcrisiscd).
Source: /local_data/qiqi/sa-ai/dragcrisis_matrix/matrix_summary.jsonl (75
Tu=0.2 rows) + the drivers run_dragcrisis_matrix.py / run_dragcrisis_extension.py.*

## Fork rule (how a case is initialized)

A "chain" is one arm run in one direction. Within a chain:
- **cold** = cold two-stage start (uniform freestream IC; staged fSlow). Every
  case in a `cold` chain is independent; and the FIRST case of any up/dn chain
  is cold.
- **up** = ascending Re; each case after the first is **warm-restarted from the
  previous (lower) Re** in the same chain.
- **dn** = descending Re; each case after the first is **warm-restarted from the
  previous (higher) Re** in the same chain.
- A few cases carry an explicit external warm anchor (noted).

Meshes (why the same Re can appear twice): `lowre` (600x121, y1=1e-3D, R=1000D),
`lowre300` (R=300D twin, far-field check), `pilot` (1200x148, y1=8e-6D, R=100D),
`highre` (1600x170, y1=1e-6D), `ultra` (1600x170, y1=1e-9D). Seam Reynolds
numbers were run on BOTH adjacent meshes on purpose (overlap validation), which
is why Fig.~18 shows two nearby points there.

## The six arms (Tu = 0.2%)

### 1. Creep arm — `lowre`, up-ladder, Re 1 -> 1000 (7 cases)
Physically steady below Re~47; a laminar validation.
| Re | dir | forked from |
|---|---|---|
| 1 | up | **cold** (chain start) |
| 3 | up | warm <- Re 1 |
| 10 | up | warm <- Re 3 |
| 30 | up | warm <- Re 10 |
| 100 | up | warm <- Re 30 |
| 300 | up | warm <- Re 100 |
| 1000 | up | warm <- Re 300 |

### 2. Creep far-field twin — `lowre300` (R=300D), up, Re 1 & 30 (2 cases)
Blockage/far-field check against arm 1. **This is the "double solid line" at
low Re in Fig.~18**: lowre (arm 1) and lowre300 (this arm) are two separate
solid up-ladder segments, coincident to ~0.5% (Re=1: 10.67 vs 10.72; Re=30:
1.710 vs 1.714).
| Re | dir | forked from |
|---|---|---|
| 1 | up | **cold** (chain start) |
| 30 | up | warm <- Re 1 |

### 3. Low arm — `pilot`, dn-continuation, Re 4e4 -> 300 (6 cases)
Below the matrix, continued DOWN from the matrix 6e4 state. **This is the range
with only a down-ladder (dashed) and no up-ladder** (Re 3000, 1e4, 2e4, 4e4);
at Re 300 & 1000 it overlaps the creep arm's up points (two dirs, two meshes).
| Re | dir | forked from |
|---|---|---|
| 4e4 | dn | warm <- **cyl_Re60000_Tu0.2_dn** (matrix 6e4 dn endpoint) |
| 2e4 | dn | warm <- Re 4e4 |
| 1e4 | dn | warm <- Re 2e4 |
| 3e3 | dn | warm <- Re 1e4 |
| 1e3 | dn | warm <- Re 3e3 |
| 300 | dn | warm <- Re 1e3 |

### 4. Matrix — `pilot`, up + dn + cold, Re 6e4 -> 2e6 (11 Re x 3 = 33 cases)
The original 84-case matrix's Tu=0.2 slice. Three independent chains:
- **up-ladder**: 6e4 cold; then 1e5<-6e4, 1.5e5<-1e5, 2e5<-1.5e5, 2.5e5<-2e5,
  3e5<-2.5e5, 3.5e5<-3e5, 4e5<-3.5e5, 5e5<-4e5, 7e5<-5e5, 1e6<-7e5, 2e6<-1e6.
- **dn-ladder**: 2e6 cold; then 1e6<-2e6, 7e5<-1e6, 5e5<-7e5, 4e5<-5e5,
  3.5e5<-4e5, 3e5<-3.5e5, 2.5e5<-3e5, 2e5<-2.5e5, 1.5e5<-2e5, 1e5<-1.5e5,
  6e4<-1e5.
- **cold**: each of the 11 Re independently cold two-stage (protocol-neutral
  reference).
(Re 6e4..2e6 therefore has up, dn AND cold — the fully-populated band.)

### 5. High arm — `highre`, Re 2e6 -> 2e7 (up + dn) + seams
- **up-ladder** (HIGH_UP): 4e6 cold; 7e6<-4e6; 1e7<-7e6; 2e7<-1e7 (stretch).
- **dn-ladder**: from the top down to the 2e6 seam; 1e7<-(top), 7e6<-1e7,
  4e6<-7e6, 2e6<-4e6.
- **seam2e6h**: highre 2e6 **cold** (protocol-neutral seam point, isolates the
  mesh-family offset from the up/dn branch spread).
- **seam4e6**: `pilot` 4e6 up, warm <- **cyl_Re2000000_Tu0.2_up** (matrix 2e6 up).
  This is the pilot point at 4e6 that is OUT of pilot y+ validity (y+~2.1) —
  a continuity check only, and the ~5% pilot-vs-highre offset you saw.
- **seam5e7h**: `highre` 5e7 **cold** — OUT of highre y+ validity (y+~3.5),
  continuity check only.

### 6. Ultra arm — `ultra`, Re 2e7 -> 1e10 (up + dn) (2026-07-28 directive)
- **up-ladder** (ULTRA_UP): 2e7 cold (this IS the highre/ultra seam, replacing
  the y+=1.4-caveated highre 2e7); 5e7<-2e7; 1e8<-5e7; 3e8<-1e8; 1e9<-3e8;
  3e9<-1e9; 1e10<-3e9.
- **dn-ladder** (ULTRA_DN): from the 1e10 state down: 3e9<-1e10; 1e9<-3e9;
  3e8<-1e9; 1e8<-3e8; 5e7<-1e8.
- **sanity1e10**: ultra 1e10 **cold** — the muRef=1e-11 float-health gate that
  had to pass before the ladders ran; it found the near-attached low-drag
  branch (Cd 0.108) vs the warm ladder's separated 0.189 (the branch
  multiplicity).

## Why Fig.~18 looks patchwork (direct answers)

1. **Double solid line at low Re** = arm 1 (`lowre`) and arm 2 (`lowre300`),
   two up-ladders on two far-field meshes, both solid, coincident. It is the
   R=300D-vs-1000D blockage check, not a duplicate run.
2. **Range with only a down-ladder (no up)** = the low arm (arm 3), Re 300-4e4,
   which was run purely as a dn-continuation from 6e4 (no up-ladder was needed
   there — the subcritical plateau is protocol-insensitive, verified where the
   matrix has both).
3. **Range with only an up-ladder** = the creep arm (arm 1), Re 1-1000: below
   shedding onset the steady solution is unique, so a single ascending pass
   suffices.
4. **Fully populated (up + dn, both solid & dashed)** = the matrix (6e4-2e6),
   the high arm (4e6-2e7), and the ultra arm (5e7-1e10), where the branch
   spread through/above the crisis is the physics of interest.
5. **Two nearby points at a seam Re** (300, 1e3, 2e6, 4e6, 2e7, 5e7) = the same
   Re run on both adjacent meshes for overlap validation; Fig.~18 draws one
   line PER mesh family (2026-07-29 fix) so these read as adjacent points, not
   the old spurious vertical kink.

## Counts
75 Tu=0.2 cases = 7 creep + 2 creep300 + 6 low + 33 matrix (11 up + 11 dn + 11
cold) + high (4 up + 4 dn + 1 seam2e6h + 1 seam4e6 + 1 seam5e7h = 11) +
ultra (7 up + 5 dn + 1 sanity = 13). (The full three-seed campaign is 173
cases; this file is Tu=0.2 only, as requested.)
