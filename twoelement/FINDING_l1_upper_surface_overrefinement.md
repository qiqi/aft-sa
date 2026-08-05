# L1 two-element mesh: upper-surface over-refinement

**Symptom.** In the L1 mesh built after the tangential metric was changed to read
the local surface spacing, the upper surface of both elements was far more
refined than the lower surface, and the total cell count jumped to 332,930
(3.4x the previous 97,554) with runtime 1.8 -> 4.4 min.

**Measured asymmetry** (`measure_l1_spacing.py case_L1_fixed`), near-wall cells
classified by nearest wall and side:

| | fore lower | fore upper | flap lower | flap upper |
|---|---|---|---|---|
| cells | 34,993 | 187,485 | 18,934 | 91,518 |
| cells per unit arc | 48,828 | **266,158** | 62,276 | **277,098** |
| median cell size, `d < 0.002` | 3.86e-4 | **8.97e-5** | 3.92e-4 | **8.96e-5** |

Upper/lower density ratio: **5.45** on the fore, **4.45** on the flap. Smallest
cell in the domain 2.86e-7. The excess is entirely near-wall: 203,338 of the
332,930 cells sat within `d < 0.002` of a surface.

## Cause: a local edge index used as a global point index

`SpaldingMetric` in `flexfoil/rans/mesher/mesh2d.cpp` obtained the local surface
spacing as

```cpp
const EdgeMinDistance<3>& info = w->tree.minDistanceToCurve(x);
const Vector3& a = w->points[info.minEdgeIdx];
const Vector3& b = w->points[info.minEdgeIdx + 1];
```

`minEdgeIdx` is **not** a global point index. `KdTreeEdgeHost::minDistanceToCurve`
sets it from `closestNode.pointIndex`, which indexes that curve's own `edges`
array; `getEdgesFromNodes` built those edges from `curve.nodeIndices`. The point
indices must be recovered through `edges[minEdgeIdx].first/.second`.

`contours.txt` lays the global point array out as

```
farfield   0 .. 359     (361 nodes, closing)
fore     360 .. 1561    (1203 nodes, closing)
flap    1562 .. 2763    (1203 nodes, closing)
```

so the fore wall's edge index is offset from its global point index by **360**
and the flap's by **1562**. Indexing `points[]` directly therefore reads the
geometry from somewhere else entirely. For both walls, edge indices 0..1201 map
to global points 0..1202, i.e.:

* edges 0..358 read the **farfield circle**, whose segments are ~1.745 long.
  `min(hLocal, hwall)` clamps these to `hwall`, so those stations silently
  reverted to the old global-constant behaviour.
* edges 359..1201 read the **fore element's contour**, shifted 360 positions.

Two consequences, both measured:

1. **Systematic upper/lower bias.** The contour is ordered lower TE -> LE ->
   upper TE. A 360-position shift hands every station the spacing belonging to a
   station 360 points closer to the lower trailing edge. The upper surface
   (indices 652..1201 on the fore) therefore reads from indices 292..841, a
   window that contains the leading-edge cluster - the finest part of the whole
   contour. Result: **57.1%** of upper-surface stations got a spacing more than
   2x finer than intended, while **55.2%** of lower-surface stations were pinned
   at `hwall` (median evaluated/intended ratio 0.44 upper, 2.62 lower).

2. **A displaced hot spot, at a predictable place.** The fore contour's finest
   segment is 2.441e-6 at its own index 600. Read 360 positions late it lands at
   index 960, which is:
   * fore: `(x, z) = (0.2794, +0.0444)`, arc 1.0023 - **upper surface, x/c 0.28**
   * flap: `(x, z) = (0.8280, +0.0732)`, arc 0.4465 - **upper surface, x/c 0.83**

   Those are exactly the two bins where the measured cell count explodes
   (61,442 cells in fore arc 0.995-1.066; 26,139 in flap arc 0.444-0.476).

Note the flap never read its own geometry at all: all 1202 of its lookups landed
in the farfield (359) or the fore contour (843).

## Fix

Go through the edge array, as the API intends:

```cpp
const Edge& e = w->tree.edges[info.minEdgeIdx];
const Vector3& a = w->points[e.first];
const Vector3& b = w->points[e.second];
```

Rebuilt and reran L0/L1 on the identical surface contours (`contours.txt` is
byte-identical between the two runs, so only the metric changed):

| L1 | cells | CL | CD | fore upper/lower density | flap upper/lower density | min cell |
|---|---|---|---|---|---|---|
| buggy index | 332,930 | 0.254 | 0.01658 | 5.45 | 4.45 | 2.86e-7 |
| **index fixed** | **250,170** | **0.305** | **0.01214** | **0.95** | **1.15** | **3.68e-6** |

Near-wall median size on the fore is now 2.28e-4 (lower) vs 2.48e-4 (upper) -
matched to 9%, against 4.3x before. L0 went 28,258 -> 66,026 cells,
CL 0.056 -> 0.217, CD 0.0467 -> 0.01842. Runtime L1 4.4 -> 3.7 min.

Full ladder with the fix in place:

| level | cells | CL | CD | y+ max | runtime | fore U/L | flap U/L |
|---|---|---|---|---|---|---|---|
| L0 | 66,026 | 0.2171 | 0.018422 | 3.7 | 1.1 min | 0.95 | 1.38 |
| L1 | 250,170 | 0.3047 | 0.012141 | 2.4 | 3.7 min | 0.95 | 1.15 |
| L2 | 931,476 | 0.2879 | 0.012063 | 1.1 | 15.0 min | 0.95 | 1.13 |

Cell count scales 3.79x then 3.72x, against a nominal 4x for halving the
spacing in 2D, so the levels now refine as intended. The upper/lower balance
holds at every level. Measured y+ is still 2-4x the nominal ladder value
(3.7 against 1.0, 2.4 against 0.5, 1.1 against 0.3) - separate issue, in the
first-cell height, not in the tangential metric.

`l1_spacing_diagnosis.pdf` shows the mechanism: row 1 has the evaluated spacing
as a copy of the intended curve shifted right by 360 indices; row 3 shows the
realised near-wall cell size tracking the *shifted* minimum, and the fixed mesh
(green) flat and symmetric about the leading edge.

## Corrects an earlier diagnosis

The previous conclusion - that propagating a fine `h_te` into the metric
"floods the domain" because `min(hLocal,hwall) + (growth-1)*d` needs 20 chords to
reach `hmax` - was wrong. `min(hLocal, hwall) <= hwall = 0.004` always, so the
starting value and hence the release rate are unchanged from before; the two
formulas differ only for `d` less than about 0.03. The extra 235,000 cells were
near-wall, not far-field. No exponential-relaxation term is needed.

## Still outstanding: the contour's own degenerate LE point

Independent of the index bug, `contour_te._build_half` places its finest points
badly, and the fix now exposes this directly to the metric:

* The quarter-sine blend has zero derivative at `s = 0.5`, so its last step
  collapses as `1/k^2`. The fore contour's smallest segment is **2.441e-6**,
  537x below its median (1.4e-3); the flap's is 1.082e-6, 541x below. Ten
  segments per element are below 2% of median. That is a degenerate point, not
  resolution, and it is why the fixed mesh still has a 3.68e-6 cell.
* The distribution is symmetric about normalized arc `s = 0.5`, but the
  geometric leading edge is not there. Fore: LE at arc fraction 0.5043, cluster
  at 0.4998 - off by 52 points. Flap (NACA 9416, heavily cambered): LE at
  0.4793, cluster at 0.4998 - off by **109 points, 0.013 chord of arc**, so the
  flap's clustering sits well onto its lower surface and the actual leading edge
  gets 2.34e-4 spacing instead of the cluster value.

Suggested next step: anchor the blend on the true LE arc fraction (found from
the dense spline, not assumed to be 0.5) and floor the step at some fraction of
`h_te` so the blend cannot produce a degenerate segment.

## Latent, elsewhere

`CurveProjection::getTangent` in
`compute/src/src/Flow360Core/Libraries/MeshDataStructures/CurveProjection.h`
makes the identical mistake (`points[info.minEdgeIdx + 1] - points[info.minEdgeIdx]`).
It has no callers in `compute` today, so nothing is broken by it, but anyone who
picks it up on a multi-curve geometry will hit exactly this.
