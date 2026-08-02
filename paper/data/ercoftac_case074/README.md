# ERCOFTAC Classic Collection Case 074 — inclined 6:1 prolate spheroid

Original **numerical** DFVLR measurement files for the inclined 6:1 prolate
spheroid: wall shear stress (magnitude and direction), surface pressure, and
boundary-layer / flow-field velocity profiles. These supersede digitization of
printed replots wherever the condition matches.

- **Source:** ERCOFTAC Classic Collection, Case 074, "Three-Dimensional
  Boundary Layer and Flow Field Data of an Inclined Prolate Spheroid",
  compiled by H.-P. Kreplin.
  <http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case074>
- **Retrieved:** 2026-08-02, 349 `.dat` files, zero download failures.
  Download URL pattern (HTTP only — HTTPS is refused by the host):
  `http://cfd.mace.manchester.ac.uk/ercoftac/lib/exe/fetch.php?media=cdata:case074:<case>:<sub>:<file>.dat`
- **Underlying reports** (DFVLR internal, neither available online):
  - Kreplin, H.-P., Vollmers, H., Meier, H. U., "Wall Shear Stress
    Measurements on an Inclined Prolate Spheroid in the DFVLR 3M x 3M Low
    Speed Wind Tunnel, Göttingen", DFVLR-AVA Rept. IB 222-84 A 33, Jan. 1985.
    (= Stock 2006 ref. 49; our bib `kreplin_1985`.)
  - Kreplin, H.-P., Vollmers, H., Meier, H. U., "Wall Shear Stress
    Measurements on an Inclined Prolate Spheroid in the ONERA F1 Wind
    Tunnel", DFVLR-AVA Rept. IB 222-84 A 34, June 1985.
    (= Stock 2006 ref. 50; **not yet in our bib**.)

## The three cases

Conditions are taken from the **file headers**, not from the web page's summary
table (which rounds F1-30 to "40e6"; the files say 43.54e6).

| dir | facility | alpha | Re_L | transition | U_inf |
|---|---|---|---|---|---|
| `nwg10/` | DLR/DFVLR 3x3 m Göttingen (NWG) | 10.0 deg | 7.70e6 | **ARTIFICIAL — tripped at X0/2A = 0.2** | 55.0 m/s |
| `nwg30/` | DLR/DFVLR 3x3 m Göttingen (NWG) | 29.7 deg | 6.54e6 | **NATURAL** | 45.0 m/s |
| `f1_30/` | CERT/ONERA F1, Le Fauga-Mauzac | 30.0 deg | 43.54e6 | *not stated in the files* | 75.8 m/s, p0 = 3.8 bar |

⚠ **`nwg10` is tripped** and is therefore useless as a natural-transition
reference — it is the case the separated-flow/RANS-validation community uses.
Do not quote it as a transition benchmark.

⚠ **`f1_30` carries no natural/artificial statement.** The Göttingen files
state it explicitly; the F1 files give only the stagnation pressure. Stock
plots *measured transition fronts* at 43.54e6 (his Fig. 17c), which implies
natural transition, but the data files do not confirm it. Verify against
report A 34 before relying on it.

## File families

| pattern | content |
|---|---|
| `<case>_cp.dat` | surface pressure: DPN, PHI, tap index, X0/L, CP |
| `<case>_cf_NN.dat` | wall shear per x-station: PHI, CF, GAMMA (~120 azimuths per station; 12 stations NWG, 11 F1) |
| `nwg10ubl_NN.dat` | boundary-layer velocity profiles (61 files; 4 x-stations x many phi) |
| `nwg30uff_NNN.dat` | flow-field velocity profiles (208 files; 11 x-stations x many phi) |
| `f1_30uff_NN.dat` | flow-field velocity profiles (42 files; 3 x-stations x 14 phi) |

Azimuth convention in the files: `PHI = 0` windward, `180` leeward, values run
about `-5` to `185` (slight overshoot past both symmetry planes).
`X0/2A` is the axial station as a fraction of body length, i.e. our `x/L`.

## Measured freestream turbulence — the numbers the transition literature does not use

The ERCOFTAC case description quotes hot-wire measurements **per facility**:

- **NWG (Göttingen, open-jet test section):** "The overall turbulence level is
  rather high compared to other wind tunnels. Values of **0.33% to 0.4%** for
  the streamwise velocity fluctuations and about **0.8%** for the vertical and
  spanwise components have been measured."
- **ONERA F1 (pressurised, closed test section):** "Hot wire measurements in
  the test section showed a low streamwise turbulence level **smaller than
  0.1%**."

Note the ordering: the tunnel that produced our benchmark data (NWG) is the
**noisier** of the two by a factor of 3–4, yet Stock's calibrated limiting
N-factors run the other way (N_TS = 8.0 for NWG, 7.0 for F1, i.e. Mack-equiv
Tu 0.106% and 0.161%). See `paper/expert_feedback.md` and
`agent-paper-review/2026-07-28-2330-spheroid-tu-literature.md`.

## Verification against our digitized replots

`paper/repro/cfd/verify_ercoftac_vs_digitized.py` (re-runnable) checks the
`nwg30` case (alpha = 29.7 deg, Re = 6.54e6) against
`data/stock2006_fig16c_digitized.json`, whose circles are the same condition:

1. **Station geometry:** the 12 true station positions agree with the
   digitized station ladder to **max 0.0020 x/L** — inside the digitizer's
   quoted +/-0.0025.
2. **Front values:** applying the paper's k = 1.5 c_f-rise criterion to the
   numerical c_f(x, phi) reproduces the digitized measured-transition symbols
   to within one station spacing (0.08) wherever the front is single-valued,
   and *exactly* at 3 of 10 azimuths. The larger residuals all sit at the
   multivalued "hook" near phi ~ 52 deg, where the printed front doubles back
   and a single-valued downstream-marching criterion cannot represent it.
3. **The c_f-minimum is NOT the detection convention** — it puts the front at
   the last station for every azimuth at this incidence. Kreplin's hot-film
   detection corresponds to the rise, not the minimum.

Conclusion: the digitization is validated, and where conditions match these
numerical files should be preferred over it.
