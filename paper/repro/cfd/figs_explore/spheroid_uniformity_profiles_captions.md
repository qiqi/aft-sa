# Captions: spheroid uniformity / Cp / BL-profile series (exploratory)

Source: `paper/repro/cfd/spheroid_uniformity_profiles.py` on the CONVERGED
full-body O-grid arm A (`case_ogridfull_L1_saai_re72a0`, 43k steps,
alpha = 0, Re_L = 7.2e6, M = 0.1); numbers for arms B/unstructured in
`spheroid_uniformity_profiles.json`.  Record:
`agent-paper-review/2026-07-28-*-spheroid-uniformity-profiles.md`.

**spheroid_uniformity_rings.png** — Circumferential uniformity: residual
azimuthal variation of the meridional velocity u_s (blue: probe-resolution
residual, 12 samples per azimuthal cell, 25-sample running-mean detrend;
black: per-cell means minus ring mean) and of Cp (green dashed, cell means)
at x/L = 0.20 / 0.42 / 0.70 (rows) and wall-normal heights n = 0.15 / 0.5 /
1.0 delta99 (columns).  Ray origins re-based on the discrete wall facets
(Moller-Trumbore).  All panels are in units of 1e-3: the flow is
azimuthally uniform to a few 1e-5 of u_s (the visible blue band is the
piecewise-linear probe-interpolation ripple at the one-cell scale, not a
flow mode; the black cell-mean line exposes the only coherent content — an
azimuthal mode-1 of amplitude ~1e-5..5e-5 at x/L = 0.70, consistent with
the arm's residual CL = +4.7e-4 transient asymmetry).  The green dCp
staircase is the float32 storage quantum of p in the slice files
(~1.7e-5 in Cp units), not a flow signal.

**spheroid_uniformity_contrast.png** — The extraction hazard displayed
once: the SAME ring (x/L = 0.42, n = 0.15 delta99) probed with rays
anchored on the analytic surface (orange; the 0105-record convention) vs
re-based on the actual wall facets (blue).  The raw convention manufactures
a one-azimuthal-cell mode of rms 3.1e-2 (p2p ~10%) — the facet-sag
wall-distance modulation, not a flow feature; re-basing collapses it ~550x
to rms 5.7e-5.

**spheroid_cp_x.png** — Surface pressure vs x: wall Cp along the
phi = 90 deg meridian (solver surface output), the azimuthal average
(coincides to <= 1.4e-5 for x/L <= 0.85), and the Bernoulli check
1 - (u_e/U_inf)^2 from the BL-edge sweep.  Suction peak (u_e maximum) at
x/L = 0.497; chi = 1 front at 0.858 marked.  Cp = (p - p_inf)/(0.5 M^2),
rho_inf = 1.

**spheroid_bl_profiles.png** — Laminar BL profile series along the
phi = 90 deg meridian: tangential velocity u_t/u_e vs wall-normal distance
n/delta99 at x/L = 0.10 ... 0.88 (the last two stations straddle the
converged chi = 1 front at 0.858).  Surface-normal rays (analytic normal,
facet-re-based origin), identical edge/integral operator as the campaign
extractions.  Thin dashed: the validated axisymmetric implicit laminar-BL
march (Blasius-validated, H = 2.5906) driven by this field's own u_e,
operator-matched, at the stations upstream of the marcher's separation
(~0.87).  Per-panel annotations: H (field vs march), delta99/L, Re_theta,
and the planar-convention kernel-coordinate maximum max_y P.  Bottom:
the same profiles in physical wall distance n/L (log scale).
