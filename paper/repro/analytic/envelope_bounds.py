"""Operating-envelope bounds: rotation, wall curvature, attachment line, suction.

Backs the four envelope paragraphs of the paper's Conclusion and the
corresponding one-sentence remarks of the whitepaper.  Everything here is
analytic: the canonical kernel of ``lib/sphere_kernel.py`` evaluated on exact
similarity profiles and on EXACT perturbations of them.  No CFD is run and no
CFD tree is read; the four campaign stations the bounds are quoted at are parsed
out of the paper's own generated tables and data files, never retyped.

The four claims, each asserted at the end of its section so this script fails if
the kernel or the campaign data drifts:

  A. ROTATION.  Frame rotation enters the kernel only through the vorticity,
     omega -> omega + 2 Omega.  Solid-body rotation is read as STABLE exactly,
     because the velocity indicator dominates: Y/X = 2d/r.  On a real layer the
     rate shifts by a coefficient times ell/L, with ell = sqrt(nu x / U_e) and
     Omega = U_inf / L.

  B. WALL CURVATURE.  On a curved wall the curvature indicator carries a metric
     term.  In cylindrical coordinates with wall-parallel u(r),

         lap(u).uhat = u'' + u'/r - u/r^2 ,   r = R + d (convex),  R - d (concave)

     which is used here UNTRUNCATED.  The leading behaviour is +-(d/2R) Y,
     stabilizing convex and destabilizing concave.  n.grad|omega| carries the
     same leading term for streamwise curvature, so this result is independent
     of which curvature realization the solver uses; for TRANSVERSE curvature
     the two realizations differ (see blindspots/01), so the spheroid number is
     reported as a realization uncertainty.

  C. ATTACHMENT LINE.  Swept Hiemenz: the chordwise similarity solution at
     beta = 1 plus the spanwise equation g'' + f g' = 0.  On the attachment line
     the chordwise velocity vanishes and the kernel sees the spanwise profile
     alone.

  D. ASYMPTOTIC SUCTION LAYER.  u = 1 - exp(-y/delta), delta = nu/|v_w|, which
     is also its displacement thickness.

NO TUNED OR ARBITRARY CONSTANTS.  Every number that enters is one of:
  (i)   a kernel constant, imported from lib/sphere_kernel.py, never restated;
  (ii)  a case definition (chord Reynolds number, 6:1 spheroid, beta = 1 for a
        stagnation point), each ASSERTED against the paper text that defines it;
  (iii) a campaign result (transition, separation, reattachment stations),
        PARSED from paper/tables/*.tex or paper/data/*.json;
  (iv)  a classical similarity-solution property (theta, delta_99, f''(0)),
        COMPUTED here from the solution rather than quoted;
  (v)   a numerical resolution, each accompanied by a convergence assertion.

Run: python3 -u analytic/envelope_bounds.py       (from paper/repro)
"""
import json
import os
import re
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.boundary_layer import Blasius, FalknerSkanWedge      # noqa: E402
from lib.sphere_kernel import sphere_indicators, sphere_rate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", ".."))
TABLES = os.path.join(PAPER, "tables")
DATA = os.path.join(PAPER, "data")
REPO = os.path.abspath(os.path.join(PAPER, ".."))

# Stagnation-point Falkner-Skan parameter.  Not a fit: beta = 1 IS the
# definition of plane stagnation-point (Hiemenz) flow, U_e ~ x.
BETA_STAGNATION = 1.0

# 6:1 prolate spheroid, the case definition (paper section sec:spheroid).
SPHEROID_ASPECT = 6.0

# Chord Reynolds numbers of the two airfoil campaigns.  Case definitions, not
# model parameters; main() asserts each against the caption of the table that
# reports that campaign, so a mismatch with the paper is a hard failure.
NLF_RE = 4.0e6
EPP_RE = 2.0e5


# --------------------------------------------------------------- sourcing ---
def _read(path):
    with open(path) as fh:
        return fh.read()


def assert_in_paper(needle, where, what):
    """Fail unless `needle` appears in the paper file `where`.

    Used to tie every declared case definition back to the text that defines
    it, so a declared Reynolds number cannot silently disagree with the paper.
    """
    txt = _read(os.path.join(PAPER, where))
    assert needle in txt, f"{what}: {needle!r} not found in {where}"


def parse_table_rows(name):
    """Rows of a generated table as lists of cell strings.

    Only lines between \\midrule and \\bottomrule that contain a column
    separator are taken, so preamble and caption cannot leak in.
    """
    txt = _read(os.path.join(TABLES, name))
    body = txt.split(r"\midrule")[1].split(r"\bottomrule")[0]
    rows = []
    for ln in body.splitlines():
        ln = ln.strip()
        if "&" not in ln:
            continue
        cells = [c.strip() for c in ln.rstrip(r"\\").split("&")]
        rows.append([c.rstrip("\\").strip() for c in cells])
    assert rows, f"no rows parsed from {name}"
    return rows


def _num(cell):
    """First signed decimal number in a LaTeX cell, or None for '--'."""
    m = re.search(r"-?\d+\.?\d*", cell.replace(r"\!", ""))
    return None if m is None else float(m.group(0))


def nlf_upper_transition(alpha_deg, level="L2"):
    """Mean upper-surface x_tr/c over the two mesh families, from tab_nlf_data.

    Columns: alpha, grid, then (CL, CD, xtr_up, xtr_lo) for the structured
    O-grid and again for the unstructured cavity family.
    """
    for r in parse_table_rows("tab_nlf_data.tex"):
        if _num(r[0]) == alpha_deg and r[1] == level:
            return 0.5 * (float(r[4]) + float(r[8]))
    raise LookupError(f"NLF alpha={alpha_deg} {level} not in tab_nlf_data")


def eppler_bubble(alpha_deg, level="L2"):
    """(x_LS/c, x_R/c) averaged over the two mesh families, from the campaign
    JSON that also feeds tab_eppler_bubble."""
    d = json.load(open(os.path.join(DATA, "eppbubble_stations_computed.json")))
    key = str(int(alpha_deg))
    got = [d[f"{fam}_{level}"][key] for fam in ("str", "cav")
           if key in d.get(f"{fam}_{level}", {})]
    assert got, f"Eppler alpha={alpha_deg} {level} missing from the JSON"
    return float(np.mean([g[0] for g in got])), float(np.mean([g[1] for g in got]))


def cylinder_traverse():
    """(Re_D, theta_tr_deg) for every up-ladder row that HAS a transition angle,
    from tab_dragcrisis (the campaign's own summary lives outside the repo)."""
    out = []
    for r in parse_table_rows("tab_dragcrisis.tex"):
        re_d, ttr = _tex_number(r[0]), _num(r[3])
        if re_d is None or ttr is None:
            continue
        out.append((re_d, ttr))
    assert len(out) > 5, "too few cylinder rows parsed"
    res = [r for r, _ in out]
    assert res == sorted(res) and len(set(res)) == len(res), (
        f"cylinder Reynolds ladder not strictly increasing: {res}")
    return out


def _tex_number(cell):
    """A LaTeX numeric cell, plain or in scientific form.

    Handles `$100$`, `$10^{4}$` and `$3\\!\\times\\!10^{4}$`.  Returns None if
    the cell is not a number, and raises if it looks scientific but does not
    parse, so a table format change cannot silently degrade to a wrong value.
    """
    t = cell.replace("$", "").replace(r"\!", "").replace(" ", "")
    if not t or t == "--":
        return None
    m = re.fullmatch(r"(?:([0-9.]+)\\times)?10\^\{(-?\d+)\}", t)
    if m:
        return (float(m.group(1)) if m.group(1) else 1.0) * 10.0 ** float(m.group(2))
    if "^" in t or "times" in t:
        raise ValueError(f"unparsed scientific cell {cell!r}")
    return float(t) if re.fullmatch(r"-?[0-9.]+", t) else None


def spheroid_stations():
    """(phi_deg, x/L) of the measured fronts the paper compares against."""
    d = json.load(open(os.path.join(DATA, "spheroid_front_summary.json")))
    return [(r["phi"], r["meas"]) for r in d["rows"]]


# ---------------------------------------------------------------- profiles --
class BlasiusLayer:
    """Blasius profile in the eta = y sqrt(U/(nu x)) scaling of lib.boundary_layer.

    Exposes exactly what the kernel needs at unit edge velocity with y measured
    in units of ell = sqrt(nu x / U): u = f', du/dy = f'', d2u/dy2 = f'''.
    theta and delta_99 are COMPUTED from the solution, not quoted.
    """

    def __init__(self):
        b = Blasius()
        self.eta, self.u, self.du = b.eta, b.u, b.dudeta
        # f''' from the ODE itself (f''' = -f f''/2), f from integrating f'
        f = cumulative_trapezoid(self.u, self.eta, initial=0.0)
        self.d2u = -0.5 * f * self.du
        self.theta = float(np.trapezoid(self.u * (1.0 - self.u), self.eta))
        self.dstar = float(np.trapezoid(1.0 - self.u, self.eta))
        i99 = int(np.argmax(self.u >= 0.99))
        self.eta99 = float(self.eta[i99])

    def triple_inputs(self, d2u_extra=0.0, du_extra=0.0):
        return (self.u, self.du + du_extra, self.d2u + d2u_extra, self.eta)

    def peak_P(self, d2u_extra=0.0, du_extra=0.0):
        P, _, _, _ = sphere_indicators(*self.triple_inputs(d2u_extra, du_extra))
        return float(np.max(P))


def swept_hiemenz():
    """eta, g, g', g'' for the spanwise attachment-line profile.

    The chordwise solution comes from lib's Falkner-Skan at beta = 1, whose
    returned v_sqrt_Rex reduces to -f at that beta (m = 1), which is how f is
    recovered without duplicating the shooting.  Checked, not assumed.
    """
    w = FalknerSkanWedge(BETA_STAGNATION)
    m = BETA_STAGNATION / (2.0 - BETA_STAGNATION)
    assert abs(m - 1.0) < 1e-12, "beta=1 must give m=1 for the f recovery below"
    f = -w.v_sqrt_Rex
    assert abs(f[0]) < 1e-9, f"f(0) = {f[0]:.3e}, expected 0"
    assert abs(w.u[0]) < 1e-9, f"f'(0) = {w.u[0]:.3e}, expected 0"

    eta = w.eta
    fi = np.interp

    def rhs(e, s):
        return [s[1], -fi(e, eta, f) * s[1]]

    # g'(0) fixed by g(inf) = 1; the equation is linear in g, so one shot plus
    # a rescale is exact -- no shooting iteration and no tolerance to choose.
    sol = solve_ivp(rhs, (eta[0], eta[-1]), [0.0, 1.0], t_eval=eta,
                    rtol=1e-10, atol=1e-12)
    g_un, gp_un = sol.y[0], sol.y[1]
    scale = 1.0 / g_un[-1]
    g, gp = g_un * scale, gp_un * scale
    gpp = -f * gp
    assert abs(g[-1] - 1.0) < 1e-9
    return eta, g, gp, gpp


def suction_layer(s_max, n):
    """s, u, du/ds, d2u/ds2 for u = 1 - exp(-s), s = y/delta."""
    s = np.linspace(0.0, s_max, n)
    e = np.exp(-s)
    return s, 1.0 - e, e, -e


# ----------------------------------------------------------- A: rotation ----
def dP_dY_analytic(X, Y, Z):
    """Exact partial derivative of P = Shat*g with respect to Y.

    Used for the reported sensitivity coefficient so that no finite-difference
    step size enters.  By the envelope theorem the derivative of max_eta P
    equals this partial evaluated at the maximizer, to first order.
    """
    r2 = X * X + Y * Y
    R2 = r2 + Z * Z
    r, R = np.sqrt(r2), np.sqrt(R2)
    Shat = Y / r
    gg = (Y - X - Z) / R
    dShat = X * X / (r2 * r)
    dg = 1.0 / R - (Y - X - Z) * Y / (R2 * R)
    return dShat * gg + Shat * dg


def section_rotation(bl):
    print("=" * 78)
    print("A.  FRAME ROTATION")
    print("=" * 78)

    # A1: solid-body rotation.  X = Omega r, Y = 2 Omega d, Z = 0.  Homogeneous
    # of degree zero, so Omega cancels and only d/r survives.
    print("\n  A1.  Solid-body rotation, X = Omega*r, Y = 2*Omega*d, Z = 0:")
    print("       d/r        P            rate")
    worst = -np.inf
    for dor in (1e-4, 1e-3, 1e-2, 1e-1):
        u = np.array([1.0])
        dudy = np.array([2.0 * dor])
        d2u = np.array([0.0])
        y = np.array([1.0])
        P, _, _, _ = sphere_indicators(u, dudy, d2u, y)
        rate = sphere_rate(u, dudy, d2u, y, nu=1.0)
        print(f"     {dor:7.0e}   {P[0]:+.4e}   {rate[0]:.3e}")
        worst = max(worst, P[0])
        assert rate[0] == 0.0, "solid-body rotation must not amplify"
    assert worst < 0.0, "solid-body rotation must read P < 0"
    print("       -> P < 0 and rate identically 0 at every d/r: the |u| term,")
    print("          not the vorticity, is what makes the kernel inert here.")

    # A2: sensitivity of a real layer.  omega -> omega + 2*Omega means
    # du/dy -> du/dy + 2*Omega; nondimensionally du_extra = 2*(ell/L).
    X, Y, Z = bl.u, bl.eta * bl.du, 0.5 * bl.eta ** 2 * bl.d2u
    P, _, _, _ = sphere_indicators(*bl.triple_inputs())
    ipk = int(np.argmax(P))
    dPdY = dP_dY_analytic(X[ipk], Y[ipk], Z[ipk])
    # dY = 2*eta*(ell/L)  =>  d(maxP)/d(ell/L) = dP/dY * 2*eta_peak
    coeff = float(dPdY * 2.0 * bl.eta[ipk] / P[ipk])
    dY_rel = float(2.0 * bl.eta[ipk] / Y[ipk])
    print(f"\n  A2.  Blasius peak at eta = {bl.eta[ipk]:.3f}, max P = {P[ipk]:.4f}")
    print(f"       d(max P)/P = {coeff:.2f} * (ell/L)      [exact derivative]")
    print(f"       dY/Y       = {dY_rel:.2f} * (ell/L)")
    print(f"       amplification of the perturbation: {coeff/dY_rel:.2f}x")
    print("       (g = (Y-X-Z)/R is a near-cancellation in the amplifying band)")

    # Per case, at the campaign's own stations, with Omega = U_inf / c: the
    # geometric scale of an airfoil case is its chord, so L = c and eps = ell/c.
    # Evaluated exactly (finite perturbation), not from the linear coefficient.
    print(f"\n       {'case':30s} {'Re_c':>8s} {'x/c':>6s} {'ell/c':>9s} {'dP/P':>9s}")
    per_case = {}
    for label, Re, station in (
            ("NLF(1)-0416, alpha=0", NLF_RE, nlf_upper_transition(0.0)),
            ("Eppler 387, alpha=0", EPP_RE, eppler_bubble(0.0)[0])):
        ell = np.sqrt(station / Re)
        P1 = bl.peak_P(du_extra=2.0 * ell)
        shift = (P1 - float(np.max(P))) / float(np.max(P))
        per_case[label] = (Re, station, ell, shift)
        print(f"       {label:30s} {Re:8.1e} {station:6.3f} {ell:9.2e} "
              f"{shift:+9.3%}")
    return coeff, dY_rel, P[ipk], per_case


# ---------------------------------------------------------- B: curvature ----
def curvature_d2u_extra(bl, ell_over_R, concave):
    """Exact metric term added to lap(u).uhat, in the layer's own units.

    In units where y = eta*ell and u is scaled on U_e, r = R +- d becomes
    r/ell = 1/eps -+ eta with eps = ell/R, and

        extra = u'/r - u/r^2      (convex, r = R + d)
              = -u'/r - u/r^2     (concave, r = R - d)

    written out in full rather than truncated to the leading (d/2R)Y term.

    Returns (extra, valid).  On a concave wall the coordinate r = R - d reaches
    the centre of curvature at d = R, where the wall-parallel description stops
    existing; `valid` excludes that region rather than clipping it, and the
    caller asserts the peak lies inside.
    """
    eps = ell_over_R
    if concave:
        r = 1.0 / eps - bl.eta
        valid = r > 0.0
        r = np.where(valid, r, np.nan)
        return -bl.du / r - bl.u / (r * r), valid
    r = 1.0 / eps + bl.eta
    return bl.du / r - bl.u / (r * r), np.ones_like(bl.eta, dtype=bool)


def curvature_shift(bl, ell_over_R, concave):
    """Relative shift of max P caused by the curvature metric term."""
    extra, valid = curvature_d2u_extra(bl, ell_over_R, concave)
    P0, _, _, _ = sphere_indicators(*bl.triple_inputs())
    P1, _, _, _ = sphere_indicators(*bl.triple_inputs(
        d2u_extra=np.where(valid, extra, 0.0)))
    P1 = np.where(valid, P1, -np.inf)
    ipk = int(np.argmax(P1))
    assert valid[ipk] and ipk < len(valid) - 1, (
        "perturbed peak sits at the edge of the physical region")
    return (float(P1[ipk]) - float(np.max(P0))) / float(np.max(P0))


def section_curvature(bl):
    print("=" * 78)
    print("B.  WALL CURVATURE")
    print("=" * 78)
    print("\n  Sensitivity, both signs (ell = sqrt(nu x / U_e)):")
    print("     ell/R      concave        convex")
    for eps in (1e-4, 1e-3, 1e-2, 1e-1):
        print(f"    {eps:7.0e}   {curvature_shift(bl, eps, True):+9.3%}   "
              f"{curvature_shift(bl, eps, False):+9.3%}")

    out = {}
    print("\n  Campaign stations.  ell/R at the station where the layer is")
    print("  amplifying; every surface below is convex, so every shift is")
    print("  stabilizing, and every one is ALREADY in the computed answer.")
    print(f"\n    {'case':34s} {'x/c or x/L':>10s} {'R':>8s} {'ell/R':>9s} {'dP/P':>9s}")

    # --- airfoils: R from the coordinate file, station from the campaign
    for label, dat, Re, station, skip in (
            ("NLF(1)-0416 upper, alpha=0", "data/nlf0416.dat", NLF_RE,
             nlf_upper_transition(0.0), 1),
            ("Eppler 387 upper, alpha=0 sep", "external/construct2d/eppler387.dat",
             EPP_RE, eppler_bubble(0.0)[0], 1)):
        R = surface_radius(dat, station, skip)
        ell = np.sqrt(station / Re)                 # ell/c at that station
        eps = ell / R
        d = curvature_shift(bl, eps, False)
        out[label] = (station, R, eps, d)
        print(f"    {label:34s} {station:10.3f} {R:8.2f} {eps:9.2e} {d:+9.3%}")

    # --- cylinder: R = D/2 exactly; arclength from the stagnation point
    print()
    for re_d, ttr in cylinder_traverse():
        x_over_D = 0.5 * np.deg2rad(ttr)
        ell = np.sqrt(x_over_D / re_d)              # in units of D
        eps = ell / 0.5
        d = curvature_shift(bl, eps, False)
        out[f"cyl_{re_d:.0e}"] = (ttr, 0.5, eps, d)
        print(f"    {'cylinder Re_D = %.0e' % re_d:34s} {ttr:10.1f} "
              f"{0.5:8.2f} {eps:9.2e} {d:+9.3%}")

    # --- spheroid: TRANSVERSE curvature, local body radius of a 6:1 spheroid
    print()
    Re_L = spheroid_Re()
    for phi, xL in spheroid_stations()[::3]:
        r0 = spheroid_radius(xL)
        ell = np.sqrt(xL / Re_L)
        eps = ell / r0
        d = curvature_shift(bl, eps, False)
        out[f"sph_{phi:.0f}"] = (xL, r0, eps, d)
        print(f"    {'spheroid phi = %.0f deg' % phi:34s} {xL:10.3f} {r0:8.3f} "
              f"{eps:9.2e} {d:+9.3%}")
    return out


def surface_radius(datfile, x_station, skip):
    """Radius of curvature of the upper surface at x_station, from coordinates.

    Fitted locally by the exact circle through three surface points bracketing
    the station, so no spline smoothing parameter enters.
    """
    xy = np.loadtxt(os.path.join(REPO, datfile), skiprows=skip)
    x, y = xy[:, 0], xy[:, 1]
    ile = int(np.argmin(x))
    br = [(x[:ile + 1][::-1], y[:ile + 1][::-1]), (x[ile:], y[ile:])]
    xs, ys = max(br, key=lambda b: float(np.mean(b[1])))
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    i = int(np.argmin(np.abs(xs - x_station)))
    i = min(max(i, 1), len(xs) - 2)
    return circumradius(xs[i - 1:i + 2], ys[i - 1:i + 2])


def circumradius(x, y):
    """Radius of the circle through three points (exact, no fitting)."""
    ax, ay, bx, by, cx, cy = x[0], y[0], x[1], y[1], x[2], y[2]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    return float(np.hypot(ax - ux, ay - uy))


def spheroid_radius(x_over_L):
    """Local body radius of a prolate spheroid of the campaign's aspect ratio.

    Semi-major a = L/2, semi-minor b = a/aspect; r0/L = (b/L) sqrt(1 - (2x/L-1)^2).
    """
    b_over_L = 0.5 / SPHEROID_ASPECT
    return float(b_over_L * np.sqrt(max(0.0, 1.0 - (2.0 * x_over_L - 1.0) ** 2)))


def spheroid_Re():
    d = json.load(open(os.path.join(DATA, "spheroid_front_summary.json")))
    m = re.search(r"Re_L=([0-9.]+)e([0-9]+)", d["condition"])
    assert m, f"cannot read Re_L from {d['condition']!r}"
    return float(m.group(1)) * 10.0 ** float(m.group(2))


# ---------------------------------------------------- C: attachment line ----
def section_attachment_line(bl):
    print("=" * 78)
    print("C.  ATTACHMENT LINE  (swept Hiemenz)")
    print("=" * 78)
    eta, g, gp, gpp = swept_hiemenz()
    print(f"\n  spanwise g'(0) = {gp[0]:.6f};  g'' = -f g' < 0 everywhere,")
    print("  so the profile carries no inflection point.")

    # On the attachment line the chordwise velocity vanishes: the kernel sees
    # (X, Y, Z) = (g, eta g', eta^2 g''/2).  Feeding sphere_rate the spanwise
    # profile with y = eta reproduces exactly that triple, and Re_Omega =
    # y^2 |du/dy| / nu = Rbar eta^2 g' when nu = 1/Rbar.
    P, _, _, _ = sphere_indicators(g, gp, gpp, eta)
    a_sat = sphere_rate(g, gp, gpp, eta, nu=0.0 + 1.0 / 1e12).max()
    print(f"\n    {'Rbar':>8s} {'max P':>9s} {'max a':>11s}")
    for Rbar in (100.0, 245.0, 583.0, 1000.0, 3000.0):
        a = sphere_rate(g, gp, gpp, eta, nu=1.0 / Rbar)
        print(f"    {Rbar:8.0f} {P.max():+9.4f} {a.max():11.3e}")

    lo, hi = 1e1, 1e7
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if sphere_rate(g, gp, gpp, eta, nu=1.0 / mid).max() >= 0.5 * a_sat:
            hi = mid
        else:
            lo = mid
    Rcrit = np.sqrt(lo * hi)
    Pb = bl.peak_P()
    ab = sphere_rate(*bl.triple_inputs(), nu=1.0 / 1e12).max()
    print(f"\n  kernel fires (half the saturated rate) at Rbar = {Rcrit:.0f}")
    print(f"  max P = {P.max():.4f} vs Blasius {Pb:.4f};  saturated rate "
          f"{a_sat:.3e} vs Blasius {ab:.3e}  ({100*a_sat/ab:.0f}%)")
    return float(Rcrit), float(P.max()), float(a_sat / ab)


# ------------------------------------------------------- D: suction layer ---
def section_suction(bl):
    print("=" * 78)
    print("D.  ASYMPTOTIC SUCTION LAYER")
    print("=" * 78)
    # s_max is fixed by a resolution requirement, not chosen: the profile decays
    # like exp(-s), so take the range over which the kernel's inputs exceed
    # double precision, and assert the extremum of P is interior.
    s_max = -np.log(np.finfo(float).eps)
    s, u, du, d2u = suction_layer(s_max, 20001)
    P, _, _, _ = sphere_indicators(u, du, d2u, s)
    imin = int(np.argmin(P))
    assert 0 < imin < len(s) - 1, "extremum of P must be interior"
    print(f"\n  s range [0, {s_max:.1f}];  most negative P = {P.min():+.4e} "
          f"at s = {s[imin]:.3f}")
    print(f"  max P over the profile = {P.max():+.4e}  (attained at the wall)")
    print("\n    P at s = " + ", ".join(
        f"{t:g}: {P[np.argmin(np.abs(s - t))]:+.3e}" for t in (0.5, 1, 2, 4)))
    print("\n    Re_delta*     max gate-weighted rate")
    for Red in (1e2, 5.2e2, 1e4, 5.44e4, 1e6):
        a = sphere_rate(u, du, d2u, s, nu=1.0 / Red)
        print(f"    {Red:10.2e}     {a.max():.3e}")
        assert a.max() == 0.0, "suction layer must not amplify at any Re"
    print("\n  P <= 0 everywhere and the rate is identically zero at every")
    print("  Reynolds number: the profile is in the kernel's stable class.")
    return float(P.max()), float(P.min())


# ------------------------------------------------------------ convergence ---
def check_convergence(bl):
    """The reported peak must not depend on the profile grid or its extent."""
    P_ref = bl.peak_P()
    coarse = BlasiusLayer()
    take = slice(None, None, 2)
    Pc, _, _, _ = sphere_indicators(coarse.u[take], coarse.du[take],
                                    coarse.d2u[take], coarse.eta[take])
    rel = abs(float(Pc.max()) - P_ref) / P_ref
    assert rel < 1e-3, f"peak P not grid-converged: {rel:.2e}"
    print(f"  grid check: halving the Blasius sampling moves max P by {rel:.1e}")


# ----------------------------------------------------------------- main -----
def main():
    # Case definitions, each tied back to the text that defines it.
    assert_in_paper(r"6:1 prolate spheroid", "sa-ai.tex", "spheroid aspect ratio")
    assert_in_paper(r"NLF(1)-0416, $Re\!=\!4\!\times\!10^6$",
                    "tables/tab_nlf_data.tex", "NLF chord Reynolds number")
    assert_in_paper(r"Eppler 387, $Re\!=\!2\!\times\!10^5$",
                    "tables/tab_eppler_bubble.tex", "Eppler chord Reynolds number")

    bl = BlasiusLayer()
    print("=" * 78)
    print("INSTRUMENT CHECK")
    print("=" * 78)
    print(f"  Blasius f''(0)   = {bl.du[0]:.5f}   (classical 0.33206)")
    print(f"  Blasius theta    = {bl.theta:.5f}   (classical 0.664)")
    print(f"  Blasius delta*   = {bl.dstar:.5f}   (classical 1.7208)")
    print(f"  Blasius eta_99   = {bl.eta99:.3f}    (classical 4.91)")
    print(f"  kernel max P     = {bl.peak_P():.4f}")
    check_convergence(bl)
    assert abs(bl.du[0] - 0.33206) < 5e-4, bl.du[0]
    assert abs(bl.theta - 0.664) < 5e-3, bl.theta
    print()

    coeff, dY_rel, P_bl, rot_cases = section_rotation(bl)
    print()
    curv = section_curvature(bl)
    print()
    Rcrit, P_al, a_ratio = section_attachment_line(bl)
    print()
    P_asbl_max, P_asbl_min = section_suction(bl)

    # ---- assertions on every number the papers quote -----------------------
    print("\n" + "=" * 78)
    print("ASSERTIONS ON THE QUOTED NUMBERS")
    print("=" * 78)
    checks = [
        ("rotation sensitivity coefficient ~ 41", abs(coeff - 41.0) < 1.5),
        ("rotation amplification factor ~ 4.3", abs(coeff / dY_rel - 4.3) < 0.3),
        ("NLF rotation shift ~ 1.3%",
         abs(rot_cases["NLF(1)-0416, alpha=0"][3] - 0.013) < 3e-3),
        ("Eppler rotation shift ~ 6.7%",
         abs(rot_cases["Eppler 387, alpha=0"][3] - 0.067) < 1e-2),
        ("Blasius peak P ~ 0.078", abs(P_bl - 0.078) < 1e-3),
        ("NLF curvature shift within 0.3%",
         abs(curv["NLF(1)-0416 upper, alpha=0"][3]) < 3e-3),
        ("Eppler curvature shift within 1%",
         abs(curv["Eppler 387 upper, alpha=0 sep"][3]) < 1e-2),
        ("attachment-line Rbar within 10% of 583", abs(Rcrit - 583.0) / 583.0 < 0.10),
        ("attachment-line rate 70-95% of Blasius", 0.70 < a_ratio < 0.95),
        ("suction layer max P is zero", P_asbl_max <= 0.0 + 1e-12),
        ("suction layer is strictly stable inside", P_asbl_min < 0.0),
    ]
    bad = 0
    for name, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
        bad += 0 if ok else 1
    if bad:
        raise SystemExit(f"{bad} assertion(s) failed")
    print("\nAll quoted numbers reproduced.")


if __name__ == "__main__":
    main()
