"""SA-AI kernel on exact profiles and exact perturbations: four analytic bounds.

Companion to pipe_kernel_analysis.py, same instrument, same style: evaluate the
canonical kernel on profiles whose linear-stability answer is known exactly, and
on exact perturbations of profiles the campaign already computed. No CFD.

  A. ROTATION.  Frame rotation at Omega enters only through the vorticity, so
     Y = d|omega| picks up 2*Omega*d while X = |u| and Z are untouched.
     - Solid-body rotation itself: X = Omega*r, Y = 2*Omega*d, Z = 0, so
       g = (Y - X - Z)/R < 0 whenever d << r  ->  rate clipped to zero.
       The kernel is inert in a rotating freestream, and it is the |u| term
       that makes it so.
     - A genuine layer on a rotating body: dY/Y = 2*Omega/u' ~ 2*delta/L
       for Omega ~ U/L, i.e. O(Re_theta/Re_L).  Quantified per case.

  B. WALL CURVATURE (Goertler / concave and convex).  In curvilinear geometry
     the solver's curvature indicator carries a metric term:
         convex   (fluid at r = R + d):  lap(u).uhat = u'' + u'/(R+d) - u/(R+d)^2
         concave  (fluid at r = R - d):  lap(u).uhat = u'' - u'/(R-d) - u/(R-d)^2
     Leading term:  dZ = +- (d/2R) * Y,  '+' convex (stabilizing, g decreases),
     '-' concave (destabilizing).  Note n.grad|omega| gives the SAME leading
     term here, unlike the pipe of file 01 -- so this result does not depend on
     which curvature realization the solver uses.  Quantified for NLF(1)-0416,
     Eppler 387, the cylinder traverse, and the spheroid.

  C. ATTACHMENT LINE (swept Hiemenz).  Exact similarity solution; kernel
     evaluated against the linear-stability critical sweep Reynolds number.

  D. ASYMPTOTIC SUCTION LAYER.  Exact exponential profile with a known critical
     Re_delta* ~ 5.4e4; also the exact realization of the u'''_w != 0
     transpiration concern of blindspots/02 section 3.

Run: python3 -u blindspots/exact_profile_bounds.py
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'paper', 'repro'))
from lib.sphere_kernel import A_MAX, RAMP_W, reom_crit, sphere_indicators

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')


# ------------------------------------------------------------------ kernel --
def kernel_from_triple(X, Y, Z, ReOm):
    """P, rate, onset, Re_Omega_crit from a triple and a Re_Omega, elementwise.

    Mirrors lib.sphere_kernel.sphere_rate but takes the triple directly, so a
    perturbation dZ can be injected exactly.
    """
    R = np.sqrt(X * X + Y * Y + Z * Z) + 1e-30
    Shat = Y / np.sqrt(X * X + Y * Y + 1e-30)
    g = (Y - X - Z) / R
    P = Shat * g
    rate = A_MAX * np.minimum(1.0, np.clip(P, 0.0, None))
    rc = reom_crit(P)
    onset = 0.5 * (1.0 + np.tanh((ReOm / rc - 1.0) / RAMP_W))
    return dict(P=P, rate=rate, onset=onset, reomc=rc, a=rate * onset)


# ---------------------------------------------------------------- profiles --
_FS_CACHE = {}


def falkner_skan(kind='blasius', eta_max=10.0, n=4001):
    """Similarity profiles, each in the scaling its own Re definition uses.

    kind='blasius':  2 f''' + f f''            = 0,  eta = y sqrt(U/(nu x))
                     f''(0) = 0.33206, so ell = sqrt(nu x / U) and
                     Re_Omega = eta^2 f''(eta) sqrt(Re_x).
    kind='hiemenz':    f''' + f f'' + 1 - f'^2 = 0,  eta = y sqrt(a/nu)
                     f''(0) = 1.23259, so delta = sqrt(nu/a), the length in
                     Rbar = W_e delta / nu.

    Getting these two scalings crossed silently rescales Re_Omega by sqrt(2);
    the peak of P is invariant under it (P is degree-0 homogeneous in the
    triple, and the triple is built on physical y), the gate is NOT.

    Returns eta, f, f', f'', f'''.  Shooting uses a shorter interval with a
    blow-up event, because an off-target f''(0) diverges and an unguarded
    integration to large eta_max spends all its time chasing the divergence.
    """
    key = (kind, eta_max, n)
    if key in _FS_CACHE:
        return _FS_CACHE[key]
    if kind == 'blasius':
        def d3(f, fp, fpp):
            return -0.5 * f * fpp
        bracket = (0.05, 1.5)
    elif kind == 'hiemenz':
        def d3(f, fp, fpp):
            return -f * fpp - (1.0 - fp * fp)
        bracket = (0.5, 3.0)
    else:                                                      # pragma: no cover
        raise ValueError(kind)

    def rhs(_eta, s):
        f, fp, fpp = s
        return [fp, fpp, d3(f, fp, fpp)]

    # Both divergence directions must be caught by events: an off-target
    # f''(0) blows up stiffly and an unguarded solve_ivp never returns.
    def hi(_eta, s):
        return 2.0 - s[1]            # f' overshoots
    hi.terminal = True

    def lo(_eta, s):
        return s[1] + 0.5            # f' dives negative
    lo.terminal = True

    eta_shoot = 8.0

    def miss(fpp0):
        sol = solve_ivp(rhs, (0.0, eta_shoot), [0.0, 0.0, fpp0],
                        rtol=1e-9, atol=1e-11, events=(hi, lo))
        if sol.t_events[0].size:      # overshot
            return +1.0
        if sol.t_events[1].size:      # undershot
            return -1.0
        return sol.y[1, -1] - 1.0

    fpp0 = brentq(miss, *bracket, xtol=1e-12)
    eta = np.linspace(0.0, eta_max, n)
    sol = solve_ivp(rhs, (0.0, eta_max), [0.0, 0.0, fpp0],
                    rtol=1e-10, atol=1e-12, dense_output=True)
    f, fp, fpp = sol.sol(eta)
    fppp = d3(f, fp, fpp)
    _FS_CACHE[key] = (eta, f, fp, fpp, fppp)
    return _FS_CACHE[key]


def blasius_triple(eta, fp, fpp, fppp):
    """(X, Y, Z) on a Blasius-family layer, U_e = 1, y in units of sqrt(nu x/U).

    X = f',  Y = eta f'',  Z = (1/2) eta^2 f'''  -- see the module docstring of
    pipe_kernel_analysis.py for the same reduction on a pipe.
    """
    return fp, eta * fpp, 0.5 * eta * eta * fppp


def swept_hiemenz(eta_max=10.0, n=4001):
    """Attachment-line similarity solution.

    f''' + f f'' + 1 - f'^2 = 0,  f(0)=f'(0)=0, f'(inf)=1   (chordwise)
    g''  + f g' = 0,              g(0)=0,       g(inf)=1     (spanwise)

    eta = y / delta with delta = sqrt(nu/a), a = dUe/dx at the attachment line.
    Returns eta, g, g', g''.
    """
    eta, f, _fp, _fpp, _fppp = falkner_skan('hiemenz', eta_max=eta_max, n=n)
    fspl = CubicSpline(eta, f)

    def rhs(e, s):
        _g, gp = s
        return [gp, -fspl(e) * gp]

    def miss(gp0):
        sol = solve_ivp(rhs, (0.0, eta_max), [0.0, gp0], rtol=1e-11, atol=1e-13)
        return sol.y[0, -1] - 1.0

    gp0 = brentq(miss, 1e-3, 10.0, xtol=1e-13)
    sol = solve_ivp(rhs, (0.0, eta_max), [0.0, gp0], rtol=1e-11, atol=1e-13,
                    dense_output=True)
    g, gp = sol.sol(eta)
    gpp = -fspl(eta) * gp
    return eta, g, gp, gpp


def asbl(s_max=30.0, n=6001):
    """Asymptotic suction layer: u = 1 - exp(-s), s = y/delta, delta = nu/|v_w|.

    delta is also the displacement thickness, so Re_delta* = U delta / nu.
    Returns s, u, y u', (1/2) y^2 u'' with y in units of delta.
    """
    s = np.linspace(0.0, s_max, n)
    e = np.exp(-s)
    return s, 1.0 - e, s * e, -0.5 * s * s * e


# ------------------------------------------------------- geometry utilities --
def surface_curvature(datfile, skip_header=1):
    """Signed radius of curvature R(x/c) of an airfoil surface, upper branch.

    Returns (x_upper, R_upper) with R > 0 meaning convex (curving away from the
    fluid).  Coordinates are splined in arclength; the leading-edge region is
    excluded because the spline there is dominated by point spacing.
    """
    xy = np.loadtxt(os.path.join(ROOT, datfile), skiprows=skip_header)
    x, y = xy[:, 0], xy[:, 1]
    # split at the minimum-x point; keep the branch with positive mean y
    ile = int(np.argmin(x))
    br = [(x[:ile + 1][::-1], y[:ile + 1][::-1]), (x[ile:], y[ile:])]
    xs, ys = max(br, key=lambda b: np.mean(b[1]))
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    keep = np.concatenate(([True], np.diff(xs) > 1e-9))
    xs, ys = xs[keep], ys[keep]
    sp = CubicSpline(xs, ys)
    d1, d2 = sp(xs, 1), sp(xs, 2)
    kappa = np.abs(d2) / (1.0 + d1 * d1) ** 1.5
    with np.errstate(divide='ignore'):
        R = 1.0 / np.maximum(kappa, 1e-12)
    return xs, R


def blasius_length_scale(x_over_c, Re_c):
    """ell = sqrt(nu x / U) in chords, and delta_99 ~ 5 ell."""
    return np.sqrt(x_over_c / Re_c)


# ------------------------------------------------------------------ A: rotation
def rotation_effect(eps_rot, sign=+1.0, kind='blasius'):
    """Relative peak-P shift when dY = sign * 2 eta eps_rot is injected.

    eps_rot = ell/L with ell = sqrt(nu x / U); the perturbation is the frame
    vorticity 2 Omega with Omega = U_inf / L.
    """
    eta, _f, fp, fpp, fppp = falkner_skan(kind)
    X, Y, Z = blasius_triple(eta, fp, fpp, fppp)
    dY = sign * 2.0 * eta * eps_rot
    big = np.full_like(X, 1e9)
    P0 = kernel_from_triple(X, Y, Z, big)['P'].max()
    P1 = kernel_from_triple(X, Y + dY, Z, big)['P'].max()
    return (P1 - P0) / P0


def report_rotation():
    print("=" * 78)
    print("A.  FRAME ROTATION")
    print("=" * 78)
    print("""
Rotation enters the triple ONLY through the vorticity: with Omega the frame (or
solid-body) rotation rate, Y = d|omega| -> Y + 2 Omega d, while X = |u| is
unchanged and Z is unchanged (lap of a solid-body field is zero).

A1. Solid-body rotation on its own is read as STABLE, exactly.
    X = Omega r,  Y = 2 Omega d,  Z = 0   ->  g = (Y - X)/R = (2d - r)/|..|
    so g < 0 for d < r/2 and the rate is clipped to zero.  Numerically:""")
    for d_over_r in (1e-4, 1e-3, 1e-2, 1e-1):
        X, Y, Z = 1.0, 2.0 * d_over_r, 0.0
        k = kernel_from_triple(np.array([X]), np.array([Y]), np.array([Z]),
                               np.array([1e9]))
        print(f"      d/r = {d_over_r:7.0e}:  P = {k['P'][0]:+.4e}   "
              f"rate = {k['rate'][0]:.3e}")
    print("""
    The |u| term is what does this: a rotating freestream has Y/X = 2d/r << 1,
    so the accumulated-inflection coordinate g is negative and the definite
    product P <= 0.  This is the SA-AI counterpart of the property Spalart
    relies on in baseline SA (production built on vorticity, which vanishes in
    an irrotational freestream): here vorticity does NOT vanish under frame
    rotation, but the rate still does, for a different and structural reason.

A2. A genuine boundary layer on a rotating body.  The frame adds 2 Omega to the
    vorticity, so in the normalized triple (U_e = 1, y in units of
    ell = sqrt(nu x / U)) the shear indicator is perturbed by

        dY = 2 Omega d / U_e = 2 eta (ell / L)      for Omega = U_inf / L

    one small parameter eps_rot = ell/L, entering exactly as the curvature term
    of section B does, and with either sign depending on the sense of rotation
    relative to the layer's own vorticity.  Propagated through the kernel to the
    amplification rate itself, not stopped at dY/Y:""")
    # sensitivity coefficient: dP/P per unit eps_rot, in the linear range
    e0 = 1e-5
    slope = rotation_effect(e0, +1.0) / e0
    eta, _f, fp, fpp, fppp = falkner_skan('blasius')
    X, Y, Z = blasius_triple(eta, fp, fpp, fppp)
    big = np.full_like(X, 1e9)
    ipk = int(np.argmax(kernel_from_triple(X, Y, Z, big)['P']))
    dYoverY = 2.0 * eta[ipk] * e0 / Y[ipk]
    print(f"""
    Sensitivity, in the linear range:   d(max P)/P  =  {slope:.2f} * (ell/L)
    At the peak-P height (eta = {eta[ipk]:.2f}) the shear itself only moves by
    dY/Y = {dYoverY/e0:.2f} * (ell/L), so the kernel AMPLIFIES the perturbation by a
    factor {slope/(dYoverY/e0):.1f}.  That is worth recording on its own: g = (Y-X-Z)/R is a
    near-cancellation in the amplifying band, so relative perturbations of the
    shear indicator arrive at P magnified several-fold.  It is the same
    conditioning property that makes the curvature term of section B visible at
    ell/R ~ 1e-3.

    Per case, with L the rotation scale.  Two columns: L = c (one radian of
    frame rotation per chord of travel -- deliberately pessimistic) and
    L = 10c (a rotor-like aspect ratio, chord one tenth of the radius):""")
    cases = [("NLF(1)-0416, a=0", 4.0e6, 0.39), ("Eppler 387", 2.0e5, 0.50),
             ("Eppler 387 (low-Re end)", 6.0e4, 0.50),
             ("Daedalus mid-span", 5.0e5, 0.50),
             ("hypothetical thick layer", 1.0e3, 0.50)]
    print(f"      {'case':26s} {'Re_L':>8s} {'x/c':>5s} {'Re_th':>6s} "
          f"{'ell/c':>9s} {'dP/P, L=c':>11s} {'dP/P, L=10c':>12s}")
    for name, Re, xc in cases:
        ell = blasius_length_scale(xc, Re)
        theta = 0.664 * xc / np.sqrt(xc * Re)       # Blasius theta/c
        d1 = rotation_effect(ell, sign=+1.0)
        d10 = rotation_effect(ell / 10.0, sign=+1.0)
        print(f"      {name:26s} {Re:8.1e} {xc:5.2f} {theta*Re:6.0f} "
              f"{ell:9.2e} {d1:+11.3%} {d10:+12.3%}")
    print("""
    ell/c = sqrt((x/c)/Re_c) and theta/c = Re_theta/Re_c are the same small
    parameter, so the bound IS O(Re_theta/Re_L) as claimed.  But the magnitude
    deserves stating plainly rather than being waved away: at the pessimistic
    L = c the rate moves by 1.3% on the NLF and 6-12% on the Eppler, because of
    the several-fold amplification above.  At a rotor-like L = 10c it is 0.1%
    and 0.6-1.2%.  So:

      - the bound is real and it is small at flight Reynolds numbers;
      - it is NOT negligible at the low-Reynolds-number end, which is exactly
        where the small-rotor application the paper's introduction advertises
        lives.  A drone rotor at chord Reynolds number 6e4 with Omega = U/10c
        carries a ~1% rate error from frame rotation alone;
      - it grows without bound as ell/L -> O(1) (last row), the same
        thick-layer limit file 01 identifies for pipe flow.

    None of this requires a rotating-frame CFD case to establish, and none of it
    threatens the campaign, whose cases are all non-rotating.  What it does is
    convert 'the kernel has no rotation sensor' from an unquantified gap into a
    number that scales, plus a statement about where the number stops being
    small.
""")


# --------------------------------------------------------------- B: curvature
def curvature_effect(eps_list, kind='blasius', concave=False):
    """Peak-P shift when dZ = -+ (d/2R) Y is injected. eps = ell/R.

    Returns list of (eps, P0_peak, P1_peak, dP/P, da/a) where a is rate*onset
    at a representative Re_x (onset saturated), so da/a == dP/P where P < 1.
    """
    eta, _f, fp, fpp, fppp = falkner_skan(kind)
    X, Y, Z = blasius_triple(eta, fp, fpp, fppp)
    sign = -1.0 if concave else +1.0
    out = []
    for eps in eps_list:
        # d/(2R) = eta * ell / (2R) = eta * eps / 2
        dZ = sign * (eta * eps / 2.0) * Y
        k0 = kernel_from_triple(X, Y, Z, np.full_like(X, 1e9))
        k1 = kernel_from_triple(X, Y, Z + dZ, np.full_like(X, 1e9))
        i0 = int(np.argmax(k0['P']))
        P0, P1 = k0['P'][i0], k1['P'].max()
        out.append((eps, P0, P1, (P1 - P0) / P0))
    return out


def report_curvature():
    print("=" * 78)
    print("B.  STREAMWISE WALL CURVATURE  (concave / convex)")
    print("=" * 78)
    eta, _f, fp, fpp, fppp = falkner_skan('blasius')
    X, Y, Z = blasius_triple(eta, fp, fpp, fppp)
    k = kernel_from_triple(X, Y, Z, np.full_like(X, 1e9))
    ipk = int(np.argmax(k['P']))
    print(f"\n  Blasius check: max P = {k['P'][ipk]:.4f} at eta = {eta[ipk]:.3f}"
          f"   (file 01 quotes the Blasius peak as 0.078)")

    print("""
  The metric term dZ = -+ (d/2R) Y is EXACT for a wall-parallel layer on a
  circularly curved wall; the sign is '-' concave (destabilizing) and '+'
  convex (stabilizing).  Injected into the triple at each height, the peak of
  the amplifying coordinate P moves by:

    eps = ell/R      dP/P concave      dP/P convex        (ell = sqrt(nu x/U))""")
    eps_list = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    cc = curvature_effect(eps_list, concave=True)
    cv = curvature_effect(eps_list, concave=False)
    for (e, _p0, _p1, dc), (_e, _q0, _q1, dv) in zip(cc, cv):
        print(f"    {e:8.1e}        {dc:+11.3%}      {dv:+11.3%}")

    print("""
  Now the campaign cases.  R is the surface radius of curvature at the
  amplifying station (splined from the coordinate file where we have one);
  ell = sqrt(nu x / U) at that station; the layer is CONVEX on every one of
  them, so the curvature term is stabilizing and the kernel under-amplifies.
""")
    rows = []
    try:
        xs, R = surface_curvature('data/nlf0416.dat')
        Rspl = CubicSpline(xs, R)
        rows.append(("NLF(1)-0416 upper, a=0", 4.0e6, 0.39,
                     float(Rspl(0.39))))
    except Exception as exc:                                   # pragma: no cover
        print(f"    [nlf0416.dat unavailable: {exc}]")
    try:
        xs, R = surface_curvature('external/construct2d/eppler387.dat')
        Rspl = CubicSpline(xs, R)
        for xc, lbl in ((0.40, "Eppler 387 upper, a=0 (pre-sep)"),
                        (0.52, "Eppler 387 upper, a=0 (sep pt)")):
            rows.append((lbl, 2.0e5, xc, float(Rspl(xc))))
    except Exception as exc:                                   # pragma: no cover
        print(f"    [eppler387.dat unavailable: {exc}]")

    print(f"    {'case':34s} {'Re_L':>8s} {'x/c':>5s} {'R/c':>7s} "
          f"{'ell/R':>9s} {'dP/P':>9s}")
    for name, Re, xc, Rc in rows:
        ell = blasius_length_scale(xc, Re)
        eps = ell / Rc
        dP = curvature_effect([eps], concave=False)[0][3]
        print(f"    {name:34s} {Re:8.1e} {xc:5.2f} {Rc:7.2f} {eps:9.2e} "
              f"{dP:+9.3%}")

    # cylinder: R = D/2, x = R*theta_tr, Re_D given
    print(f"\n    {'cylinder (convex, R=D/2)':34s} {'Re_D':>8s} {'th_tr':>5s} "
          f"{'R/D':>7s} {'ell/R':>9s} {'dP/P':>9s}")
    cyl = [(1e2, 155.0), (3e2, 139.5), (1e4, 120.0), (1e6, 100.0),
           (1e8, 95.0), (1e10, 92.0)]
    for ReD, th in cyl:
        x_over_D = 0.5 * np.deg2rad(th)      # arclength from stagnation, in D
        ell = np.sqrt(x_over_D / ReD)        # in units of D
        eps = ell / 0.5                      # R = D/2
        dP = curvature_effect([eps], concave=False)[0][3]
        print(f"    {'  Re_D = %.0e' % ReD:34s} {ReD:8.1e} {th:5.0f} "
              f"{0.5:7.2f} {eps:9.2e} {dP:+9.3%}")

    # spheroid: transverse curvature dominates; local radius r0 at x/L
    print(f"\n    {'6:1 spheroid (transverse r0)':34s} {'Re_L':>8s} {'x/L':>5s} "
          f"{'r0/L':>7s} {'ell/r0':>9s} {'dP/P':>9s}")
    for xL in (0.3, 0.6, 0.9):
        r0 = (1.0 / 12.0) * np.sqrt(1.0 - (2.0 * xL - 1.0) ** 2)  # a=L/2,b=L/12
        ell = blasius_length_scale(xL, 1.5e6)
        eps = ell / r0
        dP = curvature_effect([eps], concave=False)[0][3]
        print(f"    {'  x/L = %.1f' % xL:34s} {1.5e6:8.1e} {xL:5.2f} "
              f"{r0:7.3f} {eps:9.2e} {dP:+9.3%}")
    print("""
  Reading.  Three separate statements, and the first one matters most:

  1. This term is NOT an error the solver is missing -- it is already IN the
     computed answers.  The solver builds Z from the actual curved geometry, so
     every number above is a decomposition of what the kernel already did, not
     a correction to apply.  What the table gives is how much of each computed
     amplification rate is the curvature metric rather than the profile:
     -0.17% on the NLF, -0.4 to -0.5% on the Eppler, -0.09% on the cylinder at
     Re_D = 1e8, all negligible against the campaign's own 0.01-0.02 c
     mesh-to-mesh scatter on transition location.

  2. It is NOT negligible at the low-Reynolds-number end of the cylinder
     traverse.  At Re_D = 1e4 the curvature metric removes 10% of the rate, and
     at Re_D <= 3e2 it removes essentially all of it (ell/R = 0.13-0.23, the
     perturbation is no longer a perturbation).  The paper claims that traverse
     over ten decades, so the honest remark is that at its low-Re end the
     transition the kernel reports is being set as much by the wall's curvature
     metric as by the profile shape.  Note the traverse still produced a
     transition angle there, so this is about WHY, not about whether.

  3. On a body of revolution the transverse term is realization-dependent.
     For STREAMWISE curvature (airfoil, cylinder) the two candidate
     realizations, lap(u).uhat and n.grad|omega|, give the SAME leading term --
     verified in the docstring algebra -- so those numbers are unambiguous.
     For TRANSVERSE curvature they do not, which is exactly the open
     inconsistency file 01 records for the pipe.  So the spheroid's -3 to -7%
     is better read as a realization UNCERTAINTY in the rate than as a bias.
     (file 01 quotes ~1.6% for the spheroid mid-body; that is delta*/r0,
     against ell/r0 = 0.8% here -- the same geometry in a different thickness
     measure, not a disagreement.)

  4. Concave walls: the sign flips to destabilizing, so the kernel is NOT blind
     to Goertler-type geometry -- blindspots/02 section 5 says 'wall curvature
     radius enters no sensor', and that is wrong.  It enters through the metric
     term in Z.  What is true is a SCALING mismatch: the kernel's response is
     linear in delta/R, whereas the centrifugal instability is governed by the
     Goertler number G = Re_delta sqrt(delta/R).  At fixed delta/R the physical
     instability strengthens with Reynolds number and the kernel's response does
     not move at all, so the gap widens with Re.  Right sign, wrong law.
""")


# ------------------------------------------------------- C: attachment line --
def report_attachment_line():
    print("=" * 78)
    print("C.  ATTACHMENT LINE  (swept Hiemenz, exact)")
    print("=" * 78)
    eta, g, gp, gpp = swept_hiemenz()
    X, Y, Z = g, eta * gp, 0.5 * eta * eta * gpp
    print(f"\n  spanwise profile: g'(0) = {gp[0]:.6f}   "
          f"(g'' = -f g' < 0 everywhere: NO inflection point)")

    def peak_onset(Rbar):
        ReOm = Rbar * eta ** 2 * gp
        k = kernel_from_triple(X, Y, Z, ReOm)
        amax = k['a'].max()
        i = int(np.argmax(k['a']))
        return amax, k, i

    print("""
  eta = y/delta, delta = sqrt(nu/a);  Rbar = W_e delta / nu is the sweep
  Reynolds number.  On the attachment line the chordwise velocity vanishes, so
  the triple is built from the spanwise profile alone:
      X = g,  Y = eta g',  Z = (1/2) eta^2 g''
  and  Re_Omega = Rbar eta^2 g'(eta).

    Rbar        max P     eta(peak a)   Re_Omega     Re_Om,crit    max a""")
    for Rbar in (100.0, 245.0, 583.0, 1000.0, 3000.0, 1e4, 1e5):
        amax, k, i = peak_onset(Rbar)
        print(f"    {Rbar:9.0f}  {k['P'].max():+8.4f}   {eta[i]:9.3f}   "
              f"{Rbar*eta[i]**2*gp[i]:10.1f}   {k['reomc'][i]:10.1f}  "
              f"{amax:8.3e}")

    # critical Rbar where the peak gate reaches 0.5, and the saturated rate
    kk_p = kernel_from_triple(X, Y, Z, np.full_like(X, 1e9))['P'].max()
    a_sat = peak_onset(1e6)[0]
    try:
        Rcrit = brentq(lambda R: peak_onset(R)[1]['onset'].max() - 0.5,
                       10.0, 1e7, xtol=1e-3)
        print(f"\n  kernel's own onset (peak gate = 1/2) at Rbar = {Rcrit:.0f}")
    except ValueError:
        Rcrit = float('nan')
        print("\n  kernel's onset never reached on the scanned Rbar range")
    eta_b, _f, fp, fpp, fppp = falkner_skan('blasius')
    Xb, Yb, Zb = blasius_triple(eta_b, fp, fpp, fppp)
    kb = kernel_from_triple(Xb, Yb, Zb, eta_b ** 2 * fpp * np.sqrt(1e7))
    print(f"""
  Comparison and reading.

  Linear stability of swept Hiemenz (Hall, Malik & Poll) puts the critical
  sweep Reynolds number near Rbar = 583 and Poll's transition criterion near
  Rbar ~ 650 -- VERIFY both against the sources before quoting.  The kernel
  fires at Rbar = {Rcrit:.0f}.  That is within a few percent of the linear-stability
  value, and it was not fitted to it: the threshold's only free scale is the
  Blasius N=1 anchor.

  So blindspots/02 section 5 is wrong on this item too.  It says the model has
  'no concept of' attachment-line transition.  In fact the kernel reads the
  spanwise attachment-line profile as an amplifying shear layer -- max P =
  {kk_p:.4f} against the Blasius {kb['P'].max():.4f}, and a saturated rate of
  {a_sat:.3e} against the Blasius {kb['a'].max():.3e}, i.e. about
  {100*a_sat/kb['a'].max():.0f}% of the Blasius rate -- and it switches on at
  very close to the right Rbar.

  Why it can work at all: the attachment-line boundary layer IS a shear layer,
  and its primary instability is a viscous instability of that layer, so a
  kernel built to read viscous instability of a shear profile is not being
  asked to do anything foreign.  What it cannot know is the crossflow content
  further from the attachment line.

  Caveats, all of which must travel with the number:
   - The reduction assumes the attachment line exactly at x = 0, where the
     chordwise velocity vanishes; away from it the chordwise strain enters and
     this triple is no longer the whole story.
   - The real instability (Goertler-Haemmerlin mode) has structure in the
     chordwise direction that a wall-normal-profile kernel cannot represent, so
     agreement on the THRESHOLD is not agreement on the mechanism.
   - The confound of 02 section 5 remains and is now sharper, not resolved:
     standard SA's spurious attachment-anchored branch and this genuine
     attachment-line reading would both trip a swept leading edge, and telling
     them apart in a computed case still needs the relaminarization quench of
     file 07 or a protocol that excludes the spurious branch.
""")


# ------------------------------------------------- D: asymptotic suction BL --
def report_asbl():
    print("=" * 78)
    print("D.  ASYMPTOTIC SUCTION BOUNDARY LAYER  (exact)")
    print("=" * 78)
    s, u, Y, Z = asbl()
    X = u
    print("""
  u = 1 - exp(-y/delta), delta = nu/|v_w| = displacement thickness.
  u'' = -exp(-s)/delta^2 < 0 everywhere: no inflection point, and
  u'''_w = +1/delta^3 != 0, which is exactly the transpiration violation of
  blindspots/02 section 3 (u'''_w = (v_w/nu) u''_w).  So this profile is both a
  stability anchor and the analytic realization of that concern.

  Kernel reading, Reynolds-number-INDEPENDENT part:""")
    kk = kernel_from_triple(X, Y, Z, np.full_like(X, 1e9))
    print(f"    max over the profile of P = {kk['P'].max():+.3e}")
    print("    P at s = 0.5, 1, 2, 4     = "
          + ", ".join(f"{kk['P'][np.argmin(np.abs(s - t))]:+.3e}"
                      for t in (0.5, 1.0, 2.0, 4.0)))
    print("""
  P <= 0 everywhere, strictly negative away from the wall, so the RATE is
  identically zero and no Reynolds number can make this profile amplify.  The
  gate is irrelevant -- it does open, as the table shows, but it multiplies a
  zero, which is why quoting a 'critical Re' from the gate alone would be wrong:

    Re_delta*    max gate     max rate    max a = rate*gate""")
    for Red in (1e2, 5.2e2, 1e3, 1e4, 5.44e4, 1e5, 1e6):
        ReOm = Red * s ** 2 * np.exp(-s)
        k = kernel_from_triple(X, Y, Z, ReOm)
        print(f"    {Red:10.2e}  {k['onset'].max():9.3f}  {k['rate'].max():11.3e}"
              f"  {k['a'].max():17.3e}")
    print("""
  So the asymptotic suction layer joins the parabola family in the kernel's
  STABLE class: read as stable at every Reynolds number, whereas linear theory
  gives it a critical Re_delta* of about 5.44e4, roughly 105x the Blasius 520
  -- VERIFY against Hocking before quoting.  The error is conservative (too
  laminar), and its practical face is that the model will over-credit HLFC
  suction: a suction-shaped profile is read as indefinitely laminar.

  Two further things this profile settles, both for free:
   - It is the exact realization of the transpiration concern of blindspots/02
     section 3 (u'''_w = (v_w/nu) u''_w != 0 under suction).  The worry there
     was that P might go POSITIVE at the no-slip line, which the paper states
     never happens.  Here P(0) = 0 and P < 0 just off the wall, so for SUCTION
     on this profile the feared sign flip does not occur.  That closes the
     suction half of item 3; blowing (v_w > 0) is the half still open.
   - With plane Poiseuille and plane Couette it fixes the shape of the stable
     class: full, single-signed-curvature, inflection-free profiles are read as
     stable.  That is the model's design intent; the ASBL is the case where the
     intent costs a real, if weak, instability.
""")


# --------------------------------------------------------------------- wall --
def report_wall_sanity():
    """Blasius reference numbers, so the other three sections have a scale."""
    print("=" * 78)
    print("REFERENCE:  Blasius, same instrument")
    print("=" * 78)
    eta, _f, fp, fpp, fppp = falkner_skan('blasius')
    X, Y, Z = blasius_triple(eta, fp, fpp, fppp)
    for Rex in (1e4, 1e5, 1e6, 1e7):
        ReOm = eta ** 2 * fpp * np.sqrt(Rex)
        k = kernel_from_triple(X, Y, Z, ReOm)
        i = int(np.argmax(k['a']))
        print(f"    Re_x = {Rex:7.1e}:  max P = {k['P'].max():.4f}  "
              f"peak a = {k['a'].max():.4e} at eta = {eta[i]:.2f}  "
              f"(Re_Om = {ReOm[i]:8.1f}, crit = {k['reomc'][i]:8.1f})")
    print()


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=False)
    report_wall_sanity()
    report_rotation()
    report_curvature()
    report_attachment_line()
    report_asbl()
