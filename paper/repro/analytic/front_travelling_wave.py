"""Travelling-wave speed of the turbulence front on a FROZEN parallel profile.

An artificial transition front placed on a frozen, x-independent velocity
profile has no preferred location -- the setup is translation invariant, so the
linearisation about any steady front carries an exact zero (Goldstone) mode.
What it does have is a preferred SPEED. In the co-moving frame xi = x - c t the
nuHat equation of Eq. (aisa) becomes

    (u(y) - c) d_xi nuHat = (1/sig)[ div((c_nu,ai nu + nuHat) grad nuHat)
                                     + cb2 |grad nuHat|^2 ] + P - D

    nuHat(-inf, y) = 0            (laminar, upstream)
    nuHat(+inf, y) = nuHat_eq(y)  (the xi-independent SA equilibrium on the
                                   same frozen profile: `equilibrium` below)
    nuHat(xi, 0) = 0,   y dnuHat/dy = p nuHat  at y = ymax

The OUTER condition is Robin, not Dirichlet. Above the layer omega = 0 and the
tail obeys nuHat ~ A y^p with p = -1.109 fixed by the constants
(`far_field_exponent`); the amplitude A is free, so y nuHat' = p nuHat holds
for every member of the family and the solution stops caring where the boundary
is. Dirichlet (A = 0) and Neumann (p = 0) are the two corners of that same
condition, which is why they bracket the truth and why each is wrong on its own
side. Measured at y = 1.2 d99 with y_max = 1.5 d99: Dirichlet -10.2%,
Neumann +7.3%, Robin +0.06%. It is also ~10x CHEAPER, because the relaxation
cost goes like y_max^2 and Robin lets the box shrink instead of grow.

and c is a nonlinear eigenvalue: the one frame speed at which the heteroclinic
connection is steady. The problem is irreducibly two-dimensional because
u(y) - c CHANGES SIGN: below the critical layer y_c where u(y) = c the
co-moving advection runs upstream and carries nuHat forward into the laminar
region, while above it SA production regenerates the front. The front therefore
leans downstream with height -- it is bottom-led -- and no reduction that
assumes a single advection speed can see this.

WHAT COMES OUT (frozen Blasius, gate off, Re_theta = 150..4800):

    c = u(y_c)  with  y_c+ = 14.8

i.e. the front rides at the local velocity at the edge of the buffer layer.
c/u_tau = 14.75 +/- 0.02 over a 32x range in Re_theta; on a Blasius layer that
is c/U_e = 8.53 Re_x^(-1/4). The amplification kernel is nearly irrelevant
(<0.26%), but the HANDOVER BLEND is decisive: with sigma_P = sigma_D = 1
(baseline SA) production carries no threshold, nuHat = 0 is linearly unstable,
and the front is PULLED, not pushed -- U_F/u_tau falls 14.75 -> 1.66 and y_F+
falls 14.8 -> 1.7. Baseline SA's pulled speed 2 sqrt(cb1/sigma) u_tau =
0.90 u_tau is what the paper's stagnation section quotes; the two are different
objects, not competing estimates of one. The critical
layer is NOT at fixed chi -- chi_eq(y_c) runs 22 -> 52 across the sweep while
y_c+ does not move -- so the invariant is the wall-unit height, not c_v1.

Because c = u(y_c), the front reverses exactly when the u = 0 line climbs to
y_c ~ 0.24 delta_99: a recirculation a quarter of the way up the layer, not a
near-wall sliver. On the back-flow family of `explore_front_speed.py` that is
u_rev ~ 10%, just under the 12-30% Briggs-Bers band collected in
notes_absolute_instability_lsb.md. Keep that threshold distinct from the
chi << 1 PULLED-front threshold of front_absconv.py (3-6%): the laminar and
turbulent branches reverse at different back-flow levels.

Findings and caveats: notes_front_propagation.md. Figures and sweeps:
explore_front_speed.py. Linear pre-screen: front_absconv.py.

NOT in regenerate_all.py: one converged case is several minutes (the front
relaxes on the SA production time, ~1/(cb1 omega)). `main()` runs a single
coarse-grid Blasius case and asserts the two headline numbers; budget ~4 min.

Units throughout: nu = U_e = 1, so y is in nu/U_e and Re_x = x.

Run from paper/: python3 repro/analytic/front_travelling_wave.py
"""
import _saai  # noqa: F401  (sys.path + cwd=paper/)
import os

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq

from lib.boundary_layer import Blasius, FalknerSkanWedge
from lib.sphere_kernel import C_NU_AI, SIGMA_SA, sphere_rate
# one source for the SA constants and the handover/tie weights
from lib.wall_layer import (CNU, TAU, R_TIE, cb1, cb2, cv1, cw1, cw2, cw3,
                            kap, sig)

assert abs(CNU - C_NU_AI) < 1e-12, "wall-layer c_nu,ai != sphere-kernel c_nu,ai"
assert abs(sig - SIGMA_SA) < 1e-12, "wall-layer sigma != sphere-kernel sigma"


# ----------------------------------------------------------------- geometry
def stretched_y(ymax, ny, y1):
    """Nodes 0 = wall .. ymax, exponential stretch, first spacing ~ y1."""
    def f(s):
        return ymax * np.expm1(s / (ny - 1)) / np.expm1(s) - y1
    s = brentq(f, 1e-6, 60.0)
    return ymax * np.expm1(s * np.linspace(0.0, 1.0, ny)) / np.expm1(s)


def frozen_profile(beta, Re_theta, ny=220, ymax_del=3.0, y1=2.0, backflow=0.0,
                   uniform=False):
    """A Falkner-Skan profile frozen at the station with this Re_theta.

    backflow > 0 subtracts amp*eta*exp(1-eta) (eta = 5 y/delta_99) to open a
    recirculation; it is a knob for the reversal study, not a similarity
    solution -- see notes_front_propagation.md.

    uniform=True returns a constant-spacing grid instead of the stretched one
    (front_absconv.py needs it: eigh_tridiagonal assumes constant dy).
    """
    fs = Blasius() if beta == 0.0 else FalknerSkanWedge(beta)
    eta, ue, dudeta = np.asarray(fs.eta), np.asarray(fs.u), np.asarray(fs.dudeta)
    Rex = (Re_theta / np.trapezoid(ue * (1 - ue), eta)) ** 2
    d99 = np.interp(0.99, ue, eta) * np.sqrt(Rex)
    y = (np.linspace(1, ny, ny) * ymax_del * d99 / ny if uniform
         else stretched_y(ymax_del * d99, ny, y1))
    u = np.interp(y / np.sqrt(Rex), eta, ue, right=1.0)
    dudy = np.interp(y / np.sqrt(Rex), eta, dudeta, right=0.0) / np.sqrt(Rex)
    if backflow:
        e = 5.0 * y / d99
        u = u - backflow * e * np.exp(1.0 - e)
        dudy = np.gradient(u, y)
    d2u = np.gradient(dudy, y)
    ysafe = y.copy()
    ysafe[0] = 0.5 * y[1]          # wall node carries nuHat = 0 anyway
    return dict(y=y, ysafe=ysafe, u=u, dudy=dudy, d2u=d2u, d99=d99, Rex=Rex,
                om=np.abs(dudy) + 1e-30, beta=beta, Re_theta=Re_theta,
                backflow=backflow, utau=np.sqrt(abs(dudy[0])))



def asbl_profile(V_s, ny=220, ymax_del=8.0, y1=None):
    """Asymptotic suction boundary layer: an EXACT parallel Navier-Stokes
    solution with a wall-normal velocity, U = U_e(1 - exp(-y/delta_s)),
    v = -V_s uniform, delta_s = nu/V_s.

    Uniform v is the only way to carry wall-normal flow while keeping
    streamwise translation invariance AND mass conservation:
    du/dx = -dv/dy = 0. Any streamwise acceleration necessarily breaks the
    translation invariance and with it the travelling wave itself.

    One parameter, since u_tau^2 = nu dU/dy|_0 = U_e V_s exactly:
        V_s+ = V_s/u_tau = sqrt(V_s/U_e),   U_e+ = 1/V_s+,   delta_s+ = 1/V_s+
    so suction strength, freestream speed in wall units, and the ceiling
    U_F <= U_e are one and the same knob. Units nu = U_e = 1.
    """
    ds = 1.0 / V_s
    y = stretched_y(ymax_del * ds, ny, y1 if y1 else min(0.02 * ds, 2.0))
    u = 1.0 - np.exp(-y / ds)
    dudy = np.exp(-y / ds) / ds
    d2u = -np.exp(-y / ds) / ds**2
    ysafe = y.copy(); ysafe[0] = 0.5 * y[1]
    return dict(y=y, ysafe=ysafe, u=u, dudy=dudy, d2u=d2u,
                om=np.abs(dudy) + 1e-30, d99=np.interp(0.99, u, y),
                Rex=np.nan, beta=np.nan, Re_theta=np.nan, backflow=0.0,
                utau=np.sqrt(abs(dudy[0])), vn=-V_s, V_s=V_s)

# ------------------------------------------------------------------ sources
def ai_rate(p, gate):
    """Dimensionless a*sigma_Re from the canonical kernel (zero if gate off).

    Gate off makes the laminar state a genuine attractor, which is what turns
    the problem into a clean bistable front; gate on is the physical model.
    The measured c is the same either way.
    """
    if not gate:
        return np.zeros_like(p["y"])
    return sphere_rate(p["u"], p["dudy"], p["d2u"], p["ysafe"], nu=1.0)


def sa_terms(nh, p):
    """Blended production P and the destruction COEFFICIENT (D = Dcoef*nuHat).

    Eq. (aisa_pd): P = max[(1-sigma_P) a sigma_Re omega nuHat,
    sigma_P cb1 Stilde nuHat], D = sigma_D D_SA with the tie sigma_D =
    1 - R_TIE (1 - sigma_P).  Carrying D as a coefficient lets the time
    stepper treat this stiff near-wall sink implicitly.
    """
    y, om = p["ysafe"], p["om"]
    chi = np.maximum(nh, 0.0)
    fv1 = chi**3 / (chi**3 + cv1**3)
    fv2 = 1 - chi / (1 + chi * fv1)
    St = np.maximum(om + chi * fv2 / (kap**2 * y**2), 0.3 * om)   # solver floor
    r = np.clip(chi / (St * kap**2 * y**2 + 1e-300), 0, 10)
    g = r + cw2 * (r**6 - r)
    fw = g * ((1 + cw3**6) / (g**6 + cw3**6)) ** (1 / 6.0)
    if p.get("standard_sa"):
        # sigma_P = sigma_D = 1: baseline SA, no handover blend. Then chi = 0 is
        # LINEARLY UNSTABLE (P = cb1 Stilde chi with no threshold) and the
        # interface is a pulled front, not the pushed front the blend produces.
        sP = sD = 1.0
    else:
        sP = np.maximum(1 - np.exp(-(chi - 1) / TAU), 0.0)
        sD = 1 - R_TIE * (1 - sP)
    # optional multiplier on the SA production branch only: the
    # acceleration-quench gate q(K) of blindspots/07 enters here
    q = p.get("qprod", 1.0)
    P = np.maximum((1 - sP) * p["a_ai"] * om * chi,
                   sP * cb1 * St * chi * q)
    return P, sD * cw1 * fw * chi / y**2, St


# ------------------------------------------------------------------ numerics
def _thomas(a, b, c, d):
    """Tridiagonal solve along the LAST axis (batched over xi)."""
    n = b.shape[-1]
    cp, dp = np.empty_like(b), np.empty_like(b)
    cp[..., 0], dp[..., 0] = c[..., 0] / b[..., 0], d[..., 0] / b[..., 0]
    for i in range(1, n):
        m = b[..., i] - a[..., i] * cp[..., i - 1]
        cp[..., i] = c[..., i] / m
        dp[..., i] = (d[..., i] - a[..., i] * dp[..., i - 1]) / m
    x = np.empty_like(b)
    x[..., -1] = dp[..., -1]
    for i in range(n - 2, -1, -1):
        x[..., i] = dp[..., i] - cp[..., i] * x[..., i + 1]
    return x


def far_field_exponent():
    """Power p in the free-stream tail nuHat ~ A y^p, from the constants.

    Above the boundary layer omega = 0, so Stilde = chi f_v2/(kap^2 d^2) and
    every surviving term scales as y^(2p-2):

        (1/sig)[nuHat nuHat'' + (1+cb2)(nuHat')^2]      (diffusion + cb2)
        + (cb1 f_v2/kap^2) nuHat^2/y^2                  (production)
        - sigma_D cw1 f_w nuHat^2/y^2                   (destruction)  = 0

    The amplitude A cancels -- the tail is a one-parameter family and p is
    fixed. In the tail sigma_D -> 1, f_v2 = 1/(1+chi) -> 0, and r = 1+chi is
    past its clip so f_w saturates, leaving

        (2+cb2) p^2 - p - sig*cw1*f_w = 0   ->   p = -1.109

    Dropping the cb2 term (which scales identically) gives -1.242 and is
    wrong; the measured decay of the tail is -1.108.

    VALIDITY: assumes r > 10 at the boundary, i.e. chi >~ 9 there. Push
    y_max high enough that chi falls below that and f_w, f_v2 start varying,
    and p drifts -- so this condition is most accurate at MODEST y_max.
    """
    r = 10.0                                  # the clip in sa_terms
    g = r + cw2 * (r**6 - r)
    fw = g * ((1 + cw3**6) / (g**6 + cw3**6)) ** (1 / 6.0)
    a_, b_, c_ = (2 + cb2), -1.0, -sig * cw1 * fw
    return (-b_ - np.sqrt(b_**2 - 4 * a_ * c_)) / (2 * a_)


P_FAR = far_field_exponent()


def _ydiff(nh, y, dt, Dcoef, top_bc="dirichlet"):
    """(I - dt*Ly + dt*Dcoef) as tridiagonal bands; Ly conservative, kappa
    lagged (Picard). Dirichlet row at the wall; at the far field either
    Dirichlet (nuHat = 0) or zero-flux Neumann (a half cell with no flux
    through its top face)."""
    kn = (C_NU_AI + np.maximum(nh, 0.0)) / sig
    kf = 0.5 * (kn[..., 1:] + kn[..., :-1])
    dyf = np.diff(y)
    vol = 0.5 * (y[2:] - y[:-2])
    wm = kf[..., :-1] / dyf[:-1] / vol
    wp = kf[..., 1:] / dyf[1:] / vol
    a, b, c = np.zeros(nh.shape), np.ones(nh.shape), np.zeros(nh.shape)
    a[..., 1:-1], c[..., 1:-1] = -dt * wm, -dt * wp
    b[..., 1:-1] = 1 + dt * (wm + wp) + dt * Dcoef[..., 1:-1]
    if top_bc == "neumann":
        wN = kf[..., -1] / dyf[-1] / (0.5 * dyf[-1])
        a[..., -1] = -dt * wN
        b[..., -1] = 1 + dt * wN + dt * Dcoef[..., -1]
    elif top_bc == "robin":
        # y dnuHat/dy = p nuHat, one-sided: an algebraic row, not a march step
        a[..., -1] = -1.0
        b[..., -1] = 1.0 - P_FAR * dyf[-1] / y[-1]
    return a, b, c



def _vadv(nh, y, vn):
    """-v d(nuHat)/dy, upwinded on the sign of v (v<0 = suction toward wall,
    so information arrives from larger y). Returns 0 when vn is 0."""
    if not vn:
        return 0.0
    d = np.zeros_like(nh)
    dy = np.diff(y)
    fwd = (nh[..., 1:] - nh[..., :-1]) / dy
    if vn < 0:
        d[..., :-1] = fwd
        d[..., -1] = fwd[..., -1]
    else:
        d[..., 1:] = fwd
        d[..., 0] = fwd[..., 0]
    return -vn * d

def _cb2_y(nh, y):
    return (cb2 / sig) * np.gradient(nh, y, axis=-1) ** 2


def save_pdf(fig, path):
    """savefig, then PROVE the file opens. A mathtext parse error (or any
    other exception) part-way through savefig leaves a truncated PDF on disk
    that no viewer can open, and the traceback scrolls away in a long run.
    Write to a temporary name, verify, then move into place, so a failed run
    can never replace a good figure with a broken one."""
    tmp = path + ".tmp"
    fig.savefig(tmp, format="pdf")   # explicit: the .tmp suffix hides the type
    try:
        import fitz
        with fitz.open(tmp) as d:
            assert d.page_count >= 1 and d[0].rect.width > 1, "empty PDF"
    except ImportError:
        with open(tmp, "rb") as f:
            head, tail = f.read(5), f.read()[-1024:]
        assert head == b"%PDF-" and b"%%EOF" in tail, f"{path}: malformed PDF"
    os.replace(tmp, path)
    print(f"  wrote {path} ({os.path.getsize(path)/1024:.0f} kB, verified)")


# ------------------------------------- the xi-independent downstream state
def equilibrium_terms(nh, p, top_bc="dirichlet"):
    """The four terms of the xi-independent steady equation, and its residual.

        0 = (1/sig) d/dy[(c_nu,ai nu + nuHat) dnuHat/dy]   <- 'diff'
            + (cb2/sig) (dnuHat/dy)^2                      <- 'cb2'
            + P - D                                        <- 'P', 'D'

    Returned on the interior nodes; the two Dirichlet nodes carry zero.
    """
    y = p["y"]
    P, Dcoef, _ = sa_terms(nh, p)
    kn = (C_NU_AI + np.maximum(nh, 0.0)) / sig
    kf = 0.5 * (kn[1:] + kn[:-1])
    flux = kf * np.diff(nh) / np.diff(y)
    diff = np.zeros_like(nh)
    diff[1:-1] = (flux[1:] - flux[:-1]) / (0.5 * (y[2:] - y[:-2]))
    cb2t = _cb2_y(nh, y)
    D = Dcoef * nh
    R = diff + cb2t + P - D
    R[0] = 0.0
    if top_bc == "neumann":       # half cell, zero flux through the top face
        diff[-1] = -flux[-1] / (0.5 * (y[-1] - y[-2]))
        R[-1] = diff[-1] + cb2t[-1] + P[-1] - D[-1]
    else:
        R[-1] = 0.0           # dirichlet / robin: an imposed constraint
    return dict(diff=diff, cb2=cb2t, P=P, D=D, R=R,
                scale=np.maximum.reduce([np.abs(diff), np.abs(cb2t), P, D]))


def equilibrium(p, gate=False, nsteps=400000, tol=1e-7, verbose=False,
                top_bc="robin"):
    """Solve the xi-independent BVP -- the state the front connects TO.

    This is also the sustain test: on a frozen LAMINAR profile standard SA
    still holds a turbulent branch (chi_max ~ 67 on Blasius at Re_theta = 600,
    growing linearly in Re_theta), so laminar and turbulent are genuinely
    bistable and the front is a real object. Returns (nuHat_eq, iters, resid).
    """
    y = p["y"]
    p["a_ai"] = ai_rate(p, gate)
    # IC: the larger of the log-layer guess and the pointwise SA balance
    # nuHat ~ (cb1/cw1) omega d^2. The log-layer guess ALONE is useless where
    # the wall shear vanishes (separation limit: u_tau = 0 -> identically 0).
    nh = np.maximum(np.minimum(kap * p["utau"] * y, 0.08 * p["utau"] * p["d99"]),
                    (cb1 / cw1) * p["om"] * y**2)
    nh[0] = 0.0
    if top_bc != "neumann":
        nh[-1] = 0.0
    for it in range(nsteps):
        P, Dcoef, St = sa_terms(nh, p)
        dt = 0.4 / max(cb1 * St.max(), 1e-12)
        rhs = nh + dt * (P + _cb2_y(nh, y)
                         + _vadv(nh, y, p.get('vn', 0.0)))
        rhs[0] = 0.0
        if top_bc != "neumann":       # dirichlet and robin rows are algebraic
            rhs[-1] = 0.0
        a, b, c = _ydiff(nh, y, dt, Dcoef, top_bc)
        ab = np.zeros((3, y.size))
        ab[0, 1:], ab[1], ab[2, :-1] = c[:-1], b, a[1:]
        nh = np.maximum(solve_banded((1, 1), ab, rhs), 0.0)
        if nh.max() < 1e-8:
            break
        # Stop on the STEADY residual, not on the per-step change. The latter
        # is small merely because dt is small, and it halted the relaxation an
        # order of magnitude early at high Re_theta (2.7e-4 at Re_theta=4800).
        if it % 500 == 0:
            t = equilibrium_terms(nh, p, top_bc)
            resid = float(np.max(np.abs(t["R"])
                                 / np.maximum(t["scale"], 1e-300)))
            if verbose:
                print(f"    eq it={it} max_chi={nh.max():.5g} "
                      f"resid={resid:.2e}")
            if it > 500 and resid < tol:
                break
    t = equilibrium_terms(nh, p, top_bc)
    resid = float(np.max(np.abs(t["R"]) / np.maximum(t["scale"], 1e-300)))
    return nh, it, resid


# ------------------------------------------------------- the travelling wave
def front_position(nh, xi):
    """xi where max_y chi crosses 1 (0 upstream, turbulent downstream)."""
    m = nh.max(axis=1)
    k = np.where(m >= 1.0)[0]
    if k.size == 0 or k[0] == 0:
        return np.nan
    i = k[0]
    return xi[i - 1] + (1.0 - m[i - 1]) / (m[i] - m[i - 1] + 1e-300) * (
        xi[i] - xi[i - 1])


def travelling_wave(p, nh_eq, gate=False, c0=0.4, nx=400, Lfac=45.0,
                    nsteps=24000, mchk=200, order=2, cfl=0.4,
                    verbose=False, top_bc="robin"):
    """Relax to the travelling wave and return its speed c.

    Semi-implicit: y-diffusion and destruction implicit (one tridiagonal solve
    per xi column), xi-advection sign-aware upwind, production explicit. The
    frame speed is a fixed-point iteration -- d(x_front)/dt measured in the
    current frame IS (c_true - c), so c += drift -- with integer re-centering
    of the window so the front never reaches a boundary.

    order=2 (Beam-Warming upwind) is the default: at order=1 the numerical
    streamwise diffusion |u-c|*dxi/2 biases c low by ~0.4% on the base grid.
    """
    y, u = p["y"], p["u"]
    xi = np.linspace(-0.5, 0.5, nx) * Lfac * p["d99"]
    dxi = xi[1] - xi[0]
    q = dict(p, ysafe=p["ysafe"][None, :], om=p["om"][None, :],
             a_ai=ai_rate(p, gate)[None, :])

    nh = nh_eq[None, :] * (0.5 * (1 + np.tanh(xi / (2 * p["d99"]))))[:, None]
    nh[0], nh[-1] = 0.0, nh_eq
    c, xf_prev, v = c0, front_position(nh, xi), np.nan
    for it in range(nsteps):
        P, Dcoef, St = sa_terms(nh, q)
        adv = u[None, :] - c
        dnb, dnf = np.empty_like(nh), np.empty_like(nh)
        if order == 1:
            dnb[1:] = dnf[:-1] = (nh[1:] - nh[:-1]) / dxi
            dnb[0], dnf[-1] = dnb[1], dnf[-2]
        else:
            dnb[2:] = (3 * nh[2:] - 4 * nh[1:-1] + nh[:-2]) / (2 * dxi)
            dnb[1] = dnb[0] = (nh[1] - nh[0]) / dxi
            dnf[:-2] = (-3 * nh[:-2] + 4 * nh[1:-1] - nh[2:]) / (2 * dxi)
            dnf[-2] = dnf[-1] = (nh[-1] - nh[-2]) / dxi
        dndx = np.where(adv > 0, dnb, dnf)
        kn = (C_NU_AI + np.maximum(nh, 0.0)) / sig
        fl = 0.5 * (kn[1:] + kn[:-1]) * (nh[1:] - nh[:-1]) / dxi
        dxx = np.zeros_like(nh)
        dxx[1:-1] = (fl[1:] - fl[:-1]) / dxi
        cb2t = (cb2 / sig) * (dndx**2 + np.gradient(nh, y, axis=1) ** 2)

        dt = cfl * min(dxi / max(np.abs(adv).max(), 1e-12),
                       dxi**2 / (2 * kn.max()),
                       1.0 / max(cb1 * St.max(), 1e-12))
        rhs = nh + dt * (-adv * dndx + dxx + cb2t + P
                         + _vadv(nh, y, p.get('vn', 0.0)))
        rhs[:, 0] = 0.0
        if top_bc != "neumann":        # dirichlet / robin rows are algebraic
            rhs[:, -1] = 0.0
        nh = np.maximum(_thomas(*_ydiff(nh, y, dt, Dcoef, top_bc), rhs), 0.0)
        nh[0], nh[-1] = 0.0, nh_eq

        if (it + 1) % mchk == 0:
            xf = front_position(nh, xi)
            if not np.isfinite(xf):
                return dict(ok=False, c=np.nan)
            v = (xf - xf_prev) / (mchk * dt)          # = c_true - c
            c += v
            k = -int(round(xf / dxi))
            if k:
                nh = np.roll(nh, k, axis=0)
                nh[:k] = 0.0 if k > 0 else nh[:k]
                if k < 0:
                    nh[k:] = nh_eq
                nh[0], nh[-1] = 0.0, nh_eq
            xf_prev = xf + k * dxi
            if verbose and (it + 1) % (20 * mchk) == 0:
                print(f"    it={it+1:6d} c={c:.5f} dc={v:+.2e} "
                      f"maxchi={nh.max():.1f}")
            if it > 40 * mchk and abs(v) < 1e-6 * max(abs(c), 1e-3):
                break

    # locus of nuHat = half its local equilibrium value, per height
    xfy = np.full(y.size, np.nan)
    for j in range(y.size):
        tgt, col = 0.5 * nh_eq[j], nh[:, j]
        k = np.where(col >= tgt)[0]
        if tgt > 0 and k.size and k[0] > 0:
            i = k[0]
            xfy[j] = xi[i - 1] + (tgt - col[i - 1]) / (
                col[i] - col[i - 1] + 1e-300) * dxi
    x0 = front_position(nh, xi)
    yc = np.interp(c, u[:np.argmax(u >= c) + 1], y[:np.argmax(u >= c) + 1]) \
        if (u >= c).any() else np.nan
    return dict(ok=True, c=c, drift=v, xi=xi - x0, y=y, nh=nh, nh_eq=nh_eq,
                xfy=xfy - x0, yc=yc, dxi=dxi, iters=it)


def solve(beta=0.0, Re_theta=600.0, gate=False, backflow=0.0,
          top_bc="robin", **kw):
    """Profile -> equilibrium -> travelling wave, in one call."""
    p = frozen_profile(beta, Re_theta, ny=kw.pop("ny", 220),
                       backflow=backflow)
    nh_eq, _, _ = equilibrium(p, gate=gate, top_bc=top_bc)
    if nh_eq.max() < cv1:
        return p, nh_eq, dict(ok=False, c=np.nan, reason="branch not sustained")
    return p, nh_eq, travelling_wave(p, nh_eq, gate=gate,
                                     top_bc=top_bc, **kw)


def main():
    """One coarse Blasius case; asserts the two headline numbers. ~4 min."""
    p, nh_eq, r = solve(0.0, 600.0, ny=160, nx=200, nsteps=14000)
    c, ut = r["c"], p["utau"]
    print(f"frozen Blasius Re_theta=600, gate off, coarse grid\n"
          f"  chi_eq(max) = {nh_eq.max():.2f}   (turbulent branch sustained)\n"
          f"  c/U_e       = {c:.5f}\n"
          f"  c/u_tau     = {c/ut:.2f}\n"
          f"  y_c+        = {r['yc']*ut:.2f}\n"
          f"  drift       = {r['drift']:+.1e}   ({r['iters']+1} steps)")
    assert nh_eq.max() > cv1, "turbulent branch must be sustained"
    assert abs(c / ut - 14.75) < 0.15, f"c/u_tau = {c/ut:.3f}, expected 14.75"
    assert abs(r["yc"] * ut - 14.8) < 0.4, f"y_c+ = {r['yc']*ut:.2f}, exp 14.8"
    assert c > 0, "front must run downstream on an attached profile"
    print("  OK: c = u(y_c) with y_c+ = 14.8, front runs downstream")


if __name__ == "__main__":
    main()
