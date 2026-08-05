"""Two-element potential-flow panel solver and geometry tuner.

Python port of the browser tool, vectorised, with the geometry generators and
two design objectives made explicit so the configuration can be OPTIMISED
rather than dragged by hand.

Purpose: build a REPRODUCIBLE two-element test case for the SA-AI blindspot
campaign -- the wake of an upstream element passing close to a downstream
surface where transition on that downstream surface matters. Both sections are
public NACA definitions, so the geometry can be published in full:

  fore element  NACA 4-digit MODIFIED (Riegels/NACA Report 492 extension):
                leading-edge radius index I and max-thickness position x_m are
                free, on top of camber m, p and thickness t.
  aft element   NACA 4-digit, with max camber m allowed BEYOND the 9% the
                4-digit encoding can express.

Design targets (user, 2026-08-04):
  (1) fore element  Cp as FLAT as possible; a small favorable gradient toward
                    the trailing edge is acceptable, an adverse one is not.
  (2) aft element   NO spiky suction at its leading edge -- the flap nose must
                    not be the global suction peak of that element.

Method: Hess--Smith. Constant-strength source on every panel plus ONE constant
vortex per element; flow tangency at each panel centre, one Kutta condition per
element (Vt_first + Vt_last = 0 traversing lower TE -> LE -> upper TE).
Incompressible, inviscid: this sets the pressure distribution the RANS case
will then be run on. Viscous behaviour is the model's job, not this tool's.

Usage
    python3 panel2e.py --validate                 # NACA 0012 sanity check
    python3 panel2e.py --run                      # solve + plot the default cfg
    python3 panel2e.py --tune                     # optimise for (1) and (2)
    python3 panel2e.py --run --cfg cfg.json --dat # write element .dat files
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict, field

import numpy as np

# --------------------------------------------------------------- geometry --


def naca4_thickness(x: np.ndarray, t: float) -> np.ndarray:
    """Standard NACA 4-digit half-thickness, closed trailing edge."""
    s = np.sqrt(np.maximum(x, 0.0))
    return t*(1.4845*s - 0.6300*x - 1.7580*x**2 + 1.4215*x**3 - 0.5180*x**4)


def naca4mod_thickness(x, t, xm=0.30, ik=6.0, km=1.1, closed_te=True):
    """NACA 4-digit MODIFIED half-thickness.

    xm  position of maximum thickness (fraction of chord)
    ik  leading-edge radius index (6 = standard 4-digit, 0 = sharp, 9 = blunt)
    km  peak-curvature parameter; the curvature at x_m is -km/xm^2 * t

    Forward branch  y_t/t = a0 sqrt(x) + a1 x + a2 x^2 + a3 x^3      (x <= xm)
    Aft branch      y_t/t = d1 w + d2 w^2 + d3 w^3 ,  w = 1 - x      (x >  xm)
    matched in value, slope and curvature at x = xm.
    """
    x = np.asarray(x, float)
    wm = 1.0 - xm
    Km = min(km/(xm*xm), 2.5/(wm*wm))          # cap keeps d1 > 0
    a0 = 1.4845*np.sqrt(ik/6.0)

    R1 = 0.5 - a0*np.sqrt(xm)
    R2 = -a0/(2.0*np.sqrt(xm))
    R3 = -Km + a0/(4.0*xm**1.5)
    a3 = (R1 - R2*xm + 0.5*R3*xm*xm)/xm**3
    a2 = (R3 - 6.0*a3*xm)/2.0
    a1 = R2 - R3*xm + 3.0*a3*xm*xm

    d3 = (0.5 - 0.5*Km*wm*wm)/wm**3
    d2 = -0.5*Km - 3.0*d3*wm
    d1 = Km*wm + 3.0*d3*wm*wm

    s = np.sqrt(np.maximum(x, 0.0))
    w = 1.0 - x
    fwd = a0*s + a1*x + a2*x**2 + a3*x**3
    aft = d1*w + d2*w**2 + d3*w**3
    if not closed_te:
        aft = aft + 0.01                       # finite TE thickness option
    return t*np.maximum(np.where(x <= xm, fwd, aft), 0.0)


def naca_camber(x: np.ndarray, m: float, p: float):
    """Standard NACA 4-digit mean line. m may exceed 0.09 (beyond the code)."""
    x = np.asarray(x, float)
    if m == 0.0 or p <= 0.0:
        return np.zeros_like(x), np.zeros_like(x)
    fwd = x < p
    yc = np.where(fwd,
                  (m/p**2)*(2*p*x - x**2),
                  (m/(1-p)**2)*((1 - 2*p) + 2*p*x - x**2))
    dy = np.where(fwd,
                  (2*m/p**2)*(p - x),
                  (2*m/(1-p)**2)*(p - x))
    return yc, dy



def naca_a_camber(x, cli, a=0.8):
    """NACA a-series (6-series) mean line -- Abbott & von Doenhoff / TR-824.

    Loading is uniform from the leading edge to x = a, then falls linearly to
    zero at the trailing edge. a = 1 is exactly uniform load (but has infinite
    slope at both ends); a ~ 0.6-0.9 is what the 6-series actually uses.

    The 4-digit two-parabola mean line CANNOT give a flat Cp on both surfaces,
    which is why it is replaced here: a rooftop on both sides needs uniform
    loading, and that is what this mean line is for.
    """
    x = np.asarray(x, float)
    xc = np.clip(x, 1e-9, 1.0 - 1e-9)
    if cli == 0.0:
        return np.zeros_like(x), np.zeros_like(x)
    if a >= 0.999:                              # uniform-load closed form
        yc = -(cli/(4.0*np.pi))*((1 - xc)*np.log(1 - xc) + xc*np.log(xc))
    else:
        om = 1.0 - a
        g = -(a*a*(0.5*np.log(a) - 0.25) + 0.25)/om
        h = (0.5*om*om*np.log(om) - 0.25*om*om)/om + g
        ax = np.abs(a - xc)
        ax = np.maximum(ax, 1e-12)
        term = (0.5*(a - xc)**2*np.log(ax)
                - 0.5*(1 - xc)**2*np.log(1 - xc)
                + 0.25*(1 - xc)**2 - 0.25*(a - xc)**2)/om
        yc = (cli/(2.0*np.pi*(a + 1.0)))*(term - xc*np.log(xc) + g - h*xc)
    # slope by central differences on the same abscissae (the LE blend rolls
    # off the end behaviour anyway)
    dy = np.gradient(yc, xc)
    return yc, dy



def airfoil_nodes(n_panels, m, p, t, modified=False, xm=0.30, ik=6.0,
                  km=1.1, le_blend=0.15, mean_line='4digit',
                  cli=0.0, a_ml=0.8, te_thick=0.0):
    """Closed node loop, lower TE -> LE -> upper TE (outward normals).

    le_blend rolls the camber-slope rotation off near the nose so that strongly
    cambered sections do not self-intersect there. Set 0 to disable; it does
    perturb the leading-edge shape, so it is reported in the config.
    """
    npts = n_panels//2 + 1
    beta = np.linspace(0.0, np.pi, npts)
    x = 0.5*(1.0 + np.cos(beta))                       # 1 -> 0, cosine spaced

    yt = (naca4mod_thickness(x, t, xm, ik, km) if modified
          else naca4_thickness(x, t))
    if te_thick > 0.0:
        # open the trailing edge to a finite base: add a linear half-thickness
        # ramp, zero at the LE so the nose is untouched, te_thick/2 at the TE.
        yt = yt + 0.5*te_thick*x
    yc, dyc = (naca_a_camber(x, cli, a_ml) if mean_line == 'aseries'
               else naca_camber(x, m, p))

    blend = np.tanh(np.sqrt(np.maximum(x, 0.0))/le_blend) if le_blend > 0 else 1.0
    th = np.arctan(dyc)*blend
    xl, zl = x + yt*np.sin(th), yc - yt*np.cos(th)
    xu, zu = x - yt*np.sin(th), yc + yt*np.cos(th)

    nodes = np.empty((2*npts - 1, 2))
    nodes[:npts, 0], nodes[:npts, 1] = xl, zl           # lower: TE -> LE
    nodes[npts:, 0], nodes[npts:, 1] = xu[-2::-1], zu[-2::-1]   # upper: LE -> TE
    return nodes


def place(nodes, chord, incidence_deg, x_le, z_le):
    """Scale by chord, rotate by -incidence (nose-up positive), translate."""
    a = np.radians(incidence_deg)
    ca, sa = np.cos(a), np.sin(a)
    xs, zs = nodes[:, 0]*chord, nodes[:, 1]*chord
    return np.column_stack([x_le + xs*ca - zs*sa, z_le + xs*sa + zs*ca])


# ------------------------------------------------------------- config -----


@dataclass
class Config:
    alpha: float = -3.0
    n_panels: int = 160          # per element (even)

    # fore element: NACA 4-digit modified, LE pinned at the origin
    fore_camber: str = 'streamline'   # 'streamline' | 'aseries' | '4digit'
    fore_cam_scale: float = 1.0       # follow the streamline more/less
    fore_cli: float = 0.30      # a-series design lift coefficient
    fore_a: float = 0.80        # a-series loading extent
    fore_m: float = -0.05
    fore_p: float = 0.70
    fore_t: float = 0.10
    fore_xm: float = 0.30
    fore_ik: float = 6.0
    fore_km: float = 1.10
    fore_chord: float = 0.68
    fore_inc: float = 10.0

    # aft element: NACA 4-digit, camber unbounded
    aft_m: float = 0.13
    aft_p: float = 0.40
    aft_t: float = 0.18
    aft_chord: float = 0.30
    aft_inc: float = -15.0
    # aft LE placed explicitly so gap and overlap are free design variables
    aft_x_le: float = 0.71
    aft_z_le: float = 0.078

    le_blend: float = 0.15

    def _flap(self):
        return place(airfoil_nodes(self.n_panels, self.aft_m, self.aft_p,
                                   self.aft_t, False, le_blend=self.le_blend),
                     self.aft_chord, self.aft_inc, self.aft_x_le, self.aft_z_le)

    def _fore_on_streamline(self, n2):
        """Mean line = streamline of the FLAP-ALONE flow through the fore LE.

        User's construction (2026-08-04): solve the flap on its own, trace the
        streamline that would pass through the main element's leading edge, and
        let the main element follow it. The camber then adds no net turning to
        the flap's field, so the lower surface stops having to recover; the
        THICKNESS distribution is then the handle that cancels the residual
        acceleration toward the trailing edge, which sits near the flap's
        suction peak.
        """
        Pf, _ = solve_elements([n2], self.alpha)
        xs, zs = streamline_camber(Pf, 0.0, 0.0, self.fore_chord)

        npts = self.n_panels//2 + 1
        beta = np.linspace(0.0, np.pi, npts)
        xc = 0.5*(1.0 + np.cos(beta))*self.fore_chord       # 1 -> 0 scaled
        zc = np.interp(xc, xs, zs)*self.fore_cam_scale
        dzc = np.gradient(zc, xc)

        yt = naca4mod_thickness(xc/self.fore_chord, self.fore_t, self.fore_xm,
                                self.fore_ik, self.fore_km)*self.fore_chord
        b = (np.tanh(np.sqrt(np.maximum(xc/self.fore_chord, 0.0))/self.le_blend)
             if self.le_blend > 0 else 1.0)
        th = np.arctan(dzc)*b
        xl, zl = xc + yt*np.sin(th), zc - yt*np.cos(th)
        xu, zu = xc - yt*np.sin(th), zc + yt*np.cos(th)
        nd = np.empty((2*npts - 1, 2))
        nd[:npts, 0], nd[:npts, 1] = xl, zl
        nd[npts:, 0], nd[npts:, 1] = xu[-2::-1], zu[-2::-1]
        return nd

    def elements(self):
        if self.fore_camber == 'streamline':
            n2 = self._flap()
            return [self._fore_on_streamline(n2), n2]
        n1 = place(airfoil_nodes(self.n_panels, self.fore_m, self.fore_p,
                                 self.fore_t, True, self.fore_xm, self.fore_ik,
                                 self.fore_km, self.le_blend,
                                 mean_line='aseries', cli=self.fore_cli,
                                 a_ml=self.fore_a),
                   self.fore_chord, self.fore_inc, 0.0, 0.0)
        n2 = place(airfoil_nodes(self.n_panels, self.aft_m, self.aft_p,
                                 self.aft_t, False, le_blend=self.le_blend),
                   self.aft_chord, self.aft_inc, self.aft_x_le, self.aft_z_le)
        return [n1, n2]


# --------------------------------------------------------------- solver ---


class Panels:
    def __init__(self, elements):
        x1, z1, x2, z2, eid = [], [], [], [], []
        for k, nd in enumerate(elements):
            x1.append(nd[:-1, 0]); z1.append(nd[:-1, 1])
            x2.append(nd[1:, 0]);  z2.append(nd[1:, 1])
            eid.append(np.full(len(nd) - 1, k))
        self.x1 = np.concatenate(x1); self.z1 = np.concatenate(z1)
        self.x2 = np.concatenate(x2); self.z2 = np.concatenate(z2)
        self.eid = np.concatenate(eid)
        dx, dz = self.x2 - self.x1, self.z2 - self.z1
        self.L = np.hypot(dx, dz)
        self.tx, self.tz = dx/self.L, dz/self.L
        self.nx, self.nz = -self.tz, self.tx
        self.xc = 0.5*(self.x1 + self.x2); self.zc = 0.5*(self.z1 + self.z2)
        self.n = len(self.L)
        self.n_elem = len(elements)


def influence(px, pz, P: Panels, self_idx=None):
    """Global induced velocities at targets (M,) from every panel -> (M,N)."""
    ct, st = P.tx[None, :], P.tz[None, :]
    ex = px[:, None] - P.x1[None, :]
    ez = pz[:, None] - P.z1[None, :]
    X = ex*ct + ez*st
    Z = -ex*st + ez*ct
    L = P.L[None, :]

    r1 = np.maximum(X*X + Z*Z, 1e-30)
    r2 = np.maximum((X - L)**2 + Z*Z, 1e-30)
    db = np.arctan2(Z, X - L) - np.arctan2(Z, X)
    db = (db + np.pi) % (2*np.pi) - np.pi

    u = (0.25/np.pi)*np.log(r1/r2)
    w = (0.5/np.pi)*db
    if self_idx is not None:                    # exact self-induced values
        rows = np.arange(len(px))
        u[rows, self_idx] = 0.0
        w[rows, self_idx] = 0.5

    uv, wv = w, -u                              # vortex = source rotated 90 deg
    return (u*ct - w*st, u*st + w*ct, uv*ct - wv*st, uv*st + wv*ct)


def solve_elements(els, alpha):
    P = Panels(els)
    a = np.radians(alpha)
    Vx, Vz = np.cos(a), np.sin(a)

    dim = P.n + P.n_elem
    A = np.zeros((dim, dim))
    b = np.zeros(dim)

    us, ws, uv, wv = influence(P.xc, P.zc, P, self_idx=np.arange(P.n))
    A[:P.n, :P.n] = us*P.nx[:, None] + ws*P.nz[:, None]
    vn_v = uv*P.nx[:, None] + wv*P.nz[:, None]
    for k in range(P.n_elem):
        A[:P.n, P.n + k] = vn_v[:, P.eid == k].sum(axis=1)
    b[:P.n] = -(Vx*P.nx + Vz*P.nz)

    for k in range(P.n_elem):
        idx = np.where(P.eid == k)[0]
        f, l = idx[0], idx[-1]
        tgt = np.array([f, l])
        us2, ws2, uv2, wv2 = influence(P.xc[tgt], P.zc[tgt], P, self_idx=tgt)
        tx, tz = P.tx[tgt][:, None], P.tz[tgt][:, None]
        vt_s = (us2*tx + ws2*tz).sum(axis=0)
        vt_v = (uv2*tx + wv2*tz).sum(axis=0)
        r = P.n + k
        A[r, :P.n] = vt_s
        for j in range(P.n_elem):
            A[r, P.n + j] = vt_v[P.eid == j].sum()
        b[r] = -((Vx*P.tx[f] + Vz*P.tz[f]) + (Vx*P.tx[l] + Vz*P.tz[l]))

    sol = np.linalg.solve(A, b)
    q, gam = sol[:P.n], sol[P.n:]

    Vt = Vx*P.tx + Vz*P.tz + (us*P.tx[:, None] + ws*P.tz[:, None]) @ q
    vt_v = uv*P.tx[:, None] + wv*P.tz[:, None]
    for k in range(P.n_elem):
        Vt += gam[k]*vt_v[:, P.eid == k].sum(axis=1)
    P.Vt, P.Cp = Vt, 1.0 - Vt**2

    P.q, P.gam, P.alpha = q, gam, alpha
    cref = max(P.x2.max(), P.x1.max()) - min(P.x1.min(), P.x2.min())
    Fz = (-P.Cp*P.L*P.nz).sum()/cref
    Fx = (-P.Cp*P.L*P.nx).sum()/cref
    Cm = (-P.Cp*P.L*((P.xc - 0.25*cref)*P.nz - P.zc*P.nx)).sum()/cref**2
    res = dict(Cl=Fz*np.cos(a) - Fx*np.sin(a),
               Cd=Fz*np.sin(a) + Fx*np.cos(a), Cm=Cm, cref=cref)
    for k in range(P.n_elem):
        s = P.eid == k
        res[f'Cl{k+1}'] = (-P.Cp[s]*P.L[s]*P.nz[s]).sum()/cref
    return P, res


def solve(cfg: Config):
    return solve_elements(cfg.elements(), cfg.alpha)


def field_velocity(px, pz, P):
    """Velocity at arbitrary points from a solved Panels object."""
    px = np.atleast_1d(np.asarray(px, float))
    pz = np.atleast_1d(np.asarray(pz, float))
    us, ws, uv, wv = influence(px, pz, P)
    a = np.radians(P.alpha)
    Vx = np.cos(a) + us @ P.q
    Vz = np.sin(a) + ws @ P.q
    for k in range(P.n_elem):
        m = P.eid == k
        Vx += P.gam[k]*uv[:, m].sum(axis=1)
        Vz += P.gam[k]*wv[:, m].sum(axis=1)
    return Vx, Vz


def streamline_camber(P, x0, z0, x_end, n=200):
    """Trace the streamline through (x0, z0) to x_end; RK4 in x.

    Used to build the fore element's mean line out of the FLAP-ALONE flow: a
    camber line that follows a streamline of the flap's own field adds no
    net turning to it, so the fore element sees a smooth, already-established
    pressure field instead of imposing its own recovery.
    """
    xs = np.linspace(x0, x_end, n)
    h = xs[1] - xs[0]
    zs = np.empty(n)
    zs[0] = z0
    z = z0
    for i in range(n - 1):
        def slope(xx, zz):
            Vx, Vz = field_velocity(xx, zz, P)
            return float(Vz[0]/(Vx[0] + 1e-30))
        k1 = slope(xs[i], z)
        k2 = slope(xs[i] + 0.5*h, z + 0.5*h*k1)
        k3 = slope(xs[i] + 0.5*h, z + 0.5*h*k2)
        k4 = slope(xs[i] + h, z + h*k3)
        z += (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        zs[i + 1] = z
    return xs, zs


def surfaces(P: Panels, k: int):
    """(lower, upper) panel index arrays for element k, both LE -> TE."""
    idx = np.where(P.eid == k)[0]
    h = len(idx)//2
    return idx[:h][::-1], idx[h:]


# ----------------------------------------------------------- objectives ---


def fore_flatness(P, w0=0.03, w1=0.98, grad_free=0.20):
    """Flatness of BOTH fore-element surfaces, measured LOCALLY.

    The SNLF mechanism needs the fore element laminar on upper AND lower: its
    wake is fed by both, and a turbulent wake convecting over the aft element
    quenches the laminar separation bubble there -- the phenomenon this case
    exists to expose.

    Measured as the LOCAL gradient, not a linear fit. A fit is fooled by a long
    adverse ramp followed by a favorable dip, which nets to zero slope while
    being exactly the profile that transitions. Here any adverse dCp/d(x/c)
    is penalised pointwise, mild favorable gradient is free, and the RMS about
    the mean keeps the level flat.
    """
    lo, up = surfaces(P, 0)
    out, tot = {}, 0.0
    for tag, idx in (('up', up), ('lo', lo)):
        sx = P.xc[idx]
        sx = (sx - sx.min())/(sx.max() - sx.min() + 1e-30)
        m = (sx >= w0) & (sx <= w1)
        if m.sum() < 8:
            return 1e3, dict(rms_up=1e3, rms_lo=1e3, adv_up=1e3, adv_lo=1e3,
                             cp_up=0.0, cp_lo=0.0, dCp=0.0)
        xs, cp = sx[m], P.Cp[idx][m]
        g = np.gradient(cp, xs)
        adv = np.clip(g, 0.0, None)                   # adverse only
        fav = np.clip(-g - grad_free, 0.0, None)      # excessive acceleration
        rms = float(np.sqrt(np.mean((cp - cp.mean())**2)))
        tot += rms + 0.6*float(adv.mean()) + 0.15*float(adv.max()) \
             + 0.10*float(fav.mean())
        out['rms_' + tag] = rms
        out['adv_' + tag] = float(adv.mean())
        out['cp_' + tag] = float(cp.mean())
    out['dCp'] = out['cp_lo'] - out['cp_up']
    return tot, out


def flap_suction(P, grad_cap=12.0, nose=0.30):
    """Gradient-based spike detector for the flap upper surface.

    A leading-edge suction SPIKE is not a high suction LEVEL -- it is -Cp
    shooting up at the nose and recovering steeply immediately behind it, so
    the signature is a large ADVERSE dCp/ds just aft of the nose, per unit
    flap arc length. A clean nose rises to a rooftop and recovers gradually.

    Returns (grad_excess, peak_level); the peak level is REPORTED ONLY, not
    penalised -- user judgement (2026-08-04) is that -Cp ~ 1.7 is acceptable.
    """
    _, up = surfaces(P, 1)
    xs, zs, cp = P.xc[up], P.zc[up], P.Cp[up]
    ds = np.hypot(np.diff(xs), np.diff(zs))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] < 1e-9 or len(s) < 8:
        return 1e3, 0.0
    sn = s/s[-1]
    g = np.gradient(cp, sn)
    front = sn <= nose
    if front.sum() < 3:
        return 1e3, 0.0
    return float(max(0.0, float(np.max(g[front])) - grad_cap)), float((-cp).max())


def slot_geometry(cfg: Config):
    """(overlap, gap) between the fore trailing edge and the flap surface.

    overlap > 0 when the flap leading edge sits AHEAD of the fore TE in x --
    i.e. the elements actually form a slot rather than sitting end to end.
    gap is the shortest distance from the fore TE to the flap contour.
    """
    n1, n2 = cfg.elements()
    te1 = n1[0]                                   # first node is the lower TE
    overlap = te1[0] - cfg.aft_x_le
    gap = float(np.min(np.hypot(n2[:, 0] - te1[0], n2[:, 1] - te1[1])))
    return float(overlap), gap


def objective_single(cfg: Config, els=None, alpha=None,
                     cl_target=0.80, fore_share_min=0.0,
              flap_grad_cap=12.0, gap_lim=(0.015, 0.060), overlap_min=0.005,
              w=(3.0, 3.0, 2.0, 4.0, 20.0, 20.0)):
    """Flatness + flap smoothness, with the constraints that stop the trivial
    solution (unload the fore element, dump all loading on the flap)."""
    try:
        els = cfg.elements() if els is None else els
        P, res = solve_elements(els, cfg.alpha if alpha is None else alpha)
    except np.linalg.LinAlgError:
        return 1e6, None, None
    flat, fl = fore_flatness(P)
    spike, peak = flap_suction(P, grad_cap=flap_grad_cap)
    overlap, gap = slot_geometry(cfg)

    share = res['Cl1']/res['Cl'] if abs(res['Cl']) > 1e-6 else -1.0
    p_share = max(0.0, fore_share_min - share)
    p_cl = abs(res['Cl'] - cl_target)
    p_gap = max(0.0, gap_lim[0] - gap) + max(0.0, gap - gap_lim[1])
    p_ovl = max(0.0, overlap_min - overlap)

    J = (flat + w[1]*spike + w[2]*p_share + w[3]*p_cl
         + w[4]*p_gap + w[5]*p_ovl)
    return J, dict(J=J, flat=flat, peak=peak, spike=spike,
                   share=share, overlap=overlap, gap=gap,
                   **fl, **res), P


# --------------------------------------------------------------- tuning ---

def objective(cfg: Config, das=(-2.0, 0.0, 2.0), **kw):
    """Objective over an ALPHA RANGE, not a single point.

    A very thin fore element can be made flat at one alpha, but its leading
    edge radius is then too small to hold the stagnation point on the nose
    when alpha moves: one surface promptly develops a suction spike and
    transitions, and the whole point of the case -- a laminar wake reaching
    the flap -- is lost. Requiring flatness on BOTH surfaces across a band of
    alpha is what forces a real leading-edge radius, exactly as the width of
    an S702-style low-drag bucket does.

    The GEOMETRY is built once at the design alpha (the streamline mean line
    is traced there); only the flow solution varies over the band.
    """
    els = cfg.elements()
    tot, per, worst = 0.0, [], None
    for da in das:
        J, info, P = objective_single(cfg, els=els, alpha=cfg.alpha + da, **kw)
        if info is None:
            return 1e6, None, None
        per.append((da, J, info, P))
        tot += J
    tot /= len(das)
    worst = max(per, key=lambda r: r[1])
    # penalise the WORST alpha as well as the mean: a band is only as good as
    # its weakest station
    J = 0.5*tot + 0.5*worst[1]
    _, _, info0, P0 = min(per, key=lambda r: abs(r[0]))
    info0 = dict(info0)
    info0['J'] = J
    info0['J_mean'] = tot
    info0['J_worst'] = worst[1]
    info0['da_worst'] = worst[0]
    info0['band'] = [(d, i['adv_up'], i['adv_lo'], i['Cl'])
                     for d, _, i, _ in per]
    return J, info0, P0


TUNABLE = ['fore_cam_scale', 'fore_t', 'fore_xm', 'fore_ik', 'fore_km',
           'fore_chord',
           'aft_m', 'aft_p', 'aft_inc', 'aft_x_le', 'aft_z_le']
BOUNDS = dict(fore_cam_scale=(0.6, 1.4), fore_chord=(0.45, 0.75),
              fore_cli=(-0.2, 0.9), fore_a=(0.3, 0.999),
              fore_m=(-0.12, 0.04), fore_p=(0.2, 0.9),
              fore_t=(0.035, 0.16), fore_xm=(0.20, 0.60),
              fore_ik=(1.0, 9.0), fore_km=(0.5, 2.5), fore_inc=(0.0, 18.0),
              aft_m=(0.02, 0.20), aft_p=(0.15, 0.75), aft_inc=(-28.0, 5.0),
              aft_x_le=(0.60, 0.85), aft_z_le=(-0.02, 0.16))


def tune(cfg: Config, names=None, iters=4000, seed=0, verbose=True,
         n_start=8):
    from scipy.optimize import minimize
    names = names or TUNABLE
    lo = np.array([BOUNDS[n][0] for n in names])
    hi = np.array([BOUNDS[n][1] for n in names])
    x0 = np.array([getattr(cfg, n) for n in names], float)
    x0 = np.clip(x0, lo, hi)

    def unpack(x):
        c = Config(**asdict(cfg))
        for n, v in zip(names, np.clip(x, lo, hi)):
            setattr(c, n, float(v))
        return c

    def f(x):
        return objective(unpack(x))[0]

    rng = np.random.default_rng(seed)
    starts = [x0] + [lo + (hi - lo)*rng.random(len(names))
                     for _ in range(n_start - 1)]
    bx, bf = None, np.inf
    for si, xs in enumerate(starts):
        r = minimize(f, xs, method='Nelder-Mead',
                     options=dict(maxfev=iters, xatol=1e-4, fatol=1e-7))
        if r.fun < bf:
            bf, bx = r.fun, r.x
        if verbose:
            print('    start %d: J=%.5f%s' % (si, r.fun,
                                              '  <-- best' if r.fun == bf else ''))
    c = unpack(bx)
    J, info, P = objective(c)
    if verbose:
        print('  J=%.5f up(rms %.4f adv %.3f) lo(rms %.4f adv %.3f) dCp=%.3f '
              'spike=%.2f Cl=%.3f(sh %.2f) gap=%.4f ovl=%+.4f'
              % (J, info['rms_up'], info['adv_up'], info['rms_lo'],
                 info['adv_lo'], info['dCp'], info['spike'],
                 info['Cl'], info['share'], info['gap'], info['overlap']))
    return c, info, P


# --------------------------------------------------------------- output ---


def write_dat(cfg: Config, outdir='.', stem='twoelement'):
    for k, nd in enumerate(cfg.elements()):
        p = os.path.join(outdir, f'{stem}_elem{k+1}.dat')
        with open(p, 'w') as fh:
            fh.write(f'{stem} element {k+1}\n')
            for x, z in nd:
                fh.write('%12.8f %12.8f\n' % (x, z))
        print('wrote', p)
    p = os.path.join(outdir, f'{stem}.json')
    json.dump(asdict(cfg), open(p, 'w'), indent=2)
    print('wrote', p)


def plot(cfg: Config, P, info, path='twoelement.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(9, 7),
                           gridspec_kw=dict(height_ratios=[1, 1.4]))
    for k, nd in enumerate(cfg.elements()):
        ax[0].plot(nd[:, 0], nd[:, 1], '-', lw=1.4,
                   color=['#0284c7', '#059669'][k])
    ax[0].set_aspect('equal'); ax[0].grid(alpha=.3)
    ax[0].set_title('geometry')
    for k, col in enumerate(['#0284c7', '#059669']):
        lo, up = surfaces(P, k)
        ax[1].plot(P.xc[up], -P.Cp[up], '-', color=col, lw=1.5,
                   label=f'elem {k+1} upper')
        ax[1].plot(P.xc[lo], -P.Cp[lo], '--', color=col, lw=1.1, alpha=.7,
                   label=f'elem {k+1} lower')
    ax[1].axhline(0, color='0.6', lw=.8)
    ax[1].set_xlabel('x'); ax[1].set_ylabel(r'$-C_p$')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title(r'up rms %.4f / lo rms %.4f  $\Delta C_p$ %.3f  '
                    r'spike %.2f  $C_l$ %.3f  gap %.3f'
                    % (info['rms_up'], info['rms_lo'], info['dCp'],
                       info['spike'], info['Cl'], info['gap']))
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print('wrote', path)



def plot_xfoil(cfg: Config, P, info, path='twoelement_cp.pdf', title=None):
    """XFOIL-style figure: Cp above (suction up, axis inverted), geometry
    below at true 1:1 aspect, x axes exactly aligned."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    els = cfg.elements()
    allx = np.concatenate([e[:, 0] for e in els])
    allz = np.concatenate([e[:, 1] for e in els])
    pad = 0.04*(allx.max() - allx.min())
    x0, x1 = allx.min() - pad, allx.max() + pad
    z0, z1 = allz.min() - pad, allz.max() + pad
    X, Z = x1 - x0, z1 - z0

    W = 7.5
    L, R, B, T, GAP = 0.085, 0.015, 0.075, 0.045, 0.055   # figure fractions
    axw = 1.0 - L - R
    geo_h_in = (Z/X)*(axw*W)                    # inches for 1:1 aspect
    cp_h_in = 4.1
    H = geo_h_in + cp_h_in + (B + T + GAP)*6.0

    fig = plt.figure(figsize=(W, H))
    gh, ch = geo_h_in/H, cp_h_in/H
    axg = fig.add_axes([L, B, axw, gh])
    axc = fig.add_axes([L, B + gh + GAP, axw, ch])

    cols = ['#1f4e9c', '#1a8a5a']
    for k, (nd, c) in enumerate(zip(els, cols)):
        lo, up = surfaces(P, k)
        axc.plot(P.xc[up], P.Cp[up], '-', color=c, lw=1.6,
                 label=f'element {k+1} upper')
        axc.plot(P.xc[lo], P.Cp[lo], '--', color=c, lw=1.2, alpha=0.85,
                 label=f'element {k+1} lower')
        axg.plot(nd[:, 0], nd[:, 1], '-', color=c, lw=1.4)
        axg.fill(nd[:, 0], nd[:, 1], color=c, alpha=0.10)

    axc.axhline(0.0, color='0.55', lw=0.8, zorder=0)
    axc.set_xlim(x0, x1)
    axc.invert_yaxis()                          # suction upward, XFOIL style
    axc.set_ylabel('$C_p$')
    axc.set_xticklabels([])
    axc.grid(alpha=0.25, lw=0.6)
    axc.legend(fontsize=7.5, loc='lower right', framealpha=0.9)

    txt = (r'$\alpha$ = %.2f$^\circ$' % cfg.alpha + '\n'
           r'$C_l$ = %.4f' % info['Cl'] + '\n'
           r'$C_m$ = %.4f' % info['Cm'] + '\n'
           r'$C_{l,1}$ = %.3f   $C_{l,2}$ = %.3f' % (info['Cl1'], info['Cl2']))
    axc.text(0.013, 0.035, txt, transform=axc.transAxes, fontsize=8,
             va='bottom', ha='left', family='monospace',
             bbox=dict(fc='white', ec='0.8', lw=0.6, pad=4, alpha=0.92))
    if title:
        axc.set_title(title, fontsize=9.5)

    axg.set_xlim(x0, x1)
    axg.set_ylim(z0, z1)
    axg.set_xlabel('$x/c$')
    axg.set_yticks([])
    for sp in ('left', 'right', 'top'):
        axg.spines[sp].set_visible(False)
    axg.grid(axis='x', alpha=0.25, lw=0.6)

    fig.savefig(path)
    plt.close(fig)
    print('wrote', path)


def validate():
    """NACA 0012, single element, against thin-airfoil and published panel Cl."""
    for al in (0.0, 5.0, 10.0):
        c = Config(alpha=al, fore_m=0.0, fore_p=0.0, fore_t=0.12,
                   fore_chord=1.0, fore_inc=0.0, fore_xm=0.30, fore_ik=6.0,
                   fore_km=1.1)
        n = place(airfoil_nodes(c.n_panels, 0.0, 0.0, 0.12), 1.0, 0.0, 0.0, 0.0)
        P = Panels([n])
        cc = Config(**asdict(c))
        cc.elements = lambda: [n]                       # single element
        _, res = solve(cc)
        print('  NACA0012  alpha=%5.1f   Cl=%7.4f   (2*pi*alpha = %6.4f)'
              % (al, res['Cl'], 2*np.pi*np.radians(al)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--tune', action='store_true')
    ap.add_argument('--cfg')
    ap.add_argument('--dat', action='store_true')
    ap.add_argument('--pdf', default=None,
                    help='write an XFOIL-style Cp+geometry PDF')
    ap.add_argument('--iters', type=int, default=4000)
    a = ap.parse_args()

    cfg = Config(**json.load(open(a.cfg))) if a.cfg else Config()

    if a.validate:
        validate()
    if a.tune:
        print('tuning %d parameters ...' % len(TUNABLE))
        J0, i0, _ = objective(cfg)
        print('  start: J=%.5f up(rms %.4f adv %.3f) lo(rms %.4f adv %.3f) dCp=%.3f '
              'spike=%.2f Cl=%.3f(sh %.2f) gap=%.4f ovl=%+.4f'
              % (J0, i0['rms_up'], i0['adv_up'], i0['rms_lo'],
                 i0['adv_lo'], i0['dCp'], i0['spike'],
                 i0['Cl'], i0['share'], i0['gap'], i0['overlap']))
        cfg, info, P = tune(cfg, iters=a.iters)
        json.dump(asdict(cfg), open('twoelement_tuned.json', 'w'), indent=2)
        print('wrote twoelement_tuned.json')
        plot(cfg, P, info, 'twoelement_tuned.png')
    if a.run:
        J, info, P = objective(cfg)
        print('  Cl=%.4f  Cd=%.5f  Cm=%.4f  (fore %.3f, flap %.3f)'
              % (info['Cl'], info['Cd'], info['Cm'], info['Cl1'], info['Cl2']))
        print('  up(rms %.4f adv %.3f) lo(rms %.4f adv %.3f) dCp=%.3f '
              'spike=%.2f peak=%.2f sh=%.2f gap=%.4f ovl=%+.4f'
              % (info['rms_up'], info['adv_up'], info['rms_lo'],
                 info['adv_lo'], info['dCp'], info['spike'],
                 info['peak'], info['share'], info['gap'],
                 info['overlap']))
        plot(cfg, P, info)
        if a.pdf:
            plot_xfoil(cfg, P, info, a.pdf)
    if a.dat:
        write_dat(cfg)
