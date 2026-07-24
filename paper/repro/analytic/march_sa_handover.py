"""Frozen-profile parabolized march INCLUDING the SA handover.

Extends fig04's laminar-branch march with the full blended right-hand
side -- sigma_P-weighted SA production (f_v1/f_v2), sigma_D-tied
destruction with f_w, and nuHat self-diffusion (lagged) -- so chi(x) can
be watched through the chi=1 crossing on a frozen Falkner-Skan profile.
In these units nu=1, so nuHat IS chi; the seed is the LTPT-class
chi_inf = c_v1 e^-9.

Purpose (author discussion 2026-07-24): (1) reproduce the low-Re_theta
handover STALL (every brake scales as chi/Re_Omega); (2) tune the
amplifying-zone gate
    A = (1-sigma_P) * clip< sign(Shat)*max(0,|Shat|-1/sqrt2) * g >_0^1
applied as exp(-c*A) suspending destruction and the f_v2 correction.
Target: smallest c restoring d(ln chi)/dx slope continuity across the
handover (post/pre slope ratio >= 0.9) on the separated profile at
bubble Re_theta. Guards: Blasius unchanged (A == 0 there); the same
profile at high Re_theta already completes its handover un-gated.

Run from repro/analytic/:  python3 march_sa_handover.py
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import _saai  # noqa: F401
from fig04_shapefactor import C_NU_AI, SIGMA_SA, sphere_rate, profile_ints
from lib.boundary_layer import FalknerSkanWedge

CB1, CV1, KAP = 0.1355, 7.1, 0.41
CB2, SIG = 0.622, 2.0/3.0
CW1 = CB1/KAP**2 + (1.0 + CB2)/SIG
CW2, CW3 = 0.3, 2.0
TAU = 4.0
SEED = CV1*np.exp(-9.0)          # 8.76e-4, the LTPT-class chi_inf
SQ2I = 1.0/np.sqrt(2.0)
CHI_A = 300.0                    # gate amplitude fade scale
GATE_MODE = 'amplify'            # 'suspend' (v5) or 'amplify' (option b)


def march_sa(fs, x_max, nx=1200, ny=800, seed=SEED, gate_c=0.0, ufrac=0.0,
             x_freeze=None):
    """Return xs, Re_theta(x), chi_max(x). With x_freeze, the profile is
    held PARALLEL at that station (v=0, no thickening): Re_Omega stays
    pinned at the frozen profile's value -- the bubble caricature."""
    I_th, H = profile_ints(fs)
    x_ref = x_freeze if x_freeze is not None else x_max
    eta99 = np.interp(0.99, np.maximum.accumulate(fs.u), fs.eta)
    y_top = 8.0*eta99*np.sqrt(x_ref/fs.inviscid_at(x_ref))
    dy = y_top/ny
    yc = (np.arange(ny) + 0.5)*dy
    dx = x_max/nx
    nu = np.ones(ny)*seed
    xs = [0.0]
    chimax = [seed]
    for i in range(nx):
        x = x_freeze if x_freeze is not None else (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, np.arange(ny + 1)*dy, cellCentered=True)
        if x_freeze is not None:
            v = np.zeros_like(v)
        u = np.maximum(u, max(1e-12, ufrac*fs.inviscid_at(x)))
        om = np.abs(dudy)
        chi = nu                                        # nu_mol = 1
        sigP = np.where(chi > 1.0, 1.0 - np.exp(-(chi - 1.0)/TAU), 0.0)
        # amplifying-zone gate A (author-corrected single-factor form)
        d2u = np.gradient(dudy, yc)
        X, Y, Z = u, yc*dudy, 0.5*yc**2*d2u
        R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
        Sh = Y/np.sqrt(X*X + Y*Y + 1e-30)
        gco = (Y - X - Z)/R
        # v4 COLUMN gate, hinge form: the shear surplus <Y-X>_+ (= omega*d
        # exceeding |u|; identical zero set to |Shat|>1/sqrt2 for the
        # non-negative implemented indicators) times the Rayleigh coordinate
        # g -- both vanish at the wall state (X=Y, Z=0), so the gate is
        # quadratically insensitive to sublayer noise and exactly zero in
        # the log layer. Column max: the chi peak detaches from the
        # amplifying locus (v1-v3 falsified by this rig); faded by chi.
        Xn, Yn = X/R, Y/R
        # v5: POINTWISE hinge gate, LARGE c -- the stalled peak's gate value
        # is ~0.05 (small, not zero: measured on the sphere locus), and the
        # safety zeros are exact/quadratic, so c is free to be large.
        A = np.exp(-chi/CHI_A)*np.clip(np.maximum(Yn - Xn, 0.0)*gco, 0.0, 1.0)
        susp = np.exp(-gate_c*A) if GATE_MODE == 'suspend' else np.ones_like(A)
        # option (b): keep OUR amplification live in the gated zone instead
        # of suspending Spalart's sinks -- zero footprint on SA terms.
        w_ai = np.maximum(1.0 - sigP, 1.0 - np.exp(-gate_c*A)) \
            if GATE_MODE == 'amplify' else (1.0 - sigP)
        # SA pieces (lagged in chi)
        fv1 = chi**3/(chi**3 + CV1**3)
        fv2 = 1.0 - chi/(1.0 + chi*fv1)
        St = om + susp*chi*fv2/(KAP**2*yc**2)
        St = np.maximum(St, 0.3*om)
        r = np.clip(chi/(St*KAP**2*yc**2 + 1e-30), 0.0, 10.0)
        gw = r + CW2*(r**6 - r)
        fw = gw*((1.0 + CW3**6)/(gw**6 + CW3**6))**(1.0/6.0)
        sigD = 1.0 - (CB1/(KAP**2*CW1))*(1.0 - sigP)
        Dcoef = susp*sigD*CW1*fw*chi/yc**2              # implicit, per nuHat
        b_ai = sphere_rate(u, dudy, yc)*om              # onset-gated amplification
        Pcoef = w_ai*b_ai + sigP*CB1*St                 # explicit, per nuHat
        # variable diffusion (C_NU_AI*nu + nuHat_lagged)/SIGMA, face-averaged.
        # Gate v3: the SELF-diffusion is the measured stall brake (92% of
        # production at chi~13, ReOm=400) -- suspend it by the gate too.
        knode = (C_NU_AI + susp*chi)/SIGMA_SA/dy**2
        kface = 0.5*(knode[1:] + knode[:-1])
        vp = np.clip(v, 0, None)/dy
        vm = np.clip(-v, 0, None)/dy
        di = vp + vm
        di[:-1] += kface
        di[1:] += kface
        lo = -(vp[1:] + kface)
        up = -(vm[:-1] + kface)
        main = u/dx + di + Dcoef
        rhs = u/dx*nu + Pcoef*nu
        rhs[-1] += vm[-1]*seed
        Amat = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(Amat, rhs)
        xs.append((i + 1)*dx)
        chimax.append(float(nu.max()))
    xs = np.array(xs)
    Ue = fs.inviscid_at(np.maximum(xs, 1e-12))
    return xs, I_th*np.sqrt(xs*Ue), np.array(chimax)


def diagnose(Rt, chim, tag):
    ln = np.log(np.maximum(chim, 1e-30))
    def slope(clo, chi_hi):
        m = (chim > clo) & (chim < chi_hi)
        if m.sum() < 4:
            return np.nan
        return np.polyfit(Rt[m], ln[m], 1)[0]
    s_pre = slope(2e-3, 0.5)
    s_post = slope(2.0, 20.0)
    m30 = np.where(chim >= 30.0)[0]
    m1 = np.where(chim >= 1.0)[0]
    d30 = (Rt[m30[0]] - Rt[m1[0]]) if len(m30) and len(m1) else np.nan
    stall = chim[-1] if not len(m30) else np.nan
    print(f"  {tag}: dlnchi/dRt pre={s_pre:.4g} post={s_post:.4g} "
          f"ratio={s_post/s_pre if s_pre else np.nan:.3f}  "
          f"dRt(1->30)={d30:.0f}  end-chi={chim[-1]:.3g}"
          + (f"  STALL at chi~{stall:.2g}" if np.isfinite(stall) else ""))
    return (s_post/s_pre if s_pre else np.nan), d30


def freeze_station(fs, ReOm_target):
    """x at which the frozen profile's max local Re_Omega = target (nu=1)."""
    from scipy.optimize import brentq
    def f(lx):
        x = 10.0**lx
        eta99 = np.interp(0.99, np.maximum.accumulate(fs.u), fs.eta)
        y = np.linspace(0, 6*eta99*np.sqrt(x/fs.inviscid_at(x)), 800)
        _, u, dudy, _ = fs.at(x, y, cellCentered=True)
        yc = 0.5*(y[1:] + y[:-1])
        return np.max(yc**2*np.abs(dudy)) - ReOm_target
    return 10.0**brentq(f, 1.0, 8.0)


def main():
    sep = FalknerSkanWedge(beta=-0.1988)
    for ReOm in (130.0, 400.0, 1300.0):
        x0 = freeze_station(sep, ReOm)
        # fetch: enough for ~15 e-folds at the amplification rate
        print(f"== PARALLEL separated (H=3.98), pinned max Re_Omega = {ReOm:.0f} ==",
              flush=True)
        cs = (0.0, 30.0, 100.0, 300.0) if ReOm < 200 else (0.0, 100.0)
        for c in cs:
            xs, Rt, ch = march_sa(sep, 40*x0, gate_c=c, ufrac=0.02, x_freeze=x0)
            diagnose(Rt*0 + xs/x0, ch, f"gate c={c:4.0f} (x/x0 units)")
    print("== Blasius guard (growing layer; must be identical) ==", flush=True)
    for c in (0.0, 100.0):
        xs, Rt, ch = march_sa(FalknerSkanWedge(beta=0.0), 8e6, gate_c=c)
        diagnose(Rt, ch, f"gate c={c:4.0f}")


if __name__ == '__main__':
    main()
