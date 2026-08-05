"""Five-row surface diagnostic for the two-element case, in the same style as
the paper's 2D airfoil figures (paper/repro/cfd/regen_eppler_v2.make_cf_figure
and regen_epp_resweep_suite).

Rows, top to bottom:
    1  max Re_Omega       = max over a wall-normal probe of d^2 |omega| / nu
    2  max Omega_hat I_hat  the sphere-kernel rate coordinate
    3  max chi            = max nuHat / muRef, with the e^9 seed and c_v1 marked
    4  -Cp
    5  Cf (signed)

Conventions taken from the paper figures: upper surface blue (C0), lower red
(C3), refinement level by line width, and here the ELEMENT by line style --
fore solid, flap dashed -- with both elements drawn on the same axes against
global x, as asked.

Kernel coordinates, matching paper/repro/cfd/add_derived_to_slice.py exactly:

    X = |u|,  Y = omega d,  Z = (1/2) d^2 (n . grad omega)
    Omega_hat = Y / sqrt(X^2 + Y^2)
    I_hat     = (Y - X - Z) / sqrt(X^2 + Y^2 + Z^2)

with n the outward wall normal. The normal derivative of omega is taken ALONG
THE PROBE by finite differences rather than through a VTK gradient filter on
grad(wallDistance): the probe direction IS the wall normal, so this is the same
quantity computed more directly and without a second interpolation.

nu = muRef = Mach / Re, read from the case's Flow360.json rather than assumed.

Run:  python3 plot_paper_style.py out.pdf case_dir [case_dir ...]
"""
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D

import plot_solution_mesh as PM
import plot_cp_cf as C
from measure_l1_spacing import read_contours, surface_frame

UP_COLOR, LO_COLOR = 'C0', 'C3'
LEVEL_LW = {'L0': 0.8, 'L1': 1.6, 'L2': 2.4}
ELEM_LS = {'fore': '-', 'flap': '--'}
C_V1 = 7.1
N_LO, N_HI = -2.0, 12.0
CHI_INF = C_V1*np.exp(-9.0)
CHI_LO, CHI_HI = CHI_INF*np.exp(N_LO), CHI_INF*np.exp(N_HI)
REOMC_FLOOR = 124.6                 # sphere-kernel onset floor k*A
L_PROBE, N_PROBE = 0.01, 80         # 0.01 total chord, as the paper figures use
ELEMS = ('fore', 'flap')


def nu_of(case):
    """muRef = Mach/Re from the solver input, not assumed."""
    p = '%s/Flow360.json' % case
    if os.path.exists(p):
        d = json.load(open(p))
        try:
            return float(d['freestream']['muRef'])
        except (KeyError, TypeError):
            pass
    return 0.10/1.0e6


class Field:
    """Mid-plane interpolants for the quantities the kernel coordinates need."""

    def __init__(self, case):
        g, pts, arr = PM.read_vtu('%s/volume_proc0.vtu' % case)
        idx, tris, P2 = PM.midplane_tris(g, pts)
        T = mtri.Triangulation(P2[:, 0], P2[:, 1], tris)
        vel = arr['velocity'][idx]
        self.f = {}
        self.f['q'] = mtri.LinearTriInterpolator(
            T, np.hypot(vel[:, 0], vel[:, 2]))
        self.f['u'] = mtri.LinearTriInterpolator(T, vel[:, 0])
        self.f['w'] = mtri.LinearTriInterpolator(T, vel[:, 2])
        for nm, key in (('omega', 'vorticityMagnitude'), ('d', 'wallDistance'),
                        ('nuHat', 'nuHat')):
            if key not in arr:
                raise KeyError('%s missing %s -- rerun with the extended '
                               'VOLUME_FIELDS' % (case, key))
            self.f[nm] = mtri.LinearTriInterpolator(T, arr[key][idx])

    def __call__(self, nm, x, z):
        v = self.f[nm](np.atleast_1d(x), np.atleast_1d(z))
        return np.asarray(np.ma.filled(v, np.nan), float)


def probe_maxima(fld, cont, ile, u2, nu):
    """Wall-normal probe maxima per surface station.

    Returns dict side -> (x, Re_Omega_max, OmegaI_max, chi_max)."""
    n = len(cont)
    t = cont[np.minimum(np.arange(n)+1, n-1)] - cont[np.maximum(np.arange(n)-1, 0)]
    t /= np.maximum(np.linalg.norm(t, axis=1), 1e-30)[:, None]
    t *= np.where(np.arange(n) >= ile, 1.0, -1.0)[:, None]
    nrm = np.column_stack([-t[:, 1], t[:, 0]])
    c0 = cont.mean(axis=0)
    nrm[((cont - c0)*nrm).sum(1) < 0] *= -1.0

    dist = np.linspace(1e-6, L_PROBE, N_PROBE)
    out = {}
    second = np.arange(n) >= ile
    upper = second if u2 else ~second
    for side, msk in (('upper', upper), ('lower', ~upper)):
        ii = np.where(msk)[0]
        ii = ii[(ii > 0) & (ii < n-1)]
        if len(ii) < 10:
            continue
        P0, NN = cont[ii], nrm[ii]
        X = P0[None, :, 0] + dist[:, None]*NN[None, :, 0]
        Z = P0[None, :, 1] + dist[:, None]*NN[None, :, 1]
        q = fld('q', X.ravel(), Z.ravel()).reshape(X.shape)
        om = fld('omega', X.ravel(), Z.ravel()).reshape(X.shape)
        dd = fld('d', X.ravel(), Z.ravel()).reshape(X.shape)
        nh = fld('nuHat', X.ravel(), Z.ravel()).reshape(X.shape)
        # n . grad omega along the probe: the probe direction IS the normal.
        # omega comes from a C0 interpolant, so differencing it raw makes the
        # rate coordinate visibly jittery; smooth lightly along the probe first.
        k = np.ones(5)/5.0
        oms = np.apply_along_axis(
            lambda a: np.convolve(np.pad(a, 2, mode='edge'), k, mode='valid'),
            0, om)
        dom = np.gradient(oms, dist, axis=0)
        Xk, Yk = q, om*dd
        Zk = 0.5*dd*dd*dom
        R = np.sqrt(Xk*Xk + Yk*Yk + Zk*Zk) + 1e-30
        Om_hat = Yk/(np.sqrt(Xk*Xk + Yk*Yk) + 1e-30)
        I_hat = (Yk - Xk - Zk)/R
        with np.errstate(invalid='ignore'):
            reo = dd*dd*om/nu
            oi = Om_hat*I_hat
            chi = nh/nu
        f = lambda A: np.nanmax(np.where(np.isfinite(A), A, -np.inf), axis=0)
        out[side] = (P0[:, 0], f(reo), f(oi), f(chi))
    return out


def surface_cp_cf(case, nm, cont, ile, u2, fld=None, offset=2.0e-4):
    """Cp and SIGNED Cf per side, from the surface output. Uses CfVec if it is
    present -- that carries the direction directly, so no sign has to be
    inferred from the near-wall velocity."""
    g, pts, arr = PM.read_vtu('%s/surface_fluid_%s_proc0.vtu' % (case, nm))
    nun = [len(np.unique(np.round(pts[:, i], 9))) for i in range(3)]
    span = int(np.argmin(nun))
    keep = np.isclose(pts[:, span], np.unique(pts[:, span])[0], atol=1e-9)
    P = np.delete(pts[keep], span, axis=1)
    cp = arr['Cp'][keep]
    cfm = arr['Cf'][keep]
    if cfm.ndim > 1:
        cfm = np.linalg.norm(cfm, axis=1)
    j, upper = C.order_on_contour(P, cont, ile, u2)
    n = len(cont)
    t = cont[np.minimum(np.arange(n)+1, n-1)] - cont[np.maximum(np.arange(n)-1, 0)]
    t /= np.maximum(np.linalg.norm(t, axis=1), 1e-30)[:, None]
    t *= np.where(np.arange(n) >= ile, 1.0, -1.0)[:, None]
    tg = t[j]
    if 'CfVec' in arr:
        cv = arr['CfVec'][keep]
        cv2 = np.delete(cv, span, axis=1) if cv.shape[1] == 3 else cv
        cfs = cv2[:, 0]*tg[:, 0] + cv2[:, 1]*tg[:, 1]
    elif fld is not None:
        # CfVec is requested in the case JSON but the solver does not write it
        # to the surface vtu, so recover the sign from the near-wall tangential
        # velocity. Returning |Cf| here instead silently hides every separation.
        nn = np.column_stack([-tg[:, 1], tg[:, 0]])
        c0 = cont.mean(axis=0)
        nn[((P - c0)*nn).sum(1) < 0] *= -1.0
        qp = P + offset*nn
        uu = fld('u', qp[:, 0], qp[:, 1])
        ww = fld('w', qp[:, 0], qp[:, 1])
        dot = uu*tg[:, 0] + ww*tg[:, 1]
        sg = np.where(~np.isfinite(dot) | (np.abs(dot) < 1e-6), 1.0,
                      np.sign(dot))
        cfs = cfm*sg
    else:
        cfs = cfm
    out = {}
    for side, msk in (('upper', upper), ('lower', ~upper)):
        o = np.argsort(P[msk][:, 0])
        out[side] = (P[msk][:, 0][o], cp[msk][o], cfs[msk][o])
    return out


def parse(case):
    m = re.match(r'case_(L\d)_v2(?:_yp[\d.]+)?(?:_a([+-][\d.]+))?', case)
    return (m.group(1) if m else 'L1'), float(m.group(2) or -1.0) if m else -1.0


def main():
    out = sys.argv[1]
    cases = sys.argv[2:]
    fig, axs = plt.subplots(5, 1, figsize=(8.6, 13.0), sharex=True)
    ax_reo, ax_P, ax_chi, ax_cp, ax_cf = axs

    for case in cases:
        level, alpha = parse(case)
        lw = LEVEL_LW.get(level, 1.6)
        nu = nu_of(case)
        hdr, pts, curves = read_contours('%s/contours_L1.txt' % case)
        walls = [n for w, n in curves if w]
        fld = Field(case)
        for k, nm in enumerate(ELEMS):
            cont, seg, sarc, ile, u2 = surface_frame(pts, walls[k])
            ls = ELEM_LS[nm]
            pm = probe_maxima(fld, cont, ile, u2, nu)
            for side, col in (('upper', UP_COLOR), ('lower', LO_COLOR)):
                if side not in pm:
                    continue
                x, reo, oi, chi = pm[side]
                o = np.argsort(x)
                ax_reo.semilogy(x[o], reo[o], ls=ls, lw=lw, color=col)
                ax_P.semilogy(x[o], np.clip(oi[o], 1e-4, None), ls=ls, lw=lw,
                              color=col)
                ax_chi.semilogy(x[o], np.clip(chi[o], CHI_LO*1e-2, None),
                                ls=ls, lw=lw, color=col)
            sc = surface_cp_cf(case, nm, cont, ile, u2, fld=fld)
            for side, col in (('upper', UP_COLOR), ('lower', LO_COLOR)):
                x, cp, cf = sc[side]
                ax_cp.plot(x, -cp, ls=ls, lw=lw, color=col)
                ax_cf.plot(x, cf, ls=ls, lw=lw, color=col)

    ax_reo.axhline(REOMC_FLOOR, color='gray', ls='--', lw=0.6, alpha=0.6)
    ax_reo.set_ylim(1e2, 1e4)
    ax_reo.set_ylabel(r'$\max Re_\Omega$ (log)')
    ax_P.set_ylim(1e-3, 1.0)
    ax_P.set_ylabel(r'$\max \hat\Omega \hat I$ (log)')
    ax_chi.axhline(C_V1, color='gray', ls=':', lw=0.6, alpha=0.6)
    ax_chi.axhline(CHI_INF, color='gray', ls='-.', lw=0.6, alpha=0.6)
    ax_chi.set_ylim(CHI_LO, CHI_HI)
    ax_chi.set_ylabel(r'$\max \chi$ (log)')
    ax_cp.set_ylabel(r'$-C_p$')
    ax_cf.set_ylabel(r'$C_f$')
    ax_cf.set_ylim(-0.002, 0.008)
    ax_cf.axhline(0.0, color='gray', lw=0.6, alpha=0.6)
    ax_cf.set_xlabel('$x$')
    for a in axs:
        a.grid(alpha=0.3, which='both')

    handles = [Line2D([], [], color=UP_COLOR, lw=1.6, label='upper'),
               Line2D([], [], color=LO_COLOR, lw=1.6, label='lower'),
               Line2D([], [], color='0.3', lw=1.6, ls='-', label='fore'),
               Line2D([], [], color='0.3', lw=1.6, ls='--', label='flap')]
    handles += [Line2D([], [], color='0.3', lw=LEVEL_LW[l], label=l)
                for l in ('L0', 'L1', 'L2')
                if any(parse(c)[0] == l for c in cases)]
    ax_reo.legend(handles=handles, fontsize=7.5, ncol=4, loc='upper right')
    fig.tight_layout()
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    main()
