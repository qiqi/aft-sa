"""Literature-style surface maps for the 6:1 prolate spheroid transition case.

Replicates the standard presentations of the Kreplin/DFVLR experiments and
the transition-modeling literature that validates against them
(Krimmelbein & Krumbein; Grabe & Krumbein; Hildebrand et al. FUN3D+LASTRAC):

  (1) unrolled map, x/L vs azimuth phi (0 = windward symmetry line,
      180 = leeward), filled contours of skin-friction magnitude c_f,
      with skin-friction lines (the oil-flow analogue) overlaid --
      transition shows as the c_f rise;
  (2) same plane, wall-shear direction angle gamma_w (Kreplin's own
      quantity: angle of the wall shear vector from the local meridian,
      positive toward the leeward side);
  (3) same plane, near-wall max chi = nu_t/nu with the chi = c_v1 contour
      as the model-native transition front;
  (4) 3D perspective view of the surface colored by c_f (Hildebrand
      Fig. 15 style).

Everything is sampled parametrically off the ANALYTIC surface
(x^2/A^2 + r^2/B^2 = 1, A = L/2, B = L/12) with vtkProbeFilter rays, so it
works for any mesh family without relying on node ordering.  Mesh phi=0 is
+z = LEEWARD at alpha > 0 (ogrid_spheroid.py); the maps use the literature
convention phi_lit = 180 - phi_mesh.

c_f = muRef * |du_t/dn| / (0.5 rho_inf M^2).  The wall gradient is the
least-squares SLOPE of u_t probed at several sublayer heights: the discrete
wall sags below the analytic surface by the local chord sagitta (up to
~13 h0 at L0 midbody), which shifts every probe height by an unknown
per-ray constant -- the slope is immune to it, a fixed height is not.
(mu/mu_inf ~ 1 at the wall for M = 0.1.)

Usage:  python3 surface_map.py <case_dir> [--h0 5e-6] [--out prefix]
"""
import argparse
import os
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

L = 1.0
A = 0.5 * L
B = L / 12.0
MACH = 0.1
MUREF = MACH / 1.5e6          # default; overridden from Flow360.json
CV1 = 7.1

NX, NP = 360, 121             # x/L stations x azimuth stations
XLO, XHI = 0.004, 0.996
CHI_RAY = np.geomspace(1.2e-5, 0.04, 36)   # wall-normal chi probe stations
# sublayer heights for the wall-gradient slope fit (see docstring); kept
# below y+ ~ 7 at the highest turbulent cf of this Re ladder
SHEAR_RAY = np.array([1.5e-5, 3e-5, 4.5e-5, 6e-5, 8e-5, 1e-4])


def surface_frame(xl, phi_lit):
    """Points + local frame on the analytic surface.

    xl: x/L in (0,1) from nose; phi_lit: 0 windward .. pi leeward.
    Returns P (surface points), n (outward unit normal), t_s (meridional
    unit tangent, nose->tail), t_p (circumferential unit tangent, toward
    leeward = +phi_lit)."""
    x = -A + xl * L
    r = B * np.sqrt(np.clip(1.0 - (x / A) ** 2, 0.0, None))
    phi_mesh = np.pi - phi_lit                    # mesh: phi=0 leeward (+z)
    sin, cos = np.sin(phi_mesh), np.cos(phi_mesh)
    P = np.stack([x, r * sin, r * cos], axis=-1)
    n = np.stack([x / A**2, r * sin / B**2, r * cos / B**2], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    # meridional tangent: d/dx of (x, r(x) e_r); dr/dx = -B^2 x/(A^2 r)
    drdx = -(B**2 * x) / (A**2 * np.maximum(r, 1e-12))
    t_s = np.stack([np.ones_like(x), drdx * sin, drdx * cos], axis=-1)
    t_s /= np.linalg.norm(t_s, axis=-1, keepdims=True)
    # toward leeward = decreasing phi_mesh: d/d(-phi_mesh)
    t_p = np.stack([np.zeros_like(x), -cos, sin], axis=-1)
    return P, n, t_s, t_p


def probe(grid, pts):
    """Probe point-data arrays of `grid` at pts (N,3); returns dict + valid."""
    pd = vtk.vtkPolyData()
    vp = vtk.vtkPoints()
    vp.SetData(numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
    pd.SetPoints(vp)
    pr = vtk.vtkProbeFilter()
    pr.SetInputData(pd)
    pr.SetSourceData(grid)
    pr.Update()
    out = pr.GetOutput().GetPointData()
    res = {out.GetArray(i).GetName(): vtk_to_numpy(out.GetArray(i))
           for i in range(out.GetNumberOfArrays())}
    valid = vtk_to_numpy(pr.GetOutput().GetPointData()
                         .GetArray(pr.GetValidPointMaskArrayName())).astype(bool)
    return res, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case_dir')
    ap.add_argument('--h0', type=float, default=None,
                    help='first wall spacing /L (default: 5e-6 * 0.5^level '
                         'guessed from the case name L0/L1/L2)')
    ap.add_argument('--out', default=None, help='output prefix')
    ap.add_argument('--paper', action='store_true',
                    help='paper-grade titles (no case tag)')
    ap.add_argument('--from-npz', action='store_true',
                    help='replot from the existing .npz (skip probing)')
    args = ap.parse_args()
    case = args.case_dir.rstrip('/')
    tag = os.path.basename(case)
    h0 = args.h0
    if h0 is None:
        lev = next((int(t[1]) for t in tag.split('_') if
                    len(t) == 2 and t[0] == 'L' and t[1].isdigit()), 0)
        h0 = 5e-6 * 0.5**lev
    out = args.out or os.path.join(case, f'surface_map_{tag}')

    import json
    fs = json.load(open(os.path.join(case, 'Flow360.json')))['freestream']
    mach, muref, alpha = fs['Mach'], fs['muRef'], fs['alphaAngle']
    qinf = 0.5 * mach**2                       # rho_inf = 1

    if args.from_npz and os.path.exists(out + '.npz'):
        d = np.load(out + '.npz')
        xl, ph = d['xl'], np.radians(d['phi_deg'])
        cf, gamma, chimax = d['cf'], d['gamma_w'], d['chimax']
        us, up = d['us'], d['up']
        return plot_all(out, args, case, tag, alpha, mach, muref,
                        xl, d['phi_deg'], cf, gamma, chimax, us, up)

    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(os.path.join(case, 'volume.pvtu'))
    r.Update()
    grid = r.GetOutput()
    print(f'{tag}: {grid.GetNumberOfPoints():,} pts, h0={h0:.2e}, '
          f'alpha={alpha}, muRef={muref:.3e}')

    re_scale = 1.5e6 / (mach / muref)   # rays were designed for Re_L=1.5e6
    shear_ray = SHEAR_RAY * re_scale
    chi_ray = np.geomspace(CHI_RAY[0] * re_scale, CHI_RAY[-1], len(CHI_RAY))
    xl = np.linspace(XLO, XHI, NX)
    ph = np.linspace(0.0, np.pi, NP)
    XL, PH = np.meshgrid(xl, ph)                       # (NP, NX)
    P, n, t_s, t_p = surface_frame(XL, PH)

    # ---- wall shear: LS slope of u_t over the sublayer heights -------------
    spts = (P[..., None, :] + shear_ray[None, None, :, None] * n[..., None, :])
    res, valid = probe(grid, spts.reshape(-1, 3))
    u = res['velocity'].reshape(NP, NX, len(shear_ray), 3)
    # wall pressure (constant across the sublayer): innermost valid height;
    # Cp = (p - 1/gamma) / qinf in the a_inf,rho_inf nondimensionalization
    pr = res['p'].reshape(NP, NX, len(shear_ray))
    vmask = valid.reshape(NP, NX, len(shear_ray))
    pw = np.where(vmask, pr, np.nan)
    first = np.argmax(vmask, axis=-1)   # 0 if no height valid -> pw NaN there
    pwall = np.take_along_axis(pw, first[..., None], axis=-1)[..., 0]
    cp = (pwall - 1.0 / 1.4) / qinf
    w = valid.reshape(NP, NX, len(shear_ray)).astype(float)
    ok = w.sum(axis=-1) >= 4                  # need >= 4 heights for the fit
    print(f'  shear-ray probe valid: {w.mean()*100:.2f}% '
          f'(fit-able: {ok.mean()*100:.2f}%)')
    u_t = u - np.einsum('ijkl,ijl->ijk', u, n)[..., None] * n[..., None, :]
    u_t = np.where(w[..., None] > 0, u_t, 0.0)
    wsum = np.maximum(w.sum(axis=-1), 1e-30)
    dbar = (w * shear_ray).sum(axis=-1) / wsum
    dc = (shear_ray[None, None, :] - dbar[..., None]) * w
    dudn = np.einsum('ijk,ijkl->ijl', dc, u_t) \
        / np.maximum((dc * dc).sum(axis=-1), 1e-30)[..., None]
    us = np.einsum('ijk,ijk->ij', dudn, t_s)
    up = np.einsum('ijk,ijk->ij', dudn, t_p)
    cf = muref * np.hypot(us, up) / qinf
    gamma = np.degrees(np.arctan2(up, us))
    cf[~ok] = np.nan
    gamma[~ok] = np.nan

    # ---- near-wall max chi along wall-normal rays --------------------------
    rays = (P[..., None, :] + chi_ray[None, None, :, None] * n[..., None, :])
    resr, validr = probe(grid, rays.reshape(-1, 3))
    nut = resr['solutionTurbulence'].reshape(NP, NX, len(chi_ray))
    rho = resr['rho'].reshape(NP, NX, len(chi_ray))
    chi = rho * nut / muref
    chi[~validr.reshape(NP, NX, len(chi_ray))] = np.nan
    with np.errstate(all='ignore'):
        chimax = np.nanmax(chi, axis=-1)

    np.savez(out + '.npz', xl=xl, phi_deg=np.degrees(ph), cf=cf,
             gamma_w=gamma, chimax=chimax, us=us, up=up, cp=cp,
             alpha=alpha, mach=mach, muref=muref, h0=h0)

    plot_all(out, args, case, tag, alpha, mach, muref,
             xl, np.degrees(ph), cf, gamma, chimax, us, up)


def plot_all(out, args, case, tag, alpha, mach, muref,
             xl, phd, cf, gamma, chimax, us, up):
    """Figures in the paper's line-contour style (labeled black contours)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    XL, PH = np.meshgrid(xl, np.radians(phd))
    fig, axs = plt.subplots(3, 1, figsize=(9.6, 10.2), sharex=True,
                            constrained_layout=True)

    a = axs[0]
    lev = np.arange(0.0, 6.51, 0.5)
    cs = a.contour(xl, phd, cf * 1e3, levels=lev, colors='k',
                   linewidths=0.6)
    a.clabel(cs, levels=lev[::2], fmt='%g', fontsize=7, inline_spacing=2)
    # skin-friction lines (oil-flow analogue) in the unrolled plane
    a.streamplot(xl, phd, us, np.degrees(up / np.maximum(
        B * np.sqrt(np.clip(1 - ((-A + XL) / A)**2, 1e-6, None)), 1e-9)),
        color='0.55', linewidth=0.5, density=(2.2, 1.1), arrowsize=0.6)
    a.set_ylabel(r'$\phi$ [deg]  (0 = windward)')
    head = '' if args.paper else f'{tag}:  '
    re_l = mach / muref
    re_str = f'{re_l/10**int(np.log10(re_l)):.3g}\\times10^{int(np.log10(re_l))}'
    a.set_title(head + '$c_f\\times10^3$ (labeled) + skin-friction lines '
                f'($\\alpha={alpha:g}^\\circ$, $Re_L={re_str}$)')

    a = axs[1]
    lev = np.arange(-60, 61, 10)
    cs = a.contour(xl, phd, gamma, levels=lev, colors='k', linewidths=0.6)
    a.clabel(cs, levels=lev[::2], fmt='%g', fontsize=7, inline_spacing=2)
    a.contour(xl, phd, gamma, levels=[0], colors='k', linewidths=1.6)
    a.set_ylabel(r'$\phi$ [deg]')
    a.set_title(r'wall-shear direction $\gamma_w$ [deg] from local meridian '
                r'(positive toward leeward; dashed negative; bold: '
                r'$\gamma_w=0$)')

    a = axs[2]
    with np.errstate(all='ignore'):
        logchi = np.log10(np.maximum(chimax, 1e-8))
    major = [-3, -2, -1, 0, np.log10(CV1), np.log10(30.0), 2.0]
    minor = [v + off for v in (-3, -2, -1, 0, 1) for off in
             (np.log10(2), np.log10(5))]
    a.contour(xl, phd, logchi, levels=sorted(minor), colors='k',
              linewidths=0.35)
    cs = a.contour(xl, phd, logchi, levels=major, colors='k',
                   linewidths=0.8)
    fmt = {lv: ('$c_{v1}$' if abs(lv - np.log10(CV1)) < 1e-9 else
                ('30' if abs(lv - np.log10(30.0)) < 1e-9 else
                 f'$10^{{{lv:g}}}$')) for lv in major}
    a.clabel(cs, fmt=fmt, fontsize=7, inline_spacing=2)
    a.contour(xl, phd, chimax, levels=[CV1], colors='k', linewidths=1.6)
    a.set_ylabel(r'$\phi$ [deg]')
    a.set_xlabel(r'$x/L$')
    a.set_title(r'near-wall $\max\chi$ (log labels); bold: '
                r'$\chi=c_{v1}$ (model-native transition front)')
    for a in axs:
        a.set_ylim(0, 180)
        a.set_yticks([0, 45, 90, 135, 180])
    fig.savefig(out + '.png', dpi=140)
    fig.savefig(out + '.pdf')
    print('wrote', out + '.png/.pdf')

    # ---- figure 2: 3D perspective colored by cf (render, not a contour) ----
    from matplotlib.colors import Normalize
    fig = plt.figure(figsize=(11, 4.4))
    norm = Normalize(0, 6)
    for k, (el, azv, ttl) in enumerate(
            [(28, -125, 'leeward'), (-28, -125, 'windward')]):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        for sgn in (1, -1):
            Pm, _, _, _ = surface_frame(XL, PH)
            Y = sgn * Pm[..., 1]
            ax.plot_surface(Pm[..., 0], Y, Pm[..., 2],
                            facecolors=plt.cm.viridis(norm(cf * 1e3)),
                            rstride=2, cstride=2, shade=False,
                            antialiased=False)
        ax.view_init(el, azv)
        ax.set_box_aspect((6, 1.15, 1.15))
        ax.set_axis_off()
        ax.set_title(f'$c_f$, {ttl} side', y=0.85)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    fig.colorbar(sm, ax=fig.axes, shrink=0.6, label=r'$c_f \times 10^3$')
    fig.savefig(out + '_3d.png', dpi=140, bbox_inches='tight')
    print('wrote', out + '_3d.png')


if __name__ == '__main__':
    main()
