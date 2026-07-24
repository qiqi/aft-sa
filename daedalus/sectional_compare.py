"""Sectional comparison: RANS strip forces (wind axes) vs AVL + XFOIL strips,
transition location x_tr(eta) vs XFOIL, and transition type (separation
bubble vs attached) from both sides.

RANS strips: integrate -Cp*n_hat*dA + CfVec*dA over spanwise bins of the wall
surface (normals oriented outward per section), rotate to wind axes at alpha.
Transition proxy per strip and surface side: x/c of the max chordwise gradient
of |Cf| (the laminar->turbulent rise); bubble = contiguous Cf_x < 0 region.
XFOIL: reported x_tr + Cf(x) from DUMP (bubble = Cf < 0 run before the wake).

Usage: python3 sectional_compare.py
Outputs: /tmp/daedalus_mesh_views/sectional/*.png + printed table.
"""
import os
import re
import subprocess
import numpy as np
np.trapezoid = getattr(np, 'trapezoid', np.trapz)
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wing_geometry import SectionFamily, chord, HALF_SPAN, C_ROOT, XQC

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'avl')
OUT = '/tmp/daedalus_mesh_views/sectional'
ALPHA = np.deg2rad(4.0)
MACH, RE_ROOT, NCRIT = 0.1, 5.0e5, 13.6
NBIN = 36
ETA_C = (np.arange(NBIN) + 0.5) / NBIN

CASES = {
    'ogrid_saai': ('case_ogrid_saai', 'surface_fluid_wing.pvtu'),
    'ogrid_turb': ('case_ogrid_turb', 'surface_fluid_wing.pvtu'),
    'cavity_saai': ('case_cavity_saai', 'surface_farfield_body.pvtu'),
    'cavity_turb': ('case_cavity_turb', 'surface_farfield_body.pvtu'),
}


def load_tris(case, fn):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f'{HERE}/{case}/{fn}')
    r.Update()
    tf = vtk.vtkTriangleFilter()
    gf = vtk.vtkGeometryFilter()
    gf.SetInputData(r.GetOutput())
    gf.Update()
    tf.SetInputData(gf.GetOutput())
    tf.Update()
    pd = tf.GetOutput()
    pts = vtk_to_numpy(pd.GetPoints().GetData())
    tris = vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    P = pd.GetPointData()
    cp = vtk_to_numpy(P.GetArray('Cp'))
    cf = vtk_to_numpy(P.GetArray('CfVec'))
    return pts, tris, cp, cf


_FAM = SectionFamily(64)
_CAM_X = _FAM.xs
_CAM_Z = 0.5 * (_FAM._blend(0.0)[0] + _FAM._blend(0.0)[1])   # root camber line


def strip_data(case, fn):
    pts, tris, cp, cf = load_tris(case, fn)
    v0, v1, v2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    avec = 0.5 * np.cross(v1 - v0, v2 - v0)          # area vector, arbitrary sign
    ctr = (v0 + v1 + v2) / 3.0
    # side classification against the CAMBER LINE (the DAE aft-lower surface
    # sits above any fixed section center, which breaks point-center outward
    # tests), then orient: upper outward has +z, lower -z; near the nose
    # (|n_z| small) outward points upstream (-x).
    c_loc = chord(np.clip(np.abs(ctr[:, 1]) / HALF_SPAN, 0, 1))
    xc_t = np.clip((ctr[:, 0] - (XQC - 0.25 * c_loc)) / c_loc, 0.0, 1.0)
    z_cam = np.interp(xc_t, _CAM_X, _CAM_Z) * c_loc
    upper = ctr[:, 2] > z_cam
    want = np.where(upper, 1.0, -1.0)
    flip = np.sign(avec[:, 2]) * want                # +1 if already correct
    nose = (xc_t < 0.03) & (np.abs(avec[:, 2]) < 0.5 * np.linalg.norm(avec, axis=1))
    flip[nose] = -np.sign(avec[nose, 0])             # outward = -x at the nose
    flip[flip == 0] = 1.0
    avec = avec * flip[:, None]
    dA = np.linalg.norm(avec, axis=1)
    cp_t = cp[tris].mean(axis=1)
    cf_t = cf[tris].mean(axis=1)

    eta_t = np.abs(ctr[:, 1]) / HALF_SPAN
    ib = np.clip((eta_t * NBIN).astype(int), 0, NBIN - 1)

    cl_s = np.zeros(NBIN); cd_s = np.zeros(NBIN)
    xtr = {'upper': np.full(NBIN, np.nan), 'lower': np.full(NBIN, np.nan)}
    bub = {'upper': [(np.nan, np.nan)] * NBIN, 'lower': [(np.nan, np.nan)] * NBIN}
    for b in range(NBIN):
        m = (ib == b) & (eta_t < 0.999)
        if m.sum() < 10:
            continue
        F = (-cp_t[m, None] * avec[m]).sum(axis=0) + (cf_t[m] * dA[m, None]).sum(axis=0)
        # normalize by the CAPTURED projected planform area of the bin (sum of
        # upper-side area-vector z components): the mesh's spanwise station
        # spacing beats against uniform geometric bins otherwise (sawtooth).
        area = avec[m & upper, 2].sum()
        if area <= 0:
            continue
        L = F[2] * np.cos(ALPHA) - F[0] * np.sin(ALPHA)
        D = F[0] * np.cos(ALPHA) + F[2] * np.sin(ALPHA)
        cl_s[b] = L / area
        cd_s[b] = D / area
        # transition + bubble per side from chordwise Cf profiles
        for side, sm in (('upper', m & upper), ('lower', m & ~upper)):
            if sm.sum() < 10:
                continue
            c_b = chord(ETA_C[b])
            xc = (ctr[sm, 0] - (XQC - 0.25 * c_b)) / c_b
            cfx = cf_t[sm, 0]
            cfm = np.linalg.norm(cf_t[sm], axis=1)
            o = np.argsort(xc)
            xc, cfx, cfm = xc[o], cfx[o], cfm[o]
            grid = np.linspace(0.03, 0.97, 80)
            prof = np.interp(grid, xc, cfm)
            prof_x = np.interp(grid, xc, cfx)
            dprof = np.gradient(prof, grid)
            xtr[side][b] = grid[int(np.argmax(dprof))]
            neg = prof_x < -5e-5
            if neg.any():
                bub[side][b] = (grid[neg.argmax()],
                                grid[len(neg) - 1 - neg[::-1].argmax()])
    return cl_s, cd_s, xtr, bub


def native_strips(case):
    """Solver's own sectional forces: Y_slicing_forceDistribution.csv
    (ClipOutput.cpp): CF{x,z}_per_span = F/(q Sref) per unit span. Convert to
    local wind-axis coefficients via cl*c = CF_per_span * Sref."""
    import csv as _csv
    path = f'{HERE}/{case}/Y_slicing_forceDistribution.csv'
    rows = np.genfromtxt(path, delimiter=',', skip_header=1)
    y, cfx, cfz = rows[:, 0], rows[:, 2], rows[:, 3]
    sref = 15.42 if 'ogrid' in case else 30.84
    eta = np.abs(y) / HALF_SPAN
    keep = eta < 0.999
    eta, cfx, cfz = eta[keep], cfx[keep], cfz[keep]
    c = chord(eta)
    cl = (cfz * np.cos(ALPHA) - cfx * np.sin(ALPHA)) * sref / c
    cd = (cfx * np.cos(ALPHA) + cfz * np.sin(ALPHA)) * sref / c
    o = np.argsort(eta)
    return eta[o], cl[o], cd[o]


def avl_strips():
    rows = []
    for ln in open(f'{WORK}/fs.txt'):
        t = ln.split()
        if len(t) >= 10:
            try:
                int(t[0])
                rows.append((float(t[1]), float(t[2]), float(t[7]),
                             float(t[8]), float(t[5])))
            except ValueError:
                continue
    r = np.array([x for x in rows if x[0] > 0])
    o = np.argsort(r[:, 0])
    return r[o]     # y, chord, cl, cd_strip(noisy), ai


def xfoil_station(eta, cl, ncrit):
    c = chord(eta)
    re_c = RE_ROOT * c / C_ROOT
    fam = SectionFamily(64)
    sec = f'{WORK}/sec_st_{int(eta*100):03d}.dat'
    if not os.path.exists(sec):
        from avl_compare import write_section
        write_section(fam, eta, sec)
    dmp = f'dump_{int(eta*100):03d}.dat'
    cmds = [f'LOAD {os.path.basename(sec)}', 'PANE', 'OPER', f'MACH {MACH}',
            f'VISC {re_c:.0f}', 'VPAR', f'N {ncrit}', '', 'ITER 300',
            f'CL {cl:.4f}', f'CL {cl:.4f}', f'DUMP {dmp}', '', 'QUIT', '']
    p = subprocess.run(['xvfb-run', '-a', 'xfoil'], input='\n'.join(cmds),
                       capture_output=True, text=True, cwd=WORK, timeout=240)
    xtr_u = xtr_l = np.nan
    for ln in p.stdout.splitlines():
        low = ln.lower()
        if 'transition at x/c' in low:
            try:
                x = float(ln.split('=')[-1].split()[0])
            except ValueError:
                continue
            if 'side 1' in low:
                xtr_u = x
            elif 'side 2' in low:
                xtr_l = x
    # bubble from the BL dump: Cf < 0 runs on the airfoil (x <= 1)
    bub_u = bub_l = (np.nan, np.nan)
    try:
        arr = []
        for ln in open(f'{WORK}/{dmp}'):
            t = ln.split()
            if len(t) >= 8:
                try:
                    arr.append((float(t[1]), float(t[6])))
                except ValueError:
                    continue
        arr = np.array(arr)
        on = arr[:, 0] <= 1.0
        x_arr, cf_arr = arr[on, 0], arr[on, 1]
        # side 1 = points until x returns to LE minimum; split at argmin(x)
        i_le = int(np.argmin(x_arr))
        sides = {'u': (x_arr[:i_le + 1], cf_arr[:i_le + 1]),
                 'l': (x_arr[i_le:], cf_arr[i_le:])}
        out = {}
        for k, (xs, cfs) in sides.items():
            neg = cfs < 0
            out[k] = ((xs[neg].min(), xs[neg].max()) if neg.any()
                      else (np.nan, np.nan))
        bub_u, bub_l = out['u'], out['l']
    except Exception:
        pass
    return xtr_u, xtr_l, bub_u, bub_l


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    rans = {}
    for tag, (case, fn) in CASES.items():
        rans[tag] = strip_data(case, fn)
        print(f'{tag}: strips done', flush=True)
    av = avl_strips()
    eta_avl = av[:, 0] / HALF_SPAN

    # XFOIL stations
    etas_x = [0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.85, 0.92, 0.97]
    xf = []
    for e in etas_x:
        cl_loc = np.interp(e * HALF_SPAN, av[:, 0], av[:, 2])
        r = xfoil_station(e, cl_loc, NCRIT)
        xf.append(r)
        bu = f'{r[2][0]:.2f}-{r[2][1]:.2f}' if np.isfinite(r[2][0]) else 'none'
        print(f'  xfoil eta {e:.2f}: xtr_u {r[0]:.3f} xtr_l {r[1]:.3f} '
              f'bubble_u {bu}', flush=True)
    xf = np.array([(x[0], x[1], x[2][0], x[2][1], x[3][0], x[3][1]) for x in xf])

    STY = {'ogrid_saai': ('C0', '-'), 'cavity_saai': ('C0', '--'),
           'ogrid_turb': ('C3', '-'), 'cavity_turb': ('C3', '--')}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tag, (cl_s, cd_s, _, _) in rans.items():
        col, ls = STY[tag]
        e_n, cl_n, cd_n = native_strips(CASES[tag][0])
        ax.plot(e_n, cl_n, color=col, ls=ls, lw=1.8, label=f'RANS {tag} (native)')
        ax.plot(ETA_C, cl_s, color=col, ls=':', lw=0.9, alpha=0.7)
    ax.plot(eta_avl, av[:, 2], 'k-o', ms=2.5, lw=1, label='AVL')
    ax.set_xlabel(r'$\eta = 2y/b$'); ax.set_ylabel(r'sectional $c_l$ (wind axes)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_xlim(0, 1)
    fig.savefig(f'{OUT}/sectional_cl.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    # sectional cd: RANS vs AVL cdi + XFOIL cdp (free) at stations
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tag, (cl_s, cd_s, _, _) in rans.items():
        col, ls = STY[tag]
        e_n, cl_n, cd_n = native_strips(CASES[tag][0])
        ax.plot(e_n, cd_n, color=col, ls=ls, lw=1.8, label=f'RANS {tag} (native)')
        ax.plot(ETA_C, cd_s, color=col, ls=':', lw=0.9, alpha=0.7)
    # sectional induced drag from cl*ai (smooth, grid-converged), rescaled so
    # its integral matches the Trefftz-plane CDind: AVL's per-strip cd column
    # is the NEAR-FIELD surface integration and oscillates strip-to-strip
    # (std ~ 100% of its mean inboard, even changing sign) -- an artifact of
    # discrete-vortex leading-edge suction, not physics.
    cdi_loc = av[:, 2] * av[:, 4]
    cdi_int = 2.0 * np.trapezoid(cdi_loc * av[:, 1], av[:, 0]) / 30.84
    cdi_scale = 0.01214 / cdi_int          # Trefftz CDind / near-field integral
    print(f'cl*ai integral {cdi_int:.5f} vs Trefftz 0.01214 '
          f'-> scale {cdi_scale:.3f}')
    cdi_e = np.interp(etas_x, eta_avl, cdi_loc * cdi_scale)
    # profile cd at stations from the avl_compare run (recompute quickly)
    from avl_compare import xfoil_cd
    cdp_free, cdp_trip = [], []
    for e in etas_x:
        cl_loc = np.interp(e * HALF_SPAN, av[:, 0], av[:, 2])
        re_c = RE_ROOT * chord(e) / C_ROOT
        sec = f'sec_st_{int(e*100):03d}.dat'
        cdp_free.append(xfoil_cd(sec, re_c, cl_loc, NCRIT, False) or np.nan)
        cdp_trip.append(xfoil_cd(sec, re_c, cl_loc, NCRIT, True) or np.nan)
    ax.plot(etas_x, cdi_e + np.array(cdp_free), 'k-o', ms=3, lw=1,
            label='AVL cdi + XFOIL cdp (free)')
    ax.plot(etas_x, cdi_e + np.array(cdp_trip), 'k--s', ms=3, lw=1,
            label='AVL cdi + XFOIL cdp (tripped)')
    ax.set_xlabel(r'$\eta$'); ax.set_ylabel(r'sectional $c_d$ (wind axes)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.05)
    fig.savefig(f'{OUT}/sectional_cd.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    # transition location + bubbles
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for k, side in enumerate(('upper', 'lower')):
        ax = axs[k]
        for tag in ('ogrid_saai', 'cavity_saai'):
            col, ls = STY[tag]
            ax.plot(ETA_C, rans[tag][2][side], color=col, ls=ls,
                    label=f'RANS {tag} (Cf-rise)')
            bb = rans[tag][3][side]
            lo = np.array([b[0] for b in bb]); hi = np.array([b[1] for b in bb])
            ax.fill_between(ETA_C, lo, hi, color=col, alpha=0.15,
                            label=f'{tag} reversed-flow' if k == 0 else None)
        ax.plot(etas_x, xf[:, 0] if side == 'upper' else xf[:, 1], 'k-o', ms=4,
                label='XFOIL $x_{tr}$')
        icol = 2 if side == 'upper' else 4
        ax.fill_between(etas_x, xf[:, icol], xf[:, icol + 1], color='k',
                        alpha=0.15, label='XFOIL bubble')
        ax.set_title(f'{side} surface'); ax.set_xlabel(r'$\eta$')
        ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axs[0].set_ylabel(r'$x_{tr}/c$')
    axs[0].legend(fontsize=7, loc='lower left')
    fig.suptitle('Transition location and separation bubbles: SA-AI vs XFOIL '
                 f'(N={NCRIT})', fontsize=11)
    fig.savefig(f'{OUT}/transition_eta.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT)
