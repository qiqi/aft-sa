"""Daedalus SECTION sheets -> figs/daedalus_section_eta{10,75,92}.pdf.

The 2D airfoils' five-row whole-page diagnostic (rows: probe-max Re_Omega;
probe-max Omega_hat*I_hat; max chi with the e^N envelope N; -Cp; signed C_f,x),
now cut at fixed spanwise stations of the Daedalus wing -- built to
diagnose why the RANS sectional profile drag grows with lift while the
strips' XFOIL profile drag stays flat (the CD-vs-CL slope discrepancy of
fig:daepolar).

Columns: alpha = 4, 5, 6 deg. Curves: structured O-grid L2 (solid),
unstructured L2 where complete (dashed); upper surface blue, lower red;
dotted = the FlexFoil e^N strip at the nearest station (N envelope
truncated at its transition; Cp from Karman-Tsien-corrected u_e).
Rows 1-2 probe the volume along in-plane surface normals to 0.01 c_loc
(the 2D convention); row 3 reads the near-wall band max chi from the
committed chi_surface.npz (5% c band).

  python3 regen_daedalus_section_sheets.py [eta ...]   # default 0.10 0.75 0.92
"""
import os
import sys
import json
import pickle
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# per-family roots: the structured family is the final-kernel (fv1)
# recomputation; the cavity-L2 recomputation is HELD, so its solutions
# remain the bypass-inactive model (disclosed in Sec. VII)
D_STR = '/local_data/qiqi/sa-ai/daedalus_fv1'
D_CAV = '/home/qiqi/flexcompute/sa-ai/daedalus'
PD = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(PD, 'repro'))
sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai/daedalus')  # geometry modules
from wing_geometry import chord, HALF_SPAN, XQC          # noqa: E402
import sectional_compare as SC                            # noqa: E402

CHI_INF = 8.76e-6
AMAX, C_ON, A_ON, B_ON, K_ON, W_ON = 0.19, 2600.0, 175.0, 2.0, 0.712, 0.35
CV1 = 7.1
ALPHAS = [4, 5, 6]
CASES = {'str': (D_STR, 'case_ogrid_L2_saai_a{a}', 'surface_fluid_wing.pvtu', '-'),
         'cav': (D_CAV, 'case_cavity_L2_saai_a{a}', 'surface_farfield_body.pvtu', '--')}
SIDCOL = {'upper': 'C0', 'lower': 'C3'}
STRIPS = pickle.load(open('/home/qiqi/flexcompute/sa-ai/flow360_ai/'
                          'flexfoil_daedalus_strips.pkl', 'rb'))
N_ANCH, N_PROBE, L_PROBE = 130, 90, 0.01
c_loc_g, x_le_g = [1.0], [0.0]     # anchors, heights, probe depth (x c_loc)


def load(p):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(p)
    r.Update()
    return r.GetOutput()


def complete(case):
    fn = f'{root}/{case}/total_forces_v2.csv'
    return os.path.exists(fn) and sum(1 for _ in open(fn)) >= 2001


def section_contour(surf_pts, eta_q):
    """Ordered (x, z) upper/lower contours of the station's section."""
    y0 = eta_q * HALF_SPAN
    yy = np.abs(surf_pts[:, 1])
    dy = np.abs(yy - y0)
    band = max(np.partition(dy, 800)[800], 1e-4)
    m = dy <= band
    p = surf_pts[m]
    c_loc = float(chord(eta_q))
    x_le = XQC - 0.25 * c_loc
    xc = (p[:, 0] - x_le) / c_loc
    up = p[:, 2] >= np.interp(np.clip(xc, 0, 1), SC._CAM_X, SC._CAM_Z) * c_loc
    out = {}
    for side, sel in (('upper', up), ('lower', ~up)):
        q = p[sel]
        o = np.argsort(q[:, 0])
        out[side] = q[o]
    return out, c_loc, x_le, m


def sphere_rate_1d(u, dudy, yc):
    """The parallel-layer sphere kernel on a probe line (fig04 conventions)."""
    d2u = np.gradient(dudy, yc)
    X = u
    Y = yc * dudy
    Z = 0.5 * yc**2 * d2u
    R = np.sqrt(X * X + Y * Y + Z * Z) + 1e-30
    Omega_hat = Y / np.sqrt(X * X + Y * Y + 1e-30)
    return Omega_hat * (Y - X - Z) / R


def onset_threshold(pmax):
    """Solver onset threshold Re_Omega^c(P) = ceil*pw/sqrt(ceil^2+pw^2),
    pw = A + B/P^2, all three constants carrying the whole-equation k."""
    pf = np.maximum(pmax, 1e-6)
    ceil, A, B = K_ON * C_ON, K_ON * A_ON, K_ON * B_ON
    pw = A + B / pf**2
    return ceil * pw / np.sqrt(ceil**2 + pw**2)


def probe_station(vol, case, contours, c_loc, x_le, mu_ref):
    """Probe rows 1-2 along in-plane normals; return per-side dict.
    mu_ref: freestream kinematic viscosity in solver nondim units."""
    c_loc_g[0], x_le_g[0] = c_loc, x_le
    pr = {}
    for side, q in contours.items():
        n = len(q)
        xq = (q[:, 0] - x_le_g[0]) / c_loc_g[0]
        ok = np.where((xq > 0.01) & (xq < 0.985))[0]
        idx = ok[np.linspace(0, len(ok) - 1, min(N_ANCH, len(ok))).astype(int)]
        anch = q[idx]
        # in-plane tangent/normal from contour neighbors
        tan = q[np.minimum(idx + 2, n - 1)] - q[np.maximum(idx - 2, 0)]
        tan[:, 1] = 0.0
        tan /= (np.linalg.norm(tan, axis=1)[:, None] + 1e-30)
        # outward in-plane normal: up for the upper surface, down for the lower
        if side == 'upper':
            nrm = np.stack([-tan[:, 2], np.zeros(len(idx)), tan[:, 0]], axis=1)
        else:
            nrm = np.stack([tan[:, 2], np.zeros(len(idx)), -tan[:, 0]], axis=1)
        want = 1.0 if side == 'upper' else -1.0
        nrm[np.sign(nrm[:, 2]) != want] *= -1.0
        d = np.linspace(1e-5, L_PROBE, N_PROBE) * c_loc
        pts = (anch[:, None, :] + nrm[:, None, :] * d[None, :, None]).reshape(-1, 3)
        pd = vtk.vtkPolyData()
        vp = vtk.vtkPoints()
        vp.SetData(numpy_to_vtk(pts, deep=1))
        pd.SetPoints(vp)
        pf = vtk.vtkProbeFilter()
        pf.SetInputData(pd)
        pf.SetSourceData(vol)
        pf.Update()
        out = pf.GetOutput().GetPointData()
        vel = vtk_to_numpy(out.GetArray('velocity')).reshape(len(idx), N_PROBE, 3)
        ut = np.einsum('ijk,ik->ij', vel, tan)          # tangential profile
        reo = np.full(len(idx), np.nan)
        rate2d = np.full((len(idx), N_PROBE), np.nan)
        for i in range(len(idx)):
            u = ut[i]
            dud = np.gradient(u, d)
            reo[i] = np.nanmax(d**2 * np.abs(dud)) / mu_ref
            rate2d[i] = sphere_rate_1d(np.abs(u), dud, d)
        # neighbor-smooth Omega_hat*I_hat before the max (see repro/lib/smooth.py)
        from lib.smooth import nan_gaussian
        rate2d = nan_gaussian(rate2d, sigma=(0.005 * N_ANCH / 0.985,
                                             0.05 * N_PROBE))
        pmax = np.nanmax(rate2d, axis=1)
        xc = (anch[:, 0] - x_le) / c_loc
        pr[side] = dict(xc=xc, reo=reo, pmax=pmax)
    return pr


def surface_rows(case, surfname, contours, c_loc, x_le, mstation, mu):
    """Rows 3-5 data: max chi (npz), Cp, Cf_x at the station."""
    g = load(f'{root}/{case}/{surfname}')
    p = vtk_to_numpy(g.GetPoints().GetData())
    cp = vtk_to_numpy(g.GetPointData().GetArray('Cp'))
    cfx = vtk_to_numpy(g.GetPointData().GetArray('CfVec'))[:, 0]
    npz = np.load(f'{root}/{case}/chi_surface.npz')
    chi = npz['chi']
    rows = {}
    for side, q in contours.items():
        # match station-band surface nodes to this side by nearest xz
        pb = p[mstation]
        cpb, cfb, chb = cp[mstation], cfx[mstation], chi[mstation]
        xc_all = (pb[:, 0] - x_le) / c_loc
        up = pb[:, 2] >= np.interp(np.clip(xc_all, 0, 1), SC._CAM_X, SC._CAM_Z) * c_loc
        sel = up if side == 'upper' else ~up
        good = sel & (np.abs(cfb) < 0.1) & (xc_all > 0.005) & (xc_all < 0.995)
        o = np.argsort(xc_all[good])
        rows[side] = dict(xc=xc_all[good][o], cp=cpb[good][o], cfx=cfb[good][o],
                          chi=chb[good][o])
    return rows


def strip_ref(a, eta_q):
    s = STRIPS[float(a)]
    i = int(np.argmin(np.abs(np.asarray(s['eta']) - eta_q)))
    st = s['stations'][i]
    if st is None:
        return None
    xc = np.asarray(st['xc'], float)
    n = np.asarray(st['n'], float)
    ue = np.asarray(st['ue'], float)
    cf = np.asarray(st['cf'], float)
    m = np.isfinite(n)
    if m.sum() > 2:
        j = int(np.nanargmax(n))
        xn, nn = xc[:j + 1], n[:j + 1]
    else:
        xn, nn = xc, n
    beta = np.sqrt(1 - 0.1**2)
    cpi = 1.0 - ue**2
    cpk = cpi / (beta + 0.1**2 / 2 * cpi / (1 + beta))
    # FlexFoil dump cf is EDGE-normalized; the sheets' C_f,x row is
    # freestream-normalized -> convert by ue^2
    return dict(xn=xn, n=nn, xc=xc, cp=cpk, cf=cf * ue**2)


def make_sheet(eta_q):
    fig, axs = plt.subplots(5, len(ALPHAS), figsize=(5.76 * len(ALPHAS), 13),
                            sharex=True)
    got_cav = False
    for col, a in enumerate(ALPHAS):
        ax_reo, ax_P, ax_n, ax_cp, ax_cf = axs[:, col]
        ax_nN = ax_n.twinx()
        for fam, (root, tpl, surfname, ls) in CASES.items():
            case = tpl.format(a=a)
            if not complete(case) or not os.path.exists(f'{root}/{case}/chi_surface.npz'):
                continue
            if fam == 'cav':
                got_cav = True
            if case not in CACHE or eta_q not in CACHE[case]:
                continue
            pr, sr = CACHE[case][eta_q]
            for side in ('upper', 'lower'):
                cc = SIDCOL[side]
                ax_reo.semilogy(pr[side]['xc'], pr[side]['reo'], ls, color=cc, lw=1.5)
                ax_reo.semilogy(pr[side]['xc'], onset_threshold(pr[side]['pmax']),
                                '-.', color=cc, lw=0.8, alpha=0.6)
                ax_P.semilogy(pr[side]['xc'],
                              np.clip(pr[side]['pmax'], 1e-4, None), ls, color=cc, lw=1.5)
                ax_n.semilogy(sr[side]['xc'], np.clip(sr[side]['chi'], 1e-6, None),
                              ls, color=cc, lw=1.5)
                ax_cp.plot(sr[side]['xc'], -sr[side]['cp'], ls, color=cc, lw=1.5)
                ax_cf.plot(sr[side]['xc'], sr[side]['cfx'], ls, color=cc, lw=1.5)
        ref = strip_ref(a, eta_q)
        if ref is not None:
            ax_nN.plot(ref['xn'], ref['n'], ':', color='0.35', lw=1.4)
            ax_cp.plot(ref['xc'], -ref['cp'], ':', color='0.35', lw=1.4)
            ax_cf.plot(ref['xc'], ref['cf'], ':', color='0.35', lw=1.4)
        ax_reo.set_ylim(1e2, 1e4); ax_reo.grid(alpha=0.3, which='both')
        ax_reo.set_title(rf'$\alpha={a}^\circ$', fontsize=10)
        ax_P.set_ylim(1e-3, 1.0); ax_P.grid(alpha=0.3, which='both')
        ax_n.set_ylim(1e-6, 3e2); ax_n.grid(alpha=0.3)
        ax_n.axhline(CV1, color='gray', ls=':', lw=0.6, alpha=0.6)
        ax_nN.set_ylim(np.log(1e-6 / CHI_INF), np.log(3e2 / CHI_INF))
        ax_cp.grid(alpha=0.3)
        ax_cf.set_ylim(-0.004, 0.012)
        ax_cf.grid(alpha=0.3); ax_cf.axhline(0, color='gray', lw=0.5, alpha=0.6)
        ax_cf.set_xlabel('$x/c$')
        ax_cf.set_xlim(0, 1)
        if col == 0:
            ax_reo.set_ylabel(r'$\max Re_\Omega$ (log)')
            ax_P.set_ylabel(r'$\max \hat\Omega \hat I$ (log)')
            ax_n.set_ylabel(r'$\chi$ (log)')
            ax_cp.set_ylabel(r'$-C_p$')
            ax_cf.set_ylabel(r'$C_{f,x}$')
        if col == len(ALPHAS) - 1:
            ax_nN.set_ylabel(r'strip $N$ (linear)')
    handles = [Line2D([], [], color='C0', lw=1.5, label='upper'),
               Line2D([], [], color='C3', lw=1.5, label='lower'),
               Line2D([], [], color='0.3', ls='-', lw=1.5, label='O-grid L2')]
    if got_cav:
        handles.append(Line2D([], [], color='0.3', ls='--', lw=1.5,
                              label='unstructured L2 (where complete)'))
    handles.append(Line2D([], [], color='0.5', ls='-.', lw=0.8,
                          label=r'onset threshold $Re_\Omega^c(\hat\Omega\hat I)$'))
    handles.append(Line2D([], [], color='0.35', ls=':', lw=1.4,
                          label=r'FlexFoil strip ($N$, $C_p$, $C_f$)'))
    axs[0, 0].legend(handles=handles, fontsize=7.5, loc='lower right')
    fig.suptitle(rf'Daedalus wing section $\eta={eta_q:.2f}$ '
                 rf'($Re_c={5e5*chord(eta_q)/chord(0):.2g}$)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = f"{PD}/figs/daedalus_section_eta{int(round(eta_q*100)):02d}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print('wrote', out, flush=True)


CACHE = {}


def probe_cached(case, surfname, etas, mu):
    surf = load(f'{root}/{case}/{surfname}')
    spts = vtk_to_numpy(surf.GetPoints().GetData())
    vol = load(f'{root}/{case}/volume.pvtu')
    out = {}
    for eta_q in etas:
        contours, c_loc, x_le, mst = section_contour(spts, eta_q)
        pr = probe_station(vol, case, contours, c_loc, x_le, mu)
        sr = surface_rows(case, surfname, contours, c_loc, x_le, mst, mu)
        out[eta_q] = (pr, sr)
    del vol
    return out


if __name__ == '__main__':
    etas = [float(x) for x in sys.argv[1:]] or [0.10, 0.75, 0.92]
    for a in ALPHAS:
        for fam, (root, tpl, surfname, ls) in CASES.items():
            case = tpl.format(a=a)
            if complete(case) and os.path.exists(f'{root}/{case}/chi_surface.npz'):
                mu = json.load(open(f'{root}/{case}/Flow360.json'))['freestream']['muRef']
                CACHE[case] = probe_cached(case, surfname, etas, mu)
                print('probed', case, flush=True)
    for eta_q in etas:
        make_sheet(eta_q)
