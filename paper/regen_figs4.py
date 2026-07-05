"""Fig 8 (NACA0012) and new Fig (NLF0416): Cp + Cf at a=4, BOTH surfaces.
Color = surface (upper/lower); line style = solution (SA-AI cavity / O-grid / mfoil)."""
import pickle, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.rcParams.update({'font.size': 10.5, 'font.family': 'serif', 'axes.linewidth': 0.8})
B = "/home/qiqi/flexcompute/aft-sa/flow360"; PD = "/home/qiqi/flexcompute/aft-sa/paper"
UP, LO = 'C0', 'C3'   # surface colors
def wd(af, mesh, a):
    if mesh == 'cavity': return f"{B}/cavprop_{af}_a{a}"
    if mesh == 'ogrid':  return f"{B}/strprop_{af}_a{a}"
    return f"{B}/full_{af}_{mesh}_aftsa_m2_a{a}"
def _walk_contour(d, af, field):
    """Robust upper/lower split: nearest-neighbor walk + TE-shared exclusion.
    Returns ((x_upper, v_upper), (x_lower, v_lower)) sorted by x.
    Fixes the upper/lower confusion on cambered airfoils (e.g. NLF) where the
    naive z>0 / z<0 split mis-assigns near-TE points.
    """
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{d}/surface_fluid_{af}.pvtu"); r.Update()
    g = r.GetOutput(); pts = vtk_to_numpy(g.GetPoints().GetData()); pd = g.GetPointData()
    nm = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    arr = vtk_to_numpy(pd.GetArray(next(n for n in nm if n.lower().startswith(field))))
    v = np.linalg.norm(arr, axis=1) if (field == 'cf' and arr.ndim > 1) else arr
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    s = np.abs(y - y.min()) < 1e-6
    X, Z, V = x[s], z[s], v[s]; n = len(X); pp = np.column_stack([X, Z])
    st = int(np.argmin(X)); o = [st]; u = np.zeros(n, bool); u[st] = True
    for _ in range(n-1):
        c = o[-1]; dd = np.sum((pp - pp[c])**2, 1); dd[u] = 1e9
        nx = int(np.argmin(dd)); o.append(nx); u[nx] = True
    o = np.array(o); xo, zo, vo = X[o], Z[o], V[o]
    # Robust TE-split (handles closed-TE unstructured + open-TE O-grid).
    mid = n // 2
    window = slice(max(0, mid - n//8), min(n, mid + n//8))
    te = window.start + int(np.argmax(xo[window]))
    b1, b2 = slice(0, te), slice(te+1, n)   # exclude TE-shared point
    up, lo = (b1, b2) if zo[b1].mean() >= zo[b2].mean() else (b2, b1)
    def srt(sl):
        xs = xo[sl]; oo = np.argsort(xs); xs, vs = xs[oo], vo[sl][oo]
        if len(xs) > 2 and xs[0] < 0.001 and xs[1] > 2*xs[0]: xs, vs = xs[1:], vs[1:]
        return xs, vs
    return srt(up), srt(lo)

def surf2(d, af, field, nb=45):
    # Use the robust contour walk + TE-shared exclusion.
    (xu, vu), (xl, vl) = _walk_contour(d, af, field)
    out = []
    for xs, vs in [(xu, vu), (xl, vl)]:
        if len(vs) < 10:
            out.append((xs, vs)); continue
        # Rolling-mean smooth to suppress cavity tangential-staggering noise.
        w = max(5, len(vs) // 30)
        k = np.ones(w) / w; vsm = np.convolve(vs, k, mode='same')
        h = w // 2; xs2, vsm = xs[h:len(xs)-h], vsm[h:len(vsm)-h]
        if len(xs2) < 3:
            out.append((xs, vs)); continue
        xc = np.linspace(xs2.min(), xs2.max(), 60); vc = np.interp(xc, xs2, vsm)
        out.append((xc, vc))
    return out  # [(x,v) upper, (x,v) lower]

def cpcf(af, a, title):
    mfs = pickle.load(open(f"{PD}/data/mfoil_surf_{af}.pkl", 'rb'))[float(a)]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1))
    # Both grids equally prominent; linestyle distinguishes grid/solver (per Fig 13 spec)
    sols = [('cavity', wd(af, 'cavity', a), '--', 1.4, 1.0),
            ('ogrid',  wd(af, 'ogrid',  a), '-',  1.4, 1.0)]
    # ---- Cp (left): -Cp ----
    for nm, d, ls, lw, a_ in sols:
        (xu, cu), (xl, cl) = surf2(d, af, 'cp')
        axs[0].plot(xu, -cu, ls, color=UP, lw=lw, alpha=a_); axs[0].plot(xl, -cl, ls, color=LO, lw=lw, alpha=a_)
    axs[0].plot(mfs['x_upper'], -np.asarray(mfs['cp_upper']), ':', color=UP, lw=1.4)
    axs[0].plot(mfs['x_lower'], -np.asarray(mfs['cp_lower']), ':', color=LO, lw=1.4)
    axs[0].set_xlabel('$x/c$'); axs[0].set_ylabel('$-C_p$'); axs[0].set_xlim(0, 1); axs[0].set_title(r'$C_p$, $\alpha=%d^\circ$' % a, fontsize=10)
    # ---- Cf (right) ----
    for nm, d, ls, lw, a_ in sols:
        (xu, cu), (xl, cl) = surf2(d, af, 'cf')
        axs[1].plot(xu, cu, ls, color=UP, lw=lw, alpha=a_); axs[1].plot(xl, cl, ls, color=LO, lw=lw, alpha=a_)
    axs[1].plot(mfs['x_upper'], np.asarray(mfs['cf_upper']), ':', color=UP, lw=1.4)
    axs[1].plot(mfs['x_lower'], np.asarray(mfs['cf_lower']), ':', color=LO, lw=1.4)
    axs[1].set_xlabel('$x/c$'); axs[1].set_ylabel('$C_f$'); axs[1].set_xlim(0, 1); axs[1].set_ylim(0, None); axs[1].set_title(r'$C_f$, $\alpha=%d^\circ$' % a, fontsize=10)
    surfh = [Line2D([], [], color=UP, lw=2, label='upper'), Line2D([], [], color=LO, lw=2, label='lower')]
    solh = [Line2D([], [], color='0.3', ls='-', label='SA-AI, O-grid'), Line2D([], [], color='0.3', ls='--', label='SA-AI, unstructured'), Line2D([], [], color='0.3', ls=':', label='mfoil ($e^N$)')]
    axs[0].legend(handles=surfh, fontsize=7.5, frameon=False, loc='upper right')
    axs[1].legend(handles=solh, fontsize=7.5, frameon=False, loc='upper right')
    plt.tight_layout(pad=0.4); plt.savefig(f"{PD}/figs/{af}_cp_cf_a{a}.pdf"); plt.savefig(f"/tmp/{af}_cpcf.png", dpi=120)
    print("wrote", f"{af}_cp_cf_a{a}.pdf")

cpcf('naca0012', 4, 'NACA 0012')
cpcf('nlf0416', 0, 'NLF(1)-0416')
