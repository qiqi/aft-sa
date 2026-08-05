"""The flap's EQUIVALENT INVISCID BODY: surface + delta*, with the wake, built
from an mfoil viscous solution at the flap's own Reynolds number.

Purpose. Fine-tuning the fore element's trailing edge and the slot needs many
evaluations, which rules out RANS. The inviscid panel code is fast enough, but
the plain inviscid flap carries ~40% too much circulation (cl 1.985 against
1.459 viscous), which is exactly the error that produced the leading-edge
suction spike in the first place. Freezing the flap's delta* into its geometry
gives the panel code the right circulation without a viscous coupling of its own.

Construction. Offset every airfoil node outward along its normal by delta*, then
continue downstream as a slit of thickness delta*_wake about the wake path, and
close at the wake tip. The node loop is ordered

    wake tip (lower) -> airfoil lower TE -> LE -> airfoil upper TE -> wake tip

so the loop's own closure sits at the WAKE TIP, which is where the Kutta
condition belongs for a displacement body -- not at the airfoil trailing edge,
which is now an interior station.

The construction is validated against the cl it is supposed to reproduce; it is
not assumed to work.

Use:
    from flap_displacement import flap_body
    nodes, info = flap_body()      # in the GLOBAL frame, ready for panel2e
"""
import sys

import numpy as np

sys.path.insert(0, '/home/qiqi/flexcompute/flexfoil/mfoil')

import panel2e as M                                           # noqa: E402
import step1v_flap_viscous as V                                # noqa: E402

FLAP_CHORD, FLAP_INC, FLAP_LE = 0.30, -8.0, (0.70, 0.04)


def _outward_normals(X):
    """Unit outward normals at the nodes of a closed CW loop (2, N) -> (N, 2)."""
    P = X.T
    n = len(P)
    t = P[np.minimum(np.arange(n) + 1, n - 1)] - P[np.maximum(np.arange(n) - 1, 0)]
    t /= np.maximum(np.linalg.norm(t, axis=1), 1e-30)[:, None]
    nn = np.column_stack([-t[:, 1], t[:, 0]])
    c = P.mean(axis=0)
    nn[((P - c)*nn).sum(1) < 0] *= -1.0
    return nn


def flap_body(wake_len=None, aoa=None, re=None, ncrit=9.0, npanel=259,
              with_wake=True, verbose=True):
    """Equivalent inviscid body of the flap, in the GLOBAL frame."""
    aoa = V.AOA if aoa is None else aoa
    re = V.RE_FLAP if re is None else re
    m = V.run(aoa, re, True, ncrit=ncrit, npanel=npanel)
    N = m.foil.N
    ds = np.asarray(m.post.ds)
    Xf = np.asarray(m.foil.x)                      # (2, N), lower TE -> LE -> upper TE
    nf = _outward_normals(Xf)
    surf = Xf.T + ds[:N][:, None]*nf               # displaced airfoil surface

    info = dict(cl_mfoil=float(m.post.cl), cd_mfoil=float(m.post.cd),
                ds_te_lower=float(ds[0]), ds_te_upper=float(ds[N-1]),
                aoa=aoa, re=re)

    if with_wake:
        Xw = np.asarray(m.wake.x)                  # (2, Nw), TE -> downstream
        dsw = ds[N:N + m.wake.N]
        Pw = Xw.T
        tw = np.gradient(Pw, axis=0)
        tw /= np.maximum(np.linalg.norm(tw, axis=1), 1e-30)[:, None]
        nw = np.column_stack([-tw[:, 1], tw[:, 0]])   # +nw is the upper side
        if nw[0, 1] < 0:
            nw = -nw
        up = Pw + 0.5*dsw[:, None]*nw
        lo = Pw - 0.5*dsw[:, None]*nw
        if wake_len is not None:
            keep = (Pw[:, 0] - Pw[0, 0]) <= wake_len
            up, lo = up[keep], lo[keep]
        # wake tip (lower) -> upstream along wake lower -> airfoil lower TE
        # -> LE -> airfoil upper TE -> downstream along wake upper -> wake tip
        loop = np.vstack([lo[::-1][:-1], surf, up[1:]])
        info['n_wake'] = len(up)
    else:
        loop = surf
        info['n_wake'] = 0

    # drop duplicate consecutive points, which panel2e cannot panel
    keep = np.concatenate([[True],
                           np.linalg.norm(np.diff(loop, axis=0), axis=1) > 1e-9])
    loop = loop[keep]
    nodes = M.place(loop, FLAP_CHORD, FLAP_INC, *FLAP_LE)
    info['n_nodes'] = len(nodes)
    if verbose:
        print('flap displacement body: %d nodes (%d wake), '
              'delta*_TE lower %.5f upper %.5f, mfoil cl %.4f'
              % (info['n_nodes'], info['n_wake'], info['ds_te_lower'],
                 info['ds_te_upper'], info['cl_mfoil']))
    return nodes, info


if __name__ == '__main__':
    print('TARGETS for the flap alone, in flap-chord units')
    print('  mfoil viscous (Re=3e5, ncrit 9)   cl = 1.459')
    print('  SA-AI RANS, flap alone            cl = 1.356')
    print('  inviscid, real TE base            cl = 1.985  <- what we must NOT get')
    print()

    # plain inviscid reference, same section
    plain = M.place(M.airfoil_nodes(400, V.FLAP['m'], V.FLAP['p'], V.FLAP['t'],
                                   modified=False, te_thick=V.TE_THICK),
                    FLAP_CHORD, FLAP_INC, *FLAP_LE)
    _, r0 = M.solve_elements([plain], -1.0)
    print('%-34s cl = %.4f' % ('panel2e, plain inviscid', r0['Cl']))

    for wl, tag in ((None, 'full wake'), (0.60, 'wake 0.6c'), (0.30, 'wake 0.3c'),
                    (0.15, 'wake 0.15c')):
        nd, info = flap_body(wake_len=wl, verbose=False)
        _, r = M.solve_elements([nd], -1.0)
        print('%-34s cl = %.4f   (%d nodes)'
              % ('panel2e, displacement, ' + tag, r['Cl'], info['n_nodes']))

    nd, info = flap_body(with_wake=False, verbose=False)
    _, r = M.solve_elements([nd], -1.0)
    print('%-34s cl = %.4f   (%d nodes)'
          % ('panel2e, displacement, NO wake', r['Cl'], info['n_nodes']))
