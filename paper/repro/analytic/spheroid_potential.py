"""Exact potential flow on a 6:1 prolate spheroid at incidence, to O(alpha).

Prolate spheroidal coordinates (xi, eta, phi):

    x = c xi eta,     rho = c sqrt((xi^2-1)(1-eta^2)),
    h_xi = c sqrt((xi^2-eta^2)/(xi^2-1)),
    h_eta = c sqrt((xi^2-eta^2)/(1-eta^2)),
    h_phi = c sqrt((xi^2-1)(1-eta^2))

with c = sqrt(a^2-b^2) the focal half-distance and the body at xi = xi0 = a/c.
On the surface x = a eta and rho = b sqrt(1-eta^2), so eta = x/a runs -1 (nose)
to +1 (tail).

Axial stream U along +x:
    Phi = U c xi eta + A P1(eta) Q1(xi),    Q1(xi) = (xi/2) ln((xi+1)/(xi-1)) - 1
Transverse stream W along the transverse axis, at azimuth phi from that axis:
    Phi = W c P11(xi) P11(eta) cos phi + B P11(eta) Q11(xi) cos phi
    P11(z) = sqrt(z^2-1) or sqrt(1-z^2);
    Q11(xi) = sqrt(xi^2-1) [ (1/2) ln((xi+1)/(xi-1)) - xi/(xi^2-1) ]
Both A and B follow from d(Phi)/d(xi) = 0 on xi = xi0.

At incidence alpha the freestream is U(cos alpha, transverse sin alpha), and to
first order the surface velocity is

    u_s   = U [ f0(eta) + alpha f1(eta) cos(phi_w) ]      meridional
    u_phi = U   alpha g(eta) sin(phi_w)                   azimuthal

with phi_w measured from the WINDWARD generator, so that phi_w = 0 carries the
stagnation point and g > 0 means the surface flow spills windward -> leeward.

Nothing here is taken on trust: `verify()` checks the wall boundary condition
d(Phi)/d(xi) = 0 on the body, the far-field limit Phi -> U x + W z, the
incompressible surface c_p against 1 - (u_s^2+u_phi^2)/U^2, and the sign of the
stagnation point, and `main()` prints all of it.

Run from paper/:  python3 repro/analytic/spheroid_potential.py
"""
import numpy as np

A_AX, B_AX = 0.5, 1.0 / 12.0            # spheroid semi-axes / L
C_FOC = np.sqrt(A_AX**2 - B_AX**2)
XI0 = A_AX / C_FOC


def _L(xi):
    return 0.5 * np.log((xi + 1.0) / (xi - 1.0))


def Q1(xi):
    return xi * _L(xi) - 1.0


def dQ1(xi):
    return _L(xi) - xi / (xi**2 - 1.0)


def Q11(xi):
    return np.sqrt(xi**2 - 1.0) * (_L(xi) - xi / (xi**2 - 1.0))


def dQ11(xi):
    r = np.sqrt(xi**2 - 1.0)
    d = _L(xi) - xi / (xi**2 - 1.0)
    dd = -1.0 / (xi**2 - 1.0) - (1.0 / (xi**2 - 1.0)
                                 - 2.0 * xi**2 / (xi**2 - 1.0)**2)
    return xi / r * d + r * dd


# --- coefficients, fixed once by the wall condition on xi = xi0 ---
A_COEF = -C_FOC / dQ1(XI0)                                  # per unit U
B_COEF = -C_FOC * (XI0 / np.sqrt(XI0**2 - 1.0)) / dQ11(XI0)  # per unit W


def phi_axial(xi, eta):
    return C_FOC * xi * eta + A_COEF * eta * Q1(xi)


def phi_trans(xi, eta, phi):
    p11x, p11e = np.sqrt(xi**2 - 1.0), np.sqrt(1.0 - eta**2)
    return (C_FOC * p11x * p11e + B_COEF * p11e * Q11(xi)) * np.cos(phi)


def metrics(xi, eta):
    return (C_FOC * np.sqrt((xi**2 - eta**2) / (xi**2 - 1.0)),
            C_FOC * np.sqrt((xi**2 - eta**2) / (1.0 - eta**2)),
            C_FOC * np.sqrt((xi**2 - 1.0) * (1.0 - eta**2)))


def surface_fields(eta):
    """-> f0, f1, g at surface stations eta (|eta| < 1), all per unit U.

    f0  meridional speed at alpha = 0 (positive aft)
    f1  O(alpha) meridional perturbation, multiplies cos(phi_w)
    g   O(alpha) azimuthal speed, multiplies sin(phi_w)
    """
    xi = XI0
    _, h_eta, h_phi = metrics(xi, eta)
    # axial part: u_eta = (1/h_eta) dPhi/deta
    d_ax = C_FOC * xi + A_COEF * Q1(xi)                 # d/deta of phi_axial
    f0 = d_ax / h_eta
    # transverse part, phi measured from the TRANSVERSE axis:
    #   Phi_tr = T(xi) sqrt(1-eta^2) cos phi,  T = c P11(xi) + B Q11(xi)
    T = C_FOC * np.sqrt(xi**2 - 1.0) + B_COEF * Q11(xi)
    dT_deta = T * (-eta / np.sqrt(1.0 - eta**2))
    f1_raw = dT_deta / h_eta                 # multiplies cos(phi from trans axis)
    g_raw = -T * np.sqrt(1.0 - eta**2) / h_phi   # (1/h_phi) dPhi/dphi, sin term
    # The freestream transverse component points from the windward generator
    # toward the body, so phi_w = 0 (windward) corresponds to cos = -1 in the
    # transverse-axis frame: flip the sign of both perturbations.
    return f0, -f1_raw, -g_raw


def arc_and_radius(eta):
    """Meridional arc length from the nose (eta = -1) and body radius."""
    ee = np.linspace(-1.0, 1.0, 200001)
    _, h_eta, _ = metrics(XI0, np.clip(ee, -1 + 1e-12, 1 - 1e-12))
    s = np.concatenate([[0.0], np.cumsum(0.5 * (h_eta[1:] + h_eta[:-1])
                                         * np.diff(ee))])
    return np.interp(eta, ee, s), B_AX * np.sqrt(1.0 - eta**2)


def verify():
    out = {}
    eta = np.linspace(-0.995, 0.995, 41)
    h = 1e-6
    # 1. wall condition dPhi/dxi = 0 on xi = xi0, both modes
    d_ax = (phi_axial(XI0 + h, eta) - phi_axial(XI0 - h, eta)) / (2 * h)
    d_tr = (phi_trans(XI0 + h, eta, 0.0)
            - phi_trans(XI0 - h, eta, 0.0)) / (2 * h)
    out['wall BC axial   max|dPhi/dxi|'] = float(np.abs(d_ax).max())
    out['wall BC transv. max|dPhi/dxi|'] = float(np.abs(d_tr).max())
    # 2. far field: Phi -> U x (axial) and W z (transverse)
    xi = 4.0e3
    x = C_FOC * xi * eta
    z = C_FOC * np.sqrt((xi**2 - 1.0) * (1.0 - eta**2))
    out['far field axial   max|Phi/x - 1|'] = float(
        np.abs(phi_axial(xi, eta) / x - 1.0).max())
    out['far field transv. max|Phi/z - 1|'] = float(
        np.abs(phi_trans(xi, eta, 0.0) / z - 1.0).max())
    # 3. c_p from the O(alpha) fields against c_p from the full potential
    al = 0.02
    f0, f1, g = surface_fields(eta)
    for phw in (0.0, np.pi / 2, np.pi):
        us = f0 + al * f1 * np.cos(phw)
        up = al * g * np.sin(phw)
        cp_lin = 1.0 - (us**2 + up**2)
        # full: superpose the two exact modes at the same incidence
        phi_t = np.pi - phw                     # transverse-axis frame
        _, h_eta, h_phi = metrics(XI0, eta)
        d_ax = C_FOC * XI0 + A_COEF * Q1(XI0)
        T = C_FOC * np.sqrt(XI0**2 - 1.0) + B_COEF * Q11(XI0)
        us_f = (np.cos(al) * d_ax
                + np.sin(al) * T * (-eta / np.sqrt(1 - eta**2))
                * np.cos(phi_t)) / h_eta
        up_f = np.sin(al) * (-T * np.sqrt(1 - eta**2) / h_phi) * np.sin(phi_t)
        cp_full = 1.0 - (us_f**2 + up_f**2)
        out[f'c_p linear vs full, phi_w={np.degrees(phw):5.0f} deg  max|d|'] = \
            float(np.abs(cp_lin - cp_full).max())
    # 4. stagnation point sits on the WINDWARD generator, just aft of the nose
    en = np.linspace(-0.9999, -0.90, 20001)
    f0n, f1n, _ = surface_fields(en)
    for phw, name in ((0.0, 'windward'), (np.pi, 'leeward')):
        u = f0n + al * f1n * np.cos(phw)
        sgn = np.where(np.diff(np.sign(u)) != 0)[0]
        s_n, _ = arc_and_radius(en)
        out[f'stagnation on {name}'] = (
            f'x/L = {0.5*(1+en[sgn[0]]):.5f}, s/L = {s_n[sgn[0]]:.5f}'
            if len(sgn) else 'none (u_s never vanishes)')
    return out


def main():
    print(f'a = {A_AX}, b = {B_AX}, c = {C_FOC:.6f}, xi0 = {XI0:.6f}')
    print(f'A = {A_COEF:.6f} (per U), B = {B_COEF:.6f} (per W)\n')
    for k, v in verify().items():
        print(f'  {k:52s} {v if isinstance(v, str) else f"{v:.3e}"}')
    print(f'\n{"x/L":>6} {"f0":>9} {"f1":>9} {"g":>9} {"cp(a=0)":>9}')
    for xl in (0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80):
        eta = 2 * xl - 1
        f0, f1, g = surface_fields(np.array([eta]))
        print(f'{xl:6.2f} {f0[0]:9.4f} {f1[0]:9.4f} {g[0]:9.4f} '
              f'{1-f0[0]**2:9.4f}')


if __name__ == '__main__':
    main()
