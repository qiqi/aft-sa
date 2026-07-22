"""The three velocity-scale vectors P=d^2 grad^2 u, Omega=(omega x n)d, U=u,
comparable near the wall. For a 2D/parallel profile they are collinear, so
the local state is the signed triple s=(d^2 u'', d u', u)/R, R=|.|, a point
on S^2. Gamma and Lambda_v are two functions on this sphere (and in fact
X^2 = Lv^2/(Lv^2 + Gamma/2) shows they coordinatize it).

Plot where the canonical profiles LIVE on S^2: Blasius, adverse FS,
separation-limit, Stewartson LSB, free mixing layer, jet edge, wake.
Mark each profile's Rayleigh production peak (star) and the reference loci
(wall, dead-air crossing). Two views.

The point: for 2D these are DISTINCT points on the sphere (jet, dead-air,
mixing layer separate in X), but the amplify/suppress requirement is
non-monotonic and partly nonlocal. The genuinely-missing room is 3D: when
P, Omega, U are NOT collinear (crossflow, corner, tip), the extra rotational
invariants P.U, Omega.U and the triple product P.(Omega x U) appear -- which
(Gamma, Lambda_v) discard. That is where crossflow transition lives.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.linalg import eig
from scipy.optimize import minimize_scalar

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa
from explore_lsb_frozen_profile import build_profile

FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_explore')


def sphere_xyz(d, u, up, upp):
    """(X,Y,Z) = (d^2 u'', d u', u)/R along the profile (d = wall distance)."""
    X = d**2*upp
    Y = d*up
    Z = u
    R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-30
    return X/R, Y/R, Z/R


def fs_traj(beta, guess=None):
    pr = build_profile(beta, guess)
    Th = pr['Theta']
    d = pr['eta']/Th          # wall distance in theta units
    m = (pr['eta'] < np.interp(0.999, np.maximum.accumulate(pr['u']),
                               pr['eta'])) & (d > 0.02)
    return sphere_xyz(d[m], pr['u'][m], Th*pr['up'][m], Th*Th*pr['upp'][m])


def free_traj(uf, upf, uppf, H=12.0):
    y = np.linspace(0.2, H+8, 600)
    return sphere_xyz(y, uf(y), upf(y), uppf(y))


def rayleigh_peak_xyz(uf, upf, uppf, ymax):
    y = np.linspace(0, ymax, 900); h = y[1]-y[0]
    u, up, upp = uf(y), upf(y), uppf(y)
    n = len(y)-2
    D2 = (np.diag(np.ones(n-1), -1)-2*np.eye(n)+np.diag(np.ones(n-1), 1))/h**2

    def wi(al, vec=False):
        B = D2-al**2*np.eye(n); A = np.diag(u[1:-1])@B - np.diag(upp[1:-1])
        c, V = eig(A, B, right=True); j = int(np.argmax(np.imag(c)))
        return (V[:, j] if vec else al*float(np.imag(c[j])))
    r = minimize_scalar(lambda a: -wi(a), bounds=(0.05, 1.4),
                        method='bounded', options={'xatol': 3e-3})
    v = wi(r.x, vec=True); phi = np.zeros(len(y), complex); phi[1:-1] = v
    dphi = np.gradient(phi, h)
    p = 0.5*r.x*np.abs(np.imag(dphi*np.conj(phi)))*np.abs(up)
    jp = int(np.argmax(p))
    x, yy, z = sphere_xyz(np.array([y[jp]]), u[jp:jp+1], up[jp:jp+1],
                          upp[jp:jp+1])
    return x[0], yy[0], z[0]


def main():
    os.makedirs(FIGD, exist_ok=True)
    fig = plt.figure(figsize=(14, 6.5))
    trajs = []
    # wall-bounded FS
    for beta, guess, lab, col in [
            (0.0, None, 'Blasius', 'C0'),
            (-0.1988, None, 'sep limit', 'C1'),
            (-0.15, -0.08, 'Stewartson LSB', 'C3')]:
        X, Y, Z = fs_traj(beta, guess)
        trajs.append((lab, col, X, Y, Z))
    # free shear
    for (uf, upf, uppf), lab, col in [
        ((lambda y: np.tanh(y-12), lambda y: 1/np.cosh(y-12)**2,
          lambda y: -2*np.tanh(y-12)/np.cosh(y-12)**2), 'mixing layer', 'C2'),
        ((lambda y: 1/np.cosh(y-12)**2,
          lambda y: -2*np.sinh(y-12)/np.cosh(y-12)**3,
          lambda y: (-2+4*np.sinh(y-12)**2/np.cosh(y-12)**2)/np.cosh(y-12)**2),
         'jet edge', 'C4'),
        ((lambda y: 1-0.7/np.cosh(y-12)**2,
          lambda y: 1.4*np.sinh(y-12)/np.cosh(y-12)**3,
          lambda y: -0.7*(-2+4*np.sinh(y-12)**2/np.cosh(y-12)**2)/np.cosh(y-12)**2),
         'wake', 'C5')]:
        X, Y, Z = free_traj(uf, upf, uppf)
        trajs.append((lab, col, X, Y, Z))

    # reference points on the sphere from (Gamma, Lv):
    def pt(G, Lv, zsign=+1, ysign=+1):
        X2 = Lv**2/(Lv**2 + G/2)
        rem = 1-X2
        Y2 = G*rem/2; Z2 = rem - Y2
        X = -np.sign(Lv)*np.sqrt(X2)*ysign  # XY sign from Lv=-XY/rem>0? keep simple
        Y = ysign*np.sqrt(max(Y2, 0)); Z = zsign*np.sqrt(max(Z2, 0))
        # fix X sign so that -X*Y/rem = Lv
        X = -Lv*rem/Y if Y != 0 else np.sqrt(X2)
        return X, Y, Z
    refs = [('wall (G=1,Lv=0)', 1.0, 0.0, 'k', 'o'),
            ('dead-air (G=2,Lv=-2)', 1.999, -2.0, 'k', 'X'),
            ('mixing pk (G=2,Lv=.07)', 1.999, 0.07, 'C2', '*'),
            ('jet pk (G=2,Lv=-3.5)', 1.999, -3.5, 'C4', '*'),
            ('wake pk (G=2,Lv=-3.5)', 1.999, -3.5, 'C5', '*')]

    for vi, (elev, azim) in enumerate([(20, -60), (20, 30)]):
        ax = fig.add_subplot(1, 2, vi+1, projection='3d')
        us, vs = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        ax.plot_wireframe(np.cos(us)*np.sin(vs), np.sin(us)*np.sin(vs),
                          np.cos(vs), color='0.9', lw=0.4)
        for lab, col, X, Y, Z in trajs:
            ax.plot(X, Y, Z, color=col, lw=2.0, label=lab if vi == 0 else None)
        for lab, G, Lv, col, mk in refs:
            X, Y, Z = pt(G, Lv)
            ax.scatter([X], [Y], [Z], color=col, marker=mk, s=90,
                       edgecolors='k', zorder=6,
                       label=lab if vi == 0 else None)
        ax.set_xlabel(r"$X=d^2u''/R$ (curv.)")
        ax.set_ylabel(r"$Y=d\,u'/R$ (shear)")
        ax.set_zlabel(r"$Z=u/R$ (vel.)")
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
    fig.legend(loc='lower center', ncol=5, fontsize=7.5)
    fig.suptitle('Canonical profiles on the indicator sphere '
                 r"$(d^2u'',\,d\,u',\,u)/R$ (2D collinear reduction; "
                 r"$(\Gamma,\Lambda_v)$ coordinatize it)", fontsize=11)
    plt.tight_layout(rect=(0, 0.06, 1, 0.97))
    out = os.path.join(FIGD, 'indicator_sphere.png')
    plt.savefig(out, dpi=140)
    print('wrote', out)

    # tabulate production peaks on the sphere
    print("\nproduction-peak sphere coords (free-shear, far from wall):")
    for (uf, upf, uppf), lab in [
        ((lambda y: np.tanh(y-12), lambda y: 1/np.cosh(y-12)**2,
          lambda y: -2*np.tanh(y-12)/np.cosh(y-12)**2), 'mixing layer'),
        ((lambda y: 1/np.cosh(y-12)**2,
          lambda y: -2*np.sinh(y-12)/np.cosh(y-12)**3,
          lambda y: (-2+4*np.sinh(y-12)**2/np.cosh(y-12)**2)/np.cosh(y-12)**2),
         'jet edge'),
        ((lambda y: 1-0.7/np.cosh(y-12)**2,
          lambda y: 1.4*np.sinh(y-12)/np.cosh(y-12)**3,
          lambda y: -0.7*(-2+4*np.sinh(y-12)**2/np.cosh(y-12)**2)/np.cosh(y-12)**2),
         'wake')]:
        X, Y, Z = rayleigh_peak_xyz(uf, upf, uppf, 21.0)
        print(f"  {lab:>14}: (X,Y,Z)=({X:+.3f},{Y:+.3f},{Z:+.3f})")


if __name__ == '__main__':
    main()
