"""a_max as an eigenvalue: temporal Rayleigh problem on the tanh mixing layer.

The paper sets the free-shear amplification ceiling a_max to the ratio of the
most-amplified temporal growth rate to the peak vorticity of the
hyperbolic-tangent layer (Michalke 1964). This script derives that number from
scratch -- solve the Rayleigh equation (U-c)(phi'' - a^2 phi) - U'' phi = 0 on
U = tanh(y) as a generalized eigenproblem, maximize omega_i = a*Im(c) over the
wavenumber -- and asserts it equals the canonical A_MAX to the paper's two
digits. The ratio omega_i,max/omega_peak is independent of the velocity and
thickness normalization.
"""
import numpy as np
from scipy.linalg import eig
from scipy.optimize import minimize_scalar

import _saai  # noqa: F401  (canonical-constant plumbing)
from _saai import A_MAX


def omega_i(alpha: float, N: int = 1200, L: float = 25.0) -> float:
    """Max temporal growth rate at wavenumber alpha (omega_peak = U'(0) = 1)."""
    y = np.linspace(-L, L, N)
    h = y[1] - y[0]
    U, Upp = np.tanh(y), -2*np.tanh(y)/np.cosh(y)**2
    D2 = (np.diag(np.ones(N-1), -1) - 2*np.eye(N) + np.diag(np.ones(N-1), 1))/h**2
    B = D2 - alpha**2*np.eye(N)
    A = np.diag(U) @ B - np.diag(Upp)
    c = eig(A[1:-1, 1:-1], B[1:-1, 1:-1], right=False)   # Dirichlet far-field BCs
    return alpha*float(np.max(np.imag(c)))


def main():
    r = minimize_scalar(lambda a: -omega_i(a), bounds=(0.3, 0.6), method="bounded",
                        options={"xatol": 1e-4})
    alpha, wmax = r.x, -r.fun
    print(f"tanh layer, temporal Rayleigh problem:")
    print(f"  omega_i,max / omega_peak = {wmax:.4f}   at alpha*delta = {alpha:.3f}")
    print(f"  canonical a_max          = {A_MAX}")
    assert abs(wmax - A_MAX) < 0.005, f"eigenvalue {wmax:.4f} != a_max {A_MAX}"
    print("  OK: a_max is the tanh-layer eigenvalue to the paper's two digits.")


if __name__ == "__main__":
    main()
