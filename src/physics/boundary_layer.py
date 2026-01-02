"""Blasius and Falkner-Skan boundary layer solutions."""

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def blasiusEqn(eta, y):
    """Blasius ODE: f''' = -f''·f/2"""
    f, g, h = y
    return g, h, -h*f/2


def findh0(h0):
    """Find wall shear h0 such that f'(∞) = 1."""
    f0, g0 = 0, 0
    initVals = np.array([f0, g0, h0], dtype=object)
    res = integrate.RK45(blasiusEqn, 0, initVals, 100)
    for i in range(100):
        res.step()
    return 1 - res.y[1]


def blasius():
    """Solve Blasius. Returns eta, u, du/deta, v·sqrt(Rex)."""
    f0, g0 = 0, 0
    h0 = optimize.fsolve(findh0, 0.3)[0]
    initVals = np.array([f0, g0, h0], dtype=float)
    res = integrate.solve_ivp(blasiusEqn, (0, 10), initVals, max_step=0.01)
    return res.t, res.y[1], res.y[2], 0.5 * (res.t * res.y[1] - res.y[0])


class Blasius:
    def __init__(self):
        self.eta, self.u, self.dudeta, self.v_sqrt_Rex = blasius()

    def at(self, Rex, yGrid, cellCentered=True):
        """Return y, u, dudy, v at given Rex interpolated to yGrid."""
        y = self.eta * np.sqrt(Rex)
        dudy = self.dudeta / np.sqrt(Rex)
        v = self.v_sqrt_Rex / np.sqrt(Rex)

        yGrid = np.array(yGrid, float)
        if cellCentered:
            yGrid = 0.5 * (yGrid[1:] + yGrid[:-1])
        u = np.interp(yGrid, y, self.u)
        dudy = np.interp(yGrid, y, dudy)
        v = np.interp(yGrid, y, v)
        return yGrid, u, dudy, v


def falknerSkanEqn(eta, y, beta):
    """Falkner-Skan ODE scaled for Blasius-consistent eta."""
    f, g, h = y
    coeff_f = 1.0 / (2.0 - beta)
    coeff_beta = beta / (2.0 - beta)
    return g, h, -coeff_f * f * h - coeff_beta * (1 - g**2)


def shoot_boundary_condition(h0, beta):
    """Shooting target: f'(∞) = 1."""
    f0, g0 = 0, 0
    initVals = np.array([f0, g0, h0], dtype=object)
    res = solve_ivp(
        fun=lambda t, y: falknerSkanEqn(t, y, beta),
        t_span=(0, 10),
        y0=initVals.astype(float),
        max_step=0.1
    )
    return 1.0 - res.y[1][-1]


def solve_falkner_skan(beta, guess=None):
    """Solve Falkner-Skan. Returns eta, u, dudy, v."""
    m = beta / (2.0 - beta)

    if guess is None:
        if beta > 0.6: guess = 1.2
        elif beta < 0: guess = 0.05
        else: guess = 0.05 + beta * 2

    h0 = fsolve(shoot_boundary_condition, guess, args=(beta,))[0]

    f0, g0 = 0, 0
    initVals = np.array([f0, g0, h0], dtype=object)
    res = solve_ivp(
        fun=lambda t, y: falknerSkanEqn(t, y, beta),
        t_span=(0, 10),
        y0=initVals.astype(float),
        max_step=0.01
    )

    eta = res.t
    f = res.y[0]
    u = res.y[1]
    dudy = res.y[2]
    v = 0.5 * ((1 - m) * eta * u - (1 + m) * f)

    return eta, u, dudy, v


class FalknerSkanWedge:
    def __init__(self, beta, guess=None):
        self.beta = beta
        self.eta, self.u, self.dudeta, self.v_sqrt_Rex = solve_falkner_skan(beta, guess)

    def at(self, Rex, yGrid, cellCentered=True):
        """Return y, u, dudy, v at given Rex interpolated to yGrid."""
        if cellCentered:
            y = 0.5 * (yGrid[:-1] + yGrid[1:])
        else:
            y = yGrid

        scale = self.inviscid_at(Rex)
        eta_target = y * np.sqrt(scale / Rex)

        u_prof = np.interp(eta_target, self.eta, self.u)
        du_prof = np.interp(eta_target, self.eta, self.dudeta)
        v_prof = np.interp(eta_target, self.eta, self.v_sqrt_Rex)

        u = u_prof * scale
        dudy = du_prof * (scale**1.5) / np.sqrt(Rex)
        v = v_prof * np.sqrt(scale / Rex)

        return y, u, dudy, v

    def inviscid_at(self, Rex):
        """Inviscid edge velocity at Rex."""
        return Rex**(self.beta / (2.0 - self.beta))
