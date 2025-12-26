import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

def blasiusEqn(eta, y):
    "RHS of the Blasius equations f''' = -f'' * f/2 for f, g=f', h=f''"
    f, g, h = y
    return g, h, -h*f/2

def findh0(h0):
    "Find the velocity slope at the wall"
    f0, g0 = 0, 0
    initVals = np.array([f0, g0, h0], dtype=object)
    res = integrate.RK45(blasiusEqn, 0, initVals, 100)
    for i in range(100):
        res.step()
    return 1 - res.y[1]

def blasius():
    """
    Solve for the Blasius equations.
    Returns eta, u, du/deta, and v * sqrt(Rex)
    """
    f0, g0 = 0, 0
    h0 = optimize.fsolve(findh0, 0.3)[0]
    initVals = np.array([f0, g0, h0], dtype=float)
    res = integrate.solve_ivp(blasiusEqn, (0, 10), initVals, max_step=0.01)
    return res.t, res.y[1], res.y[2], 0.5 * (res.t * res.y[1] - res.y[0])

class Blasius:
    def __init__(self):
        self.eta, self.u, self.dudeta, self.v_sqrt_Rex = blasius()

    def at(self, Rex, yGrid, cellCentered=True):
        '''
        return y, u, dudy
        Here y is in the unit of nu/Uinf
        u is in the unit of Uinf
        '''
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
    """
    Falkner-Skan ODE scaled to match Blasius definition of eta.
    eta = y * sqrt(Ue / (nu * x))
    ODE: f''' + (1/(2-beta)) * f * f'' + (beta/(2-beta)) * (1 - f'^2) = 0
    """
    f, g, h = y

    # Coefficients for Blasius consistency
    coeff_f = 1.0 / (2.0 - beta)
    coeff_beta = beta / (2.0 - beta)

    return g, h, -coeff_f * f * h - coeff_beta * (1 - g**2)

def shoot_boundary_condition(h0, beta):
    """
    Target function to find correct wall shear h0.
    """
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
    """
    Solves Falkner-Skan. Returns: eta, u, dudy, v
    """
    # 1. Determine m from beta
    m = beta / (2.0 - beta)

    # 2. Find h0 (wall shear)
    if guess is None:
        if beta > 0.6: guess = 1.2
        elif beta < 0: guess = 0.05
        else: guess = 0.05 + beta * 2

    h0 = fsolve(shoot_boundary_condition, guess, args=(beta,))[0]

    # 3. Solve ODE
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

    # 4. Calculate v (Scaled Normal Velocity)
    v = 0.5 * ((1 - m) * eta * u - (1 + m) * f)

    return eta, u, dudy, v

class FalknerSkanWedge:
    def __init__(self, beta, guess=None):
        self.beta = beta
        # Solve and cache the profile using the existing solver function
        self.eta, self.u, self.dudeta, self.v_sqrt_Rex = solve_falkner_skan(beta, guess)
        # print(self.u)

    def at(self, Rex, yGrid, cellCentered=True):
        '''
        return y, u, dudy
        Here y is in the unit of nu/Uref
        u is in the unit of Uref
        Rex = x Uref / nu
        '''
        # 1. Determine the calculation points y
        if cellCentered:
            y = 0.5 * (yGrid[:-1] + yGrid[1:])
        else:
            y = yGrid

        # 2. Scaling Factors for Wedge Flow
        # For wedge: Ue ~ x^m. In terms of Rex: Ue/Uref ~ Rex**(beta/(2-beta))
        # This factor converts local similarity units to global reference units.
        scale = self.inviscid_at(Rex)

        # 3. Calculate similarity variable eta
        # eta = y_phys * sqrt(Ue / nu x)
        #     = (y_code * nu/Uref) * sqrt( Ue / (nu * (Rex * nu / Uref)) )
        #     = y_code * sqrt(Re / Uref * Rex)
        eta_target = y * np.sqrt(scale / Rex)

        # 4. Interpolate (Piecewise Linear)
        # Use defaults (extrapolates end values) which works for u->1, dudy->0
        u_prof = np.interp(eta_target, self.eta, self.u)
        du_prof = np.interp(eta_target, self.eta, self.dudeta)
        v_prof = np.interp(eta_target, self.eta, self.v_sqrt_Rex)

        # 5. Rescale to physical/reference units
        u = u_prof * scale
        dudy = du_prof * (scale**1.5) / np.sqrt(Rex)
        v = v_prof * np.sqrt(scale / Rex)

        return y, u, dudy, v

    def inviscid_at(self, Rex):
        u = Rex**(self.beta / (2.0 - self.beta))
        return u
