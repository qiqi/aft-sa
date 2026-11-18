"""Canonical SA-AI amplification kernel on the RP2 indicator space.

SINGLE SOURCE for the repro folder. Mirrors ModelConstants.h (the sa-ai
consistency test `tests/test_constants_consistency.py` pins these against the
header text) and SAAiTransition.h::__aiRateFromXYZ.

    X = u,  Y = y u',  Z = y^2 u''/2          (velocity, shear, curvature)
    U_loc = |(X,Y,Z)|                         (local velocity scale)
    Omega_hat = Y/sqrt(X^2+Y^2),  I_hat = (Y - X - Z)/U_loc
    OmegaI = Omega_hat * I_hat                (projectively even product)

    rate  = a_max * clip(OmegaI, 0, 1)               [LINEAR in OmegaI]
    Re_Omega_crit = softmin_n(REOM_CEIL, REOM_A + REOM_B * OmegaI^-2)
    sigma_Re = (1 + tanh((Re_Omega/Re_Omega_crit - 1)/RAMP_W))/2  (onset gate)
    a        = rate * sigma_Re                [dimensionless]

The normalized triples are often drawn on an indicator sphere, but the model's
state space is RP2: antipodal triples are identical. Every modeled scalar must
therefore be even under (X,Y,Z)->(-X,-Y,-Z); P satisfies that requirement.

There is NO sigmoid, NO Q-gate and NO separate favorable-gradient treatment:
the favorable branch is held near-laminar by the definite product P <~ 0.

The shape (CEIL, A, B) = (2600, 175, 2) is fixed by the LST neutral-point graze
of the Falkner-Skan family (fig02_onset_graze.py, NEVER refit). The single
WHOLE-EQUATION scale K_ANCHOR multiplies the entire threshold (softmin_n is
degree-1 homogeneous, so all three stored constants carry it): the residual
laminar diffusion delays net growth, so inception must be permitted earlier
than the inviscid graze.

This module REPLACES the retired Gamma-sigmoid + Q4-gate kernel that used to
live in lib/aft_sources.py (removed 2026-07-30). That kernel's constants
(g_c, s, reOmegaFloor, K_lambda, K_r, c_A, c_V, c_2) are no longer in the
paper; ModelConstants.h keeps its fields only for layout stability and
old-run replay.
"""
import numpy as np

# --- constants, verbatim from ModelConstants.h -----------------------------
A_MAX = 0.19            # ai_rateScale; Michalke free-shear eigenvalue 0.1897
K_ANCHOR = 0.712        # whole-equation drain-compensation scale
REOM_CEIL = 2600.0*K_ANCHOR   # ai_reOmCeil  = 1851.2
REOM_A = 175.0*K_ANCHOR       # ai_reOmA     = 124.6
REOM_B = 2.0*K_ANCHOR         # ai_reOmB     = 1.424
REOM_N = 2.0                  # soft-min sharpness
RAMP_W = 0.35                 # ai_rampWidth
C_NU_AI = 1.0/6.0             # ai_nuLamScale: c_nu,ai in the SA diffusion only
SIGMA_SA = 2.0/3.0            # SA sigma
C_V1 = 7.1                    # standard SA c_v1 (transition fires at chi=c_v1)


def sphere_indicators(u, dudy, d2u, y):
    """(OmegaI, Omega_hat, I_hat, U_loc) at wall distances y."""
    X, Y, Z = u, y*dudy, 0.5*y*y*d2u
    U_loc = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
    Omega_hat = Y/np.sqrt(X*X + Y*Y + 1e-30)
    I_hat = (Y - X - Z)/U_loc
    return Omega_hat*I_hat, Omega_hat, I_hat, U_loc


def reom_crit(OmegaI):
    """Soft-min onset threshold Re_Omega,crit(OmegaI)."""
    pw = REOM_A + REOM_B*np.maximum(OmegaI, 1e-6)**(-2.0)
    return (REOM_CEIL**(-REOM_N) + pw**(-REOM_N))**(-1.0/REOM_N)


def sphere_rate(u, dudy, d2u, y, nu=1.0, want_parts=False):
    """Dimensionless rate x onset. Caller forms the source as b = rate*|dudy|.

    nu is the laminar kinematic viscosity in the SAME units as u, y (so
    nu = 1/Re_L when u is scaled on U_e and y on L). fig04_shapefactor's
    normalisation is nu = 1.
    """
    OmegaI, _, _, _ = sphere_indicators(u, dudy, d2u, y)
    rate = A_MAX*np.minimum(1.0, np.clip(OmegaI, 0.0, None))
    ReOm = y*y*np.abs(dudy)/nu
    rc = reom_crit(OmegaI)
    sigma_Re = 0.5*(1.0 + np.tanh((ReOm/rc - 1.0)/RAMP_W))
    if want_parts:
        return rate*sigma_Re, dict(OmegaI=OmegaI, rate=rate,
                                   sigma_Re=sigma_Re, ReOm=ReOm, reomc=rc)
    return rate*sigma_Re
