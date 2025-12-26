import numpy as np

def dN_dRe_theta(H, Mach=0.0):
    """
    Calculate dN/dRe_theta using the authoritative Drela-Giles (1987) correlation.
    Source: Eqn 29 and Eqn 9 from Drela & Giles (1987).
    """

    # 1. Convert H to Kinematic Shape Factor Hk (Eqn 9)
    # For low speed (Mach ~ 0), Hk approx H.
    Hk = (H - 0.290 * Mach**2) / (1.0 + 0.113 * Mach**2)

    # 2. Calculate the slope (Eqn 29)
    # The term inside the tanh handles the "knee" of the stability curve.
    inner_term = 2.4 * Hk - 3.7 + 2.5 * np.tanh(1.5 * Hk - 4.65)

    # The hypotenuse form (sqrt(x^2 + 0.25)) prevents the rate from dropping
    # to exactly zero, maintaining a tiny 'floor' rate.
    slope = 0.01 * np.sqrt(inner_term**2 + 0.25)

    return slope

def Re_theta0(H, Mach=0.0):
    """
    Calculate the Critical Momentum Thickness Reynolds Number (Re_theta0)
    using the authoritative Drela-Giles (1987) correlation.

    Source: Eqn 30 and Eqn 9 from Drela & Giles (1987).
    """

    # 1. Convert H to Kinematic Shape Factor Hk (Eqn 9) [cite: 1539]
    # H_k is defined based on the velocity profile shape only.
    Hk = (H - 0.290 * Mach**2) / (1.0 + 0.113 * Mach**2)

    # 2. Safety Clip for Hk
    # The correlation has singularities at Hk = 1.0.
    # Physical Hk is usually > 2.0 for laminar flow.
    # We clip it slightly above 1.0 to prevent division by zero errors.
    Hk = np.maximum(Hk, 1.05)

    # 3. Calculate log10(Re_theta0) (Eqn 30) [cite: 1650]
    term1 = (1.415 / (Hk - 1)) - 0.489
    term2 = np.tanh((20.0 / (Hk - 1)) - 12.9)
    term3 = (3.295 / (Hk - 1)) + 0.44

    log_Re_theta0 = term1 * term2 + term3

    # 4. Return the actual Reynolds number
    return 10.0**log_Re_theta0

def compute_nondimensional_spatial_rate(H, Mach=0.0):
    """
    Calculates the Nondimensional Spatial Envelope Growth Rate: theta * dN/dx.

    This combines the Envelope Slope (Eq 29) with the Falkner-Skan
    spatial transformation (Eqs 33, 34) from Drela & Giles (1987).

    Returns:
        theta_dN_dx: The growth rate of N per unit (x/theta).
                     Values: ~0.002 (Blasius) to ~0.03 (Separation).
    """

    # --- 1. Kinematic Shape Factor Hk (Eq 9) ---
    Hk = (H - 0.290 * Mach**2) / (1.0 + 0.113 * Mach**2)

    # Clip Hk for safety (Singularity at Hk=1.0)
    # Physical limit for attached laminar flow is Hk > 2.0
    Hk = np.maximum(Hk, 1.05)

    # --- 2. Envelope Slope dN/dRe_theta (Eq 29) ---
    # The intrinsic sensitivity of the profile to instability
    inner_term = 2.4 * Hk - 3.7 + 2.5 * np.tanh(1.5 * Hk - 4.65)
    dN_dRe = 0.01 * np.sqrt(inner_term**2 + 0.25)

    # --- 3. Spatial Transformation Factors (Eqs 32-34) ---
    # We need to compute the factor: 0.5 * l * (m + 1)
    # To avoid dividing by zero when l->0, we compute (l*m) directly.

    # l(Hk) from Eq 33
    l_Hk = (6.54 * Hk - 14.07) / (Hk**2)

    # product l(Hk) * m(Hk) derived from Eq 34
    # m = (Term - 0.068) / l  ->  l*m = Term - 0.068
    term_m = 0.058 * ((Hk - 4.0)**2) / (Hk - 1.0)
    l_times_m = term_m - 0.068

    # The geometric conversion factor: dRe_theta / d(x/theta)
    # Factor = 0.5 * (l*m + l)
    spatial_conversion = 0.5 * (l_times_m + l_Hk)

    # Note: In strong favorable gradients (Hk < 2.2), l_Hk becomes negative
    # physically implying x is negative (reverse integration).
    # We clip the conversion to 0 to prevent negative growth rates in stagnation.
    spatial_conversion = np.maximum(spatial_conversion, 0.0)

    # --- 4. Final Calculation ---
    theta_dN_dx = dN_dRe * spatial_conversion

    return theta_dN_dx
