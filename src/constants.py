"""
Global constants for the RANS solver.

This module defines constants used throughout the codebase to ensure
consistency in array shapes and indexing.
"""

# Number of ghost cell layers in each direction
# Required for 4th-order JST scheme (5-point stencil = 2 cells on each side)
NGHOST = 2

# State vector components
P_IDX = 0   # Pressure
U_IDX = 1   # x-velocity
V_IDX = 2   # y-velocity
NU_IDX = 3  # SA turbulent viscosity
N_VARS = 4  # Total number of state variables


def get_interior_slice():
    """
    Return the slice for interior cells in the state array.
    
    With NGHOST ghost cells on each side:
    - Q.shape = (NI + 2*NGHOST, NJ + 2*NGHOST, N_VARS)
    - Interior cells: Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
    
    Returns
    -------
    tuple of slices
        (slice(NGHOST, -NGHOST), slice(NGHOST, -NGHOST))
    """
    return (slice(NGHOST, -NGHOST), slice(NGHOST, -NGHOST))


def get_q_shape(NI: int, NJ: int) -> tuple:
    """
    Get the shape of the state array Q.
    
    Parameters
    ----------
    NI, NJ : int
        Number of interior cells in I and J directions.
        
    Returns
    -------
    tuple
        Shape of Q array: (NI + 2*NGHOST, NJ + 2*NGHOST, N_VARS)
    """
    return (NI + 2 * NGHOST, NJ + 2 * NGHOST, N_VARS)

