"""Global constants for the RANS solver."""

NGHOST = 2

P_IDX = 0
U_IDX = 1
V_IDX = 2
NU_IDX = 3
N_VARS = 4


def get_interior_slice():
    """Return the slice for interior cells: Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]"""
    return (slice(NGHOST, -NGHOST), slice(NGHOST, -NGHOST))


def get_q_shape(NI: int, NJ: int) -> tuple:
    """Get the shape of the state array Q: (NI + 2*NGHOST, NJ + 2*NGHOST, N_VARS)"""
    return (NI + 2 * NGHOST, NJ + 2 * NGHOST, N_VARS)
