"""Global constants for the RANS solver."""

from typing import Tuple

NGHOST: int = 2

P_IDX: int = 0
U_IDX: int = 1
V_IDX: int = 2
NU_IDX: int = 3
N_VARS: int = 4


def get_interior_slice() -> Tuple[slice, slice]:
    """Return the slice for interior cells: Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]"""
    return (slice(NGHOST, -NGHOST), slice(NGHOST, -NGHOST))


def get_q_shape(NI: int, NJ: int) -> Tuple[int, int, int]:
    """Get the shape of the state array Q: (NI + 2*NGHOST, NJ + 2*NGHOST, N_VARS)"""
    return (NI + 2 * NGHOST, NJ + 2 * NGHOST, N_VARS)
