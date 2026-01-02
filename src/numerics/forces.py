"""
Aerodynamic force computation for 2D airfoil simulations.

This module computes lift and drag coefficients by integrating
pressure and viscous stresses over the airfoil surface.

Also provides surface distributions:
- Cp: Pressure coefficient
- Cf: Skin friction coefficient
"""

import numpy as np
from numba import njit
from typing import NamedTuple

from src.constants import NGHOST


class AeroForces(NamedTuple):
    """Aerodynamic force coefficients."""
    CL: float      # Lift coefficient
    CD: float      # Total drag coefficient
    CD_p: float    # Pressure drag coefficient
    CD_f: float    # Friction (viscous) drag coefficient
    CL_p: float    # Pressure lift coefficient
    CL_f: float    # Friction lift coefficient
    Fx: float      # Body-axis x-force
    Fy: float      # Body-axis y-force


class SurfaceData(NamedTuple):
    """Surface distribution data."""
    x: np.ndarray    # x-coordinates (NI,)
    y: np.ndarray    # y-coordinates (NI,)
    Cp: np.ndarray   # Pressure coefficient (NI,)
    Cf: np.ndarray   # Skin friction coefficient (NI,)


@njit(cache=True, fastmath=True)
def _compute_surface_forces_kernel(
    Q: np.ndarray,
    Sj_x: np.ndarray,
    Sj_y: np.ndarray,
    volume: np.ndarray,
    mu_eff: np.ndarray,
    n_wake: int = 0,
) -> tuple:
    """
    Compute pressure and viscous forces on airfoil surface (j=0).
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector [p, u, v, nu_t].
    Sj_x, Sj_y : ndarray, shape (NI, NJ+1)
        J-face normal vectors (area-weighted).
    volume : ndarray, shape (NI, NJ)
        Cell volumes.
    mu_eff : ndarray, shape (NI, NJ)
        Effective viscosity (laminar + turbulent).
    n_wake : int
        Number of wake cells at each end of i-direction.
        Only cells from n_wake to NI-n_wake are on the airfoil surface.
        
    Returns
    -------
    Fx_p, Fy_p : float
        Pressure force components (body axes).
    Fx_v, Fy_v : float
        Viscous force components (body axes).
    """
    NI = Q.shape[0] - 2 * NGHOST  # NGHOST ghost layers on each side
    
    Fx_p = 0.0
    Fy_p = 0.0
    Fx_v = 0.0
    Fy_v = 0.0
    
    # Determine airfoil cell range (exclude wake cells)
    i_start = n_wake if n_wake > 0 else 0
    i_end = NI - n_wake if n_wake > 0 else NI
    
    # Iterate over airfoil surface faces only (j=0 boundary, excluding wake)
    for i in range(i_start, i_end):
        # Face normal at j=0 (points into domain, away from wall)
        # Note: Sj has shape (NI, NJ+1), j=0 is the wall face
        Sx = Sj_x[i, 0]
        Sy = Sj_y[i, 0]
        area = np.sqrt(Sx*Sx + Sy*Sy)
        
        if area < 1e-14:
            continue
            
        # Unit normal (pointing into domain)
        nx = Sx / area
        ny = Sy / area
        
        # ===== Pressure Force =====
        # Pressure at wall face (average of interior and ghost)
        # Ghost cell: p_ghost = p_interior (Neumann BC for pressure)
        # So p_wall = p_interior
        p_wall = Q[i + NGHOST, NGHOST, 0]  # Interior cell (NGHOST offset, j=NGHOST is first interior)
        
        # Force on body from pressure: F = -p * S (pressure pushes in)
        # The normal S points into domain (away from body)
        # Pressure force on body = -p * (-n) * area = p * n * area = p * S
        # Wait, let's be careful:
        # - S_j at j=0 points from ghost to interior (into domain, away from wall)
        # - Pressure acts on the wall surface
        # - Force on fluid from wall = +p * S (fluid pushed away from wall)
        # - Force on body from fluid = -p * S (Newton's 3rd law)
        Fx_p += -p_wall * Sx
        Fy_p += -p_wall * Sy
        
        # ===== Viscous Force =====
        # At wall (j=0), we need velocity gradients
        # Using ghost cell formulation:
        #   u_ghost = -u_interior (no-slip: u_wall = 0)
        #   du/dy ≈ (u_interior - u_ghost) / dy = 2*u_interior / dy
        
        # Get cell size in wall-normal direction
        # dy ≈ volume / dx, where dx ≈ area of i-face
        # For simplicity, use: dy ≈ sqrt(volume / (NI cells per unit chord))
        # Better: dy = volume[i,0] / |S_i| but we don't have S_i here
        # Use volume and area: dy ≈ volume / area_tangent
        # Since j-face area ≈ dx, we have volume ≈ dx * dy
        # So dy ≈ volume / (area of j-face tangent component)
        # For simplicity, approximate: dy ≈ 2 * volume[i,0] / area
        # (factor 2 because volume extends to cell center, not face)
        
        vol = volume[i, 0]
        dy = vol / area  # Approximate wall-normal distance to cell center
        
        # Velocity at first interior cell
        u_int = Q[i + NGHOST, NGHOST, 1]
        v_int = Q[i + NGHOST, NGHOST, 2]
        
        # Wall velocity gradients (assuming wall at y=0 locally)
        # du/dn = (u_int - 0) / (dy/2) = 2*u_int / dy (to cell center)
        # Actually, ghost cell is at -dy/2, interior at +dy/2
        # So gradient = (u_int - u_ghost) / dy = (u_int - (-u_int)) / dy = 2*u_int/dy
        dudn = 2.0 * u_int / dy
        dvdn = 2.0 * v_int / dy
        
        # Convert to Cartesian gradients using normal direction
        # dn = (nx, ny), so d/dn = nx * d/dx + ny * d/dy
        # For wall-parallel flow, approximate:
        # du/dx ≈ 0 (no streamwise gradient at wall for attached flow)
        # du/dy ≈ dudn * ny (if n is mostly in y)
        # This is approximate; proper implementation would use full gradient
        
        # Simple approximation: assume wall normal is mostly in y
        # dudy ≈ dudn, dvdy ≈ dvdn, dudx ≈ 0, dvdx ≈ 0
        # Wall shear stress: tau_xy = mu * (du/dy + dv/dx) ≈ mu * du/dn
        
        # More accurate: project onto wall
        # Tangent vector: tx = -ny, ty = nx
        # Velocity gradient in normal direction
        # tau_wall = mu * du_tangent/dn
        
        # Stress tensor at wall (incompressible):
        # tau_xx = 2*mu*du/dx
        # tau_yy = 2*mu*dv/dy  
        # tau_xy = mu*(du/dy + dv/dx)
        
        # At wall, du/dx and dv/dx are small, so:
        # tau_xy ≈ mu * du/dy
        # Project velocity gradient onto normal to get du/dn
        # Then tau_wall ≈ mu * du/dn (tangent direction)
        
        # Effective viscosity at wall
        mu = mu_eff[i, 0]
        
        # Wall shear stress magnitude (simplified)
        # tau = mu * |du/dn| where u is tangential velocity
        # Tangential velocity: u_t = u*tx + v*ty = -u*ny + v*nx
        # u_t_int = -u_int * ny + v_int * nx
        # du_t/dn = -dudn * ny + dvdn * nx
        
        # Shear stress vector on wall (acts tangent to wall, in flow direction)
        # F_viscous = tau * tangent * area = mu * du_t/dn * (tx, ty) * area
        # = mu * (-dudn*ny + dvdn*nx) * (-ny, nx) * area
        
        # Simpler: stress tensor approach
        # tau_ij * n_j = stress on face with normal n
        # tau_xx*nx + tau_xy*ny, tau_xy*nx + tau_yy*ny
        
        # At wall with no-slip:
        # du/dx ≈ 0 (continuity + no-slip)
        # dv/dx ≈ small  
        # du/dy = dudn / ny (if normal is (0,1))
        # But for general normal:
        
        # Use the gradient in normal direction directly
        # The shear traction on the wall:
        # f_x = mu * (2*dudx*nx + (dudy+dvdx)*ny)
        # f_y = mu * ((dudy+dvdx)*nx + 2*dvdy*ny)
        
        # Approximate dudx ≈ 0, dvdy ≈ 0 at wall (incompressible no-slip)
        # dudy + dvdx ≈ 2 * (dudn*ny + dvdn*nx) ??? 
        
        # Let's use a simpler model:
        # Shear stress = mu * velocity_gradient_at_wall
        # tau_wall = mu * V_int / (dy/2) = 2 * mu * V_int / dy
        # Direction: tangent to wall, in direction of flow
        
        # The viscous traction on the wall:
        tau_x = mu * dudn
        tau_y = mu * dvdn
        
        # Viscous force on body (opposite to traction on fluid)
        # Force on fluid from wall = -tau (friction opposes flow)
        # Force on body from fluid = +tau
        Fx_v += tau_x * area
        Fy_v += tau_y * area
    
    return Fx_p, Fy_p, Fx_v, Fy_v


def compute_aerodynamic_forces(
    Q: np.ndarray,
    metrics,
    mu_laminar: float,
    mu_turb: np.ndarray = None,
    alpha_deg: float = 0.0,
    chord: float = 1.0,
    rho_inf: float = 1.0,
    V_inf: float = 1.0,
    n_wake: int = 0,
) -> AeroForces:
    """
    Compute aerodynamic force coefficients.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector [p, u, v, nu_t].
    metrics : object
        Grid metrics with Sj_x, Sj_y, volume attributes.
    mu_laminar : float
        Laminar (molecular) viscosity.
    mu_turb : ndarray, optional
        Turbulent viscosity field, shape (NI, NJ).
        If None, assumes laminar flow.
    alpha_deg : float
        Angle of attack in degrees.
    chord : float
        Chord length for normalization.
    rho_inf : float
        Freestream density.
    V_inf : float
        Freestream velocity magnitude.
    n_wake : int
        Number of wake cells at each end of i-direction.
        Forces are only computed for airfoil cells (excluding wake).
        
    Returns
    -------
    forces : AeroForces
        Named tuple with CL, CD, CD_p, CD_f, etc.
    """
    NI = Q.shape[0] - 2 * NGHOST  # NGHOST ghost layers on each side
    NJ = Q.shape[1] - 2 * NGHOST  # NGHOST ghost layers on each side
    
    # Build effective viscosity array
    if mu_turb is None:
        mu_eff = np.full((NI, NJ), mu_laminar)
    else:
        mu_eff = mu_laminar + np.maximum(0.0, mu_turb)
    
    # Compute forces using kernel
    Fx_p, Fy_p, Fx_v, Fy_v = _compute_surface_forces_kernel(
        Q, metrics.Sj_x, metrics.Sj_y, metrics.volume, mu_eff, n_wake
    )
    
    # Total body-axis forces
    Fx = Fx_p + Fx_v
    Fy = Fy_p + Fy_v
    
    # Rotate to wind axes
    alpha = np.deg2rad(alpha_deg)
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    
    # Drag = Fx*cos(α) + Fy*sin(α)
    # Lift = Fy*cos(α) - Fx*sin(α)
    D_total = Fx * cos_a + Fy * sin_a
    L_total = Fy * cos_a - Fx * sin_a
    
    D_p = Fx_p * cos_a + Fy_p * sin_a
    L_p = Fy_p * cos_a - Fx_p * sin_a
    
    D_f = Fx_v * cos_a + Fy_v * sin_a
    L_f = Fy_v * cos_a - Fx_v * sin_a
    
    # Dynamic pressure
    q_inf = 0.5 * rho_inf * V_inf**2
    
    # Normalize by dynamic pressure and chord
    # Note: For 2D, reference area = chord * 1 (unit span)
    ref_area = chord * 1.0
    
    CL = L_total / (q_inf * ref_area)
    CD = D_total / (q_inf * ref_area)
    CD_p = D_p / (q_inf * ref_area)
    CD_f = D_f / (q_inf * ref_area)
    CL_p = L_p / (q_inf * ref_area)
    CL_f = L_f / (q_inf * ref_area)
    
    return AeroForces(
        CL=CL,
        CD=CD,
        CD_p=CD_p,
        CD_f=CD_f,
        CL_p=CL_p,
        CL_f=CL_f,
        Fx=Fx,
        Fy=Fy,
    )


@njit(cache=True, fastmath=True)
def _compute_surface_cp_cf_kernel(
    Q: np.ndarray,
    Sj_x: np.ndarray,
    Sj_y: np.ndarray,
    volume: np.ndarray,
    mu_eff: np.ndarray,
    p_inf: float,
    q_inf: float,
    Cp_out: np.ndarray,
    Cf_out: np.ndarray,
) -> None:
    """
    Compute surface Cp and Cf distributions.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector.
    Sj_x, Sj_y : ndarray, shape (NI, NJ+1)
        J-face normals.
    volume : ndarray, shape (NI, NJ)
        Cell volumes.
    mu_eff : ndarray, shape (NI, NJ)
        Effective viscosity.
    p_inf : float
        Freestream pressure.
    q_inf : float
        Dynamic pressure (0.5 * rho * V^2).
    Cp_out : ndarray, shape (NI,)
        Output: pressure coefficient.
    Cf_out : ndarray, shape (NI,)
        Output: skin friction coefficient.
    """
    NI = Q.shape[0] - 2 * NGHOST  # NGHOST ghost layers on each side
    
    for i in range(NI):
        # Face normal at j=0
        Sx = Sj_x[i, 0]
        Sy = Sj_y[i, 0]
        area = np.sqrt(Sx*Sx + Sy*Sy)
        
        if area < 1e-14:
            Cp_out[i] = 0.0
            Cf_out[i] = 0.0
            continue
        
        # ===== Pressure coefficient =====
        p_wall = Q[i + NGHOST, NGHOST, 0]
        Cp_out[i] = (p_wall - p_inf) / q_inf
        
        # ===== Skin friction coefficient =====
        # Wall-normal distance
        vol = volume[i, 0]
        dy = vol / area
        
        # Velocity at first interior cell
        u_int = Q[i + NGHOST, NGHOST, 1]
        v_int = Q[i + NGHOST, NGHOST, 2]
        
        # Velocity gradients at wall
        dudn = 2.0 * u_int / dy
        dvdn = 2.0 * v_int / dy
        
        # Wall shear stress magnitude
        mu = mu_eff[i, 0]
        tau_mag = mu * np.sqrt(dudn*dudn + dvdn*dvdn)
        
        # Skin friction coefficient
        Cf_out[i] = tau_mag / q_inf


def compute_surface_distributions(
    Q: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    metrics,
    mu_laminar: float,
    mu_turb: np.ndarray = None,
    p_inf: float = 0.0,
    rho_inf: float = 1.0,
    V_inf: float = 1.0,
) -> SurfaceData:
    """
    Compute surface Cp and Cf distributions.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector.
    X, Y : ndarray, shape (NI+1, NJ+1)
        Grid node coordinates.
    metrics : object
        Grid metrics with Sj_x, Sj_y, volume.
    mu_laminar : float
        Laminar viscosity.
    mu_turb : ndarray, optional
        Turbulent viscosity, shape (NI, NJ).
    p_inf : float
        Freestream pressure.
    rho_inf : float
        Freestream density.
    V_inf : float
        Freestream velocity magnitude.
        
    Returns
    -------
    SurfaceData
        Named tuple with x, y, Cp, Cf arrays.
    """
    NI = Q.shape[0] - 2 * NGHOST  # NGHOST ghost layers on each side
    NJ = Q.shape[1] - 2 * NGHOST  # NGHOST ghost layers on each side
    
    # Build effective viscosity
    if mu_turb is None:
        mu_eff = np.full((NI, NJ), mu_laminar)
    else:
        mu_eff = mu_laminar + np.maximum(0.0, mu_turb)
    
    # Dynamic pressure
    q_inf = 0.5 * rho_inf * V_inf**2
    if q_inf < 1e-14:
        q_inf = 1e-14  # Prevent division by zero
    
    # Allocate outputs
    Cp = np.zeros(NI)
    Cf = np.zeros(NI)
    
    # Compute distributions
    _compute_surface_cp_cf_kernel(
        Q, metrics.Sj_x, metrics.Sj_y, metrics.volume,
        mu_eff, p_inf, q_inf, Cp, Cf
    )
    
    # Surface coordinates (cell-centered at j=0)
    x_surf = 0.5 * (X[:-1, 0] + X[1:, 0])
    y_surf = 0.5 * (Y[:-1, 0] + Y[1:, 0])
    
    return SurfaceData(x=x_surf, y=y_surf, Cp=Cp, Cf=Cf)


def create_surface_vtk_fields(
    Q: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    metrics,
    mu_laminar: float,
    mu_turb: np.ndarray = None,
    p_inf: float = 0.0,
    rho_inf: float = 1.0,
    V_inf: float = 1.0,
) -> dict:
    """
    Create Cp and Cf fields for VTK output (full grid arrays).
    
    Returns dictionary with 'Cp' and 'Cf' as (NI, NJ) arrays.
    Values are only meaningful at j=0 (surface); rest filled with NaN.
    """
    from src.constants import NGHOST
    NI = Q.shape[0] - 2 * NGHOST  # NGHOST ghosts on each side
    NJ = Q.shape[1] - 2 * NGHOST  # NGHOST ghosts on each side
    
    # Get surface distributions
    surf = compute_surface_distributions(
        Q, X, Y, metrics, mu_laminar, mu_turb, p_inf, rho_inf, V_inf
    )
    
    # Create full-grid arrays (NaN for non-surface cells)
    Cp_field = np.full((NI, NJ), np.nan)
    Cf_field = np.full((NI, NJ), np.nan)
    
    # Fill surface row (j=0)
    Cp_field[:, 0] = surf.Cp
    Cf_field[:, 0] = surf.Cf
    
    return {'SurfaceCp': Cp_field, 'SurfaceCf': Cf_field}

