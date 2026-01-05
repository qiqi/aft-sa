"""
VTK file writer for CFD solution data.
"""

import numpy as np
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from ._array_utils import sanitize_array

if TYPE_CHECKING:
    from .plotter import Snapshot
    from ..grid.metrics import FVMMetrics


def write_vtk(
    filename: Path,
    grid_metrics: 'FVMMetrics',
    snapshot: 'Snapshot',
    wall_distance: Optional[np.ndarray] = None,
    u_inf: float = 1.0,
    v_inf: float = 0.0,
    nu_laminar: float = 1e-6,
) -> None:
    """Save snapshot data to VTK legacy format (structured grid).
    
    Parameters
    ----------
    filename : Path
        Output VTK file path.
    grid_metrics : FVMMetrics
        Grid metrics containing cell centers.
    snapshot : Snapshot
        Snapshot data to save.
    wall_distance : np.ndarray, optional
        Wall distance field.
    u_inf, v_inf : float
        Freestream velocity components.
    nu_laminar : float
        Laminar kinematic viscosity.
    """
    xc = grid_metrics.xc
    yc = grid_metrics.yc
    ni, nj = xc.shape
    
    # Prepare all field data
    p_data = sanitize_array(snapshot.p, fill_value=0.0)
    u_data = sanitize_array(snapshot.u, fill_value=0.0)
    v_data = sanitize_array(snapshot.v, fill_value=0.0)
    nu_data = sanitize_array(snapshot.nu, fill_value=0.0)
    chi_data = np.maximum(nu_data, 0.0) / nu_laminar
    vel_mag = np.sqrt((u_data + u_inf)**2 + (v_data + v_inf)**2)
    
    with open(filename, 'w') as f:
        # VTK header
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"CFD Solution - Iteration {snapshot.iteration}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {ni} {nj} 1\n")
        
        # Grid points (cell centers)
        f.write(f"POINTS {ni * nj} double\n")
        for j in range(nj):
            for i in range(ni):
                f.write(f"{xc[i, j]:.10e} {yc[i, j]:.10e} 0.0\n")
        
        # Cell data (scalar fields)
        f.write(f"\nPOINT_DATA {ni * nj}\n")
        
        # Write scalar fields
        _write_scalar_field(f, "pressure", p_data, ni, nj)
        _write_scalar_field(f, "u_velocity", u_data, ni, nj)
        _write_scalar_field(f, "v_velocity", v_data, ni, nj)
        _write_scalar_field(f, "nuHat", nu_data, ni, nj)
        _write_scalar_field(f, "chi", chi_data, ni, nj)
        _write_scalar_field(f, "velocity_magnitude", vel_mag, ni, nj)
        
        # Optional fields
        if snapshot.C_pt is not None:
            cpt_data = sanitize_array(snapshot.C_pt, fill_value=0.0)
            _write_scalar_field(f, "C_pt", cpt_data, ni, nj)
        
        if snapshot.residual_field is not None:
            res_data = sanitize_array(snapshot.residual_field, fill_value=1e-12)
            _write_scalar_field(f, "residual", res_data, ni, nj)
        
        if wall_distance is not None:
            wd_data = sanitize_array(wall_distance, fill_value=0.0)
            _write_scalar_field(f, "wall_distance", wd_data, ni, nj)
        
        # AFT diagnostic fields
        if snapshot.Re_Omega is not None:
            re_omega_data = sanitize_array(snapshot.Re_Omega, fill_value=1.0)
            _write_scalar_field(f, "Re_Omega", re_omega_data, ni, nj)
        
        if snapshot.Gamma is not None:
            gamma_data = sanitize_array(snapshot.Gamma, fill_value=0.0)
            _write_scalar_field(f, "Gamma", gamma_data, ni, nj)
        
        if snapshot.is_turb is not None:
            is_turb_data = sanitize_array(snapshot.is_turb, fill_value=0.0)
            _write_scalar_field(f, "is_turb", is_turb_data, ni, nj)
        
        # Velocity vector field
        f.write("\nVECTORS velocity double\n")
        for j in range(nj):
            for i in range(ni):
                u_abs = u_data[i, j] + u_inf
                v_abs = v_data[i, j] + v_inf
                f.write(f"{u_abs:.10e} {v_abs:.10e} 0.0\n")
    
    print(f"Saved VTK file to: {filename}")


def _write_scalar_field(f, name: str, data: np.ndarray, ni: int, nj: int) -> None:
    """Write a scalar field to VTK file."""
    f.write(f"\nSCALARS {name} double 1\n")
    f.write("LOOKUP_TABLE default\n")
    for j in range(nj):
        for i in range(ni):
            f.write(f"{data[i, j]:.10e}\n")
