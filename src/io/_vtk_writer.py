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


def write_vts(
    filename: Path,
    X: np.ndarray,
    Y: np.ndarray,
    snapshot: 'Snapshot',
    wall_distance: Optional[np.ndarray] = None,
    u_inf: float = 1.0,
    v_inf: float = 0.0,
    nu_laminar: float = 1e-6,
) -> None:
    """Write a single VTS file (structured grid) for one snapshot."""
    ni, nj = X.shape
    cell_ni = ni - 1
    cell_nj = nj - 1

    if snapshot.p.shape != (cell_ni, cell_nj):
        raise ValueError("Snapshot shape does not match grid cell dimensions.")

    # Prepare field data
    p_data = sanitize_array(snapshot.p, fill_value=0.0)
    u_data = sanitize_array(snapshot.u, fill_value=0.0)
    v_data = sanitize_array(snapshot.v, fill_value=0.0)
    nu_data = sanitize_array(snapshot.nu, fill_value=0.0)
    chi_data = np.maximum(nu_data, 0.0) / nu_laminar
    vel_mag = np.sqrt((u_data + u_inf)**2 + (v_data + v_inf)**2)

    def flat(arr: np.ndarray) -> str:
        return " ".join(f"{val:.10e}" for val in arr.flatten(order='F'))

    with open(filename, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <StructuredGrid WholeExtent="0 {ni - 1} 0 {nj - 1} 0 0">\n')
        f.write(f'    <Piece Extent="0 {ni - 1} 0 {nj - 1} 0 0">\n')

        # Cell data
        f.write('      <CellData>\n')
        _write_vts_scalar(f, "pressure", p_data)
        _write_vts_scalar(f, "u_velocity", u_data)
        _write_vts_scalar(f, "v_velocity", v_data)
        _write_vts_scalar(f, "nuHat", nu_data)
        _write_vts_scalar(f, "chi", chi_data)
        _write_vts_scalar(f, "velocity_magnitude", vel_mag)

        if snapshot.C_pt is not None:
            cpt_data = sanitize_array(snapshot.C_pt, fill_value=0.0)
            _write_vts_scalar(f, "C_pt", cpt_data)

        if snapshot.residual_field is not None:
            res_data = sanitize_array(snapshot.residual_field, fill_value=1e-12)
            _write_vts_scalar(f, "residual", res_data)

        if wall_distance is not None:
            wd_data = sanitize_array(wall_distance, fill_value=0.0)
            _write_vts_scalar(f, "wall_distance", wd_data)

        if snapshot.Re_Omega is not None:
            re_omega_data = sanitize_array(snapshot.Re_Omega, fill_value=1.0)
            _write_vts_scalar(f, "Re_Omega", re_omega_data)

        if snapshot.Gamma is not None:
            gamma_data = sanitize_array(snapshot.Gamma, fill_value=0.0)
            _write_vts_scalar(f, "Gamma", gamma_data)

        if snapshot.is_turb is not None:
            is_turb_data = sanitize_array(snapshot.is_turb, fill_value=0.0)
            _write_vts_scalar(f, "is_turb", is_turb_data)

        if snapshot.amplification_ratio is not None:
            amp_data = sanitize_array(snapshot.amplification_ratio, fill_value=0.0)
            _write_vts_scalar(f, "amplification_ratio", amp_data)

        # Velocity vector (absolute)
        u_abs = u_data + u_inf
        v_abs = v_data + v_inf
        vel_vec = np.stack([u_abs, v_abs, np.zeros_like(u_abs)], axis=2)
        f.write('        <DataArray type="Float64" Name="velocity" NumberOfComponents="3" format="ascii">\n')
        f.write(f"          {flat(vel_vec)}\n")
        f.write('        </DataArray>\n')
        f.write('      </CellData>\n')

        # Points (grid nodes)
        f.write('      <Points>\n')
        points = np.stack([X, Y, np.zeros_like(X)], axis=2)
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write(f"          {flat(points)}\n")
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')

        f.write('    </Piece>\n')
        f.write('  </StructuredGrid>\n')
        f.write('</VTKFile>\n')


def write_vts_series(
    output_dir: Path,
    base_name: str,
    X: np.ndarray,
    Y: np.ndarray,
    snapshots: list,
    wall_distance: Optional[np.ndarray] = None,
    u_inf: float = 1.0,
    v_inf: float = 0.0,
    nu_laminar: float = 1e-6,
) -> Path:
    """Write a VTS time series plus PVD collection file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pvd_entries = []
    for snap in snapshots:
        vtk_name = f"{base_name}_{snap.iteration:06d}.vts"
        vtk_path = output_dir / vtk_name
        write_vts(vtk_path, X, Y, snap, wall_distance, u_inf, v_inf, nu_laminar)
        pvd_entries.append((snap.iteration, vtk_name))

    pvd_path = output_dir / f"{base_name}.pvd"
    with open(pvd_path, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for timestep, filename in pvd_entries:
            f.write(f'    <DataSet timestep="{timestep}" file="{filename}"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

    return pvd_path


def _write_vts_scalar(f, name: str, data: np.ndarray) -> None:
    """Write scalar DataArray for VTS cell data."""
    flat = " ".join(f"{val:.10e}" for val in data.flatten(order='F'))
    f.write(f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="1" format="ascii">\n')
    f.write(f"          {flat}\n")
    f.write('        </DataArray>\n')


def _write_scalar_field(f, name: str, data: np.ndarray, ni: int, nj: int) -> None:
    """Write a scalar field to VTK file."""
    f.write(f"\nSCALARS {name} double 1\n")
    f.write("LOOKUP_TABLE default\n")
    for j in range(nj):
        for i in range(ni):
            f.write(f"{data[i, j]:.10e}\n")
