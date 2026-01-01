"""
I/O module for CFD solver.

Provides file readers and writers for various formats,
as well as plotting utilities.
"""

from .output import write_vtk, write_vtk_series, VTKWriter

from .plotting import (
    plot_flow_field,
    plot_residual_history,
    plot_surface_distributions,
)

__all__ = [
    'write_vtk', 
    'write_vtk_series', 
    'VTKWriter',
    'plot_flow_field',
    'plot_residual_history',
    'plot_surface_distributions',
]

