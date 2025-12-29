"""
I/O module for CFD solver.

Provides file readers and writers for various formats.
"""

from .output import write_vtk, write_vtk_series, VTKWriter

__all__ = ['write_vtk', 'write_vtk_series', 'VTKWriter']

