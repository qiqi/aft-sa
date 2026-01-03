"""
Grid generation and processing module.
"""

from .mesher import (
    Construct2DWrapper,
    Construct2DError,
    GridOptions,
    estimate_first_cell_height
)

from .plot3d import (
    read_plot3d,
    read_plot3d_ascii,
    read_plot3d_binary,
    compute_cell_centers,
    compute_cell_volumes,
    compute_face_normals_i,
    compute_face_normals_j,
    compute_wall_distance,
    compute_wall_distance_fast,
    compute_wall_distance_gridpoints,
    check_grid_quality,
    GridMetrics,
    StructuredGrid
)

from .metrics import (
    MetricComputer,
    FVMMetrics,
    GCLValidation,
    compute_metrics
)

from .loader import (
    load_or_generate_grid,
    find_construct2d_binary
)

__all__ = [
    'Construct2DWrapper',
    'Construct2DError', 
    'GridOptions',
    'estimate_first_cell_height',
    'read_plot3d',
    'read_plot3d_ascii',
    'read_plot3d_binary',
    'compute_cell_centers',
    'compute_cell_volumes',
    'compute_face_normals_i',
    'compute_face_normals_j',
    'compute_wall_distance',
    'compute_wall_distance_fast',
    'compute_wall_distance_gridpoints',
    'check_grid_quality',
    'GridMetrics',
    'StructuredGrid',
    'MetricComputer',
    'FVMMetrics',
    'GCLValidation',
    'compute_metrics',
    'load_or_generate_grid',
    'find_construct2d_binary',
]
