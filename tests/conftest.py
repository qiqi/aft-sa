"""
Shared pytest fixtures for the test suite.

This module provides session-scoped fixtures for expensive operations
like grid generation, JAX kernel JIT compilation, and mfoil baseline computations.

Multi-GPU Support:
    Run tests across multiple GPUs with: pytest -n 8 --dist=loadgroup
    Tests in the same @pytest.mark.xdist_group will run on the same worker/GPU.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import os


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONSTRUCT2D_BIN = PROJECT_ROOT / "bin" / "construct2d"
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# Multi-GPU Support
# =============================================================================

def pytest_configure(config):
    """Configure pytest with multi-GPU support."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Assign tests to GPU workers based on xdist worker id."""
    # This runs after collection, can be used to modify test distribution
    pass


@pytest.fixture(scope="session")
def gpu_id(worker_id):
    """Get GPU ID for this worker (for multi-GPU testing with pytest-xdist).
    
    Usage with pytest-xdist:
        pytest -n 8  # Run on 8 workers, each gets a different GPU
        
    Returns GPU ID 0-7 based on worker_id (gw0-gw7).
    For master process (no xdist), returns GPU 0.
    """
    if worker_id == "master":
        return 0
    # worker_id is 'gw0', 'gw1', etc.
    try:
        gpu = int(worker_id.replace("gw", "")) % 8
    except ValueError:
        gpu = 0
    return gpu


@pytest.fixture(scope="session", autouse=True)
def configure_gpu_for_worker(worker_id):
    """Auto-configure CUDA_VISIBLE_DEVICES based on xdist worker.
    
    This runs once per worker session and sets the GPU before JAX imports.
    """
    if worker_id == "master":
        gpu_id = 0
    else:
        try:
            gpu_id = int(worker_id.replace("gw", "")) % 8
        except ValueError:
            gpu_id = 0
    
    # Set before any JAX imports
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    yield gpu_id


@pytest.fixture(scope="session")
def worker_id(request):
    """Get pytest-xdist worker id, or 'master' if not using xdist."""
    if hasattr(request.config, 'workerinput'):
        return request.config.workerinput['workerid']
    return "master"


# =============================================================================
# Construct2D availability check
# =============================================================================

def construct2d_available() -> bool:
    """Check if construct2d binary is available."""
    return CONSTRUCT2D_BIN.exists() and bool(CONSTRUCT2D_BIN.stat().st_mode & 0o111)


# =============================================================================
# NACA 0012 airfoil coordinates
# =============================================================================

NACA0012_COORDS = """NACA 0012
1.000000  0.001260
0.950000  0.011490
0.900000  0.020560
0.800000  0.035270
0.700000  0.046030
0.600000  0.053140
0.500000  0.056150
0.400000  0.054710
0.300000  0.048500
0.200000  0.037100
0.100000  0.021610
0.050000  0.013260
0.025000  0.008650
0.012500  0.005820
0.000000  0.000000
0.012500 -0.005820
0.025000 -0.008650
0.050000 -0.013260
0.100000 -0.021610
0.200000 -0.037100
0.300000 -0.048500
0.400000 -0.054710
0.500000 -0.056150
0.600000 -0.053140
0.700000 -0.046030
0.800000 -0.035270
0.900000 -0.020560
0.950000 -0.011490
1.000000 -0.001260
"""


# =============================================================================
# Grid Generation Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def naca0012_airfoil_file(tmp_path_factory):
    """
    Create a NACA 0012 airfoil coordinate file.
    
    Session-scoped: created once, shared across all tests.
    """
    tmp_dir = tmp_path_factory.mktemp("airfoil")
    airfoil_file = tmp_dir / "naca0012.dat"
    airfoil_file.write_text(NACA0012_COORDS)
    return airfoil_file


@pytest.fixture(scope="session")
def construct2d_wrapper():
    """
    Get a Construct2DWrapper if binary is available.
    
    Returns None if construct2d is not available.
    """
    if not construct2d_available():
        return None
    
    from src.grid.mesher import Construct2DWrapper
    return Construct2DWrapper(str(CONSTRUCT2D_BIN))


@pytest.fixture(scope="session")
def naca0012_coarse_grid(construct2d_wrapper, naca0012_airfoil_file, tmp_path_factory):
    """
    Generate a coarse NACA 0012 grid for quick tests.
    
    Session-scoped: generated once, shared across all tests.
    Grid size: 64x16 cells (power-of-2 for multigrid, nodes = cells + 1)
    
    Returns None if construct2d is not available.
    """
    if construct2d_wrapper is None:
        return None
    
    from src.grid.mesher import GridOptions
    
    tmp_dir = tmp_path_factory.mktemp("grid_coarse")
    options = GridOptions(
        n_surface=65,   # 64 cells
        n_normal=17,    # 16 cells
        topology='CGRD',
    )
    
    X, Y = construct2d_wrapper.generate(
        str(naca0012_airfoil_file),
        options=options,
        working_dir=str(tmp_dir),
        verbose=False
    )
    
    # Return both grid and path to .p3d file
    grid_file = tmp_dir / "naca0012.p3d"
    return {
        'X': X,
        'Y': Y,
        'path': str(grid_file),
        'ni': X.shape[0],
        'nj': X.shape[1],
    }


@pytest.fixture(scope="session")
def naca0012_medium_grid(construct2d_wrapper, naca0012_airfoil_file, tmp_path_factory):
    """
    Generate a medium NACA 0012 grid for validation tests.
    
    Session-scoped: generated once, shared across all tests.
    Grid size: 128x32 cells (power-of-2 for multigrid, nodes = cells + 1)
    
    Returns None if construct2d is not available.
    """
    if construct2d_wrapper is None:
        return None
    
    from src.grid.mesher import GridOptions
    
    tmp_dir = tmp_path_factory.mktemp("grid_medium")
    options = GridOptions(
        n_surface=129,  # 128 cells
        n_normal=33,    # 32 cells
        topology='CGRD',
        farfield_radius=20.0,
    )
    
    X, Y = construct2d_wrapper.generate(
        str(naca0012_airfoil_file),
        options=options,
        working_dir=str(tmp_dir),
        verbose=False
    )
    
    grid_file = tmp_dir / "naca0012.p3d"
    return {
        'X': X,
        'Y': Y,
        'path': str(grid_file),
        'ni': X.shape[0],
        'nj': X.shape[1],
    }


# =============================================================================
# mfoil Baseline Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def mfoil_baseline_re10k():
    """
    mfoil baseline results for NACA 0012 at Re=10,000, α=0° (laminar).
    
    Session-scoped: computed once, shared across all tests.
    """
    from src.validation.mfoil import mfoil
    
    M = mfoil(naca='0012', npanel=199)
    M.param.ncrit = 1000.0  # Force laminar
    M.param.doplot = False
    M.param.verb = 0
    M.setoper(alpha=0.0, Re=10000)
    
    try:
        M.solve()
        converged = True
    except Exception:
        converged = False
        return {
            'cl': np.nan, 'cd': np.nan,
            'cdf': np.nan, 'cdp': np.nan,
            'converged': False
        }
    
    return {
        'cl': M.post.cl,
        'cd': M.post.cd,
        'cdf': M.post.cdf,
        'cdp': M.post.cdp,
        'converged': converged,
        'reynolds': 10000,
        'alpha': 0.0,
    }


@pytest.fixture(scope="session")
def mfoil_baseline_re100k():
    """
    mfoil baseline results for NACA 0012 at Re=100,000, α=0° (laminar).
    
    Session-scoped: computed once, shared across all tests.
    """
    from src.validation.mfoil import mfoil
    
    M = mfoil(naca='0012', npanel=199)
    M.param.ncrit = 1000.0  # Force laminar
    M.param.doplot = False
    M.param.verb = 0
    M.setoper(alpha=0.0, Re=100000)
    
    try:
        M.solve()
        converged = True
    except Exception:
        converged = False
        return {
            'cl': np.nan, 'cd': np.nan,
            'cdf': np.nan, 'cdp': np.nan,
            'converged': False
        }
    
    return {
        'cl': M.post.cl,
        'cd': M.post.cd,
        'cdf': M.post.cdf,
        'cdp': M.post.cdp,
        'converged': converged,
        'reynolds': 100000,
        'alpha': 0.0,
    }


# =============================================================================
# JAX Kernel Fixtures (Session-Scoped for JIT Sharing)
# =============================================================================

@pytest.fixture(scope="session")
def jax_flux_kernels():
    """
    Session-scoped JAX flux computation kernels.
    
    Pre-compiles and caches the JIT-compiled flux functions to avoid
    re-compilation overhead in each test.
    """
    from src.numerics.fluxes import compute_fluxes_jax, _compute_fluxes_jax_impl
    from src.physics.jax_config import jax, jnp
    
    # Trigger JIT compilation with a small dummy array
    NI, NJ = 16, 8
    nghost = 2
    Q = jnp.zeros((NI + 2*nghost, NJ + 2*nghost, 4))
    Q = Q.at[:, :, 1].set(1.0)
    Si_x = jnp.ones((NI + 1, NJ))
    Si_y = jnp.zeros((NI + 1, NJ))
    Sj_x = jnp.zeros((NI, NJ + 1))
    Sj_y = jnp.ones((NI, NJ + 1))
    
    # Warmup call to trigger JIT
    _ = compute_fluxes_jax(Q, Si_x, Si_y, Sj_x, Sj_y, beta=10.0, k4=0.04, nghost=nghost)
    
    return {
        'compute_fluxes_jax': compute_fluxes_jax,
        '_compute_fluxes_jax_impl': _compute_fluxes_jax_impl,
    }


@pytest.fixture(scope="session")
def jax_gradient_kernels():
    """Session-scoped JAX gradient computation kernels."""
    from src.numerics.gradients import compute_gradients_jax
    from src.physics.jax_config import jax, jnp
    
    # Trigger JIT compilation
    NI, NJ = 16, 8
    nghost = 2
    Q = jnp.zeros((NI + 2*nghost, NJ + 2*nghost, 4))
    Q = Q.at[:, :, 1].set(1.0)
    Si_x = jnp.ones((NI + 1, NJ))
    Si_y = jnp.zeros((NI + 1, NJ))
    Sj_x = jnp.zeros((NI, NJ + 1))
    Sj_y = jnp.ones((NI, NJ + 1))
    volume = jnp.ones((NI, NJ))
    
    _ = compute_gradients_jax(Q, Si_x, Si_y, Sj_x, Sj_y, volume, nghost)
    
    return {
        'compute_gradients_jax': compute_gradients_jax,
    }


@pytest.fixture(scope="session")
def jax_batch_kernels():
    """
    Session-scoped batch JAX kernels for batch solver tests.
    
    Pre-compiles vmap-wrapped kernels for batch processing.
    """
    from src.solvers.batch import (
        compute_fluxes_batch,
        compute_gradients_batch,
        compute_timestep_batch,
        compute_viscous_fluxes_batch,
        smooth_residual_batch,
    )
    from src.physics.jax_config import jax, jnp
    
    # Warmup with small batch
    NI, NJ = 16, 8
    nghost = 2
    n_batch = 2
    Q_batch = jnp.zeros((n_batch, NI + 2*nghost, NJ + 2*nghost, 4))
    Q_batch = Q_batch.at[:, :, :, 1].set(1.0)
    Si_x = jnp.ones((NI + 1, NJ))
    Si_y = jnp.zeros((NI + 1, NJ))
    Sj_x = jnp.zeros((NI, NJ + 1))
    Sj_y = jnp.ones((NI, NJ + 1))
    volume = jnp.ones((NI, NJ))
    
    # Trigger JIT compilation
    _ = compute_fluxes_batch(Q_batch, Si_x, Si_y, Sj_x, Sj_y, beta=10.0, k4=0.04, nghost=nghost)
    
    return {
        'compute_fluxes_batch': compute_fluxes_batch,
        'compute_gradients_batch': compute_gradients_batch,
        'compute_timestep_batch': compute_timestep_batch,
        'compute_viscous_fluxes_batch': compute_viscous_fluxes_batch,
        'smooth_residual_batch': smooth_residual_batch,
    }


# =============================================================================
# Skip Markers
# =============================================================================

requires_construct2d = pytest.mark.skipif(
    not construct2d_available(),
    reason="construct2d binary not available"
)

slow = pytest.mark.slow


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def simple_uniform_grid():
    """Create a simple uniform Cartesian grid for testing."""
    NI, NJ = 20, 20
    x = np.linspace(0, 1, NI + 1)
    y = np.linspace(0, 1, NJ + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


@pytest.fixture
def simple_state_vector(simple_uniform_grid):
    """Create a simple state vector for testing."""
    X, Y = simple_uniform_grid
    NI, NJ = X.shape[0] - 1, X.shape[1] - 1
    
    # State with ghost cells: (NI+2, NJ+2, 4)
    Q = np.zeros((NI + 2, NJ + 2, 4))
    
    # Initialize with uniform freestream
    Q[:, :, 0] = 0.0   # pressure
    Q[:, :, 1] = 1.0   # u-velocity
    Q[:, :, 2] = 0.0   # v-velocity
    Q[:, :, 3] = 0.0   # nu_tilde
    
    return Q

