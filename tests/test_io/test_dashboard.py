import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
from src.io.dashboard.data import Snapshot, DataManager

def test_snapshot_creation():
    """Test creating a Snapshot object."""
    snap = Snapshot(
        iteration=10,
        residual=np.array([1e-3, 1e-3, 1e-3, 1e-3]),
        cfl=1.0,
        p=np.zeros((10, 10)),
        u=np.ones((10, 10)),
        v=np.zeros((10, 10)),
        nu=np.zeros((10, 10)),
        vel_max=1.5,
        p_min=-0.1,
        p_max=0.5
    )
    
    assert snap.iteration == 10
    assert snap.cfl == 1.0
    assert snap.u.shape == (10, 10)

def test_datamanager_store_snapshot():
    """Test storing a snapshot via DataManager."""
    dm = DataManager(p_inf=0.0, u_inf=1.0, v_inf=0.0)
    
    # Create dummy data (with ghost cells)
    # NGHOST=2 defined in src.constants
    from src.constants import NGHOST
    ni, nj = 10, 10
    shape = (ni + 2*NGHOST, nj + 2*NGHOST, 4)
    Q = np.zeros(shape)
    
    residual_history = [1e-3, 1e-4]
    
    dm.store_snapshot(
        Q=Q,
        iteration=1,
        residual_history=residual_history,
        cfl=0.5
    )
    
    assert len(dm.snapshots) == 1
    snap = dm.snapshots[0]
    assert snap.iteration == 1
    # Check that ghosts were stripped
    assert snap.p.shape == (ni, nj)
    
    # Check residual history parsing
    assert len(dm.residual_history) == 2
    # The parsing logic converts scalar to 4-component array
    assert dm.residual_history[0].shape == (4,)
    assert dm.residual_history[0][0] == 1e-3

def test_datamanager_divergence():
    """Test divergence snapshot storage."""
    dm = DataManager()
    shape = (14, 14, 4) # 10x10 + ghosts
    Q = np.zeros(shape)
    
    dm.store_snapshot(
        Q=Q,
        iteration=100,
        residual_history=[1e-3],
        is_divergence_dump=True
    )
    
    assert len(dm.snapshots) == 0
    assert len(dm.divergence_snapshots) == 1
    assert dm.divergence_snapshots[0].iteration == 100
