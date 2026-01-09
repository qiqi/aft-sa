import sys
import os
import time
from pathlib import Path
import jax

# Add src to path
sys.path.append(os.getcwd())

from src.solvers.rans_solver import RANSSolver
from src.config.loader import load_yaml
from src.grid.loader import load_or_generate_grid
from loguru import logger

def test_dynamic_params():
    """Verify that changing params does not trigger JIT recompilation."""
    
    config_path = "config/examples/naca0012_newton.yaml"
    logger.info(f"Loading config from {config_path}...")
    
    sim_config = load_yaml(config_path)
    # Convert to solver config
    config = sim_config.to_solver_config()
    
    # Override for test speed
    config.max_iter = 0 # Manual stepping
    config.gmres_restart = 10
    config.gmres_maxiter = 20
    
    # Load Grid
    logger.info("Loading/Generating grid...")
    X, Y = load_or_generate_grid(
        sim_config.grid.airfoil,
        n_surface=sim_config.grid.n_surface,
        n_wake=sim_config.grid.n_wake,
        y_plus=sim_config.grid.y_plus,
        gradation=sim_config.grid.gradation,
        reynolds=sim_config.flow.reynolds,
        farfield_radius=sim_config.grid.farfield_radius,
        wake_fan_factor=sim_config.grid.wake_fan_factor,
        wake_fan_k=sim_config.grid.wake_fan_k,
        max_first_cell=0.001, # Simplified default
        project_root=Path(os.getcwd()),
        verbose=False
    )
    
    logger.info("Initializing Solver...")
    solver = RANSSolver(grid_data=(X, Y), config=config)
    
    # Warmup (triggers first compilation)
    logger.info("Warming up (First Compilation)...")
    start_time = time.time()
    solver.step() # Iteration 1
    logger.info(f"Step 1 time: {time.time() - start_time:.4f}s")
    
    # Run a few steps
    for i in range(5):
        solver.step()
    
    # 2. Change Parameters (mocking dynamic update)
    logger.info("\nChanging Physics Parameters (Alpha & Chi_inf)...")
    
    # Change Angle of Attack (modifies params passed to JIT)
    solver.config.alpha = 5.0
    
    # Change AFT parameter (modifies params passed to JIT)
    solver.config.chi_inf = 5.0
    
    # 3. Run Steps - Measure Time
    logger.info("Running Step with New Params...")
    start_time = time.time()
    solver.step()
    first_step_time = time.time() - start_time
    logger.info(f"First Dynamic Step Time: {first_step_time:.4f}s")
    
    # Run more steps
    start_time = time.time()
    solver.step()
    second_step_time = time.time() - start_time
    logger.info(f"Second Dynamic Step Time: {second_step_time:.4f}s")
    
    # Check if recompilation occurred
    if first_step_time > 2.0 and first_step_time > 5 * second_step_time:
        logger.error("POSSIBLE RECOMPILATION DETECTED!")
        print("FAIL: Recompilation likely occurred.")
        sys.exit(1)
    else:
        logger.info("SUCCESS: No significant recompilation delay detected.")
        print("PASS: Dynamic update works.")

if __name__ == "__main__":
    test_dynamic_params()
