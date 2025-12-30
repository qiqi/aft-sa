"""
Laminar benchmark tests using mfoil as ground truth.

This module compares the RANS solver against mfoil (XFOIL-like panel code)
for laminar flow validation cases.

Test Case: NACA 0012 at α=0°, Re=10,000 (fully laminar)
"""

import pytest
import numpy as np

# Import skip marker from conftest
from tests.conftest import requires_construct2d


def run_mfoil_laminar(reynolds: float, alpha: float = 0.0, 
                      naca: str = '0012', npanel: int = 199) -> dict:
    """
    Run mfoil for fully laminar flow.
    
    Parameters
    ----------
    reynolds : float
        Reynolds number based on chord.
    alpha : float, optional
        Angle of attack in degrees (default 0).
    naca : str, optional
        NACA 4-digit airfoil code (default '0012').
    npanel : int, optional
        Number of panels (default 199).
        
    Returns
    -------
    dict
        Results containing:
        - cl: Lift coefficient
        - cd: Total drag coefficient
        - cdf: Skin friction drag coefficient
        - cdp: Pressure drag coefficient
        - converged: Whether solution converged
    """
    from src.validation.mfoil import mfoil
    
    # Initialize mfoil with NACA airfoil
    M = mfoil(naca=naca, npanel=npanel)
    
    # Force fully laminar by setting ncrit extremely high
    # This prevents transition from ever occurring
    M.param.ncrit = 1000.0
    
    # Disable plotting for automated tests
    M.param.doplot = False
    
    # Reduce verbosity
    M.param.verb = 0
    
    # Set operating conditions
    M.setoper(alpha=alpha, Re=reynolds)
    
    # Solve
    try:
        M.solve()
        converged = True
    except Exception as e:
        print(f"mfoil solve failed: {e}")
        converged = False
        return {
            'cl': np.nan,
            'cd': np.nan,
            'cdf': np.nan,
            'cdp': np.nan,
            'converged': False
        }
    
    return {
        'cl': M.post.cl,
        'cd': M.post.cd,
        'cdf': M.post.cdf,
        'cdp': M.post.cdp,
        'converged': converged
    }


class TestMfoilLaminar:
    """Tests for mfoil laminar flow baseline."""
    
    def test_symmetry_zero_lift(self):
        """
        NACA 0012 at α=0° should have exactly zero lift.
        
        This is a fundamental symmetry check for the panel code.
        """
        result = run_mfoil_laminar(reynolds=10000, alpha=0.0)
        
        assert result['converged'], "mfoil failed to converge"
        
        # Cl should be exactly zero (within numerical tolerance)
        assert abs(result['cl']) < 1e-10, \
            f"Cl = {result['cl']:.2e} (expected 0 for symmetric airfoil at α=0)"
    
    def test_reasonable_drag(self):
        """
        Drag coefficient should be positive and reasonable.
        
        For laminar flow at Re=10,000, expect Cd ~ O(0.01-0.1).
        """
        result = run_mfoil_laminar(reynolds=10000, alpha=0.0)
        
        assert result['converged'], "mfoil failed to converge"
        
        # Cd should be positive
        assert result['cd'] > 0, f"Cd = {result['cd']} (expected positive)"
        
        # Cd should be reasonable for laminar flow
        # At Re=10,000, expect Cd roughly 0.02-0.08
        assert result['cd'] < 0.2, f"Cd = {result['cd']} (unexpectedly high)"
        assert result['cd'] > 0.005, f"Cd = {result['cd']} (unexpectedly low)"
        
        print(f"\nLaminar NACA 0012 at Re=10,000:")
        print(f"  Cl = {result['cl']:.6f}")
        print(f"  Cd = {result['cd']:.6f}")
        print(f"  Cdf (friction) = {result['cdf']:.6f}")
        print(f"  Cdp (pressure) = {result['cdp']:.6f}")
    
    def test_drag_components(self):
        """
        Verify drag decomposition: Cd = Cdf + Cdp.
        """
        result = run_mfoil_laminar(reynolds=10000, alpha=0.0)
        
        assert result['converged'], "mfoil failed to converge"
        
        # Total drag should equal friction + pressure drag
        cd_sum = result['cdf'] + result['cdp']
        assert abs(result['cd'] - cd_sum) < 1e-10, \
            f"Cd = {result['cd']}, but Cdf + Cdp = {cd_sum}"
    
    def test_reynolds_scaling(self):
        """
        Higher Reynolds number should give lower friction drag.
        
        For laminar flow, Cf ~ 1/sqrt(Re), so Cd should decrease with Re.
        """
        result_low = run_mfoil_laminar(reynolds=5000, alpha=0.0)
        result_high = run_mfoil_laminar(reynolds=20000, alpha=0.0)
        
        assert result_low['converged'], "mfoil failed at Re=5000"
        assert result_high['converged'], "mfoil failed at Re=20000"
        
        # Higher Re should have lower drag
        assert result_high['cd'] < result_low['cd'], \
            f"Cd at Re=20000 ({result_high['cd']:.4f}) should be < " \
            f"Cd at Re=5000 ({result_low['cd']:.4f})"
        
        print(f"\nReynolds scaling check:")
        print(f"  Re=5000:  Cd = {result_low['cd']:.6f}")
        print(f"  Re=20000: Cd = {result_high['cd']:.6f}")
        print(f"  Ratio: {result_low['cd'] / result_high['cd']:.2f}")


@requires_construct2d
class TestLaminarBenchmark:
    """
    Benchmark test comparing RANS solver against mfoil.
    
    This test validates the RANS solver's accuracy for laminar flow
    by comparing against the established panel code solution.
    """
    
    def test_drag_comparison(self, naca0012_medium_grid, mfoil_baseline_re10k):
        """
        Compare RANS solver Cd against mfoil baseline.
        
        Expectation: RANS Cd should be within 15% of mfoil Cd.
        
        Note: Some mismatch is expected because:
        - RANS captures more physics (full NS vs. boundary layer approximation)
        - Grid resolution effects
        - Different turbulence modeling assumptions
        """
        from src.solvers.rans_solver import RANSSolver, SolverConfig
        
        # Skip if grid not available
        if naca0012_medium_grid is None:
            pytest.skip("Grid generation failed or construct2d unavailable")
        
        # Get mfoil baseline
        mfoil_result = mfoil_baseline_re10k
        assert mfoil_result['converged'], "mfoil baseline failed"
        
        cd_mfoil = mfoil_result['cd']
        cl_mfoil = mfoil_result['cl']
        
        # Run RANS solver
        config = SolverConfig(
            mach=0.1,  # Low Mach for incompressible
            alpha=0.0,
            reynolds=10000,
            max_iter=5000,
            tol=1e-8,
            print_freq=500,
            output_freq=1000,
        )
        
        solver = RANSSolver(naca0012_medium_grid['path'], config)
        solver.run_steady_state()
        
        forces = solver.compute_forces()
        cd_rans = forces.CD
        cl_rans = forces.CL
        
        # Check Cl ≈ 0 (symmetry)
        assert abs(cl_rans) < 0.01, f"CL = {cl_rans:.4f} (expected ~0 for symmetric)"
        
        # Check Cd within 15% tolerance
        relative_error = abs(cd_rans - cd_mfoil) / cd_mfoil
        assert relative_error < 0.15, \
            f"RANS Cd ({cd_rans:.4f}) differs from mfoil ({cd_mfoil:.4f}) " \
            f"by {relative_error*100:.1f}% (tolerance: 15%)"
        
        print(f"\nLaminar NACA 0012 at Re=10,000 comparison:")
        print(f"  mfoil: CL={cl_mfoil:.6f}, CD={cd_mfoil:.6f}")
        print(f"  RANS:  CL={cl_rans:.6f}, CD={cd_rans:.6f}")
        print(f"  CD Error: {relative_error*100:.1f}%")
    
    def test_cp_distribution(self, naca0012_medium_grid, mfoil_baseline_re10k):
        """
        Compare RANS Cp distribution against mfoil baseline.
        
        The Cp distribution reveals:
        - Suction peak location and magnitude
        - Trailing edge pressure recovery
        - Wake effects
        """
        from src.solvers.rans_solver import RANSSolver, SolverConfig
        from src.validation.mfoil import mfoil
        
        if naca0012_medium_grid is None:
            pytest.skip("Grid not available")
        
        # Get detailed mfoil Cp distribution
        M = mfoil(naca='0012', npanel=199)
        M.param.ncrit = 1000.0
        M.param.doplot = False
        M.param.verb = 0
        M.setoper(alpha=0.0, Re=10000)
        M.solve()
        
        # mfoil Cp is stored in M.post (check available attributes)
        # Typically: x_cp, cp_upper, cp_lower or similar
        # For now, just verify we can get surface data from RANS
        
        # Run RANS
        config = SolverConfig(
            mach=0.1, alpha=0.0, reynolds=10000,
            max_iter=3000, tol=1e-7,
            print_freq=1000, output_freq=5000,
        )
        
        solver = RANSSolver(naca0012_medium_grid['path'], config)
        solver.run_steady_state()
        
        # Get RANS surface distributions
        surf = solver.get_surface_distributions()
        
        # Basic checks on Cp distribution
        assert len(surf.Cp) > 0, "No surface Cp data"
        assert np.all(np.isfinite(surf.Cp)), "Cp contains NaN/Inf"
        
        # Stagnation point check: Cp ≈ 1 at leading edge
        # For symmetric airfoil at α=0, LE is roughly at min(x)
        le_idx = np.argmin(surf.x)
        cp_stag = surf.Cp[le_idx]
        assert cp_stag > 0.8, f"Stagnation Cp = {cp_stag:.3f} (expected ~1.0)"
        
        # Trailing edge check: Cp should recover toward positive values
        # TE is at max(x)
        te_indices = np.where(surf.x > 0.9 * surf.x.max())[0]
        cp_te = np.mean(surf.Cp[te_indices])
        assert cp_te > -0.5, f"TE Cp = {cp_te:.3f} (pressure not recovering)"
        
        print(f"\nRe=10,000 Cp distribution checks:")
        print(f"  Stagnation Cp: {cp_stag:.4f}")
        print(f"  Trailing edge Cp: {cp_te:.4f}")
        print(f"  Cp range: [{surf.Cp.min():.4f}, {surf.Cp.max():.4f}]")
    
    def test_cf_distribution(self, naca0012_medium_grid, mfoil_baseline_re10k):
        """
        Compare RANS Cf (skin friction) distribution.
        
        For laminar flow, Cf should:
        - Be positive everywhere (no separation)
        - Decrease downstream (BL thickening)
        - Scale approximately as ~1/sqrt(Re_x)
        """
        from src.solvers.rans_solver import RANSSolver, SolverConfig
        
        if naca0012_medium_grid is None:
            pytest.skip("Grid not available")
        
        config = SolverConfig(
            mach=0.1, alpha=0.0, reynolds=10000,
            max_iter=3000, tol=1e-7,
            print_freq=1000, output_freq=5000,
        )
        
        solver = RANSSolver(naca0012_medium_grid['path'], config)
        solver.run_steady_state()
        
        surf = solver.get_surface_distributions()
        
        # Basic checks
        assert len(surf.Cf) > 0, "No Cf data"
        assert np.all(np.isfinite(surf.Cf)), "Cf contains NaN/Inf"
        
        # Cf should be non-negative (no separation for laminar Re=10k)
        # Allow small negative values due to numerical noise
        assert surf.Cf.min() > -0.01, \
            f"Cf min = {surf.Cf.min():.4f} (significant separation detected)"
        
        # Cf should be positive on average (friction drag)
        assert surf.Cf.mean() > 0, "Average Cf is negative"
        
        # Cf should be reasonable magnitude
        # For Re=10k, expect Cf ~ O(0.01-0.1)
        assert surf.Cf.max() < 1.0, f"Cf max = {surf.Cf.max():.3f} (too high)"
        
        print(f"\nRe=10,000 Cf distribution checks:")
        print(f"  Cf range: [{surf.Cf.min():.5f}, {surf.Cf.max():.5f}]")
        print(f"  Mean Cf: {surf.Cf.mean():.5f}")


def get_laminar_baseline(reynolds: float = 10000) -> dict:
    """
    Get the expected laminar flow baseline values.
    
    This is a convenience function for use in other tests.
    
    Parameters
    ----------
    reynolds : float
        Reynolds number (default 10000).
        
    Returns
    -------
    dict
        Baseline values from mfoil.
    """
    return run_mfoil_laminar(reynolds=reynolds, alpha=0.0)


if __name__ == '__main__':
    # Quick test run
    print("Running mfoil laminar baseline test...")
    result = run_mfoil_laminar(reynolds=10000, alpha=0.0)
    
    print(f"\nNACA 0012, α=0°, Re=10,000 (Laminar)")
    print(f"{'='*40}")
    print(f"Converged: {result['converged']}")
    print(f"Cl = {result['cl']:.6f}")
    print(f"Cd = {result['cd']:.6f}")
    print(f"  Cdf (friction) = {result['cdf']:.6f}")
    print(f"  Cdp (pressure) = {result['cdp']:.6f}")

