"""
Boundary Layer Solvers with JAX JIT compilation for GPU acceleration.

This module provides solvers for:
- Blasius flat plate boundary layer
- Falkner-Skan (pressure gradient) boundary layer
- Flat plate with SA-AFT transition model
"""

import numpy as np

from src.physics.jax_config import jax, jnp
from src.physics.boundary_layer import Blasius, FalknerSkanWedge
from src.physics.laminar import amplification, compute_nondimensional_amplification_rate, Re_Omega
from src.physics.spalart_allmaras import spalart_allmaras_amplification, fv1


def generate_stretched_grid(n, total_length, ratio):
    """Generate grid with geometric progression."""
    if abs(ratio - 1.0) < 1e-6:
        return np.linspace(0, total_length, n + 1, dtype=np.float64)
    h0 = total_length * (1 - ratio) / (1 - ratio**n)
    indices = np.arange(n + 1)
    grid = h0 * (1 - ratio**indices) / (1 - ratio)
    return np.array(grid, np.float64)


@jax.jit
def _build_M_A(u_cell, v_grid, nu_grid, dy_vol, dy_dual):
    """Build M, A matrices for equation M df/dx + A f = 0. Batched (B, ny)."""
    B = u_cell.shape[0]
    _ny = u_cell.shape[1]
    
    M = jax.vmap(jnp.diag)(u_cell)
    coeff = v_grid / dy_dual[1:]
    zeros = jnp.zeros((B, 1))
    
    coeff_p = jnp.concatenate([zeros, jnp.clip(coeff, min=0.0)], axis=-1)
    coeff_m = jnp.concatenate([jnp.clip(-coeff, min=0.0), zeros], axis=-1)
    
    A_adv_p = jax.vmap(jnp.diag)(coeff_p)
    A_adv_p = A_adv_p - jax.vmap(lambda x: jnp.diag(x, k=-1))(coeff_p[:, 1:])
    
    A_adv_m = jax.vmap(jnp.diag)(coeff_m)
    A_adv_m = A_adv_m - jax.vmap(lambda x: jnp.diag(x, k=1))(coeff_m[:, :-1])
    
    A_adv = A_adv_p + A_adv_m
    
    nu_val_up = nu_grid[:, 1:]
    denom_up = dy_vol[:-1] * dy_dual[1:]
    D_up = nu_val_up / denom_up
    
    nu_val_down = nu_grid
    denom_down = dy_vol * dy_dual
    D_down = nu_val_down / denom_down
    
    D_up_padded = jnp.concatenate([D_up, zeros], axis=-1)
    
    diag_main = D_down + D_up_padded
    diag_lower = -D_down[:, 1:]
    diag_upper = -D_up
    
    A_diff = jax.vmap(jnp.diag)(diag_main) + \
             jax.vmap(lambda x: jnp.diag(x, k=-1))(diag_lower) + \
             jax.vmap(lambda x: jnp.diag(x, k=1))(diag_upper)
    
    return M, A_adv + A_diff


class NuHatBlasiusSolver:
    """Blasius boundary layer solver."""
    
    def __init__(self):
        self.dx = 20_000
        self.nx = 200
        self.dy = 50
        self.ny = 200
        self.blasius = Blasius()
    
    def build_M_A_b(self, x):
        """Build M(x), A(x), b for the semi-discrete system."""
        y_grid = self.dy * np.arange(self.ny + 1)
        y_cell, u, dudy, v = self.blasius.at(x, y_grid, cellCentered=True)
        
        u_jax = jnp.array(u)
        dudy_jax = jnp.array(dudy)
        y_cell_jax = jnp.array(y_cell)
        
        M = jnp.diag(u_jax)
        
        coeff = np.concatenate([np.zeros(1), (v[1:] + v[:-1]) / 2]) / self.dy
        A = jnp.diag(jnp.array(coeff)) - jnp.diag(jnp.array(coeff[1:]), k=-1)
        
        sigma = 2./3
        A = A + (2 * jnp.eye(self.ny) - jnp.eye(self.ny, k=1) - jnp.eye(self.ny, k=-1)) / self.dy**2
        A = A.at[0, 0].add(1 / self.dy**2)
        A = A.at[-1, -1].add(-1 / self.dy**2)
        A = A / sigma
        
        b = amplification(u_jax, dudy_jax, y_cell_jax) * dudy_jax
        
        return M, A, b
    
    def __call__(self, nuHat0):
        """March nuHat from 0 to self.nx * self.dx."""
        nuHat = [jnp.ones(self.ny) * nuHat0]
        
        for n in range(self.nx):
            M, A, b = self.build_M_A_b((n + 0.5) * self.dx)
            L = M / self.dx + A
            rhs = (M / self.dx) @ nuHat[-1] + b * nuHat[-1]
            nuHat.append(jax.scipy.linalg.solve(L, rhs))
        
        return jnp.stack(nuHat, axis=0)


class NuHatFalknerSkanSolver:
    """Falkner-Skan boundary layer solver."""
    
    def __init__(self, beta=0.0):
        self.dx = 5_000
        self.nx = 200
        self.dy = 50
        self.ny = 200
        self.fs = FalknerSkanWedge(beta)
    
    def build_M_A_b(self, x):
        """Build M(x), A(x), b for the semi-discrete system."""
        y_grid = self.dy * np.arange(self.ny + 1)
        y_cell, u, dudy, v = self.fs.at(x, y_grid, cellCentered=True)
        
        u_jax = jnp.array(u)
        dudy_jax = jnp.array(dudy)
        y_cell_jax = jnp.array(y_cell)
        
        M = jnp.diag(u_jax)
        
        coeff = np.concatenate([np.zeros(1), (v[1:] + v[:-1]) / 2]) / self.dy
        A = jnp.diag(jnp.array(coeff)) - jnp.diag(jnp.array(coeff[1:]), k=-1)
        
        sigma = 2./3
        A = A + (2 * jnp.eye(self.ny) - jnp.eye(self.ny, k=1) - jnp.eye(self.ny, k=-1)) / self.dy**2
        A = A.at[0, 0].add(1 / self.dy**2)
        A = A.at[-1, -1].add(-1 / self.dy**2)
        A = A / sigma
        
        b = amplification(u_jax, dudy_jax, y_cell_jax) * dudy_jax
        
        return M, A, b
    
    def __call__(self, nuHat0):
        """March nuHat from 0 to self.nx * self.dx."""
        nuHat = [jnp.ones(self.ny) * nuHat0]
        
        for n in range(self.nx):
            M, A, b = self.build_M_A_b((n + 0.5) * self.dx)
            L = M / self.dx + A
            rhs = (M / self.dx) @ nuHat[-1] + b * nuHat[-1]
            nuHat.append(jax.scipy.linalg.solve(L, rhs))
        
        return jnp.stack(nuHat, axis=0)


class NuHatFlatPlateSolver:
    """Flat plate boundary layer solver with SA-AFT transition."""
    
    def __init__(self):
        self.nx = 160
        self.ny = 80
        self.L_plate = 4_000_000.0
        self.H_domain = 50000.0
        self.ratio_x = 1.0
        self.ratio_y = 1.05
        self.x0 = 50_000
        
        x_grid_rel = generate_stretched_grid(self.nx, self.L_plate, self.ratio_x)
        self.dx_list = jnp.array(x_grid_rel[1:] - x_grid_rel[:-1])
        self.x_grid = self.x0 + x_grid_rel
        
        y_grid_np = generate_stretched_grid(self.ny, self.H_domain, self.ratio_y)
        y_cell_np = 0.5 * (y_grid_np[1:] + y_grid_np[:-1])
        dy_vol_np = y_grid_np[1:] - y_grid_np[:-1]
        dy_dual_np = np.zeros(self.ny)
        dy_dual_np[0] = y_cell_np[0]
        dy_dual_np[1:] = y_cell_np[1:] - y_cell_np[:-1]
        
        self.y_cell = jnp.array(y_cell_np)
        self.dy_vol = jnp.array(dy_vol_np)
        self.dy_dual = jnp.array(dy_dual_np)
        
        blasius = Blasius()
        dx_0 = float(self.dx_list[0])
        
        y_cell_0, u0, dudy0, _ = blasius.at(self.x0 - dx_0 / 2, y_grid_np, cellCentered=True)
        _, _, _, v0 = blasius.at(self.x0 - dx_0 / 2, y_grid_np, cellCentered=False)
        
        self.u0_template = jnp.array(u0)
        self.v0_template = jnp.array(v0[1:-1])
    
    def build_u_system(self, u, vgrid, nuHat):
        """Build M, A for momentum equation."""
        B = u.shape[0]
        nu_t = nuHat * fv1(nuHat)[0]
        nu_eff = 1.0 + nu_t
        ones = jnp.ones((B, 1))
        nu_eff_avg = (nu_eff[:, 1:] + nu_eff[:, :-1]) / 2
        nu_eff_grid = jnp.concatenate([ones, nu_eff_avg], axis=-1)
        return _build_M_A(u, vgrid, nu_eff_grid, self.dy_vol, self.dy_dual)
    
    def build_nuhat_system(self, u, vgrid, nuHat):
        """Build M, A for nuHat equation."""
        sigma = 2./3
        Cb2 = 0.622
        B = u.shape[0]
        dnuHat_dy = (nuHat[:, 1:] - nuHat[:, :-1]) / self.dy_dual[1:]
        v_eff = vgrid - dnuHat_dy * Cb2 / sigma
        nu_eff = (1.0 + nuHat) / sigma
        ones = jnp.ones((B, 1))
        nu_eff_avg = (nu_eff[:, 1:] + nu_eff[:, :-1]) / 2
        nu_eff_grid = jnp.concatenate([ones, nu_eff_avg], axis=-1)
        return _build_M_A(u, v_eff, nu_eff_grid, self.dy_vol, self.dy_dual)
    
    def step_u(self, u0, v0, nuHat, dx):
        """Step the velocity equation."""
        _B = u0.shape[0]
        M, A = self.build_u_system(u0, v0, nuHat)
        L = M / dx + A
        rhs_vec = (M / dx) @ u0[:, :, None]
        u_pred = jax.scipy.linalg.solve(L, rhs_vec)[:, :, 0]
        dudx = (u_pred - u0) / dx
        integrand = dudx[:, :-1] * self.dy_vol[:-1]
        v_pred = -jnp.cumsum(integrand, axis=1)
        return u_pred, v_pred
    
    def step_nuhat(self, u, v, nuHat0, dx):
        """Step the nuHat equation with Newton iteration."""
        _B = u.shape[0]
        dudy_wall = u[:, :1] / self.y_cell[0]
        dudy_interior = (u[:, 1:] - u[:, :-1]) / self.dy_dual[1:]
        dudy = jnp.concatenate([dudy_wall, dudy_interior], axis=1)
        
        M, A = self.build_nuhat_system(u, v, nuHat0)
        delta_nuHat = jnp.zeros_like(nuHat0)
        
        is_turb = jnp.clip(1 - jnp.exp(-(nuHat0 - 1) / 4), min=0.0)
        Gamma = jnp.abs(dudy) * self.y_cell / jnp.abs(u)
        a_aft = compute_nondimensional_amplification_rate(Re_Omega(dudy, self.y_cell), Gamma)
        a_aft = a_aft * dudy
        
        for iiter in range(45):
            nuHat1 = nuHat0 + delta_nuHat
            
            (aNuHat_p, aNuHat_p_grad), (aNuHat_m, aNuHat_m_grad) = \
                spalart_allmaras_amplification(dudy, nuHat1, self.y_cell)
            
            aNuHat_p = aNuHat_p * is_turb + a_aft * nuHat0 * (1 - is_turb)
            aNuHat_p_grad = aNuHat_p_grad * is_turb
            
            rhs = (M @ delta_nuHat[:, :, None])[:, :, 0] / dx + \
                  (A @ nuHat1[:, :, None])[:, :, 0] - aNuHat_p + aNuHat_m
            
            L = (M / dx + A + jax.vmap(jnp.diag)(-aNuHat_p_grad + aNuHat_m_grad))
            
            implicit_update = jax.scipy.linalg.solve(L, rhs[:, :, None])[:, :, 0]
            delta_nuHat = delta_nuHat - implicit_update * 0.9
            
            batch_norms = jnp.linalg.norm(implicit_update, axis=1)
            max_err = jnp.max(batch_norms)
            
            if max_err < 1E-4:
                break
        
        return nuHat0 + delta_nuHat
    
    def __call__(self, nuHat_input):
        """
        Run the flat plate solver.
        
        Parameters
        ----------
        nuHat_input : float, list, or array
            Initial turbulence intensity (can be batched).
            
        Returns
        -------
        u : jnp.ndarray
            Velocity field (nx+1, B, ny).
        v : jnp.ndarray
            Normal velocity field (nx+1, B, ny-1).
        nuHat : jnp.ndarray
            SA working variable (nx+1, B, ny).
        """
        if isinstance(nuHat_input, (float, int)):
            nuHat_input = jnp.array([nuHat_input])
        elif isinstance(nuHat_input, list):
            nuHat_input = jnp.array(nuHat_input)
        
        if nuHat_input.ndim == 0:
            nuHat_input = nuHat_input[None]
        
        B = nuHat_input.shape[0]
        
        # Initialize
        current_nuHat = jnp.broadcast_to(nuHat_input[:, None], (B, self.ny)) * 1.0
        current_u = jnp.broadcast_to(self.u0_template[None, :], (B, self.ny))
        current_v = jnp.broadcast_to(self.v0_template[None, :], (B, self.ny - 1))
        
        u_history = [current_u]
        v_history = [current_v]
        nuHat_history = [current_nuHat]
        
        for n in range(self.nx):
            dx_curr = self.dx_list[n]
            
            u1, v1 = self.step_u(u_history[-1], v_history[-1], nuHat_history[-1], dx_curr)
            u_history.append(u1)
            v_history.append(v1)
            
            nuHat1 = self.step_nuhat(u_history[-1], v_history[-1], nuHat_history[-1], dx_curr)
            nuHat_history.append(nuHat1)
        
        u = jnp.stack(u_history, axis=0)
        v = jnp.stack(v_history, axis=0)
        nuHat = jnp.stack(nuHat_history, axis=0)
        
        return u, v, nuHat

