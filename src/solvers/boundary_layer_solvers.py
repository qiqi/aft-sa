import torch
import torch.nn as nn
import numpy as np

from src.physics.boundary_layer import Blasius, FalknerSkanWedge
from src.physics.laminar import amplification, compute_nondimensional_amplification_rate, Re_Omega
from src.physics.spalart_allmaras import spalart_allmaras_amplification, fv1

def build_M_A(u_cell, v_grid, nu_grid, dy_vol, dy_dual):
    """
    Build M, A for the equation: M df/dx + A f = 0 on a variable grid.
    Supports Batch dimension (B, ny).

    Args:
        u_cell: (B, ny)
        v_grid: (B, ny-1)
        nu_grid: (B, ny)
        dy_vol: (ny,) 1D Tensor
        dy_dual: (ny,) 1D Tensor
    """
    # Batch size
    B = u_cell.shape[0]
    ny = u_cell.shape[1]

    # Mass matrix: u ∂u/∂x term
    # Input: (B, ny) -> Output: (B, ny, ny) diagonal matrices
    M = torch.diag_embed(u_cell)

    # ---- v * ∂/∂y with upwind (v>0) ----
    # v is defined at faces 1..ny-1.
    # coeff shape: (B, ny-1)
    coeff = v_grid / dy_dual[1:]

    zeros = torch.zeros((B, 1), dtype=u_cell.dtype, device=u_cell.device)

    coeff_p = torch.cat([zeros, torch.clamp(coeff, min=0.0)], dim=-1) # (B, ny)
    coeff_m = torch.cat([torch.clamp(-coeff, min=0.0), zeros], dim=-1) # (B, ny)

    # Matrix Construction using diag_embed
    # Main diagonal
    A_adv_p = torch.diag_embed(coeff_p)
    # Lower diagonal (offset -1). Note: we slice input to size ny-1
    A_adv_p -= torch.diag_embed(coeff_p[:, 1:], offset=-1)

    # Main diagonal
    A_adv_m = torch.diag_embed(coeff_m)
    # Upper diagonal (offset 1). Note: we slice input to size ny-1
    A_adv_m -= torch.diag_embed(coeff_m[:, :-1], offset=1)

    A_adv = A_adv_p + A_adv_m

    # ---- variable viscosity: ∂/∂y ( nu_eff ∂/∂y ) ----
    # nu_grid is (B, ny).

    # Up fluxes (faces 1 to ny-1)
    nu_val_up = nu_grid[:, 1:] # (B, ny-1)
    denom_up = dy_vol[:-1] * dy_dual[1:] # (ny-1) broadcasts
    D_up = nu_val_up / denom_up # (B, ny-1)

    # Down fluxes (faces 0 to ny-1)
    nu_val_down = nu_grid
    denom_down = dy_vol * dy_dual
    D_down = nu_val_down / denom_down # (B, ny)

    # Assemble Tridiagonal Diffusion Matrix
    # Main diagonal: D_down + D_up (padded)
    # Note: We need to handle the shapes carefully for addition
    # D_down is (B, ny), D_up is (B, ny-1).
    # We append a 0 to D_up to align with D_down for the main diagonal addition
    D_up_padded = torch.cat([D_up, zeros], dim=-1)

    diag_main = D_down + D_up_padded

    diag_lower = -D_down[:, 1:] # (B, ny-1)
    diag_upper = -D_up          # (B, ny-1)

    A_diff = torch.diag_embed(diag_main) + \
             torch.diag_embed(diag_lower, offset=-1) + \
             torch.diag_embed(diag_upper, offset=1)

    return M, A_adv + A_diff

def generate_stretched_grid(n, total_length, ratio):
    if abs(ratio - 1.0) < 1e-6:
        return np.linspace(0, total_length, n + 1, dtype=np.float32)
    h0 = total_length * (1 - ratio) / (1 - ratio**n)
    indices = np.arange(n + 1)
    grid = h0 * (1 - ratio**indices) / (1 - ratio)
    return np.array(grid, np.float32)

class NuHatBlasiusSolver(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.dx = 20_000
        self.nx = 200
        self.dy = 50
        self.ny = 200
        self.blasius = Blasius()

    def build_M_A_b(self, x, device='cpu'):
        """
        Build M(x), A(x) for the semi-discrete system
            M q' + A q = b.
        - M is diag(u)
        - A includes v * d/dy (upwind, v>0) and d^2/dy^2
        - b is the source term
        """
        y_grid = self.dy * np.arange(self.ny + 1)
        y_cell, u, dudy, v = self.blasius.at(x, y_grid, cellCentered=True)
        M = np.diag(u)
        # ---- v * d/dy with upwind (v>0) ----
        coeff = np.concatenate([np.zeros(1), (v[1:] + v[:-1]) / 2]) / self.dy
        A = np.diag(coeff) - np.diag(coeff[1:], -1)
        # ---- viscosity -----
        sigma = 2./3
        A += (2 * np.eye(self.ny) - np.eye(self.ny, k=1) - np.eye(self.ny, k=-1)) / self.dy**2
        A[0,0] += 1 / self.dy**2   # zero value at wall (first cell reflected)
        A[-1,-1] -= 1 / self.dy**2 # zero y-derivative at max-y
        A /= sigma
        # ---- source term -----
        b = amplification(u, dudy, y_cell) * dudy
        return (torch.tensor(M, dtype=torch.float32, device=device),
                torch.tensor(A, dtype=torch.float32, device=device),
                torch.tensor(b, dtype=torch.float32, device=device))

    def forward(self, nuHat0):
        """
        March nuHat from 0 to self.nx * self.dx
        nuHat0: initial nuHat at x=0
        returns: nuHat over x: [Nx, Ny]
        """
        nuHat = [torch.ones(self.ny, dtype=torch.float32) * nuHat0]
        for n in range(self.nx):
            M, A, b = self.build_M_A_b((n + 0.5) * self.dx)
            # Left-hand side matrix: (M / dx + A)
            L = M / self.dx + A  # [Ny, Ny]
            # Right-hand side: (M / dx) * nuHat_n + b
            rhs = (M / self.dx) @ nuHat[-1] + b * nuHat[-1]
            # solve for q_{n+1}: L q_{n+1} = rhs
            # torch.linalg.solve expects [Ny,Ny] and [Ny, batch]
            nuHat.append(torch.linalg.solve(L, rhs))

        nuHat = torch.stack(nuHat, dim=0)  # [Nx, Ny]
        return nuHat


class NuHatFalknerSkanSolver(nn.Module):
    def __init__(self, beta=0.0, device="cpu"):
        super().__init__()
        self.dx = 5_000
        self.nx = 200
        self.dy = 50
        self.ny = 200
        self.fs = FalknerSkanWedge(beta)

    def build_M_A_b(self, x, device='cpu'):
        """
        Build M(x), A(x) for the semi-discrete system
            M q' + A q = b.
        - M is diag(u)
        - A includes v * d/dy (upwind, v>0) and d^2/dy^2
        - b is the source term
        """
        y_grid = self.dy * np.arange(self.ny + 1)
        y_cell, u, dudy, v = self.fs.at(x, y_grid, cellCentered=True)
        M = np.diag(u)
        # ---- v * d/dy with upwind (v>0) ----
        coeff = np.concatenate([np.zeros(1), (v[1:] + v[:-1]) / 2]) / self.dy
        A = np.diag(coeff) - np.diag(coeff[1:], -1)
        # ---- viscosity -----
        sigma = 2./3
        A += (2 * np.eye(self.ny) - np.eye(self.ny, k=1) - np.eye(self.ny, k=-1)) / self.dy**2
        A[0,0] += 1 / self.dy**2   # zero value at wall (first cell reflected)
        A[-1,-1] -= 1 / self.dy**2 # zero y-derivative at max-y
        A /= sigma
        # ---- source term -----
        b = amplification(u, dudy, y_cell) * dudy
        return (torch.tensor(M, dtype=torch.float32, device=device),
                torch.tensor(A, dtype=torch.float32, device=device),
                torch.tensor(b, dtype=torch.float32, device=device))

    def forward(self, nuHat0):
        """
        March nuHat from 0 to self.nx * self.dx
        nuHat0: initial nuHat at x=0
        returns: nuHat over x: [Nx, Ny]
        """
        nuHat = [torch.ones(self.ny, dtype=torch.float32) * nuHat0]
        for n in range(self.nx):
            M, A, b = self.build_M_A_b((n + 0.5) * self.dx)
            # Left-hand side matrix: (M / dx + A)
            L = M / self.dx + A  # [Ny, Ny]
            # Right-hand side: (M / dx) * nuHat_n + b
            rhs = (M / self.dx) @ nuHat[-1] + b * nuHat[-1]
            # solve for q_{n+1}: L q_{n+1} = rhs
            # torch.linalg.solve expects [Ny,Ny] and [Ny, batch]
            nuHat.append(torch.linalg.solve(L, rhs))

        nuHat = torch.stack(nuHat, dim=0)  # [Nx, Ny]
        return nuHat


class NuHatFlatPlateSolver(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

        # Grid Parameters
        self.nx = 160
        self.ny = 80
        self.L_plate = 4_000_000.0
        self.H_domain = 50000.0

        self.ratio_x = 1.0
        self.ratio_y = 1.05

        self.device = device
        self.x0 = 50_000

        # Generate X Grid
        x_grid_rel = generate_stretched_grid(self.nx, self.L_plate, self.ratio_x)
        self.dx_list = torch.tensor(x_grid_rel[1:] - x_grid_rel[:-1], dtype=torch.float32, device=device)
        self.x_grid = self.x0 + x_grid_rel

        # Generate Y Grid
        y_grid_np = generate_stretched_grid(self.ny, self.H_domain, self.ratio_y)
        y_cell_np = 0.5 * (y_grid_np[1:] + y_grid_np[:-1])
        dy_vol_np = y_grid_np[1:] - y_grid_np[:-1]
        dy_dual_np = np.zeros(self.ny)
        dy_dual_np[0] = y_cell_np[0]
        dy_dual_np[1:] = y_cell_np[1:] - y_cell_np[:-1]

        self.register_buffer('y_cell', torch.tensor(y_cell_np, dtype=torch.float32))
        self.register_buffer('dy_vol', torch.tensor(dy_vol_np, dtype=torch.float32))
        self.register_buffer('dy_dual', torch.tensor(dy_dual_np, dtype=torch.float32))

        # Initial Conditions (Blasius)
        # Note: We still calculate these as 1D vectors, but will expand them in forward()
        blasius = Blasius()
        dx_0 = self.dx_list[0].item()

        y_cell_0, u0, dudy0, _ = blasius.at(self.x0 - dx_0 / 2, y_grid_np, cellCentered=True)
        _,_,_, v0 = blasius.at(self.x0 - dx_0 / 2, y_grid_np, cellCentered=False)

        self.register_buffer('u0_template', torch.tensor(u0, dtype=torch.float32))
        self.register_buffer('v0_template', torch.tensor(v0[1:-1], dtype=torch.float32))

    # -------------------------------------------------------------------------
    # u-equation
    # -------------------------------------------------------------------------
    def build_u_system(self, u, vgrid, nuHat):
        # Assumed shape: (B, ny)
        B = u.shape[0]

        # nuHat * fv1(nuHat)[0] -> Assuming fv1 handles element-wise ops
        nu_t = nuHat * fv1(nuHat)[0]
        nu_eff = 1.0 + nu_t

        # Interpolate to faces. (B, ny) -> (B, ny+1) effectively, but we map to indices
        # We need ones at the wall (index 0)
        ones = torch.ones((B, 1), dtype=nu_eff.dtype, device=self.device)
        nu_eff_avg = (nu_eff[:, 1:] + nu_eff[:, :-1]) / 2

        nu_eff_grid = torch.cat([ones, nu_eff_avg], dim=-1) # (B, ny)

        return build_M_A(u, vgrid, nu_eff_grid, self.dy_vol, self.dy_dual)

    # -------------------------------------------------------------------------
    # nuHat-equation
    # -------------------------------------------------------------------------
    def build_nuhat_system(self, u, vgrid, nuHat):
        sigma = 2./3
        Cb2 = 0.622
        B = u.shape[0]

        # Gradient on non-uniform grid.
        # nuHat: (B, ny). dy_dual: (ny,) -> broadcasts to (B, ny) via [1:]
        dnuHat_dy = (nuHat[:, 1:] - nuHat[:, :-1]) / self.dy_dual[1:]

        v_eff = vgrid - dnuHat_dy * Cb2 / sigma

        nu_eff = (1.0 + nuHat) / sigma

        ones = torch.ones((B, 1), dtype=torch.float32, device=self.device)
        nu_eff_avg = (nu_eff[:, 1:] + nu_eff[:, :-1]) / 2

        nu_eff_grid = torch.cat([ones, nu_eff_avg], dim=-1)
        return build_M_A(u, v_eff, nu_eff_grid, self.dy_vol, self.dy_dual)

    # -------------------------------------------------------------------------
    # Steps
    # -------------------------------------------------------------------------
    def step_u(self, u0, v0, nuHat, dx):
        """
        u0: (B, ny)
        """
        B = u0.shape[0]
        M, A = self.build_u_system(u0, v0, nuHat)

        # Implicit Step
        L = M / dx + A

        # RHS prep: (B, N, N) @ (B, N, 1) -> (B, N, 1)
        rhs_vec = (M / dx) @ u0.unsqueeze(-1)

        u_pred = torch.linalg.solve(L, rhs_vec) # Returns (B, N, 1)
        u_pred = u_pred.squeeze(-1) # (B, N)

        dudx = (u_pred - u0) / dx

        # Continuity integration on variable grid
        # dudx[:-1] is slice on spatial dim 1. dy_vol[:-1] is (ny-1).
        # We broadcast and multiply, then cumsum along dim 1
        integrand = dudx[:, :-1] * self.dy_vol[:-1]
        v_pred = -torch.cumsum(integrand, dim=1)

        return u_pred, v_pred

    def step_nuhat(self, u, v, nuHat0, dx):
        # u: (B, ny)
        B = u.shape[0]

        # Derivative at wall (y=0): u[:, :1] / y_cell[0]
        dudy_wall = u[:, :1] / self.y_cell[0] # (B, 1)
        dudy_interior = (u[:, 1:] - u[:, :-1]) / self.dy_dual[1:] # (B, ny-1)
        dudy = torch.cat((dudy_wall, dudy_interior), dim=1) # (B, ny)

        M, A = self.build_nuhat_system(u, v, nuHat0)
        delta_nuHat = torch.zeros_like(nuHat0)

        # static source terms
        is_turb = torch.clamp(1 - torch.exp(-(nuHat0 - 1) / 4), min=0.0)
        Gamma = torch.abs(dudy) * self.y_cell / torch.abs(u)
        a_aft = compute_nondimensional_amplification_rate(Re_Omega(dudy, self.y_cell), Gamma)
        '''
        print(Re_Omega(dudy, self.y_cell))
        print(Gamma)
        print(a_aft)
        '''
        a_aft = a_aft * dudy

        final_norm = 0.0

        for iiter in range(45):
            nuHat1 = nuHat0 + delta_nuHat

            # Source terms
            (aNuHat_p, aNuHat_p_grad), (aNuHat_m, aNuHat_m_grad) = \
                spalart_allmaras_amplification(dudy, nuHat1, self.y_cell)

            aNuHat_p = aNuHat_p * is_turb + a_aft * nuHat0 * (1 - is_turb)
            aNuHat_p_grad = aNuHat_p_grad * is_turb

            # aNuHat_m = aNuHat_m * is_turb
            # aNuHat_m_grad = aNuHat_m_grad * is_turb

            # RHS: (B, N, 1) logic
            # M @ delta (B, N) -> needs unsqueeze
            rhs = (M @ delta_nuHat.unsqueeze(-1)).squeeze(-1) / dx + \
                  (A @ nuHat1.unsqueeze(-1)).squeeze(-1) - aNuHat_p + aNuHat_m

            # Linearization (Jacobian)
            # diag_embed handles (B, N) -> (B, N, N)
            L = (M / dx + A + torch.diag_embed(-aNuHat_p_grad + aNuHat_m_grad))

            implicit_update = torch.linalg.solve(L, rhs.unsqueeze(-1)).squeeze(-1)
            delta_nuHat = delta_nuHat - implicit_update * 0.9

            # Check convergence across batch (take max error)
            batch_norms = torch.linalg.norm(implicit_update, dim=1)
            max_err = torch.max(batch_norms).detach().cpu().numpy()
            final_norm = max_err

            if max_err < 1E-4:
                break
        else:
            # Optional: Warning if not converged
            pass

        return nuHat0 + delta_nuHat

    def forward(self, nuHat_input):
        """
        nuHat_input: Can be float, (1,), or (B,) tensor representing Tu intensities
        """
        # Parse Input to Standardize Batch
        if isinstance(nuHat_input, (float, int)):
            nuHat_input = torch.tensor([nuHat_input], dtype=torch.float32, device=self.device)
        elif isinstance(nuHat_input, list):
            nuHat_input = torch.tensor(nuHat_input, dtype=torch.float32, device=self.device)

        # Ensure it is at least 1D
        if nuHat_input.dim() == 0:
            nuHat_input = nuHat_input.unsqueeze(0)

        B = nuHat_input.shape[0]

        # Expand Initial Conditions for Batch
        # (B, ny)
        current_nuHat = nuHat_input.unsqueeze(1).expand(B, self.ny) * 1.0
        current_u = self.u0_template.unsqueeze(0).expand(B, self.ny)
        current_v = self.v0_template.unsqueeze(0).expand(B, self.ny - 1)

        # Storage
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

        # Stack logic: Time dimension first or batch?
        # Usually (Time/Space_X, Batch, Space_Y).
        # Original was (Space_X, Space_Y).
        # Let's return (Space_X, Batch, Space_Y)
        u = torch.stack(u_history, dim=0)
        v = torch.stack(v_history, dim=0)
        nuHat = torch.stack(nuHat_history, dim=0)

        return u, v, nuHat
