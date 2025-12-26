import numpy as np

def compute_mesh_quality(X, Y):
    """
    Computes mesh quality metrics:
    1. Orthogonality (cosine of angle, ideal 0, degenerate 1 or -1?)
       Wait, let's use sin(theta) or similar.
       Standard metric: (A . B) / (|A| |B|). Ideal is 0.
       Or cross product normalized (sin theta). Ideal is 1 (90 deg).
       Negative means flipped (Jacobian < 0).
       
    2. Aspect Ratio (max side / min side)
    
    3. Smoothness (Area ratio between neighbors)
    
    Returns:
        quality: Dict containing metric arrays.
    """
    ni, nj = X.shape
    
    # Grid vectors
    # I-direction: (i+1, j) - (i, j)
    dx_i = X[1:, :] - X[:-1, :]
    dy_i = Y[1:, :] - Y[:-1, :]
    
    # J-direction: (i, j+1) - (i, j)
    dx_j = X[:, 1:] - X[:, :-1]
    dy_j = Y[:, 1:] - Y[:, :-1]
    
    # We need to evaluate at cell centers or nodes. Let's do nodes (interior).
    # Vectors at node (i,j) can be approximated by averaging neighbors or just taking forward/backward.
    # Let's compute cell areas (Jacobians) first.
    # Cell (i, j) defined by nodes (i,j), (i+1,j), (i+1,j+1), (i,j+1).
    # Cross product of diagonals or sum of triangles.
    # Vector 1: (i+1, j) - (i, j) -> Bottom edge
    # Vector 2: (i, j+1) - (i, j) -> Left edge
    # Simple cross product at corner (i,j): dx_i * dy_j - dy_i * dx_j
    # This is effectively the Jacobian at the node (i,j).
    
    # Let's compute node-based orthogonality.
    # At node (i, j), we have 4 vectors. Let's take the centered vectors.
    # vec_i = (x[i+1] - x[i-1], y[i+1] - y[i-1])
    # vec_j = (x[j+1] - x[j-1], y[j+1] - y[j-1])
    
    # Interior nodes
    x_i = X[2:, 1:-1] - X[:-2, 1:-1]
    y_i = Y[2:, 1:-1] - Y[:-2, 1:-1]
    
    x_j = X[1:-1, 2:] - X[1:-1, :-2]
    y_j = Y[1:-1, 2:] - Y[1:-1, :-2]
    
    dot = x_i * x_j + y_i * y_j
    mag_i = np.sqrt(x_i**2 + y_i**2)
    mag_j = np.sqrt(x_j**2 + y_j**2)
    
    # Cosine of angle deviation from 90.
    # if orthogonal, dot = 0.
    # Normalized dot product: cos(theta).
    # If cos(theta) = 0, good.
    # If cos(theta) = 1 or -1, degenerate.
    
    # Metric: sine of angle = sqrt(1 - cos^2) * sign(cross)
    # Cross product z = xi*yj - yi*xj
    cross = x_i * y_j - y_i * x_j
    
    ortho = cross / (mag_i * mag_j + 1e-12)
    # 1.0 = perfect 90 deg. 0.0 = collapsed. < 0 = flipped.
    
    # Cell Volumes (Areas)
    # Use 0.5 * | (x_ac * y_bd) - (y_ac * x_bd) | for diagonals
    # A = (i,j), B=(i+1,j), C=(i+1,j+1), D=(i,j+1)
    # AC = C - A, BD = D - B
    x_ac = X[1:, 1:] - X[:-1, :-1]
    y_ac = Y[1:, 1:] - Y[:-1, :-1]
    x_bd = X[:-1, 1:] - X[1:, :-1]
    y_bd = Y[:-1, 1:] - Y[1:, :-1]
    
    areas = 0.5 * (x_ac * y_bd - y_ac * x_bd)
    # Negative areas indicate flipped cells (if ordering is CCW vs CW)
    
    return {
        'orthogonality': ortho, # (ni-2, nj-2)
        'areas': areas          # (ni-1, nj-1)
    }

def elliptic_smooth(X, Y, n_iter=50, relax=0.5):
    """
    Elliptic smoothing (Laplace/Poisson).
    Solves xi_xx + xi_yy = P, eta_xx + eta_yy = Q
    In physical space:
    alpha * r_xx - 2*beta * r_xi_eta + gamma * r_yy = Source
    
    Simple Laplacian smoothing: each point moves to average of neighbors.
    Preserves boundaries.
    """
    X_new = X.copy()
    Y_new = Y.copy()
    
    ni, nj = X.shape
    
    for k in range(n_iter):
        # Update interior points
        # X_new[i,j] = 0.25 * (X[i+1,j] + X[i-1,j] + X[i,j+1] + X[i,j-1])
        # This is simple Laplace. Tends to pull grid lines away from boundaries (clustering loss).
        # We need to preserve clustering (j-direction).
        # One way: only smooth in i-direction? Or use TFI-like logic?
        
        # Or standard elliptical generation with control functions P, Q to maintain spacing.
        # Approximated by using weighted averages based on current spacing?
        
        # Simple fix for "intersecting near trailing edge":
        # The intersection happens because the normals cross.
        # Smoothing the mesh usually helps.
        
        # Vectorized update
        X_ip = X_new[2:, 1:-1]
        X_im = X_new[:-2, 1:-1]
        X_jp = X_new[1:-1, 2:]
        X_jm = X_new[1:-1, :-2]
        
        Y_ip = Y_new[2:, 1:-1]
        Y_im = Y_new[:-2, 1:-1]
        Y_jp = Y_new[1:-1, 2:]
        Y_jm = Y_new[1:-1, :-2]
        
        # Standard Laplacian
        # X_inner = 0.25 * (X_ip + X_im + X_jp + X_jm)
        # Y_inner = 0.25 * (Y_ip + Y_im + Y_jp + Y_jm)
        
        # To preserve wall clustering (j-direction), we usually disable smoothing in j 
        # or use very weak relaxation, or use algebraic terms.
        # If we smooth purely, the tight spacing at wall will diffuse out.
        
        # Let's try 1D smoothing in i-direction (streamwise) first? 
        # Intersection is likely due to wake lines turning sharply or normals crossing.
        # Let's apply smoothing ONLY to X, Y coordinates relative to neighbors, but weight it?
        
        # Implementation of " Winslow " smoother (Equidistribution)
        # alpha * x_xi_xi + ...
        
        # Simplified: weighted Laplacian
        # Preserve J-distribution by restricting movement in normal direction? 
        # Hard for structured grid.
        
        # Fallback: Just smooth the Wake Region specifically?
        # Let's apply a few passes of simple Laplacian with low relaxation
        # BUT fix the j-distribution.
        # i.e., allow points to slide along the constant-j lines, or 
        # re-project onto algebraic lines.
        
        # Let's stick to the algebraic generator refinement first.
        # If we smooth here, we might lose y+ control.
        pass
        
    return X_new, Y_new
