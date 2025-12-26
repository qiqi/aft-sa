import torch

def fv1(nuHat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (fv1, d(fv1)/d(nuHat)).
    """
    cv1 = 7.1
    chi = nuHat
    chi3 = chi ** 3
    denom = chi3 + cv1 ** 3

    val = chi3 / denom

    # Derivative: d/dchi [chi^3 / (chi^3 + cv1^3)]
    # = (3chi^2 * denom - chi^3 * 3chi^2) / denom^2
    # = 3chi^2 * (denom - chi^3) / denom^2
    # = 3chi^2 * cv1^3 / denom^2
    grad = (3 * chi**2 * cv1**3) / (denom ** 2)

    return val, grad

def fv2(nuHat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (fv2, d(fv2)/d(nuHat)).
    """
    chi = nuHat
    fv1_val, fv1_grad = fv1(nuHat)

    denom = 1.0 + chi * fv1_val
    val = 1.0 - chi / denom

    # Derivative of fv2 = 1 - chi / (1 + chi*fv1)
    # Let term = chi / denom
    # d(term)/dchi = (1*denom - chi*(fv1 + chi*fv1_grad)) / denom^2
    #              = (1 + chi*fv1 - chi*fv1 - chi^2*fv1_grad) / denom^2
    #              = (1 - chi^2*fv1_grad) / denom^2
    # fv2_grad = - d(term)/dchi
    term_grad = (1.0 - chi**2 * fv1_grad) / (denom ** 2)
    grad = -term_grad

    return val, grad

def _S_tilde(dudy: torch.Tensor, nuHat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (S_tilde, d(S_tilde)/d(nuHat)).
    """
    kappa = 0.41
    omega = dudy.abs()
    fv2_val, fv2_grad = fv2(nuHat)

    # Constants wrt nuHat
    inv_k2y2 = 1.0 / (kappa ** 2 * y ** 2)

    # S_tilde = Omega + nuHat * inv_k2y2 * fv2
    # S_tilde_raw to check for clamping
    s_tilde_term = nuHat * inv_k2y2 * fv2_val
    S_tilde_raw = omega + s_tilde_term

    # Derivative:
    # d/dnuHat = inv_k2y2 * [ 1 * fv2 + nuHat * fv2' ]
    grad_raw = inv_k2y2 * (fv2_val + nuHat * fv2_grad)

    # Apply clamp: min=1e-16
    # If S_tilde_raw < 1e-16, value is 1e-16 and grad is 0
    val = torch.clamp(S_tilde_raw, min=1e-16)
    grad = torch.where(S_tilde_raw < 1e-16, torch.zeros_like(grad_raw), grad_raw)

    return val, grad

def r(dudy: torch.Tensor, nuHat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (r, d(r)/d(nuHat)).
    """
    kappa = 0.41
    S_t, S_t_grad = _S_tilde(dudy, nuHat, y)

    denom_factor = kappa ** 2 * y ** 2
    denom = S_t * denom_factor

    r_raw = nuHat / denom

    # Derivative of nuHat / (S_t * C) where C = k^2 y^2
    # d/dnuHat = (1 * (S_t*C) - nuHat * (S_t_grad*C)) / (S_t*C)^2
    #          = C(S_t - nuHat*S_t_grad) / C^2 S_t^2
    #          = (S_t - nuHat*S_t_grad) / (C * S_t^2)
    grad_raw = (S_t - nuHat * S_t_grad) / (denom_factor * S_t ** 2)

    # Apply clamp: max=10.0
    val = torch.clamp(r_raw, max=10.0)
    grad = torch.where(r_raw > 10.0, torch.zeros_like(grad_raw), grad_raw)

    return val, grad

def g(dudy: torch.Tensor, nuHat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (g, d(g)/d(nuHat)).
    """
    cw2 = 0.3
    r_val, r_grad = r(dudy, nuHat, y)

    # g = r + cw2 * (r^6 - r)
    val = r_val + cw2 * (r_val ** 6 - r_val)

    # Chain rule: dg/dnuHat = dg/dr * dr/dnuHat
    # dg/dr = 1 + cw2 * (6r^5 - 1)
    dg_dr = 1.0 + cw2 * (6.0 * r_val ** 5 - 1.0)

    grad = dg_dr * r_grad
    return val, grad

def fw(dudy: torch.Tensor, nuHat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (fw, d(fw)/d(nuHat)).
    """
    cw3 = 2.0
    g_val, g_grad = g(dudy, nuHat, y)

    g6 = g_val ** 6
    c6 = cw3 ** 6
    denom = g6 + c6
    top = 1.0 + c6

    ratio = top / denom
    radicand = ratio ** (1.0 / 6.0)

    val = g_val * radicand

    # Derivative dfw/dg
    # fw = g * ((1+c6)/(g^6+c6))^(1/6)
    # Using logarithmic differentiation or quotient rules efficiently:
    # dfw/dg = radicand * [ c6 / (g^6 + c6) ]
    # (Derived from: fw' = R^(1/6) * (1 - g^6/(g^6+c6)) )

    # Note: radicand is (top/denom)^(1/6)
    # term2 = c6 / denom
    dfw_dg = radicand * (c6 / denom)

    grad = dfw_dg * g_grad
    return val, grad

def spalart_allmaras_amplification(
    dudy: torch.Tensor,
    nuHat: torch.Tensor,
    y: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns ((Production, dP/dnuHat), (Destruction, dD/dnuHat)).
    """
    cb1 = 0.1355
    cb2 = 0.622
    sigma = 2.0 / 3.0
    kappa = 0.41
    cw1 = cb1 / (kappa ** 2) + (1.0 + cb2) / sigma

    # Production = cb1 * S_tilde * nuHat
    S_t, S_t_grad = _S_tilde(dudy, nuHat, y)
    prod_val = cb1 * S_t * nuHat

    # d(Prod)/dnuHat = cb1 * (S_t_grad * nuHat + S_t * 1)
    prod_grad = cb1 * (S_t_grad * nuHat + S_t)

    # Destruction = cw1 * fw * (nuHat / y)^2
    fw_val, fw_grad = fw(dudy, nuHat, y)

    term_sq = (nuHat / y) ** 2
    dest_val = cw1 * fw_val * term_sq

    # d(Dest)/dnuHat
    # = cw1 * [ fw_grad * (nuHat/y)^2 + fw_val * d/dnuHat((nuHat/y)^2) ]
    # d/dnuHat((nuHat/y)^2) = 2 * (nuHat/y) * (1/y) = 2 * nuHat / y^2

    term_sq_grad = 2.0 * nuHat / (y ** 2)
    dest_grad = cw1 * (fw_grad * term_sq + fw_val * term_sq_grad)

    return (prod_val, prod_grad), (dest_val, dest_grad)
