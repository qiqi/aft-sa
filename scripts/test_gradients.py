import torch
import numpy as np
from src.physics.spalart_allmaras import spalart_allmaras_amplification

def run():
    print("--- Verifying Spalart-Allmaras Analytical Gradients ---")

    # Use double precision for strict numerical checking
    torch.set_default_dtype(torch.float64)

    # 1. Create Inputs
    dudy = torch.tensor([0.0, 10.0, 100.0, 50.0])
    y = torch.tensor([1.0, 0.1, 0.01, 0.5])

    # nuHat is the variable we are differentiating with respect to
    nuHat = torch.tensor([1e-5, 0.5, 5.0, 20.0], requires_grad=True)

    # 2. Run the Modified Forward Pass (Analytical)
    # This returns ((Prod, dProd/dNu), (Dest, dDest/dNu))
    (prod_val, prod_grad_analytic), (dest_val, dest_grad_analytic) = \
        spalart_allmaras_amplification(dudy, nuHat, y)

    # 3. Compute PyTorch Autograd (Automatic)
    
    # Verify Production
    prod_grad_autograd = torch.autograd.grad(
        outputs=prod_val,
        inputs=nuHat,
        grad_outputs=torch.ones_like(prod_val),
        retain_graph=True
    )[0]

    # Verify Destruction
    dest_grad_autograd = torch.autograd.grad(
        outputs=dest_val,
        inputs=nuHat,
        grad_outputs=torch.ones_like(dest_val),
        retain_graph=False
    )[0]

    # 4. Compare Results
    tol = 1e-8

    prod_diff = (prod_grad_analytic - prod_grad_autograd).abs().max()
    dest_diff = (dest_grad_analytic - dest_grad_autograd).abs().max()

    print(f"\nProduction Gradient Difference (Max Abs): {prod_diff:.2e}")
    if prod_diff < tol:
        print("✅ Production gradient matches Autograd.")
    else:
        print("❌ Production gradient mismatch!")

    print(f"Destruction Gradient Difference (Max Abs): {dest_diff:.2e}")
    if dest_diff < tol:
        print("✅ Destruction gradient matches Autograd.")
    else:
        print("❌ Destruction gradient mismatch!")

    print("\n--- Detailed Dump ---")
    print("NuHat:          ", nuHat.detach().numpy())
    print("Prod Analytic:  ", prod_grad_analytic.detach().numpy())
    print("Prod Autograd:  ", prod_grad_autograd.detach().numpy())
    print("Dest Analytic:  ", dest_grad_analytic.detach().numpy())
    print("Dest Autograd:  ", dest_grad_autograd.detach().numpy())

if __name__ == "__main__":
    run()
