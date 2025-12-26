# AFT-SA Transition Model

This project implements a transition model based on Amplification Factor Transport (AFT) coupled with the Spalart-Allmaras (SA) turbulence model. It explores the evolution of a transition variable $\hat{\nu}$ (nuHat) in Blasius and Falkner-Skan boundary layers.

## Project Structure

*   **`src/`**: Source code for physics models and solvers.
    *   `physics/`:
        *   `boundary_layer.py`: Blasius and Falkner-Skan similarity solutions.
        *   `spalart_allmaras.py`: Spalart-Allmaras turbulence model terms and gradients.
        *   `correlations.py`: Drela-Giles $e^N$ correlations.
    *   `solvers/`:
        *   `boundary_layer_solvers.py`: PyTorch-based parabolic solvers for $\hat{\nu}$ transport.

*   **`scripts/`**: Executable scripts to generate plots and run simulations.
    *   `plot_blasius_growth.py`: Plots amplification rates on Blasius profiles.
    *   `plot_falkner_skan.py`: Plots Falkner-Skan velocity and shear profiles.
    *   `plot_falkner_skan_gamma.py`: Visualizes the transition phase space ($\Gamma$ vs $Re_\Omega$).
    *   `plot_drela_correlation.py`: Plots Drela-Giles growth rates.
    *   `check_inner_layer.py`: Verifies that the transition model doesn't contaminate the turbulent inner layer.
    *   `test_gradients.py`: Validates analytical gradients for the SA model against PyTorch autograd.
    *   `run_blasius_transport.py`: Solves for $\hat{\nu}$ evolution on a Blasius boundary layer.
    *   `run_falkner_skan_transport.py`: Solves for $\hat{\nu}$ evolution on a Falkner-Skan wedge.
    *   `run_flat_plate.py`: Full simulation of a flat plate boundary layer with coupled u-momentum and $\hat{\nu}$ transport.

*   **`docs/`**: Documentation and theory.
    *   `blasius_transition.md`: Theory regarding Blasius transition and the AFT model.
    *   `turbulent_inner_layer.md`: Analysis of the model's behavior in the turbulent wall layer.

## Setup

Ensure you have the required dependencies installed:

```bash
pip install numpy scipy matplotlib torch
```

## Running Scripts

You can run any script from the root directory using python module execution or direct script execution (ensuring `src` is in PYTHONPATH).

Example:

```bash
# Run the flat plate simulation
python -m scripts.run_flat_plate

# Plot Blasius growth rates
python -m scripts.plot_blasius_growth
```
