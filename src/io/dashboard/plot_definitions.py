"""
Shared plot definitions and data helpers for the CFD dashboard.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .._array_utils import sanitize_array, to_json_safe_list
from .._html_components import compute_chi_log
from src.numerics.aft_sources import compute_aft_amplification_rate
from .data import Snapshot


@dataclass(frozen=True)
class ContourPlotSpec:
    """Simple contour plot definition used across layout, traces, and animation."""

    name: str
    color_key: str
    title: str


def get_standard_contour_specs(has_cpt: bool, has_res_field: bool) -> List[ContourPlotSpec]:
    """Return standard (non-AFT) contour plots in trace order."""
    return [
        ContourPlotSpec("pressure", "pressure", "Δp"),
        ContourPlotSpec("u_vel", "velocity", "Δu"),
        ContourPlotSpec("v_vel", "velocity", "Δv"),
        ContourPlotSpec("cpt" if has_cpt else "nu", "pressure", "C_pt" if has_cpt else "ν̃"),
        ContourPlotSpec("residual" if has_res_field else "vel_mag", "residual", "log₁₀(R)"),
        ContourPlotSpec("amplification_rate", "amplification_rate", "log₁₀(a)"),
        ContourPlotSpec("chi", "chi", "log₁₀(χ)"),
    ]


def get_aft_contour_specs(include_amplification_ratio: bool) -> List[ContourPlotSpec]:
    """Return AFT-specific contour plots."""
    specs = [
        ContourPlotSpec("re_omega", "re_omega", "log₁₀(Re_Ω)"),
        ContourPlotSpec("gamma", "gamma", "Γ"),
        ContourPlotSpec("is_turb", "is_turb", "is_turb"),
    ]
    if include_amplification_ratio:
        specs.append(ContourPlotSpec("amplification_ratio", "amplification_ratio", "log₁₀(P/ν̃)"))
    return specs


def get_contour_plot_names(
    has_cpt: bool,
    has_res_field: bool,
    has_aft: bool,
    include_wall_dist: bool,
    include_amplification_ratio: bool,
) -> List[str]:
    """Return all contour plot names in a stable order."""
    names = [spec.name for spec in get_standard_contour_specs(has_cpt, has_res_field)]
    if has_aft:
        names.extend(spec.name for spec in get_aft_contour_specs(include_amplification_ratio))
    if include_wall_dist:
        names.append("wall_dist")
    return names


def build_standard_contour_data(
    snapshot: Snapshot,
    nu_laminar: float,
    has_cpt: bool,
    has_res_field: bool,
) -> Dict[str, np.ndarray]:
    """Compute standard contour plot data arrays from a snapshot."""
    field2 = sanitize_array(snapshot.C_pt if has_cpt and snapshot.C_pt is not None else snapshot.nu)
    if has_res_field and snapshot.residual_field is not None:
        field5 = np.log10(sanitize_array(snapshot.residual_field, fill_value=1e-12) + 1e-12)
    else:
        field5 = np.sqrt(snapshot.u**2 + snapshot.v**2)

    chi_log = compute_chi_log(snapshot.nu, nu_laminar)
    chi_log_safe = to_json_safe_list(chi_log)

    data = {
        "pressure": sanitize_array(snapshot.p).flatten(),
        "u_vel": sanitize_array(snapshot.u).flatten(),
        "v_vel": sanitize_array(snapshot.v).flatten(),
        "cpt": field2.flatten(),
        "nu": field2.flatten(),
        "residual": field5.flatten(),
        "vel_mag": field5.flatten(),
        "chi": chi_log_safe,
    }

    if snapshot.Re_Omega is not None and snapshot.Gamma is not None:
        Re_Omega = sanitize_array(snapshot.Re_Omega, fill_value=0.0)
        Gamma = sanitize_array(snapshot.Gamma, fill_value=0.0)
        raw_data = np.array(compute_aft_amplification_rate(Re_Omega, Gamma)).flatten()
        data["amplification_rate"] = np.log10(np.maximum(raw_data, 1e-12))
    else:
        data["amplification_rate"] = np.zeros_like(snapshot.p).flatten()

    return data


def build_aft_contour_data(snapshot: Snapshot) -> Dict[str, np.ndarray]:
    """Compute AFT contour plot data arrays from a snapshot."""
    if snapshot.Re_Omega is None or snapshot.Gamma is None:
        return {}

    Re_Omega_safe = np.maximum(sanitize_array(snapshot.Re_Omega, fill_value=10.0), 1.0)
    Gamma_safe = sanitize_array(snapshot.Gamma, fill_value=0.0)
    is_turb_safe = (
        sanitize_array(snapshot.is_turb, fill_value=0.0)
        if snapshot.is_turb is not None else np.zeros_like(Gamma_safe)
    )
    if snapshot.amplification_ratio is not None:
        ratio = sanitize_array(snapshot.amplification_ratio, fill_value=0.0)
        amplification_ratio = np.log10(np.maximum(ratio, 1e-12))
    else:
        amplification_ratio = np.zeros_like(Gamma_safe)

    return {
        "re_omega": np.log10(Re_Omega_safe),
        "gamma": Gamma_safe,
        "is_turb": is_turb_safe,
        "amplification_ratio": amplification_ratio,
    }
