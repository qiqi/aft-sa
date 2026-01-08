"""
Interactive HTML dashboard for CFD results.
"""

from .facade import PlotlyDashboard
from .data import Snapshot

__all__ = ["PlotlyDashboard", "Snapshot"]
