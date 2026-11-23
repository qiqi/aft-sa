"""Colours for the Falkner-Skan family, the same in every figure.

The family is ordered by pressure gradient and has a meaningful zero at
Blasius, so it takes a two-sided map: cool for favorable, warm for adverse,
neutral at Blasius. The map is matplotlib's `coolwarm` with its brightness
lowered uniformly, so that its pale middle becomes a grey that is visible on a
white page; the ends darken with it.
"""
from matplotlib import colormaps

DARKEN = 0.72            # uniform brightness factor applied to coolwarm
BETA_FAVORABLE = 0.30    # the most favorable profile drawn, at the cool end
BETA_SEPARATION = 0.1988  # incipient separation, just short of the warm end


def beta_color(beta, separated=False):
    """Colour of the Falkner-Skan profile with wedge parameter beta. The
    reversed-flow (Stewartson) branch takes the warm end itself."""
    if separated:
        x = 1.0
    elif beta >= 0.0:
        x = 0.5 - 0.48*min(beta/BETA_FAVORABLE, 1.0)**0.6
    else:
        x = 0.5 + 0.42*min(-beta/BETA_SEPARATION, 1.0)**0.6
    r, g, b, _ = colormaps['coolwarm'](x)
    return (DARKEN*r, DARKEN*g, DARKEN*b)
