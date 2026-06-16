"""Shared plotting colors for experiment figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


TAB10_COLORS = tuple(plt.get_cmap("tab10").colors)


def tab10_colors(count: int):
    """Return `count` colors from the shared tab10 cycle."""
    return [TAB10_COLORS[index % len(TAB10_COLORS)] for index in range(count)]
