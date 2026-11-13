"""
Shared watermark stamp for this project's output charts/tables, so every
script doesn't redefine the same three lines -- see scripts/plot_*.py for
callers. Figure-level (fig.text, not ax.text) specifically so it lands in
the actual top-right corner of the saved PNG regardless of where any
individual script's axes happen to sit within the figure.
"""

import matplotlib.pyplot as plt

HANDLE = "@RazzberryYams"


def add_watermark(fig: plt.Figure, color: str = "#898781") -> None:
    fig.text(0.99, 0.985, HANDLE, color=color, fontsize=10, alpha=0.7, ha="right", va="top", zorder=10)
