"""
Shared stamps for this project's output charts/tables, so every script
doesn't redefine the same few lines -- see scripts/plot_*.py for callers.
Both are figure-level (fig.text, not ax.text) specifically so they land
in the actual corners of the saved PNG regardless of where any individual
script's axes happen to sit within the figure.
"""

import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

HANDLE = "@RazzberryYams"
REPO_ROOT = Path(__file__).resolve().parent


def add_watermark(fig: plt.Figure, color: str = "#898781") -> None:
    fig.text(0.99, 0.985, HANDLE, color=color, fontsize=10, alpha=0.7, ha="right", va="top", zorder=10)


def _git_stamp() -> str:
    """Short commit hash HEAD was at when the image was generated, with
    a '+dirty' suffix if the working tree had uncommitted changes at that
    moment -- the point of this stamp (per its origin: James's own
    commit-then-regenerate workflow) is to catch exactly that case, where
    an image doesn't actually match any committed version of the code."""
    try:
        commit_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no git info"
    return f"{commit_hash}+dirty" if dirty else commit_hash


def add_provenance_stamp(fig: plt.Figure, color: str = "#898781") -> None:
    stamp = f"{_git_stamp()} · generated {datetime.now():%Y-%m-%d %H:%M}"
    fig.text(0.99, 0.015, stamp, color=color, fontsize=9, alpha=0.7, ha="right", va="bottom", zorder=10)
