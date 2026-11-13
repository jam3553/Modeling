"""
A punchier, social-media-shareable cut of plot_residuals.py's chart,
filtered down to 9 marquee 2026 Senate battlegrounds instead of all 33
states -- same underlying model and numbers, just styled to grab
attention (bigger type, direct value labels, a callout on the single
biggest upset) rather than for careful side-by-side state-by-state
reading. See plot_residuals.py for the full model/chart-design rationale
this borrows from.

Run from the project root: .venv/bin/python scripts/plot_key_states_clickbait.py
"""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_model import BASE_PREDICTORS, fit_baseline_model
from chart_watermark import add_watermark
from real_counties import load_real_swing_table

BASE = Path(__file__).resolve().parent.parent
PREDICTORS_WITH_HISTORY = BASE_PREDICTORS + ["dem_share_2024"]

# 9 marquee 2026 Senate races -- chosen for being widely-watched
# battlegrounds, NOT because they're the biggest residuals in
# state_residuals.csv (compare: the full ranking's actual extremes are
# Rhode Island/West Virginia at the top and Minnesota/Idaho at the
# bottom -- none of those are here).
KEY_STATES = ["TX", "GA", "MI", "OH", "NC", "NE", "IA", "MS", "ME", "NJ", "SC", "MN", "FL"]

BLUE = "#1f6fe0"
RED = "#e8342f"
BLACK = "#0b0b0b"
MUTED = "#898781"
AXIS = "#c3c2b7"
GRIDLINE = "#e1e0d9"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"


class _SplitColorHandler(HandlerBase):
    def __init__(self, color_left, color_right, **kwargs):
        self.color_left = color_left
        self.color_right = color_right
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        half = width / 2
        left = mpatches.Rectangle((-xdescent, -ydescent), half, height, facecolor=self.color_left, transform=trans)
        right = mpatches.Rectangle((-xdescent + half, -ydescent), half, height, facecolor=self.color_right, transform=trans)
        return [left, right]


if __name__ == "__main__":
    real_df = load_real_swing_table()
    results = fit_baseline_model(real_df, predictors=PREDICTORS_WITH_HISTORY)

    fit_df = real_df.loc[real_df["excluded_multiparty_split"] == 0].copy()
    fit_df["resid"] = results.resid
    fit_df["predicted_swing"] = results.fittedvalues

    by_state = (
        fit_df.groupby("state")
        .agg(
            mean_residual=("resid", "mean"),
            n_counties=("resid", "size"),
            mean_actual_swing=("dem_swing", "mean"),
            mean_predicted_swing=("predicted_swing", "mean"),
        )
    )
    plot_data = by_state.loc[KEY_STATES].sort_values("mean_residual", ascending=True)

    colors = [BLUE if v >= 0 else RED for v in plot_data["mean_residual"]]
    bar_height = 0.3
    offset = bar_height
    group_spacing = 3 * bar_height + 1.2 * bar_height
    y = [i * group_spacing for i in range(len(plot_data))]

    fig, ax = plt.subplots(figsize=(12.5, 10 * len(KEY_STATES) / 9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # Plotted on the D-R two-party MARGIN scale, not the Dem-coalition
    # SHARE scale -- margin = D - R = 2*D - 1, so a change in margin is
    # exactly 2x the same change in share. Same underlying numbers as
    # plot_residuals.py, just doubled, which is why "clickbait" wanted
    # it: it makes every gap look twice as dramatic without changing
    # what's actually true.
    MARGIN_SCALE = 200

    ax.barh([i + offset for i in y], plot_data["mean_actual_swing"] * MARGIN_SCALE, height=bar_height, color=BLACK, zorder=3)
    ax.barh([i for i in y], plot_data["mean_predicted_swing"] * MARGIN_SCALE, height=bar_height, color=MUTED, zorder=3)
    ax.barh([i - offset for i in y], plot_data["mean_residual"] * MARGIN_SCALE, height=bar_height, color=colors, zorder=3)

    # Direct value labels on the difference bars -- the headline number
    # in a clickbait graphic needs to be readable without cross-checking
    # a legend.
    for yi, v in zip(y, plot_data["mean_residual"]):
        val = v * MARGIN_SCALE
        label = f"{val:+.1f}"
        ax.text(
            val + (0.25 if val >= 0 else -0.25), yi - offset, label,
            ha="left" if val >= 0 else "right", va="center",
            fontsize=14, fontweight="bold", color=BLUE if val >= 0 else RED, zorder=6,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_data.index)
    ax.set_ylim(-group_spacing, (len(plot_data) - 1) * group_spacing + group_spacing)

    ax.axvline(0, color=AXIS, linewidth=1.5)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.tick_params(axis="y", colors=INK, labelsize=18)
    ax.tick_params(axis="x", colors=MUTED, labelsize=13)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_xlabel("Change in D-R two-party margin, 2026 Senate vs. 2024 president (points)", color=MUTED, fontsize=12)

    best = plot_data["mean_residual"].idxmax()
    worst = plot_data["mean_residual"].idxmin()
    ax.set_title(
        f"{len(KEY_STATES)} notable Senate races:\n{best} beat the model, {worst} got blown out\n"
        "actual vs. projected D-R margin swing, and the gap between them",
        color=INK, fontsize=20, fontweight="bold", loc="left", pad=20, linespacing=1.3,
    )

    difference_swatch = mpatches.Patch(label="Difference")
    legend_handles = [
        mpatches.Patch(color=BLACK, label="Actual swing"),
        mpatches.Patch(color=MUTED, label="Projected Swing"),
        difference_swatch,
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", bbox_to_anchor=(1.02, 0.0),
        frameon=False, fontsize=14, labelcolor=INK,
        handler_map={difference_swatch: _SplitColorHandler(BLUE, RED)},
    )

    caption = (
        "Model projects each county's change in Dem-coalition vote share (2026 Senate vs. 2024 president) from six\n"
        "demographic variables (% with a bachelor's degree+, % Black, % Hispanic, % White, median age, median household\n"
        f"income) plus 2024 Dem presidential vote share, then doubled to the D-R margin scale. These {len(KEY_STATES)} states\n"
        "were picked as notable 2026 Senate races, not because they're the biggest over/underperformers in the full 33-state model -- see state_residuals.png for that."
    )
    fig.text(0.01, 0.01, caption, color=MUTED, fontsize=9.5, ha="left", va="bottom")

    # Constant ~1in bottom margin for the caption regardless of fig
    # height, so it doesn't leave growing dead whitespace as the state
    # count (and so figure height) grows.
    bottom_frac = 1.0 / fig.get_size_inches()[1]
    plt.tight_layout(rect=(0, bottom_frac, 1, 1))
    add_watermark(fig)
    out_path = BASE / "key_states_clickbait.png"
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved plot -> {out_path}")
