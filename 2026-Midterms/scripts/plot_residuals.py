"""
Answers the README's "which states had the highest residual" analysis
question: fits the real baseline model (demographics + 2024 Dem vote
share -- "voting history" -- as predictors of dem_swing, the change in
the Dem-coalition's own two-party vote share from 2024 president to 2026
Senate), computes each state's mean actual swing, mean model-projected
swing, and the residual between them, and outputs a sorted CSV plus a
chart.

Chart design: three separate horizontal bars per state, all on the
model's own outcome scale (swing, not vote-share level) --
  1. Actual swing -- solid black.
  2. Model-projected swing -- black/white hatched, same length scale as
     (1), so the two are visually paired as "two estimates of the same
     quantity" rather than looking like a share-level comparison.
  3. The residual (actual - projected) -- blue if positive, red if
     negative -- given its own bar rather than only implied by the gap
     between (1) and (2), since at swing magnitudes (a few points) a
     implied-gap reading between two long bars is hard to see precisely.
This replaces an earlier version of this chart that displayed vote-share
LEVELS (dem_share_2026 projected vs. actual) instead of swings -- moved
back to swing because the model's actual fitted outcome is dem_swing,
and showing levels overstated how much of the vote share the model
explains (most of a vote share level is persistence from 2024, not
anything in this model).

Run from the project root: .venv/bin/python scripts/plot_residuals.py
"""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.legend_handler import HandlerBase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_model import BASE_PREDICTORS, fit_baseline_model
from chart_watermark import add_watermark
from real_counties import load_real_swing_table

BASE = Path(__file__).resolve().parent.parent

# Demographics + 2024 Dem vote share ("voting history"), no incumbency --
# tested separately and found statistically indistinguishable from zero
# even controlling for just demographics (see conversation history), so
# dropped rather than carried as dead weight.
PREDICTORS_WITH_HISTORY = BASE_PREDICTORS + ["dem_share_2024"]

# Diverging pair from the dataviz skill's reference palette (light mode):
# blue <-> red poles -- chosen deliberately since it also matches the
# political-direction convention of this data.
BLUE = "#2a78d6"   # actual swing more Democratic-coalition than projected
RED = "#e34948"    # actual swing less Democratic-coalition than projected
BLACK = "#0b0b0b"  # actual swing
MUTED = "#898781"
AXIS = "#c3c2b7"
GRIDLINE = "#e1e0d9"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"


class _SplitColorHandler(HandlerBase):
    """Draws a legend swatch as two side-by-side color blocks, for a
    single "Difference" entry standing in for the blue/red bars."""

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

    # Montana is excluded from the fit itself (excluded_multiparty_split=1
    # -- the Bankhead/Bodnar anti-R vote split means dem_swing there isn't
    # comparable to a clean two-candidate race, per state_effect_options.md
    # and the original model-spec decision, with an empirical robustness
    # check in scripts/compare_montana_handling.py). Drop it here too
    # rather than let pandas index alignment silently produce a NaN row.
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
        .sort_values("mean_residual", ascending=False)
    )

    csv_path = BASE / "state_residuals.csv"
    by_state.to_csv(csv_path)
    print(f"Saved {len(by_state)} states -> {csv_path}")
    print(by_state)

    # --- Plot ---------------------------------------------------------
    # Sized and weighted for mobile: thicker bars (bar_height closer to
    # the full row spacing of 1.0, vs. the original 0.26) and larger
    # fonts throughout, since this is meant to be legible at phone-screen
    # size, not just on a desktop monitor.
    plot_data = by_state.sort_values("mean_residual", ascending=True)  # ascending so biggest is at top when plotted horizontally
    colors = [BLUE if v >= 0 else RED for v in plot_data["mean_residual"]]
    bar_height = 0.3
    offset = bar_height  # bars within a cluster touch, no gap between them
    # Group spacing > 3*bar_height opens up a visible gap between each
    # state's 3-bar cluster and the next -- sized here to ~1.2 bar-widths.
    group_spacing = 3 * bar_height + 1.2 * bar_height
    y = [i * group_spacing for i in range(len(plot_data))]

    fig, ax = plt.subplots(figsize=(11, 17 * group_spacing), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.barh(
        [i + offset for i in y], plot_data["mean_actual_swing"] * 100,
        height=bar_height, color=BLACK, zorder=3,
    )
    ax.barh(
        [i for i in y], plot_data["mean_predicted_swing"] * 100,
        height=bar_height, color=MUTED, zorder=3,
    )
    ax.barh(
        [i - offset for i in y], plot_data["mean_residual"] * 100,
        height=bar_height, color=colors, zorder=3,
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

    ax.tick_params(axis="y", colors=INK, labelsize=14)
    ax.tick_params(axis="x", colors=MUTED, labelsize=13)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_xlabel("Change in Dem-coalition vote share, 2026 Senate vs. 2024 president (points)", color=MUTED, fontsize=13)
    ax.set_title(
        "2026 Senate races:\nactual vs. projected swing,\nand the gap between them",
        color=INK, fontsize=24, fontweight="bold", loc="left", pad=22, linespacing=1.35,
    )

    # "Difference" gets ONE legend entry, not two -- a single swatch
    # split blue/red reads as "sign of the gap" on its own without
    # needing separate "more Dem"/"less Dem" rows, and keeps the legend
    # short enough to sit back inside the axes (lower right) instead of
    # needing its own strip below the plot.
    difference_swatch = mpatches.Patch(label="Difference")
    legend_handles = [
        mpatches.Patch(color=BLACK, label="Actual swing"),
        mpatches.Patch(color=MUTED, label="Projected Swing"),
        difference_swatch,
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", bbox_to_anchor=(1.02, 0.0),
        frameon=False, fontsize=15, labelcolor=INK,
        handler_map={difference_swatch: _SplitColorHandler(BLUE, RED)},
    )

    caption = (
        "Model projects each county's change in Dem-coalition vote share (2026 Senate vs. 2024 president) from six\n"
        "demographic variables (% with a bachelor's degree+, % Black, % Hispanic, % White, median age, median household\n"
        f"income) plus 2024 Dem presidential vote share, as its voting-history baseline. n={len(fit_df)} counties,\n"
        f"{len(by_state)} states. Montana excluded: anti-R vote split between a Democrat and a competing independent."
    )
    fig.text(0.01, 0.01, caption, color=MUTED, fontsize=10.5, ha="left", va="bottom")

    # Reserve a constant ~1.3in of bottom margin for the caption rather
    # than a fixed fraction -- a fixed fraction would leave growing dead
    # whitespace as fig height grows with group_spacing.
    bottom_frac = 1.3 / fig.get_size_inches()[1]
    plt.tight_layout(rect=(0, bottom_frac, 1, 1))
    add_watermark(fig)
    out_path = BASE / "state_residuals.png"
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved plot -> {out_path}")
