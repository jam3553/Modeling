"""
Partial-residual (component-plus-residual) plots for EVERY predictor in the
baseline model, to check whether each one's assumed-linear relationship with
dem_swing actually looks linear in the real data -- generalizes the one-off
pct_hispanic check (see conversation history for why that check started:
pct_hispanic's lowess smooth bent well away from the model's straight line
in the sparse, high-Hispanic-share tail).

THEORY -- what a partial residual plot shows:
For predictor x_j with fitted coefficient beta_j, the partial
(component-plus-residual) residual is:
    partial_resid_i = resid_i + beta_j * x_j,i
where resid_i is the model's actual residual (y_i - fitted_i, which already
nets out every OTHER predictor's effect). Adding beta_j * x_j,i back in
isolates "what's left in y after accounting for every predictor except
x_j's own effect" -- so plotting partial_resid against x_j is a clean look
at the SHAPE of x_j's relationship with the outcome, net of all other
covariates. If the model's linear assumption for x_j is right, this should
look like a straight line with slope beta_j, plus noise. A lowess smooth
that systematically bends away from that line is evidence the linear term
is misspecified for that predictor.

CAVEAT that applies to all of these, not just pct_hispanic: every
predictor here has its own density -- some (median_age, median_income_k)
are fairly evenly spread, others (pct_black, pct_hispanic) are
right-skewed with a long sparse tail. The county-count histogram under
each panel exists so a bend in the lowess line can be checked against how
much data actually backs it before reading anything into it.

Output: one PNG per predictor in partial_residuals/<predictor>.png.

Run from the project root: .venv/bin/python scripts/plot_partial_residuals.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_model import BASE_PREDICTORS, fit_baseline_model
from chart_watermark import add_provenance_stamp
from real_counties import load_real_swing_table

BASE = Path(__file__).resolve().parent.parent
PREDICTORS_WITH_HISTORY = BASE_PREDICTORS + ["dem_share_2024"]

MUTED = "#898781"
INK = "#0b0b0b"
BLUE = "#2a78d6"
AXIS = "#c3c2b7"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# Display label and x-axis scale factor for each predictor -- the pct_*
# shares are stored as [0, 1] fractions and read better at 0-100 scale;
# median_age/median_income_k are already in their natural units (years,
# $1,000s) and dem_share_2024 is a [0, 1] share like the pct_* variables.
DISPLAY = {
    "pct_college": ("% with a bachelor's degree+ (county)", 100),
    "pct_black": ("% Black (county)", 100),
    "pct_hispanic": ("% Hispanic (county)", 100),
    "pct_white": ("% White (county)", 100),
    "median_age": ("Median age (county, years)", 1),
    "median_income_k": ("Median household income (county, $1,000s)", 1),
    "dem_share_2024": ("2024 Dem presidential two-party vote share (county)", 100),
}


def plot_partial_residual(fit_df, results, predictor: str, out_dir: Path) -> Path:
    label, x_scale = DISPLAY[predictor]
    beta = results.params[predictor]

    x = fit_df[predictor]
    partial_resid = fit_df["resid"] + beta * x

    smoothed = lowess(partial_resid, x, frac=0.3, return_sorted=True)

    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 9), facecolor=SURFACE,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08}, sharex=True,
    )
    ax.set_facecolor(SURFACE)
    ax_hist.set_facecolor(SURFACE)

    # Marker size scaled by swing_weight (same weight the model itself
    # used) so visually-dominant points in the scatter are the same ones
    # that dominated the fit.
    marker_size = 8 + 50 * (fit_df["swing_weight"] / fit_df["swing_weight"].max())
    ax.scatter(
        x * x_scale, partial_resid * 100,
        s=marker_size, color=MUTED, alpha=0.35, edgecolors="none", zorder=2,
    )

    x_line = np.array([x.min(), x.max()])
    # The CPR line passes through (mean(x), mean(partial_resid)) with
    # slope beta -- i.e. it's just the model's own linear term, re-anchored
    # to sit among the partial residuals it was computed from.
    intercept_term = (partial_resid - beta * x).mean()
    ax.plot(
        x_line * x_scale, (beta * x_line + intercept_term) * 100,
        color=INK, linewidth=2.2, label="Model's linear fit", zorder=4,
    )
    ax.plot(
        smoothed[:, 0] * x_scale, smoothed[:, 1] * 100,
        color=BLUE, linewidth=2.6, label="Lowess smooth (no shape assumed)", zorder=5,
    )

    ax.axhline(0, color=AXIS, linewidth=1)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.set_ylabel("Partial residual on dem_swing (points),\nnet of every other predictor", color=MUTED, fontsize=10.5)
    ax.set_title(
        f"Is {predictor}'s effect on Dem swing actually linear?",
        fontsize=17, fontweight="bold", loc="left", color=INK, pad=14,
    )
    ax.legend(frameon=False, loc="best", fontsize=11, labelcolor=INK)

    bin_edges = np.linspace(x.min(), x.max(), 25) * x_scale
    ax_hist.hist(x * x_scale, bins=bin_edges, color=MUTED, edgecolor=SURFACE, zorder=2)
    ax_hist.set_ylabel("# counties", color=MUTED, fontsize=9.5)
    ax_hist.set_xlabel(label, color=MUTED, fontsize=11)
    ax_hist.tick_params(colors=MUTED, labelsize=9.5)
    for spine in ["top", "right"]:
        ax_hist.spines[spine].set_visible(False)
    ax_hist.xaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax_hist.set_axisbelow(True)

    caption = (
        f"Partial-residual plot from the real-data baseline model (n={len(fit_df)} counties, "
        f"fitted coefficient on {predictor} = {beta:.4f}).\n"
        "If the black line (model's linear assumption) and blue line (unconstrained lowess) diverge, "
        "that's evidence the linear term is misspecified --\n"
        "but check the county-count panel below before trusting any divergence where counties are sparse."
    )
    fig.text(0.01, 0.01, caption, color=MUTED, fontsize=9.5, ha="left", va="bottom")

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    add_provenance_stamp(fig)
    out_path = out_dir / f"{predictor}.png"
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    real_df = load_real_swing_table()
    results = fit_baseline_model(real_df, predictors=PREDICTORS_WITH_HISTORY)

    fit_df = real_df.loc[real_df["excluded_multiparty_split"] == 0].copy()
    fit_df["resid"] = results.resid

    out_dir = BASE / "partial_residuals"
    out_dir.mkdir(exist_ok=True)

    for predictor in PREDICTORS_WITH_HISTORY:
        out_path = plot_partial_residual(fit_df, results, predictor, out_dir)
        print(f"Saved {predictor} -> {out_path}")
