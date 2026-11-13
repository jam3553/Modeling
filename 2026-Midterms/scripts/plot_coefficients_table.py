"""
Coefficient table (point estimates, cluster-robust standard errors, 95%
confidence intervals, p-values) for the baseline model used throughout
this project's charts (plot_residuals.py, plot_partial_residuals.py,
plot_key_states_clickbait.py): demographics + 2024 Dem presidential vote
share, no incumbency, no state fixed effects -- see baseline_model.py for
the full specification rationale and compute_confidence_intervals.py for
the earlier, printed-not-plotted version of this same computation (that
one used BASE_PREDICTORS only, without dem_share_2024 -- this version
matches what the actual chart scripts fit).

All coefficients are reported in POINTS OF DEM_SWING (the outcome,
converted from a [0, 1] share to percentage points by x100). For the
share predictors (pct_college, pct_black, pct_hispanic, pct_white,
dem_share_2024), that's per a full 0->100 percentage-point change in
that share -- a big, often unrealistic hypothetical (no real county goes
from 0% to 100% Black, say), but it's the standard way a raw regression
coefficient on a [0, 1] share is reported; divide by 10 for a "per 10
percentage point" read if that's more intuitive. median_age and
median_income_k are already in their natural units (years, $1,000s), so
their coefficient is just "points of dem_swing per 1 year" / "per
$1,000" with no such caveat.

Run from the project root: .venv/bin/python scripts/plot_coefficients_table.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_model import BASE_PREDICTORS, fit_baseline_model
from real_counties import load_real_swing_table

BASE = Path(__file__).resolve().parent.parent
PREDICTORS_WITH_HISTORY = BASE_PREDICTORS + ["dem_share_2024"]

INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#e34948"
HEADER_BG = "#eceae2"
ROW_ALT_BG = "#f5f4ef"

DISPLAY = {
    "Intercept": ("Intercept", ""),
    "pct_college": ("% with bachelor's degree+", "per 100pp share"),
    "pct_black": ("% Black", "per 100pp share"),
    "pct_hispanic": ("% Hispanic", "per 100pp share"),
    "pct_white": ("% White", "per 100pp share"),
    "median_age": ("Median age", "per year"),
    "median_income_k": ("Median household income", "per $1,000"),
    "dem_share_2024": ("2024 Dem presidential share", "per 100pp share"),
}


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


if __name__ == "__main__":
    real_df = load_real_swing_table()
    results = fit_baseline_model(real_df, predictors=PREDICTORS_WITH_HISTORY)
    fit_df = real_df.loc[real_df["excluded_multiparty_split"] == 0]

    ci = results.conf_int(alpha=0.05)
    ci.columns = ["lo", "hi"]

    order = ["Intercept"] + PREDICTORS_WITH_HISTORY
    rows = []
    for term in order:
        name, unit = DISPLAY[term]
        coef = results.params[term] * 100
        se = results.bse[term] * 100
        lo = ci.loc[term, "lo"] * 100
        hi = ci.loc[term, "hi"] * 100
        p = results.pvalues[term]
        rows.append([name, unit, f"{coef:+.2f}", f"{se:.2f}", f"[{lo:+.2f}, {hi:+.2f}]", f"{p:.3f}{stars(p)}"])

    col_labels = ["Predictor", "Unit", "Coefficient", "Std. Error", "95% CI", "p-value"]
    col_widths = [0.26, 0.14, 0.14, 0.14, 0.17, 0.15]

    # Laid out in fixed, explicitly-budgeted axes-fraction bands (margin /
    # title / gap / table / gap / caption / margin) rather than
    # plt.tight_layout -- a matplotlib Table's own rendered height is NOT
    # something tight_layout accounts for, and text height estimates for
    # the title/caption need their own real inch budget (not just
    # "however much space is left") or they overlap the table.
    margin_in = 0.3
    title_in = 0.9
    gap_in = 0.25
    row_in = 0.45
    table_in = row_in * (len(rows) + 1)
    caption_in = 1.0
    fig_height = margin_in + title_in + gap_in + table_in + gap_in + caption_in + margin_in

    fig, ax = plt.subplots(figsize=(12, fig_height), facecolor=SURFACE)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.0, 1 - margin_in / fig_height, "Baseline model coefficients\ndem_swing ~ demographics + 2024 Dem presidential share",
        fontsize=15, fontweight="bold", color=INK, ha="left", va="top", linespacing=1.4,
    )

    table_top = 1 - (margin_in + title_in + gap_in) / fig_height
    table_bbox = [0.0, table_top - table_in / fig_height, 1.0, table_in / fig_height]
    table = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths, cellLoc="left", bbox=table_bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(fontweight="bold", color=INK)
        cell.set_edgecolor(GRIDLINE)
        cell.PAD = 0.02

    for i, row in enumerate(rows, start=1):
        bg = ROW_ALT_BG if i % 2 == 0 else SURFACE
        coef_val = float(row[2])
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor(GRIDLINE)
            cell.set_text_props(color=INK)
            cell.PAD = 0.02
            if j == 2:
                cell.set_text_props(color=BLUE if coef_val >= 0 else RED, fontweight="bold")

    n_clusters = fit_df["state"].nunique()
    caption = (
        f"WLS, weighted by swing_weight; standard errors cluster-robust by state (n={len(fit_df)} counties, "
        f"{n_clusters} state clusters). R-squared = {results.rsquared:.3f}.\n"
        "Coefficients are in points of dem_swing (2026 Senate minus 2024 president two-party Dem share). "
        "* p<0.05, ** p<0.01, *** p<0.001. With this few clusters, treat\n"
        "the clustered CIs as reasonably but not fully trustworthy -- see baseline_model.py's cluster_se docstring."
    )
    ax.text(0.0, margin_in / fig_height, caption, color=MUTED, fontsize=9, ha="left", va="bottom", linespacing=1.6)

    out_path = BASE / "coefficients_table.png"
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved table -> {out_path}")
