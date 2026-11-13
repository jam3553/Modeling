"""
A 95% PREDICTION interval for an individual county's dem_swing -- not to
be confused with the CONFIDENCE interval on the coefficients/mean
response (see compute_confidence_intervals.py). A confidence interval
asks "how sure are we about the average relationship"; a prediction
interval asks "how far off could one new county's actual value be from
the model's point prediction," which also has to include the residual
noise for that single observation, not just parameter uncertainty.

IMPORTANT FINDING, confirmed empirically rather than assumed: statsmodels'
built-in results.get_prediction().summary_frame()'s "obs_ci" columns (its
own attempt at a prediction interval) come back nonsensical for this
model -- roughly 1,336 percentage points wide, for a variable bounded
around +/-35 points. It does not correctly combine WLS weighting with
cluster-robust covariance for that calculation. The mean-prediction CI
columns are fine; only "obs_ci" is broken. This script computes the
individual-observation interval by hand instead, and verifies it against
empirical coverage before trusting it.

Run from the project root: .venv/bin/python scripts/compute_prediction_intervals.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_model import fit_baseline_model
from real_counties import load_real_swing_table

if __name__ == "__main__":
    pd.set_option("display.width", 120)

    real_df = load_real_swing_table()
    fit_df = real_df.loc[real_df["excluded_multiparty_split"] == 0].copy()
    results = fit_baseline_model(real_df)
    pred = results.get_prediction(fit_df)

    # THEORY: Var(new observation) = Var(mean prediction) + Var(residual
    # for this observation). pred.var_pred_mean already correctly uses the
    # cluster-robust covariance (that part of get_prediction is fine). For
    # WLS, Var(residual_i) = scale / weight_i -- the model's own
    # homoskedasticity-after-weighting assumption, using the SAME scale
    # estimate as the fit (not whatever get_prediction's broken "obs"
    # column used internally).
    var_mean = pred.var_pred_mean
    weights = fit_df["swing_weight"].values
    var_resid = results.scale / weights
    var_obs = var_mean + var_resid
    se_obs = np.sqrt(var_obs)

    z = stats.norm.ppf(0.975)
    fit_df["pred_mean"] = pred.predicted_mean
    fit_df["pi_lower"] = pred.predicted_mean - z * se_obs
    fit_df["pi_upper"] = pred.predicted_mean + z * se_obs
    fit_df["pi_width"] = fit_df["pi_upper"] - fit_df["pi_lower"]

    print("results.scale:", results.scale)
    print("\nTypical 95% prediction-interval width (percentage points):")
    print((fit_df["pi_width"] * 100).describe())

    print("\nSample rows (actual dem_swing vs predicted + 95% PI, in points):")
    sample = fit_df[["state", "dem_swing", "pred_mean", "pi_lower", "pi_upper"]].head(5).copy()
    for c in ["dem_swing", "pred_mean", "pi_lower", "pi_upper"]:
        sample[c] = (sample[c] * 100).round(1)
    print(sample)

    # Coverage check: what fraction of ACTUAL observed swings fall inside
    # their own 95% PI? Should land close to 95% if the hand-computed
    # interval is correctly calibrated -- this is the actual verification,
    # not just "the formula looks right."
    inside = (fit_df["dem_swing"] >= fit_df["pi_lower"]) & (fit_df["dem_swing"] <= fit_df["pi_upper"])
    print(f"\nEmpirical coverage (should be ~95% if correct): {inside.mean():.1%}")
