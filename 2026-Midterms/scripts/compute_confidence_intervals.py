"""
95% confidence intervals for the real baseline model's coefficients
(cluster-robust, clustered by state). Confirms the project's own
documented caveat in baseline_model.py: with ~27 clusters, this is right
at the edge of "enough clusters to trust," not deep into safe territory.

Run from the project root: .venv/bin/python scripts/compute_confidence_intervals.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_model import fit_baseline_model, BASE_PREDICTORS
from real_counties import load_real_swing_table

if __name__ == "__main__":
    pd.set_option("display.width", 120)

    real_df = load_real_swing_table()
    results = fit_baseline_model(real_df)  # cluster_se=True by default, clustered by state

    ci = results.conf_int(alpha=0.05)
    ci.columns = ["ci_lower_95", "ci_upper_95"]

    table = pd.DataFrame({
        "coef": results.params.reindex(BASE_PREDICTORS),
        "std_err": results.bse.reindex(BASE_PREDICTORS),
        "ci_lower_95": ci["ci_lower_95"].reindex(BASE_PREDICTORS),
        "ci_upper_95": ci["ci_upper_95"].reindex(BASE_PREDICTORS),
        "p_value": results.pvalues.reindex(BASE_PREDICTORS),
    })
    print(table)

    n_clusters = real_df.loc[real_df["excluded_multiparty_split"] == 0, "state"].nunique()
    print(f"\nn_clusters (states) = {n_clusters}")
