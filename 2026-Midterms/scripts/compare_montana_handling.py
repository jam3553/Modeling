"""
Robustness check on the Montana exclusion decision (see
state_effect_options.md and pipeline_walkthrough.md for the original
design rationale: Alani Bankhead/D and Seth Bodnar/independent split the
anti-Republican vote, so dem_coalition_share doesn't mean the same thing
there as in a clean two-candidate race).

Tries two ways of forcing Montana into the model instead of excluding it,
and compares R^2 and Montana's own residual against the baseline
(excluded) model:

1. "Simple D-vs-R": dem_votes = Bankhead only, rep_votes = the Republican
   only, dropping Bodnar the same way any other third-party candidate
   gets dropped (this is already how load_real_swing_table() computes
   Montana's numbers -- this variant just stops excluding the row from
   fitting).
2. "Incumbent + independent combined": dem_votes = Bankhead only,
   rep_votes = the Republican + Bodnar combined -- i.e. treat the race as
   Bankhead vs. everyone else, since Bodnar was backed by some national
   Democratic figures specifically to pull voters FROM Bankhead, so his
   votes don't obviously belong with either side.

Finding: both degrade the fit relative to exclusion (R^2 0.432 -> 0.384
and -> 0.316 respectively), and variant 2 puts Montana's own residual at
roughly 5x the magnitude of any other state's in the ranking -- empirical
support for the exclusion, not just the theoretical argument.

Run from the project root: .venv/bin/python scripts/compare_montana_handling.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline_model import fit_baseline_model, compute_swing_weight
from county_utils import county_key
from real_counties import load_real_swing_table

BASE = Path(__file__).resolve().parent.parent
TRUE_DEMO_COLS = ["pct_college", "pct_black", "pct_hispanic", "pct_white", "median_age", "median_income_k"]


def build_mt_variant_simple_d_vs_r(real_df: pd.DataFrame) -> pd.DataFrame:
    """Just stop excluding Montana -- its dem_swing/swing_weight were
    already computed with dem_votes=Bankhead, rep_votes=Republican only."""
    variant = real_df.copy()
    variant.loc[variant["state"] == "MT", "excluded_multiparty_split"] = 0
    return variant


def build_mt_variant_incumbent_plus_independent(real_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild Montana's two-party split as Bankhead (D) vs. Republican +
    Bodnar (independent) combined, then splice it into the real table."""
    mt_raw = pd.read_csv(BASE / "data" / "raw" / "MT_senate_2026.csv")
    mt_raw["county_key"] = county_key(mt_raw["county_name"])
    mt_raw["bucket"] = mt_raw["candidate_name"].map({
        "Alani Bankhead": "dem_votes",
        "Kurt Alme": "rep_votes",
        "Seth Bodnar": "rep_votes",
    }).fillna("other")

    mt_combined = (
        mt_raw[mt_raw["bucket"] != "other"]
        .groupby(["county_key", "bucket"])["votes"].sum()
        .unstack("bucket")
        .reset_index()
    )
    name_map = mt_raw.drop_duplicates("county_key").set_index("county_key")["county_name"]
    mt_combined["county_name"] = mt_combined["county_key"].map(name_map)
    mt_combined["state"] = "MT"
    mt_combined["total_two_party_votes"] = mt_combined["dem_votes"] + mt_combined["rep_votes"]
    mt_combined["dem_share_2026"] = mt_combined["dem_votes"] / mt_combined["total_two_party_votes"]

    census = pd.read_csv(BASE / "data" / "raw" / "census_demographics.csv")
    pres2024 = pd.read_csv(BASE / "data" / "raw" / "pres_2024_county.csv")
    religiosity = pd.read_csv(BASE / "data" / "raw" / "state_religiosity.csv").rename(
        columns={"pct_highly_religious": "religiosity"}
    )

    mt_combined = mt_combined.merge(census.drop(columns=["county_name"]), on=["state", "county_key"], how="left")
    mt_combined = mt_combined.merge(
        pres2024[["state", "county_key", "total_two_party_votes_2024", "dem_share_2024"]],
        on=["state", "county_key"], how="left",
    )
    mt_combined = mt_combined.merge(religiosity, on="state", how="left")
    mt_combined["dem_swing"] = mt_combined["dem_share_2026"] - mt_combined["dem_share_2024"]
    mt_combined["swing_weight"] = compute_swing_weight(
        mt_combined["total_two_party_votes_2024"], mt_combined["total_two_party_votes"]
    )
    mt_combined["is_coalition_independent"] = 0
    mt_combined["excluded_multiparty_split"] = 0

    cols = real_df.columns
    return pd.concat([real_df[real_df["state"] != "MT"], mt_combined[cols]], ignore_index=True)


if __name__ == "__main__":
    pd.set_option("display.width", 120)

    real_df = load_real_swing_table()
    results_excl = fit_baseline_model(real_df)  # baseline: MT excluded

    variant_1 = build_mt_variant_simple_d_vs_r(real_df)
    results_variant_1 = fit_baseline_model(variant_1)

    variant_2 = build_mt_variant_incumbent_plus_independent(real_df)
    results_variant_2 = fit_baseline_model(variant_2)

    print("=== Coefficients across the three treatments of Montana ===")
    print(pd.DataFrame({
        "excluded (baseline)": results_excl.params.reindex(TRUE_DEMO_COLS),
        "simple D-vs-R": results_variant_1.params.reindex(TRUE_DEMO_COLS),
        "incumbent+independent": results_variant_2.params.reindex(TRUE_DEMO_COLS),
    }))

    print(f"\nR^2 excluded:              {results_excl.rsquared:.4f}")
    print(f"R^2 simple D-vs-R:         {results_variant_1.rsquared:.4f}")
    print(f"R^2 incumbent+independent: {results_variant_2.rsquared:.4f}")

    fit_df_2 = variant_2.loc[variant_2["excluded_multiparty_split"] == 0].copy()
    fit_df_2["resid"] = results_variant_2.resid
    mt_resid = fit_df_2.loc[fit_df_2["state"] == "MT", "resid"].mean()
    print(f"\nMontana's own mean residual under 'incumbent+independent': {mt_resid:.4f}")
    print("(compare to every other state's residual being within roughly +/-0.04 -- see state_residuals.csv)")
