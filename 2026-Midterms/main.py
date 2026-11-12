"""
Runs the current pipeline end to end: data download (idempotent) -> toy
data generation -> fit the baseline model on both toy and real tables ->
diagnostics.

    .venv/bin/python main.py            # normal run, skips existing downloads
    .venv/bin/python main.py --force     # re-runs the download scripts too

This is the CURRENT (pre-refactor) codebase wired together as one script,
not the class-based ingest/model/analysis split in class_architecture.md --
that's a separate, larger follow-up.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
MODEL_TABLES_DIR = BASE / "data" / "model_tables"


def download_data(force: bool = False) -> None:
    """
    Runs the two automated ingestion scripts, skipping each one whose
    output already exists unless force=True.

    The 2026 Senate county CSVs (data/raw/{STATE}_senate_2026.csv) are NOT
    handled here -- they were gathered via agent-dispatched NBC scraping,
    not a standing script yet (see class_architecture.md Section 9: making
    ingest/senate_2026_nbc.py a real, deterministic script is future work).
    This function only covers the two pieces that already ARE deterministic
    scripts today.
    """
    census_out = BASE / "data" / "raw" / "census_demographics.csv"
    if force or not census_out.exists():
        print("Fetching Census demographics (33 states, ~1-2 min)...")
        subprocess.run([sys.executable, "fetch_census_demographics.py"], cwd=BASE, check=True)
    else:
        print(f"Skipping Census fetch -- {census_out.name} already exists.")

    pres2024_out = BASE / "data" / "raw" / "pres_2024_county.csv"
    if force or not pres2024_out.exists():
        print("Building 2024 presidential county data...")
        subprocess.run([sys.executable, "build_pres_2024_county.py"], cwd=BASE, check=True)
    else:
        print(f"Skipping 2024 presidential build -- {pres2024_out.name} already exists.")


def build_toy_data() -> pd.DataFrame:
    """
    Real demographics (all 32 non-AK Class-2 states, real 2026 vote totals
    where on file, a population-proportional proxy elsewhere) with a
    SIMULATED dem_swing from toy_data.TRUE_COEFS. See toy_data.py for the
    plasmode design rationale.
    """
    from real_counties import load_all_class2_states_county_frame
    from toy_data import simulate_demographics_for_real_counties

    frame = load_all_class2_states_county_frame()
    toy_df = simulate_demographics_for_real_counties(frame)

    MODEL_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_TABLES_DIR / "toy_counties.csv"
    toy_df.to_csv(out_path, index=False)
    print(f"Toy table: {len(toy_df):,} counties across {toy_df['state'].nunique()} states -> {out_path}")
    return toy_df


def build_real_data() -> pd.DataFrame:
    """
    Real demographics AND a real, observed dem_swing (2026 Senate minus
    2024 presidential, per county) -- the first point in this project
    where the outcome itself isn't simulated. Covers the states with real
    2026 results on file, minus Maine/Massachusetts/New Hampshire/Rhode
    Island (town-level source data, can't merge against county-level
    Census yet -- see class_architecture.md Section 10).
    """
    from real_counties import load_real_swing_table

    real_df = load_real_swing_table()

    MODEL_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_TABLES_DIR / "real_counties.csv"
    real_df.to_csv(out_path, index=False)
    print(f"Real table: {len(real_df):,} counties across {real_df['state'].nunique()} states -> {out_path}")
    return real_df


def fit_models(toy_df: pd.DataFrame, real_df: pd.DataFrame):
    """Fits the identical baseline spec (demographics only, no state FE,
    clustered SEs) on both tables -- the point being that the fitting code
    doesn't know or care which table is real."""
    from baseline_model import fit_baseline_model

    toy_results = fit_baseline_model(toy_df)
    real_results = fit_baseline_model(real_df)
    return toy_results, real_results


def run_diagnostics(toy_df: pd.DataFrame, real_df: pd.DataFrame, toy_results, real_results) -> dict:
    from baseline_model import compute_vif, test_single_interaction

    diagnostics = {}
    for name, df in [("toy", toy_df), ("real", real_df)]:
        diagnostics[name] = {
            "vif": compute_vif(df),
            "interaction": test_single_interaction(df, "pct_college", "pct_black"),
        }
    return diagnostics


def _print_report(toy_df, real_df, toy_results, real_results, diagnostics) -> None:
    from toy_data import TRUE_COEFS

    pd.set_option("display.width", 120)
    true_demo = pd.Series(TRUE_COEFS).drop(["intercept", "college_x_black"])

    print("\n" + "=" * 78)
    print("TOY MODEL (simulated outcome -- validates the fitting code, not real findings)")
    print("=" * 78)
    print(f"n = {len(toy_df):,} counties, {toy_df['state'].nunique()} states, R^2 = {toy_results.rsquared:.4f}")
    print(pd.DataFrame({
        "true": true_demo,
        "recovered": toy_results.params.reindex(true_demo.index),
        "std_err": toy_results.bse.reindex(true_demo.index),
    }))
    print("\nVIF:")
    print(diagnostics["toy"]["vif"])
    interaction = diagnostics["toy"]["interaction"]
    print(f"\ncollege x black interaction: coef={interaction['coefficient']:.4f} "
          f"(true={TRUE_COEFS['college_x_black']}), p={interaction['p_value']:.4f}")

    print("\n" + "=" * 78)
    print("REAL MODEL (observed 2026 Senate vs. 2024 presidential swing -- actual findings)")
    print("=" * 78)
    print(f"n = {len(real_df):,} counties, {real_df['state'].nunique()} states, R^2 = {real_results.rsquared:.4f}")
    print(pd.DataFrame({
        "coef": real_results.params.reindex(true_demo.index),
        "std_err": real_results.bse.reindex(true_demo.index),
        "p_value": real_results.pvalues.reindex(true_demo.index),
    }))
    print("\nVIF:")
    print(diagnostics["real"]["vif"])
    interaction = diagnostics["real"]["interaction"]
    print(f"\ncollege x black interaction: coef={interaction['coefficient']:.4f}, p={interaction['p_value']:.4f}")


def main(force_download: bool = False) -> dict:
    download_data(force=force_download)
    toy_df = build_toy_data()
    real_df = build_real_data()
    toy_results, real_results = fit_models(toy_df, real_df)
    diagnostics = run_diagnostics(toy_df, real_df, toy_results, real_results)
    _print_report(toy_df, real_df, toy_results, real_results, diagnostics)

    return {
        "toy_df": toy_df,
        "real_df": real_df,
        "toy_results": toy_results,
        "real_results": real_results,
        "diagnostics": diagnostics,
    }


if __name__ == "__main__":
    main(force_download="--force" in sys.argv)
