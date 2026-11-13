"""
Loads real 2026 Senate county-level results (gathered state by state into
data/raw/<STATE>_senate_2026.csv) and reduces them to the "county skeleton"
the toy model needs: which counties actually exist, and how many two-party
votes were actually cast in each.

This is deliberately a SMALL module. The point isn't to build the real
model yet -- real ACS demographics (pct_college, pct_black, etc.) aren't
merged in here. The point is to stop FAKING the things we don't need to
fake: county identities and vote-count scale are facts, not something a toy
model should be drawing from a lognormal. See toy_data.py's
simulate_demographics_for_real_counties() for how this skeleton gets used.

Each raw file is expected to have columns: county_name, candidate_name,
party, votes -- exactly what the research agents were told to produce.
State is inferred from the filename (<STATE>_senate_2026.csv), not stored
in the file itself.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from candidate import Candidate, Party

DATA_DIR = Path(__file__).parent / "data" / "raw"

# THEORY note tying back to the model spec discussion: candidate.py's
# Candidate.effective_party is where the "Osborne flag" logic from the
# model spec actually has to live for real data -- it can't be done from
# the party LABEL alone (see that module's docstring: Idaho's coalition
# candidate and a minor independent share the identical "ind" label).
#
#   MT: HAS a real Democrat (Alani Bankhead, stayed on the ballot) AND a
#       competing independent (Seth Bodnar, backed by some national
#       Democratic figures) splitting the anti-Republican vote -- this is
#       the original "Montana-type split" example from the very first
#       model-spec conversation. Unlike NE/SD/ID, there's a real D
#       candidate already counted correctly; the issue is that
#       dem_coalition_share means something different here (two
#       candidates competing for the same voters) than in a clean two-way
#       race, so per the original design decision this race is flagged
#       for exclusion from the main regression by default, not combined.
# This is a RACE-level fact (which election had a spoiler-split), not a
# per-candidate one, so it stays here rather than moving into Candidate.
EXCLUDED_MULTIPARTY_STATES = {"MT"}


def load_state_file(path: Path) -> pd.DataFrame:
    state = path.stem.split("_")[0]  # "OR_senate_2026.csv" -> "OR"
    df = pd.read_csv(path)
    df["state"] = state
    df["party_class"] = [
        Candidate(name=name, state=state, raw_party=Party.from_label(party)).effective_party.value
        for name, party in zip(df["candidate_name"], df["party"])
    ]
    return df


def load_real_county_skeleton(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Returns one row per county actually on file, with:
      state, county_name, dem_votes, rep_votes, total_two_party_votes

    total_two_party_votes = dem_votes + rep_votes (third-party/write-in
    votes excluded, consistent with the dem_coalition_share definition
    used throughout -- see baseline_model.py).
    """
    files = sorted(data_dir.glob("*_senate_2026.csv"))
    if not files:
        raise FileNotFoundError(f"No *_senate_2026.csv files found in {data_dir}")

    raw = pd.concat([load_state_file(f) for f in files], ignore_index=True)

    # Guard against exactly the bug we kept hitting while gathering these
    # states: every new state file brought its own party-label spelling
    # (R/Republican, D/Democrat/DFL/dem, gop...), and an unrecognized
    # label silently falls into "OTHER" rather than erroring -- which
    # means a state can silently end up with dem_votes or rep_votes of 0
    # instead of failing loudly. Check for that here rather than relying
    # on manually eyeballing each state's totals.
    class_totals = raw.groupby(["state", "party_class"])["votes"].sum()
    for state in raw["state"].unique():
        for required in ("D", "R"):
            total = class_totals.get((state, required), 0)
            if total == 0:
                unrecognized = sorted(raw.loc[raw["state"] == state, "party"].unique())
                raise ValueError(
                    f"{state}: no votes classified as '{required}' -- a party label probably "
                    f"isn't recognized by candidate.Party.from_label(). Raw labels seen: {unrecognized}"
                )

    major_party_only = raw.loc[raw["party_class"].isin(["D", "R"])]
    pivoted = (
        major_party_only.groupby(["state", "county_name", "party_class"])["votes"]
        .sum()
        .unstack("party_class")
        .rename(columns={"D": "dem_votes", "R": "rep_votes"})
        .fillna(0)
        .reset_index()
    )
    pivoted["dem_votes"] = pivoted["dem_votes"].astype(int)
    pivoted["rep_votes"] = pivoted["rep_votes"].astype(int)
    pivoted["total_two_party_votes"] = pivoted["dem_votes"] + pivoted["rep_votes"]

    # Join key for merging against other sources (e.g. Census ACS
    # demographics in fetch_census_demographics.py), which don't always
    # agree on capitalization for the exact same county (observed:
    # "Mcleod" here vs. "McLeod" in Census; "Lac Qui Parle" vs. "Lac qui
    # Parle"). county_name is kept as-is for display; county_key is what
    # merges should actually use.
    pivoted["county_key"] = pivoted["county_name"].str.lower().str.strip()

    # is_coalition_independent: every county in NE/SD/ID is flagged, since
    # the "D" numerator there is actually an independent's votes (Osborn/
    # Bengs/Achilles -- see candidate.Candidate's coalition overrides), not
    # a real Democrat's. excluded_multiparty_split: every county in MT is
    # flagged, so fit_baseline_model's existing filter
    # (df["excluded_multiparty_split"] == 0) drops it by default, per the
    # original Montana-split design decision -- see EXCLUDED_MULTIPARTY_STATES.
    pivoted["is_coalition_independent"] = pivoted["state"].isin(Candidate.coalition_states()).astype(int)
    pivoted["excluded_multiparty_split"] = pivoted["state"].isin(EXCLUDED_MULTIPARTY_STATES).astype(int)

    return pivoted


# ME/MA/NH/RI report 2026 Senate results by town/city, not county -- they
# don't use counties as functional government units. That used to make
# this a structural, not-yet-fixed mismatch against Census's county-keyed
# demographics (see pipeline_walkthrough.md and class_architecture.md
# Section 10), so those 4 states were excluded here. They're no longer
# excluded: build_town_county_senate_2026.py rolls their raw files up from
# town level to county level (backing up the original town-level files to
# data/raw/town_level_backup/) using a Census county-subdivision crosswalk,
# so data/raw/{ME,MA,NH,RI}_senate_2026.csv are now county-level like every
# other state's file, and need no special-casing here at all.


def load_real_counties_with_census(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    load_real_county_skeleton() merged with fetch_census_demographics.py's
    output (data/raw/census_demographics.csv), joined on ["state",
    "county_key"] rather than county_name (see the capitalization note
    above). Raises if any county fails to match -- silently dropping
    unmatched counties would quietly shrink the sample without anyone
    noticing.
    """
    skeleton = load_real_county_skeleton(data_dir)
    census = pd.read_csv(data_dir / "census_demographics.csv")

    merged = skeleton.merge(census.drop(columns=["county_name"]), on=["state", "county_key"], how="left")
    unmatched = merged[merged["pct_college"].isna()]
    if len(unmatched):
        raise ValueError(f"{len(unmatched)} counties had no Census match:\n{unmatched[['state', 'county_name']]}")

    return merged


def load_real_swing_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    The REAL model table: load_real_counties_with_census() merged with
    build_pres_2024_county.py's output, with dem_swing and swing_weight
    actually computed from observed 2024/2026 results -- not simulated.

    This is the first point in the project where the outcome itself is
    real rather than drawn from TRUE_COEFS (see pipeline_walkthrough.md
    Section 5, which noted this had never happened yet). Only covers the
    states where real 2026 results are on file AND county-level 2024 data
    exists -- currently every state with a *_senate_2026.csv (Alaska was
    never fetched for 2026 in the first place, so it's absent here without
    needing a separate exclusion).
    """
    from baseline_model import compute_swing_weight

    real = load_real_counties_with_census(data_dir)
    pres2024 = pd.read_csv(data_dir / "pres_2024_county.csv")

    merged = real.merge(
        pres2024[["state", "county_key", "total_two_party_votes_2024", "dem_share_2024"]],
        on=["state", "county_key"],
        how="left",
    )
    unmatched = merged[merged["dem_share_2024"].isna()]
    if len(unmatched):
        raise ValueError(f"{len(unmatched)} counties had no 2024 presidential match:\n{unmatched[['state', 'county_name']]}")
    merged["total_two_party_votes_2024"] = merged["total_two_party_votes_2024"].astype(int)

    merged["dem_share_2026"] = merged["dem_votes"] / merged["total_two_party_votes"]
    merged["dem_swing"] = merged["dem_share_2026"] - merged["dem_share_2024"]
    merged["swing_weight"] = compute_swing_weight(merged["total_two_party_votes_2024"], merged["total_two_party_votes"])

    religiosity = pd.read_csv(data_dir / "state_religiosity.csv").rename(columns={"pct_highly_religious": "religiosity"})
    merged = merged.merge(religiosity, on="state", how="left")

    # A handful of tiny-population counties (e.g. Kenedy County, TX, pop.
    # 52) have no reliable Census estimate for one or more demographic
    # variables. Dropping them here (rather than leaving it to statsmodels'
    # default missing='drop' during fitting) matters specifically because
    # cluster-robust SEs need `groups` to be the same length as the
    # fitted design matrix -- a silent patsy-level drop desyncs the two
    # and fails with an opaque numpy error, not a clear one.
    demographic_cols = ["pct_college", "pct_black", "pct_hispanic", "pct_white", "median_age", "median_income_k"]
    before = len(merged)
    merged = merged.dropna(subset=demographic_cols)
    dropped = before - len(merged)
    if dropped:
        print(f"Dropped {dropped} county row(s) with missing Census estimates (expected: tiny-population counties like Kenedy, TX, pop. 52)")

    return merged


def load_all_class2_states_county_frame(data_dir: Path = DATA_DIR, rng=None) -> pd.DataFrame:
    """
    County-level frame for the Class-2 Senate states that can support a
    SWING outcome (2024-to-2026), with a SIMULATED total_two_party_votes
    (2026) for the states we don't have real 2026 Senate results for yet
    (only OR/WY/MN/MS/NM have real returns on file) and REAL
    total_two_party_votes_2024 / dem_share_2024 for every county that has
    one (see build_pres_2024_county.py).

    Alaska is dropped entirely: build_pres_2024_county.py excludes it
    (2024 results there are reported by State House District, not by
    county-equivalent -- a genuinely different geography, not just a
    naming mismatch), so there is no way to compute a county-level swing
    for it with data this project has.

    Unlike simulate_counties()'s fully-fake county structure, the counties
    here are 100% real (real names, real demographics) -- only the 2026
    vote COUNT is a stand-in, built proportionally to each county's actual
    Census population rather than drawn independently of it. This is
    still a rough proxy, not a turnout model: it assumes a flat ~35-50%
    of total population cast a two-party vote, with no adjustment for
    age-eligibility, registration, or actual historical turnout by state.
    It exists only to give the toy model's heteroskedasticity-by-county-size
    pattern (see toy_data.py, compute_swing_weight in baseline_model.py) a
    realistic SHAPE while we don't yet have real 2026 vote totals for most
    of these states -- swap this out once real returns land for all 33.
    """
    rng = rng or np.random.default_rng(seed=7)

    census = pd.read_csv(data_dir / "census_demographics.csv")
    before = len(census)
    census = census.dropna(subset=["pct_college", "pct_black", "pct_hispanic", "pct_white", "median_age", "median_income_k"])
    dropped = before - len(census)
    if dropped:
        print(f"Dropped {dropped} county row(s) with missing Census estimates (expected: tiny-population counties like Kenedy, TX, pop. 52)")

    census = census[census["state"] != "AK"].copy()

    turnout_fraction = rng.uniform(0.35, 0.50, size=len(census))
    census["total_two_party_votes"] = (census["total_population"] * turnout_fraction).clip(lower=50).astype(int)

    pres2024 = pd.read_csv(data_dir / "pres_2024_county.csv")
    merged = census.merge(
        pres2024[["state", "county_key", "total_two_party_votes_2024", "dem_share_2024"]],
        on=["state", "county_key"],
        how="left",
    )
    unmatched = merged[merged["total_two_party_votes_2024"].isna()]
    if len(unmatched):
        raise ValueError(f"{len(unmatched)} counties had no 2024 presidential match:\n{unmatched[['state', 'county_name']]}")
    merged["total_two_party_votes_2024"] = merged["total_two_party_votes_2024"].astype(int)

    return merged


def load_real_swing_table_with_incumbency(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    load_real_swing_table() merged with data/raw/incumbency_2026.csv
    (columns: state, incumbent_party, is_appointed, incumbent_status),
    joined on "state" alone (incumbency is a state-level fact, not a
    county-level one). See baseline_model.py's EXTENDED_PREDICTORS for how
    incumbent_status gets used as a categorical regressor.

    Added as a SEPARATE function rather than folded into
    load_real_swing_table() itself, on purpose: every existing caller of
    load_real_swing_table() (scripts/plot_residuals.py, the original
    6-variable baseline comparison, etc.) keeps getting the exact same
    table it always has, untouched.

    incumbent_status has three levels:
      D_incumbent / R_incumbent -- a Democrat/Republican who was ELECTED
          to that seat in a prior election and is running again.
      no_incumbent -- either an open seat (prior holder retired, lost
          renomination, or otherwise isn't on the 2026 general-election
          ballot), OR a sitting senator who was APPOINTED to fill a
          vacancy rather than elected to that seat (e.g. SC's Darline
          Graham Nordone, OK's Alan Armstrong) -- an appointee counts as
          no_incumbent even though they're technically the sitting
          senator, because they haven't banked the electoral incumbency
          advantage (prior campaigns, name recognition from winning that
          specific seat) the other two levels are meant to capture. A
          senator who was ORIGINALLY appointed but has SINCE won an
          election to that same seat (e.g. NE's Pete Ricketts, appointed
          2023 then won the 2024 special) is treated as a normal elected
          incumbent instead, per the same logic -- see
          data/raw/incumbency_2026.csv's is_appointed column, and the
          research notes behind it, for the full per-state reasoning.
    """
    swing = load_real_swing_table(data_dir)
    incumbency = pd.read_csv(data_dir / "incumbency_2026.csv")

    merged = swing.merge(incumbency, on="state", how="left")
    unmatched = merged[merged["incumbent_status"].isna()]
    if len(unmatched):
        raise ValueError(
            f"{len(unmatched)} counties had no incumbency match (state missing from "
            f"incumbency_2026.csv):\n{unmatched[['state', 'county_name']]}"
        )
    return merged


if __name__ == "__main__":
    skeleton = load_real_county_skeleton()
    print(f"Loaded {len(skeleton)} real counties across {skeleton['state'].nunique()} states:")
    print(skeleton.groupby("state").agg(n_counties=("county_name", "size"), total_votes=("total_two_party_votes", "sum")))
    print("\nSample rows:")
    print(skeleton.head(10))
