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

DATA_DIR = Path(__file__).parent / "data" / "raw"

# Real-world party labels are messy and inconsistent across states (fusion
# tickets, state-specific party names, write-ins coded as "N/A"). We only
# need to know, for each row, whether it's the Democratic-coalition
# candidate, the Republican candidate, or neither -- everything else
# (Libertarian, Green, independents not aligned with either major party,
# write-ins) is excluded from the two-party vote total.
DEM_PARTY_LABELS = {
    "democrat",
    "democratic",
    "democrat, working families",
    "democratic-farmer-labor",
    "d",
    "dem",
}
REP_PARTY_LABELS = {
    "republican",
    "r",
    "gop",
}

# THEORY note tying back to the model spec discussion: this is where the
# "Osborne flag" / "Montana-split" logic from the model spec actually has
# to live for real data -- it can't be done from the party LABEL alone,
# because in these races the coalition candidate and a minor independent
# share the exact same party string ("ind" or "other"). Confirmed against
# real 2026 results (checked via web search + the raw party-label dump,
# not assumed):
#   NE: no Democrat on the ballot (withdrew after primary). Dan Osborn
#       (labeled "other" by the source, not even "ind") is the de facto
#       opposition candidate -- the project's original prototype case.
#   SD: no Democrat on the ballot. Brian Bengs (SD's 2022 Democratic
#       Senate nominee) is running as "ind" -- same pattern as NE.
#   ID: no Democrat on the ballot (withdrew in July). TWO independents
#       ran: Todd Achilles (33.5%, well-funded, clearly the main
#       challenger) is the coalition candidate; Natalie Fleming (6.9%, no
#       FEC filings, a repeat minor candidate) is NOT -- same "ind" party
#       label, different coalition status, hence the candidate-name-level
#       override below rather than a party-string-level one.
# Each entry's votes get added to the Democratic-coalition numerator as if
# they were "D" -- see is_coalition_independent in load_real_county_skeleton.
COALITION_INDEPENDENTS = {
    ("NE", "dan osborn"),
    ("SD", "brian bengs"),
    ("ID", "todd achilles"),
}

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
EXCLUDED_MULTIPARTY_STATES = {"MT"}


def _classify_party(state: str, candidate_name: str, raw_party: str) -> str:
    if (state, str(candidate_name).strip().lower()) in COALITION_INDEPENDENTS:
        return "D"
    key = str(raw_party).strip().lower()
    if key in DEM_PARTY_LABELS:
        return "D"
    if key in REP_PARTY_LABELS:
        return "R"
    return "OTHER"


def load_state_file(path: Path) -> pd.DataFrame:
    state = path.stem.split("_")[0]  # "OR_senate_2026.csv" -> "OR"
    df = pd.read_csv(path)
    df["state"] = state
    df["party_class"] = [
        _classify_party(state, name, party) for name, party in zip(df["candidate_name"], df["party"])
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
                    f"isn't in DEM_PARTY_LABELS/REP_PARTY_LABELS. Raw labels seen: {unrecognized}"
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
    # Bengs/Achilles -- see COALITION_INDEPENDENTS above), not a real
    # Democrat's. excluded_multiparty_split: every county in MT is
    # flagged, so fit_baseline_model's existing filter
    # (df["excluded_multiparty_split"] == 0) drops it by default, per the
    # original Montana-split design decision -- see EXCLUDED_MULTIPARTY_STATES.
    coalition_states = {state for state, _ in COALITION_INDEPENDENTS}
    pivoted["is_coalition_independent"] = pivoted["state"].isin(coalition_states).astype(int)
    pivoted["excluded_multiparty_split"] = pivoted["state"].isin(EXCLUDED_MULTIPARTY_STATES).astype(int)

    return pivoted


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
    census = census.dropna(subset=["pct_college", "pct_black", "pct_hispanic", "pct_asian", "median_age", "median_income_k"])
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


if __name__ == "__main__":
    skeleton = load_real_county_skeleton()
    print(f"Loaded {len(skeleton)} real counties across {skeleton['state'].nunique()} states:")
    print(skeleton.groupby("state").agg(n_counties=("county_name", "size"), total_votes=("total_two_party_votes", "sum")))
    print("\nSample rows:")
    print(skeleton.head(10))
