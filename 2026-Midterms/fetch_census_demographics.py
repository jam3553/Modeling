"""
Pulls real ACS demographics -- both county-level and state-level -- for the
33 states with a 2026 Class 2 U.S. Senate race (the scope decided on: House
races have unopposed-seat issues and Governor races don't exist in every
state, so for now the project is standardized on Senate as the one outcome
that's consistently defined across states; see conversation history).

One script, one process, run locally -- not farmed out to research agents
per state, specifically to keep this cheap. ~33 HTTP requests to a stable
government API is a non-event; the earlier per-state agent dispatches (one
state at a time) were the expensive part of this project so far.

Variables pulled (ACS 2023 5-year estimates, verified against the Census
API's own variable metadata before use -- see conversation history):
  B15003_001E, _022E, _023E, _024E, _025E
      Educational Attainment for the Population 25 Years and Over:
      total, bachelor's, master's, professional school, doctorate.
      pct_college = (022+023+024+025) / 001
  B03002_001E, _003E, _004E, _006E, _012E
      Hispanic or Latino Origin by Race:
      total; White alone, not Hispanic; Black alone, not Hispanic;
      Asian alone, not Hispanic; Hispanic or Latino (any race).
      White-alone-not-Hispanic is the OMITTED REFERENCE CATEGORY in the
      regression (see baseline_model.py) -- not pulled as its own column
      here because we never use it directly, only implicitly.
  B01002_001E   Median age, total population.
  B19013_001E   Median household income (inflation-adjusted dollars).

Output: data/raw/census_demographics.csv (county-level, one row per
county) and data/raw/census_demographics_state_level.csv (one row per
state, pulled directly from Census's own state-level aggregates rather
than recomputed from the county rows, so it's not vulnerable to any
weighting mistake on our end).
"""

from pathlib import Path

import pandas as pd
import requests

from county_utils import county_key, strip_county_suffix

KEY = Path(__file__).parent.joinpath(".census_api_key").read_text().strip()
ACS_YEAR = 2023

# The 33 states with a 2026 Class 2 U.S. Senate race (confirmed via web
# search earlier in this project -- Wikipedia/Green Papers agree on this
# list and it matches the known Class 2 seat count).
CLASS2_SENATE_STATE_FIPS = {
    "AL": "01", "AK": "02", "AR": "05", "CO": "08", "DE": "10", "GA": "13",
    "ID": "16", "IL": "17", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
    "ME": "23", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MT": "30",
    "NE": "31", "NH": "33", "NJ": "34", "NM": "35", "NC": "37", "OK": "40",
    "OR": "41", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48",
    "VA": "51", "WV": "54", "WY": "56",
}

VARIABLES = [
    "B15003_001E", "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E",
    "B03002_001E", "B03002_004E", "B03002_006E", "B03002_012E",
    "B01002_001E", "B19013_001E",
]


def _fetch_raw(for_clause: str, in_clause: str = None) -> pd.DataFrame:
    url = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
    params = {"get": "NAME," + ",".join(VARIABLES), "for": for_clause, "key": KEY}
    if in_clause:
        params["in"] = in_clause
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    rows = resp.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    for v in VARIABLES:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    # Census uses large negative sentinel codes (e.g. -666666666) for
    # estimates that couldn't be computed (usually tiny-population
    # counties/tracts). Treat those as missing rather than as real values.
    return df.replace(-666666666, pd.NA)


def _derive_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Shared derivation logic for both county- and state-level pulls."""
    pop_25plus = df["B15003_001E"]
    bachelors_plus = df["B15003_022E"] + df["B15003_023E"] + df["B15003_024E"] + df["B15003_025E"]
    total_pop = df["B03002_001E"]

    return pd.DataFrame(
        {
            "pct_college": bachelors_plus / pop_25plus,
            "pct_black": df["B03002_004E"] / total_pop,
            "pct_hispanic": df["B03002_012E"] / total_pop,
            "pct_asian": df["B03002_006E"] / total_pop,
            "median_age": df["B01002_001E"],
            "median_income_k": df["B19013_001E"] / 1000.0,
            "total_population": total_pop,
        }
    )


def fetch_county_level(state_abbr: str, fips: str) -> pd.DataFrame:
    df = _fetch_raw(for_clause="county:*", in_clause=f"state:{fips}")

    # NAME comes back as e.g. "Baker County, Oregon", "Acadia Parish,
    # Louisiana", or "Anchorage Municipality, Alaska" -- county-equivalent
    # naming is NOT uniform across states (Louisiana parishes, Alaska
    # boroughs/census areas/municipalities). rsplit on the LAST comma
    # strips the trailing ", <state>" regardless of what precedes it.
    # The shared suffix-stripping/collision logic lives in county_utils.py
    # -- see that module's docstring for why (it was duplicated into a
    # second ingestion script once, which is exactly what it's meant to
    # prevent going forward).
    name_no_state = df["NAME"].str.rsplit(", ", n=1).str[0]
    state_col = pd.Series(state_abbr, index=name_no_state.index)
    county_name = strip_county_suffix(name_no_state, state_col)

    out = _derive_demographics(df)
    out.insert(0, "state", state_abbr)
    out.insert(1, "county_name", county_name)
    out.insert(2, "county_key", county_key(county_name))
    return out


def fetch_state_level(state_abbr: str, fips: str) -> pd.DataFrame:
    df = _fetch_raw(for_clause=f"state:{fips}")
    out = _derive_demographics(df)
    out.insert(0, "state", state_abbr)
    return out


if __name__ == "__main__":
    # Sanity check against a typo'd FIPS code: if e.g. "35" (New Mexico)
    # were accidentally swapped with another state's code, the API would
    # happily return data for the WRONG state under our label, and
    # nothing downstream would catch it. Verify the state name embedded
    # in Census's own NAME field actually contains the state we asked
    # for, abbreviation case aside.
    STATE_FULL_NAMES = {
        "AL": "Alabama", "AK": "Alaska", "AR": "Arkansas", "CO": "Colorado", "DE": "Delaware",
        "GA": "Georgia", "ID": "Idaho", "IL": "Illinois", "IA": "Iowa", "KS": "Kansas",
        "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MA": "Massachusetts", "MI": "Michigan",
        "MN": "Minnesota", "MS": "Mississippi", "MT": "Montana", "NE": "Nebraska",
        "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NC": "North Carolina",
        "OK": "Oklahoma", "OR": "Oregon", "RI": "Rhode Island", "SC": "South Carolina",
        "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "VA": "Virginia",
        "WV": "West Virginia", "WY": "Wyoming",
    }

    county_frames, state_frames = [], []
    for i, (abbr, fips) in enumerate(CLASS2_SENATE_STATE_FIPS.items(), 1):
        print(f"[{i}/{len(CLASS2_SENATE_STATE_FIPS)}] Fetching {abbr} (FIPS {fips})...")

        county_df = fetch_county_level(abbr, fips)
        raw_names = requests.get(
            f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5",
            params={"get": "NAME", "for": "county:*", "in": f"state:{fips}", "key": KEY},
            timeout=30,
        ).json()
        if not any(STATE_FULL_NAMES[abbr] in row[0] for row in raw_names[1:]):
            raise ValueError(f"FIPS mismatch: expected {STATE_FULL_NAMES[abbr]} for {abbr}/{fips}")
        county_frames.append(county_df)

        state_frames.append(fetch_state_level(abbr, fips))

    county_combined = pd.concat(county_frames, ignore_index=True)
    state_combined = pd.concat(state_frames, ignore_index=True)

    county_out = Path(__file__).parent / "data" / "raw" / "census_demographics.csv"
    state_out = Path(__file__).parent / "data" / "raw" / "census_demographics_state_level.csv"
    county_combined.to_csv(county_out, index=False)
    state_combined.to_csv(state_out, index=False)

    print(f"\nCounty-level: {len(county_combined)} counties across {county_combined['state'].nunique()} states -> {county_out}")
    print(county_combined.groupby("state").size())
    print(f"\nState-level: {len(state_combined)} states -> {state_out}")
    print("\nMissing-value check (county-level):")
    print(county_combined.isna().sum())
