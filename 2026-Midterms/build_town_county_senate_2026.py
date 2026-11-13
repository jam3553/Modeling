"""
Rolls up ME/MA/NH/RI's 2026 Senate returns from town/city level (how all
four states actually report results -- they don't use counties as
functional government units) to county level, so they join against Census
demographics (county-keyed everywhere) exactly like every other state's
*_senate_2026.csv already does. See real_counties.py's comment just above
load_real_counties_with_census for why these 4 states used to be excluded.

The crosswalk is built from the Census API's "county subdivision" geography
(ACS 2023 5-year, same vintage as fetch_census_demographics.py), which
returns each town/city already paired with its parent county's FIPS code
-- e.g. `["Auburn city, Androscoggin County, Maine", "23", "001", "02060"]`.
No county_name string-matching against the election file is needed for the
county side; only the TOWN name has to be matched between the two sources.

Normalization needed to match town names between sources is deliberately
MINIMAL: Census's NAME field always carries a trailing LSAD (legal/
statistical area description) word -- "city", "town", or (for a handful of
unincorporated New Hampshire places) "township" -- that the hand-gathered
election files never include, so stripping exactly those three words is
enough for ~99% of towns. Maine's and New Hampshire's other place-type
words (plantation, "UT"/unorganized territory, location, grant, purchase,
gore) are deliberately NOT stripped, because they're sometimes the only
thing distinguishing two genuinely different places that happen to share a
base name -- e.g. Maine has both "Unity" (a town) and "Unity UT" (an
unorganized territory), and both "Lincoln" (a town) and "Lincoln
plantation"; New Hampshire has both "Wentworth" (a town) and "Wentworth
location". Stripping those words too would collide these onto the same
key and misassign one's votes to the other's county. The two remaining
Maine-specific exceptions this still doesn't resolve (a genuine
type-change and a same-name reservation split across two counties) are
resolved explicitly below (EXPLICIT_OVERRIDES), not guessed at.

Two rows per state could not be geographically assigned at all -- Maine's
"Overseas Absentee" and Rhode Island's "Rhode Island" -- statewide
absentee/catch-all buckets, not town subdivisions (confirmed: neither
appears anywhere in the Census county-subdivision list for its state).
Both are small relative to the state total (Maine: ~4.8k/710k = 0.7%;
Rhode Island: ~0.8k/360k = 0.2%). Rather than silently dropping them (which
would break the exact-reconciliation check this script runs, and quietly
undercount real cast votes) or inventing a 51st/40th pseudo-county (which
would break the Census join the whole point of this script is to fix),
each candidate's bucket is distributed across that state's real counties
in proportion to the candidate's OWN already-assigned county vote shares
(largest-remainder rounding to keep every county's vote count an integer
while the per-candidate total is conserved exactly). This is a modeling
assumption -- it assumes absentee voters' candidate preferences are
distributed across counties the same way in-town voters' are -- flagged
here rather than buried, same spirit as this project's other documented
judgment calls (Montana's multiparty split, Alaska's geography drop).

Before overwriting data/raw/{ME,MA,NH,RI}_senate_2026.csv, the original
town-level files are copied to data/raw/town_level_backup/ (run once;
re-running this script is a no-op backup-wise since it won't overwrite an
existing backup).
"""

import shutil
import sys
from pathlib import Path

import pandas as pd
import requests

from county_utils import county_key, strip_county_suffix

KEY = Path(__file__).parent.joinpath(".census_api_key").read_text().strip()
ACS_YEAR = 2023
DATA_DIR = Path(__file__).parent / "data" / "raw"
BACKUP_DIR = DATA_DIR / "town_level_backup"

STATE_FIPS = {"ME": "23", "MA": "25", "NH": "33", "RI": "44"}

UNASSIGNABLE_ROWS = {
    "ME": {"Overseas Absentee"},
    "RI": {"Rhode Island"},
}

GENERIC_LSAD_WORDS = {"city", "town", "township"}


def strip_lsad(name: str) -> str:
    words = name.split()
    while words and words[-1].lower() in GENERIC_LSAD_WORDS:
        words = words[:-1]
    return " ".join(words).lower()


# (state, stripped-town-key) -> correct county name. See module docstring.
EXPLICIT_OVERRIDES = {
    # Census's 2023 vintage still has this place as "Drew Plantation"
    # (confirmed: "Drew plantation, Penobscot County, Maine" in the raw
    # API response); Maine's own 2026 returns call it "Drew unorganized
    # territory" -- Drew Plantation deorganized at some point between
    # Census's vintage and the election. Same place, different government
    # type label; resolved by hand rather than guessed at.
    ("ME", "drew unorganized territory"): "Penobscot",
    # Census returns "Penobscot Indian Island Reservation" TWICE for
    # Maine: once under Aroostook County (population 0 -- confirmed via a
    # separate population pull -- a non-contiguous, uninhabited trust-land
    # parcel) and once under Penobscot County (population 532 -- Indian
    # Island/Old Town, the actual reservation community that casts
    # votes). The stripped base name is identical for both, so the
    # crosswalk can't disambiguate automatically; Penobscot County is the
    # real one.
    ("ME", "penobscot indian island reservation"): "Penobscot",
}


def fetch_town_county_crosswalk(state_abbr: str) -> pd.DataFrame:
    """
    Returns one row per Census county subdivision in `state_abbr`, with
    columns: town_raw, town_key (stripped per strip_lsad), county_name
    (Census's county name, " County" suffix removed -- NOT yet run through
    county_utils' further suffix-stripping/collision logic, which happens
    once at the very end against the canonical census_demographics.csv
    list).
    """
    fips = STATE_FIPS[state_abbr]
    resp = requests.get(
        f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5",
        params={"get": "NAME", "for": "county subdivision:*", "in": f"state:{fips}", "key": KEY},
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json()[1:]

    out = []
    for name, _state, _county_fips, _subdiv_fips in rows:
        if name.startswith("County subdivisions not defined"):
            continue  # placeholder row for a county with no further subdivision breakdown
        town_part, county_part, _state_part = name.split(", ")
        county_name = county_part[: -len(" County")] if county_part.endswith(" County") else county_part
        out.append({"town_raw": town_part, "town_key": strip_lsad(town_part), "county_name": county_name})

    df = pd.DataFrame(out)

    # Guard against a FUTURE vintage silently introducing a new ambiguous
    # base-name collision (two different real county subdivisions that
    # strip down to the same key but sit in different counties) beyond the
    # one already known and hand-resolved above.
    collisions = df.groupby("town_key")["county_name"].nunique()
    unexplained = collisions[collisions > 1]
    unexplained = [k for k in unexplained.index if (state_abbr, k) not in EXPLICIT_OVERRIDES]
    if unexplained:
        raise ValueError(f"{state_abbr}: new unresolved town-key collision(s) across counties: {unexplained}")

    return df


def distribute_unassignable_votes(county_votes: pd.Series, bucket_votes: int) -> pd.Series:
    """
    Spreads `bucket_votes` (a single candidate's statewide unassignable
    total) across `county_votes` (that same candidate's existing per-county
    vote counts) in proportion to each county's existing share, using
    largest-remainder rounding so the integer allocations sum to EXACTLY
    bucket_votes (not just approximately, which floor/round alone would
    risk breaking the exact-reconciliation check this script runs).
    """
    if bucket_votes == 0 or county_votes.sum() == 0:
        return pd.Series(0, index=county_votes.index)

    shares = county_votes / county_votes.sum()
    raw = shares * bucket_votes
    floors = raw.apply(int)
    remainder = bucket_votes - floors.sum()
    # Give the leftover votes (from rounding down) to the counties with the
    # largest fractional remainders -- the standard largest-remainder /
    # Hamilton apportionment method.
    fractional = (raw - floors).sort_values(ascending=False)
    bump = fractional.index[:remainder]
    floors.loc[bump] += 1
    assert floors.sum() == bucket_votes
    return floors


def rollup_state(state_abbr: str) -> pd.DataFrame:
    town_df = pd.read_csv(DATA_DIR / f"{state_abbr}_senate_2026.csv")
    crosswalk = fetch_town_county_crosswalk(state_abbr)
    key_to_county = {}
    for town_key, county_name in zip(crosswalk["town_key"], crosswalk["county_name"]):
        key_to_county.setdefault(town_key, county_name)
    key_to_county.update({k: v for (st, k), v in EXPLICIT_OVERRIDES.items() if st == state_abbr})

    unassignable = UNASSIGNABLE_ROWS.get(state_abbr, set())
    assignable = town_df[~town_df["county_name"].isin(unassignable)].copy()
    bucket = town_df[town_df["county_name"].isin(unassignable)].copy()

    assignable["town_key"] = assignable["county_name"].apply(strip_lsad)
    unmatched_towns = sorted(set(assignable.loc[~assignable["town_key"].isin(key_to_county), "county_name"]))
    if unmatched_towns:
        raise ValueError(
            f"{state_abbr}: {len(unmatched_towns)} town(s) didn't match the Census county-subdivision "
            f"crosswalk and aren't in UNASSIGNABLE_ROWS or EXPLICIT_OVERRIDES: {unmatched_towns}"
        )
    assignable["county_name_resolved"] = assignable["town_key"].map(key_to_county)

    county_level = (
        assignable.groupby(["county_name_resolved", "candidate_name", "party"])["votes"]
        .sum()
        .reset_index()
        .rename(columns={"county_name_resolved": "county_name"})
    )

    # Distribute each unassignable row's votes across counties by that
    # SAME candidate's existing county vote shares (not overall turnout
    # share) -- see module docstring.
    for _, row in bucket.iterrows():
        if row["votes"] == 0:
            continue
        candidate_rows = county_level[county_level["candidate_name"] == row["candidate_name"]]
        if candidate_rows.empty:
            raise ValueError(
                f"{state_abbr}: unassignable row for candidate {row['candidate_name']!r} "
                f"({row['votes']} votes) has no matching county-level rows to distribute against."
            )
        allocation = distribute_unassignable_votes(
            candidate_rows.set_index("county_name")["votes"], int(row["votes"])
        )
        for cname, add_votes in allocation.items():
            if add_votes == 0:
                continue
            mask = (county_level["county_name"] == cname) & (county_level["candidate_name"] == row["candidate_name"])
            county_level.loc[mask, "votes"] += add_votes

    return county_level[["county_name", "candidate_name", "party", "votes"]]


def verify_reconciliation(state_abbr: str, town_df: pd.DataFrame, county_df: pd.DataFrame) -> None:
    town_totals = town_df.groupby("candidate_name")["votes"].sum()
    county_totals = county_df.groupby("candidate_name")["votes"].sum()
    # Align indices; a candidate missing from one side entirely is itself
    # a mismatch, not something fillna(0) should paper over silently.
    if set(town_totals.index) != set(county_totals.index):
        raise ValueError(
            f"{state_abbr}: candidate set differs between town-level and county-level "
            f"({set(town_totals.index) ^ set(county_totals.index)})"
        )
    mismatches = town_totals[town_totals != county_totals]
    if len(mismatches):
        raise ValueError(f"{state_abbr}: per-candidate vote totals don't reconcile:\n{mismatches}")


if __name__ == "__main__":
    BACKUP_DIR.mkdir(exist_ok=True)

    for state in sorted(STATE_FIPS):
        src = DATA_DIR / f"{state}_senate_2026.csv"
        backup = BACKUP_DIR / f"{state}_senate_2026_town_level.csv"
        if not backup.exists():
            shutil.copy2(src, backup)
            print(f"Backed up {src} -> {backup}")
        else:
            print(f"Backup already exists for {state}, leaving it alone: {backup}")

    for state in sorted(STATE_FIPS):
        print(f"\n=== {state} ===")
        town_df = pd.read_csv(BACKUP_DIR / f"{state}_senate_2026_town_level.csv")
        county_df = rollup_state(state)
        verify_reconciliation(state, town_df, county_df)
        print(f"{state}: reconciled OK -- {town_df['county_name'].nunique()} towns -> {county_df['county_name'].nunique()} counties")

        out_path = DATA_DIR / f"{state}_senate_2026.csv"
        county_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    print("\nAll 4 states rolled up and reconciled.")
