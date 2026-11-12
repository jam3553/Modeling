"""
Cleans the raw 2024 county-level presidential results (downloaded from
tonmcg/US_County_Level_Election_Results_08-24, a commonly-used open mirror
of MEDSL's official countypres data -- the official Harvard Dataverse copy
is gated behind a guestbook form, this mirror isn't) into the canonical
format used elsewhere in this project.

Alaska is DROPPED: this source reports Alaska's 2024 results by STATE HOUSE
DISTRICT (40 of them), not by borough/census area (the ~30 county-equivalents
Census demographics are keyed on). These are genuinely different
geographies with no clean one-to-one crosswalk, not just a naming
mismatch -- confirmed by comparing row counts against the Census pull (40
vs 30) before assuming anything. Every other one of the 33 Class-2 states
matched Census's county count exactly.
"""

from pathlib import Path

import pandas as pd

NAME_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arkansas": "AR", "Colorado": "CO", "Delaware": "DE",
    "Georgia": "GA", "Idaho": "ID", "Illinois": "IL", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Montana": "MT", "Nebraska": "NE",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "North Carolina": "NC",
    "Oklahoma": "OK", "Oregon": "OR", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Virginia": "VA",
    "West Virginia": "WV", "Wyoming": "WY",
}

if __name__ == "__main__":
    base = Path(__file__).parent
    raw = pd.read_csv(base / "data" / "external" / "pres_2024_county.csv")

    raw["state"] = raw["state_name"].map(NAME_TO_ABBR)
    raw = raw.dropna(subset=["state"])
    raw = raw[raw["state"] != "AK"]  # see module docstring

    raw["county_key"] = (
        raw["county_name"].str.replace(r"\s+(County|Parish)$", "", regex=True).str.lower().str.strip()
    )

    out = pd.DataFrame(
        {
            "state": raw["state"],
            "county_key": raw["county_key"],
            "dem_votes_2024": raw["votes_dem"],
            "rep_votes_2024": raw["votes_gop"],
            "total_two_party_votes_2024": raw["votes_dem"] + raw["votes_gop"],
            "dem_share_2024": raw["votes_dem"] / (raw["votes_dem"] + raw["votes_gop"]),
        }
    )

    out_path = base / "data" / "raw" / "pres_2024_county.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} counties across {out['state'].nunique()} states (AK excluded) -> {out_path}")
