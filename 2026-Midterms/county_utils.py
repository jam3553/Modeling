"""
Shared county-name normalization. Pulled out after the exact same
suffix-stripping + collision-restoration logic had to be duplicated into a
second file (build_pres_2024_county.py duplicating
fetch_census_demographics.py) -- this is the one place it should live.

Two independent sources disagree on how far to go:
  - Census/NBC: strip the county-equivalent type word (County, Parish,
    Borough, Census Area, Municipality, City and Borough, city) UNLESS
    doing so would collide two differently-bordered places onto the same
    name within a state (Virginia: Fairfax/Franklin/Richmond/Roanoke each
    exist as both a county and a same-named independent city).
  - Some sources (e.g. the tonmcg 2024 presidential mirror) always keep
    "city" on every Virginia independent city, whether or not it collides.
Normalizing both to the FIRST convention (strip unless it collides) is
what lets every ingestion script join on the same county_key.
"""

import pandas as pd

SUFFIX_PATTERN = r"\s+(County|Parish|Borough|Census Area|Municipality|City and Borough|city)$"


def strip_county_suffix(names_without_state: pd.Series, state: pd.Series) -> pd.Series:
    """
    names_without_state: county names with any trailing ", <state>" already
        removed (Census's raw NAME field needs that done first -- see
        fetch_census_demographics.py -- since state names vary in how many
        commas/words they contain, which this function doesn't need to know).
    state: parallel Series of state abbreviations, scoping the collision
        check to within the same state.
    """
    stripped = names_without_state.str.replace(SUFFIX_PATTERN, "", regex=True)
    was_city = names_without_state.str.strip().str.endswith("city")
    collided = stripped.groupby(state).transform(lambda s: s.duplicated(keep=False))
    return stripped.mask(collided & was_city, stripped + " city")


def county_key(county_name: pd.Series) -> pd.Series:
    return county_name.str.lower().str.strip()
