"""
Synthetic county-level election dataset, used to validate the baseline model
in baseline_model.py BEFORE we touch real (messy, incomplete) 2026 data.

Why bother with fake data at all?
---------------------------------
When you fit a regression on real data, you can never be fully sure whether
a disappointing result (bad fit, a coefficient with the "wrong" sign, huge
standard errors) is because (a) your model/code is wrong, or (b) the world
is just noisy/confounded in a way your model can't capture. A toy dataset
breaks that ambiguity: WE pick the true coefficients when generating the
data, so after fitting we can directly compare "what the model recovered"
against "what we know is true." If the pipeline can't recover known
coefficients from data we built to satisfy the model's own assumptions,
there's a bug -- no need to even look at real data yet.

The generative model below is deliberately built to mirror the *theoretical*
assumptions behind the WLS weighting scheme used in baseline_model.py, so
that running this file also doubles as a demonstration of why WLS (weighted
by votes cast) is the right tool versus plain OLS.

OUTCOME CHANGE (see conversation history): the outcome is now dem_swing =
2026 Senate dem-coalition share MINUS 2024 presidential dem two-party
share, per county -- not a raw level. Everything below was rescaled for
that: TRUE_COEFS are swing-sized (a few points per predictor, not 20-30),
the per-state true effect is a random "swing deviation" rather than
anchored to 2024's real level (see STATE_SWING_DEVIATION for why), and the
noise/WLS weight now account for BOTH years' sample sizes via
baseline_model.compute_swing_weight(), not just one.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from baseline_model import compute_swing_weight

RNG = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# 1. "True" coefficients for the data-generating process (DGP).
#
# These are the numbers a correctly-specified WLS fit should recover
# (approximately -- there's still sampling noise, just like in real life).
# Rescaled for a SWING outcome (percentage-point change from 2024 to
# 2026), not a level: real observed swings across the 5 states with
# actual 2026 results ranged from +1.9 to +10.6 points (see conversation
# history), so these coefficients are picked to be in that ballpark for
# the FULL range of a predictor -- still exaggerated enough to spot easily
# in a recovery check, but not the old level-scale (0.10-0.30) which would
# imply a demographic shift alone could swing a county 20-30 points.
# ---------------------------------------------------------------------------
TRUE_COEFS = {
    "intercept": 0.05,       # average swing at reference demographic levels
    "pct_college": 0.06,
    "pct_black": 0.05,
    "pct_hispanic": 0.03,
    "pct_asian": 0.02,
    "median_age": -0.0008,    # per year of age
    "median_income_k": 0.0003,  # per $1,000 of median household income
    # Same "diploma divide" interaction as before, rescaled: college's
    # effect on swing is smaller in high-pct_black counties than the
    # additive pct_college + pct_black terms alone would suggest.
    "college_x_black": -0.03,
}

STATE_ABBRS = [
    # Alaska excluded: build_pres_2024_county.py drops it because its 2024
    # presidential results are reported by State House District, not by
    # county-equivalent -- a genuinely different geography, not a naming
    # mismatch -- so there's no way to compute a county-level swing for it.
    "AL","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY",
]

# 2024 presidential results by state (Harris, Trump), pulled from Wikipedia's
# "2024 United States presidential election" results-by-state table.
# Cross-checked by summing all 50 states + DC and comparing to the page's
# infobox national totals (independently sourced on that same page):
#   Harris sum 75,019,230 vs. infobox 75,017,613 (diff 1,617 of 75M)
#   Trump sum  77,303,568 vs. infobox 77,302,580 (diff    988 of 77M)
# Both within ~0.002%, consistent with minor post-certification updates
# rather than a transcription error, so treated as reliable.
PRES_2024_VOTES = {
    "AL": (772412, 1462616), "AK": (140026, 184458), "AZ": (1582860, 1770242), "AR": (396905, 759241),
    "CA": (9276179, 6081697), "CO": (1728159, 1377441), "CT": (992053, 736918), "DE": (289758, 214351),
    "FL": (4683038, 6110125), "GA": (2548017, 2663117), "HI": (313044, 193661), "ID": (274972, 605246),
    "IL": (3062863, 2449079), "IN": (1163603, 1720347), "IA": (707278, 927019), "KS": (544853, 758802),
    "KY": (704043, 1337494), "LA": (766870, 1208505), "ME": (435652, 377977), "MD": (1902577, 1035550),
    "MA": (2126518, 1251303), "MI": (2736533, 2816636), "MN": (1656979, 1519032), "MS": (466668, 747744),
    "MO": (1200599, 1751986), "MT": (231906, 352079), "NE": (369995, 564816), "NV": (705197, 751205),
    "NH": (418488, 395523), "NJ": (2220713, 1968215), "NM": (478802, 423391), "NY": (4619195, 3578899),
    "NC": (2715375, 2898423), "ND": (112327, 246505), "OH": (2533699, 3180116), "OK": (499599, 1036213),
    "OR": (1240600, 919480), "PA": (3423042, 3543308), "RI": (285156, 214406), "SC": (1028452, 1483747),
    "SD": (146859, 272081), "TN": (1056265, 1966865), "TX": (4835250, 6393597), "UT": (562566, 883818),
    "VT": (235791, 119395), "VA": (2335395, 2075085), "WA": (2245849, 1530923), "WV": (214309, 533556),
    "WI": (1668229, 1697626), "WY": (69527, 192633),
}

_national_harris = sum(v[0] for v in PRES_2024_VOTES.values())
_national_trump = sum(v[1] for v in PRES_2024_VOTES.values())
_national_two_party_dem_share = _national_harris / (_national_harris + _national_trump)

# Kept as REFERENCE data only -- NOT used in the swing DGP's math anymore
# (see STATE_SWING_DEVIATION below for what replaced it). STATE_LEAN_2024
# was the right anchor for a LEVEL outcome (how Democratic a state is
# overall); it says nothing about how much a state SWINGS from one
# election to the next, which is what the DGP needs now.
STATE_LEAN_2024 = {
    state: (harris / (harris + trump)) - _national_two_party_dem_share
    for state, (harris, trump) in PRES_2024_VOTES.items()
}

# Real state-level religiosity (Pew). Also kept as reference/diagnostic
# data, not wired into the DGP -- see state_effect_options.md: religiosity
# belongs to a later diagnostic stage (does the swing residual correlate
# with it), not the generating process or the fitted baseline itself.
RELIGIOSITY = (
    pd.read_csv(Path(__file__).parent / "data" / "raw" / "state_religiosity.csv")
    .set_index("state")["pct_highly_religious"]
    .to_dict()
)

# Each state's TRUE random deviation in SWING (not level) -- this is
# exactly what the project wants left in the residual: "how much more or
# less did this state swing than its demographics predicted," which is
# where state environment + candidate quality are supposed to show up.
# Drawn randomly rather than anchored to the 5 real states' observed mean
# swings on purpose: those observed means already MIX demographics +
# state effect + noise together (raw averages, not a clean
# decomposition), so plugging them in directly would double-count
# demographic signal into what's supposed to be a demographics-independent
# term. Scale (std=0.04) is calibrated to roughly match the spread
# actually seen across those 5 states' mean swings (+1.9 to +10.6 points;
# std of those 5 numbers is ~0.04).
STATE_SWING_DEVIATION = {state: RNG.normal(0, 0.04) for state in STATE_ABBRS}


def _draw_fake_demographics() -> dict:
    """
    Draw FAKE demographic covariates for a county. Used only by
    simulate_counties() (fully synthetic). simulate_demographics_for_real_counties()
    uses real ACS values instead and never calls this -- real counties
    already have authentic covariation baked in, so there's nothing to
    fake there.

    NOTE on covariation vs. interaction (see the conversation with the
    user): everything in THIS function is about covariation -- how the
    X's relate to EACH OTHER before the outcome is even generated. None
    of it changes any variable's EFFECT on dem_coalition_share; that's a
    separate concern, handled entirely in _simulate_outcome()'s
    "college_x_black" term. A covariate can be correlated with everything
    else here and still enter the outcome model purely additively (as
    college, income do), and conversely two variables could be drawn
    completely independently here and still interact in the outcome model
    (as college and black do, below).

    Draw order matters: median_age is drawn first as the "root" variable,
    then pct_college is nudged by age, then median_income_k is nudged by
    BOTH pct_college and median_age. These magnitudes are illustrative,
    not fit to data -- "vaguely realistic" direction and rough scale, not
    calibrated elasticities.
    """
    median_age = np.clip(RNG.normal(39, 6), 22, 65)

    # Mild NEGATIVE covariation between age and college share: counties
    # with large college-educated populations tend to skew a bit younger
    # (university towns, urban professional counties), so nudge the
    # college draw down for older counties and up for younger ones. The
    # 0.003 scale means a county 10 years older than the 39 mean loses
    # about 3 points of college share on average -- a real but secondary
    # effect next to the Beta(2,5) draw's own spread.
    pct_college = np.clip(RNG.beta(2, 5) - 0.003 * (median_age - 39), 0, 1)

    # pct_college and median_income_k are positively correlated, as in
    # real county data (this was already here). Added: a mild POSITIVE
    # age effect on top -- older counties tend to have somewhat higher
    # median household income (more years in the workforce accumulated
    # before the estimate catches retirement-income effects), independent
    # of their college share.
    median_income_k = np.clip(
        35 + 120 * pct_college + 0.3 * (median_age - 39) + RNG.normal(0, 10), 15, 160
    )

    # Race/ethnicity shares. Real ACS data has pct_white + pct_black
    # + pct_hispanic + pct_asian + pct_other = 1. We only model
    # black/hispanic/asian explicitly and treat white+other as the
    # OMITTED REFERENCE CATEGORY (see baseline_model.py for why this
    # matters -- including all categories plus an intercept gives
    # perfect multicollinearity, the "dummy variable trap").
    remaining = 1.0
    pct_black = np.clip(RNG.beta(1.5, 6) * remaining, 0, remaining)
    remaining -= pct_black
    pct_hispanic = np.clip(RNG.beta(1.5, 6) * remaining, 0, remaining)
    remaining -= pct_hispanic
    pct_asian = np.clip(RNG.beta(1.2, 10) * remaining, 0, remaining)

    return {
        "pct_college": pct_college,
        "pct_black": pct_black,
        "pct_hispanic": pct_hispanic,
        "pct_asian": pct_asian,
        "median_age": median_age,
        "median_income_k": median_income_k,
    }


def _simulate_outcome(state_swing_deviation: float, religiosity: float, demographics: dict, n_2024: int, n_2026: int, reference_votes: int) -> dict:
    """
    Given a state's TRUE random swing deviation (see STATE_SWING_DEVIATION),
    its religiosity value (carried through for later diagnostic use only --
    NOT part of the true generating formula, see state_effect_options.md),
    and a county's demographic covariates (real OR fake -- this function
    doesn't care which), generate a "true" dem_swing from TRUE_COEFS plus
    heteroskedastic noise.

    Factored out so that both simulate_counties() (fake demographics) and
    simulate_demographics_for_real_counties() (real ACS demographics) run
    the EXACT same DGP on the outcome side -- the only difference between
    them is where the demographic covariates come from.
    """
    true_swing = (
        TRUE_COEFS["intercept"]
        + state_swing_deviation
        + TRUE_COEFS["pct_college"] * demographics["pct_college"]
        + TRUE_COEFS["pct_black"] * demographics["pct_black"]
        + TRUE_COEFS["pct_hispanic"] * demographics["pct_hispanic"]
        + TRUE_COEFS["pct_asian"] * demographics["pct_asian"]
        + TRUE_COEFS["median_age"] * demographics["median_age"]
        + TRUE_COEFS["median_income_k"] * demographics["median_income_k"]
        # THE interaction term (as opposed to everything in
        # _draw_fake_demographics(), which is covariation): college's
        # effect on dem_swing is not independent of pct_black.
        + TRUE_COEFS["college_x_black"] * demographics["pct_college"] * demographics["pct_black"]
    )

    # --- Heteroskedastic noise, now for a DIFFERENCED outcome ---------
    # THEORY: dem_swing = share_2026 - share_2024, each itself a sample
    # proportion from its own election with its own sample size. See
    # compute_swing_weight()'s docstring in baseline_model.py for the full
    # derivation of Var(swing) = p(1-p) * (1/n_2026 + 1/n_2024), and why
    # the resulting WLS weight is n_2026*n_2024/(n_2026+n_2024), not just
    # n_2026. We use that SAME formula here for the noise scale, so the
    # simulated heteroskedasticity pattern is exactly what the WLS fit
    # will assume -- reference_weight = reference_votes/2 approximates
    # compute_swing_weight(reference_votes, reference_votes), i.e. "a
    # typical county where both years had a similar, typical vote count."
    swing_weight = compute_swing_weight(n_2024, n_2026)
    base_sigma = 0.025
    reference_weight = reference_votes / 2
    noise_sd = base_sigma * np.sqrt(reference_weight / swing_weight)
    noise = RNG.normal(0, noise_sd)

    observed_swing = float(np.clip(true_swing + noise, -0.35, 0.35))

    return {
        "dem_swing": observed_swing,
        "total_two_party_votes": n_2026,
        "total_two_party_votes_2024": n_2024,
        "swing_weight": swing_weight,
        "religiosity": religiosity,
        **demographics,
    }


def simulate_counties(n_states: int = 50, reference_votes: int = 10_000) -> pd.DataFrame:
    """
    Build a FULLY SYNTHETIC county-level dataset -- fake states, fake
    counties, fake vote totals -- with a known true relationship between
    demographics and Democratic-coalition two-party vote share.

    Kept around for exercising the pipeline at full 50-state scale (e.g.
    the interaction/VIF demo in baseline_model.py). For anything meant to
    resemble the real 2026 states we're actually gathering data for, use
    simulate_demographics_for_real_counties() instead, which anchors the
    geography and vote totals to real data and only fakes the demographics
    and outcome -- see that function's docstring for why this distinction
    matters.

    Parameters
    ----------
    n_states : number of states to simulate (<= 49; Alaska is excluded
        from STATE_ABBRS -- see its comment -- so 49 is the max, not 50)
    reference_votes : the "typical" county vote-total scale; used to set the
        noise level (see the heteroskedasticity discussion below).
    """
    rows = []
    states = STATE_ABBRS[:n_states]

    for state in states:
        # Real states vary enormously in county count (Delaware has 3,
        # Texas has 254), and that spread is right-skewed, not uniform --
        # most states cluster in the 50-100 range, with a long tail of a
        # few outliers (Texas, Georgia) rather than an even spread across
        # the range. RNG.integers(low, high) draws a DISCRETE UNIFORM
        # integer -- every value in [low, high) equally likely -- which
        # doesn't capture that skew. RNG.lognormal is the distribution of
        # exp(X) for X ~ Normal(mean, sigma): strictly positive and
        # right-skewed by construction, which is a better match for a
        # count like this (and is exactly the same distribution shape
        # already used for total_two_party_votes below).
        n_counties = int(np.clip(RNG.lognormal(mean=np.log(50), sigma=0.9), 3, 260))

        state_swing_deviation = STATE_SWING_DEVIATION[state]
        religiosity = RELIGIOSITY[state]

        for i in range(n_counties):
            county_id = f"{state}-{i:03d}"

            # --- Vote totals -------------------------------------------------
            # Lognormal spread mimics the real world: most counties are
            # small, a few are enormous. This range (~a few hundred to
            # several hundred thousand) is what makes the WLS weighting
            # choice actually matter.
            n_2026 = int(np.clip(RNG.lognormal(mean=8.5, sigma=1.3), 300, 1_200_000))
            # Presidential turnout typically runs higher than a midterm
            # Senate race in the same county -- a modest, illustrative
            # multiplier, not a calibrated turnout model.
            n_2024 = int(n_2026 * RNG.uniform(1.1, 1.4))

            demographics = _draw_fake_demographics()
            row = _simulate_outcome(state_swing_deviation, religiosity, demographics, n_2024, n_2026, reference_votes)
            row["state"] = state
            row["county_id"] = county_id
            # Placeholders mirroring the real-data flags discussed for
            # Nebraska (Osborne-type independent counted into the "D"
            # numerator) and Montana (D + spoiler independent splitting the
            # anti-R vote). They're inert in the toy DGP -- just here so
            # the filtering / flag-passthrough logic in baseline_model.py
            # gets exercised on *something* before real data arrives.
            row["is_coalition_independent"] = 0
            row["excluded_multiparty_split"] = 0
            rows.append(row)

    df = pd.DataFrame(rows)

    # Mark one toy state as the "Nebraska-like" coalition-independent case,
    # and one as the "Montana-like" multiparty-split case, purely so the
    # downstream filtering code has a non-trivial case to filter.
    df.loc[df["state"] == states[-1], "is_coalition_independent"] = 1
    df.loc[df["state"] == states[-2], "excluded_multiparty_split"] = 1

    return df


DEMOGRAPHIC_COLUMNS = ["pct_college", "pct_black", "pct_hispanic", "pct_asian", "median_age", "median_income_k"]


def simulate_demographics_for_real_counties(real_counties_with_census: pd.DataFrame, reference_votes: int = 10_000) -> pd.DataFrame:
    """
    Simulate a "true" dem_swing on top of REAL counties with REAL ACS
    demographics (actual states, actual counties, real-or-proxy 2026
    vote totals AND real 2024 presidential vote totals/shares, merged
    with actual Census demographics -- see real_counties.py,
    fetch_census_demographics.py, build_pres_2024_county.py).

    This is a "plasmode" simulation design: the covariates (X) are 100%
    real, only the outcome (y = dem_swing) is simulated from TRUE_COEFS.
    That's deliberate -- if we simulated BOTH X and y, we'd just be back
    to inventing our own (weaker) approximation of real collinearity
    structure; if we used the REAL y too, there'd be no known truth left
    to validate the fitting code against, and this would just be the real
    model instead of a toy one. Plasmode designs split the difference:
    realistic covariates, with a known ground truth kept only on the
    outcome side. The VIF/multicollinearity diagnostics you get from
    running this are a genuine read on these states' real collinearity,
    not a guess.

    Expects real_counties_with_census to have columns: state, county_name,
    total_two_party_votes (2026), total_two_party_votes_2024, plus all of
    DEMOGRAPHIC_COLUMNS (i.e. real_counties.load_all_class2_states_county_frame()'s
    output, or load_real_county_skeleton() merged with the same sources).
    """
    state_swing_deviations = {
        state: STATE_SWING_DEVIATION[state] for state in real_counties_with_census["state"].unique()
    }

    rows = []
    for _, county in real_counties_with_census.iterrows():
        state = county["state"]
        n_2026 = int(county["total_two_party_votes"])
        n_2024 = int(county["total_two_party_votes_2024"])
        demographics = {col: county[col] for col in DEMOGRAPHIC_COLUMNS}

        row = _simulate_outcome(state_swing_deviations[state], RELIGIOSITY[state], demographics, n_2024, n_2026, reference_votes)
        row["state"] = state
        row["county_id"] = f"{state}-{county['county_name']}"
        row["is_coalition_independent"] = 0
        row["excluded_multiparty_split"] = 0
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = simulate_counties()
    out_path = "toy_counties.csv"
    data.to_csv(out_path, index=False)
    print(f"Simulated {len(data):,} counties across {data['state'].nunique()} states -> {out_path}")
    print(data.head())
