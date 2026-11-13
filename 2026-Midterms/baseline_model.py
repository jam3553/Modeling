"""
Baseline cross-sectional model: Democratic-coalition two-party vote share
explained by county demographics (college share, race/ethnicity shares,
median age, median income), plus either state fixed effects OR a single
state-level religiosity covariate (see include_state_fe / include_religiosity
below) -- not both, and the choice matters for what this model can honestly
be used for.

WHY THIS CHOICE EXISTS -- state fixed effects vs. residual:
A saturated C(state) fixed effect absorbs EVERYTHING specific to a state --
and, crucially, forces the WITHIN-STATE AVERAGE RESIDUAL to ~0 by
construction (that's what "saturated" means: the fixed effect exactly
matches each state's mean). That's fatal for this project's actual goal:
the residual is supposed to BE the state-environment-plus-candidate-quality
signal that later analysis decomposes. If state FE is included, that
signal isn't in the residual anymore -- it's been absorbed into the FE
parameter, where it's much less directly usable (an injected true
candidate-quality term in the toy DGP correlated almost identically with
the fitted state-FE coefficient, 0.513, and with the no-FE model's
state-mean residual, 0.519 -- same information, different container, but
only the residual is the thing later analysis is built around). So state
FE is OFF by default, and so is the religiosity covariate: pulling
religiosity out as a baseline regressor re-creates the same problem on a
smaller scale (prematurely explaining away state-level signal that should
stay in the residual for later study). Religiosity's real role is as a
DIAGNOSTIC applied to the residual afterward -- see state_effect_options.md
-- not as something fit into the baseline itself. Both flags remain
available for that kind of secondary analysis.

This module is deliberately written against a plain pandas DataFrame with a
fixed set of expected column names, so the exact same code runs on
toy_counties.csv today and on real 2026 county data later -- only the input
file changes.

Expected columns
----------------
dem_swing                 : float, the outcome variable -- 2026 Senate
                             dem-coalition share MINUS 2024 presidential
                             dem two-party share, per county. Can be
                             negative. (Previously this was a raw level,
                             "dem_coalition_share"; renamed when the
                             project switched to a swing outcome -- see
                             conversation history for why.)
total_two_party_votes    : int, 2026 two-party vote count (kept for
                             reference/display; NOT the WLS weight by
                             default anymore -- see weight_col below)
swing_weight              : float, compute_swing_weight(n_2024, n_2026)
                             per county -- the actual default WLS weight
                             now that the outcome is a difference, not a
                             single-year level (see compute_swing_weight's
                             docstring for the full derivation)
pct_college               : float in [0, 1]
pct_black                 : float in [0, 1]
pct_hispanic              : float in [0, 1]
pct_white                 : float in [0, 1]
median_age                : float, years
median_income_k           : float, median household income in $1,000s
state                      : str, two-letter state abbreviation
religiosity                : float, % "highly religious" in that state
                              (only required when include_religiosity=True)
is_coalition_independent  : 0/1 flag (currently just carried through, not
                             yet used as a regressor -- see note in
                             fit_baseline_model)
excluded_multiparty_split : 0/1 flag; rows with 1 are dropped before fitting
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Reference categories, spelled out once so it's obvious what every
# coefficient is measured RELATIVE TO.
#
# THEORY -- the "dummy variable trap":
# Race/ethnicity shares in a county sum to ~1 (white + black + hispanic +
# asian + AIAN + NHPI + some-other-race + two-or-more-races = 1, per the
# Census's own B03002 breakdown -- see fetch_census_demographics.py). If
# you put every one of those shares into a regression alongside an
# intercept, the columns are perfectly linearly dependent (any one share
# = 1 - sum of the rest), so the design matrix is singular and OLS/WLS
# can't even be computed (statsmodels will usually drop one automatically
# and warn, but better to make the choice explicitly). The fix is to
# always omit at least one category and interpret every included race
# coefficient as "relative to" whatever's omitted.
#
# Here pct_black, pct_hispanic, and pct_white are explicit, so the omitted
#/reference category is everyone else: Asian alone, American Indian/Alaska
# Native alone, Native Hawaiian/Pacific Islander alone, some-other-race
# alone, and two-or-more-races, all non-Hispanic. That is NOT "Asian" on
# its own -- it's a multi-group leftover bucket, same caveat as the one
# that used to apply to "white" before pct_white was added explicitly (see
# fetch_census_demographics.py's docstring for county-level examples of
# how large and non-uniform this kind of leftover bucket can get). The
# same omit-one-category logic applies to any set of shares that sum to 1
# (e.g. age buckets), which is why median_age is used here as a single
# continuous variable instead of a full set of age-bucket shares -- no
# omission needed for a single continuous regressor.
BASE_PREDICTORS = [
    "pct_college",
    "pct_black",
    "pct_hispanic",
    "pct_white",
    "median_age",
    "median_income_k",
]

# EXTENDED_PREDICTORS -- BASE_PREDICTORS plus two more hypotheses:
#
#   dem_share_2024: the county's own 2024 presidential Dem two-party share,
#       as a CONTINUOUS predictor of dem_swing (2026 minus 2024). Tests
#       "redder places swung more D this time" -- i.e. a negative
#       coefficient here.
#
#       CAVEAT, worth internalizing before reading anything into that
#       coefficient's sign: dem_swing is LITERALLY share_2026 -
#       share_2024, so Cov(dem_swing, share_2024) =
#       Cov(share_2026, share_2024) - Var(share_2024). That difference is
#       negative whenever share_2024 isn't a PERFECT predictor of
#       share_2026 (Cov(share_2026, share_2024) <= sqrt(Var(share_2026) *
#       Var(share_2024)) <= Var(share_2024) unless the two variances are
#       equal and the correlation is exactly 1) -- which it never will be,
#       since candidates, turnout, and the national environment all differ
#       between a presidential year and a midterm. So a negative
#       coefficient on dem_share_2024 is the MECHANICALLY EXPECTED sign
#       here, not evidence of some real "overcorrection"/mean-reversion
#       behavioral effect. Don't oversell it.
#
#   incumbent_status: a 3-level categorical (D_incumbent / R_incumbent /
#       no_incumbent), entered via C(..., Treatment(reference=...)) with
#       "no_incumbent" as the explicit reference category (patsy's default
#       alphabetical-first reference would otherwise pick "D_incumbent",
#       since uppercase D/R sort before lowercase "no_incumbent" in ASCII
#       -- Treatment(reference=...) overrides that). See
#       data/raw/incumbency_2026.csv and real_counties.py's
#       load_real_swing_table_with_incumbency() for how this column is
#       built and merged in; "no_incumbent" covers both genuinely open
#       seats and sitting senators who were APPOINTED rather than elected
#       to that seat (and so lack the electoral incumbency advantage the
#       other two levels are meant to capture).
EXTENDED_PREDICTORS = BASE_PREDICTORS + [
    "dem_share_2024",
    "C(incumbent_status, Treatment(reference='no_incumbent'))",
]


def _build_formula(predictors, include_state_fe: bool = False, include_religiosity: bool = False) -> str:
    rhs = " + ".join(predictors)
    if include_state_fe:
        # C(state) tells statsmodels' formula API to expand `state` into a
        # full set of dummy variables (one per state, one dropped
        # automatically as the reference) rather than treating it as a
        # number. This is the fixed-effects term: it gives every state its
        # own intercept, absorbing whatever is specific to that state (its
        # baseline partisan lean, its own political history, anything NOT
        # captured by the five demographic covariates) so that those
        # omitted state-level factors don't get wrongly attributed to the
        # demographic coefficients (omitted variable bias). See
        # state_effect_options.md: religiosity was tried as a replacement
        # for exactly this reason (avoiding candidate-quality
        # contamination), but turned out not to help with only one
        # election cycle of data -- so state FE is back as the default
        # until multiple cycles are available to actually exploit a
        # slow-moving covariate's advantage.
        rhs += " + C(state)"
    if include_religiosity:
        rhs += " + religiosity"
    return f"dem_swing ~ {rhs}"


def compute_swing_weight(n_2024, n_2026):
    """
    The correct WLS weight when the outcome is a SWING (dem_swing =
    share_2026 - share_2024), as opposed to a single-year level.

    THEORY -- extending the single-proportion variance formula to a difference:
    For one year's share, Var(share_c) = p_c(1-p_c)/n_c (see
    fit_baseline_model's docstring for the full derivation of that piece).
    A swing is the DIFFERENCE of two such proportions -- one from 2024,
    one from 2026 -- each with its own sample size. For two random
    variables, Var(A - B) = Var(A) + Var(B) - 2*Cov(A, B). Treating the two
    elections as independent draws (a simplification -- the same places
    are voting both times, so there's probably some real covariance, but
    there's no clean way to estimate it from vote totals alone, so this is
    the standard simplifying assumption, not a claim that it's exactly
    true) sets Cov(A, B) = 0, leaving:

        Var(swing_c) = Var(share_2026,c) + Var(share_2024,c)
                     ~ p(1-p)/n_2026,c + p(1-p)/n_2024,c
                     = p(1-p) * (1/n_2026,c + 1/n_2024,c)

    Using the same "treat p(1-p) as roughly constant" simplification as
    the single-year case, the WLS weight (inverse variance, up to that
    constant) is:

        weight_c = 1 / (1/n_2026,c + 1/n_2024,c) = (n_2026,c * n_2024,c) / (n_2026,c + n_2024,c)

    This is exactly HALF the harmonic mean of n_2024 and n_2026 (the
    harmonic mean of two numbers a, b is 2ab/(a+b); this is ab/(a+b)).
    Intuition for why it's not just "add the two n's together" or
    "average them": a county that's small in EITHER year is a noisy
    estimate in that year no matter how large the other year's n is --
    the combined weight is dominated by whichever year had fewer votes,
    the way a chain is only as strong as its weakest link. Plug in
    n_2024 = n_2026 (equal sample sizes both years) and this reduces to
    n/2 -- exactly half the precision of a single-year weight using the
    same n, which makes sense: you're now estimating a DIFFERENCE of two
    noisy things instead of one noisy thing, so you've lost precision
    even before any real signal shows up.
    """
    return (n_2026 * n_2024) / (n_2026 + n_2024)


def fit_baseline_model(
    df: pd.DataFrame,
    predictors=None,
    include_state_fe: bool = False,
    include_religiosity: bool = False,
    use_weights: bool = True,
    weight_col: str = "swing_weight",
    cluster_se: bool = True,
):
    """
    Fit the baseline WLS (or OLS) model and return the fitted statsmodels
    results object.

    THEORY -- why weight by total_two_party_votes at all?
    -------------------------------------------------------
    dem_coalition_share for county c is, underneath, a sample proportion:
    it's the fraction of n_c individual two-party votes that went to the
    Democratic-coalition candidate. For a sample proportion with true
    probability p_c and n_c independent trials, basic probability theory
    gives:

        Var(observed share_c) = p_c * (1 - p_c) / n_c

    So counties with fewer total votes are NOISIER estimates of "what that
    county's demographics imply," purely from sampling variance -- nothing
    to do with the county being unusual. Plain OLS assumes every
    observation has the SAME error variance (homoskedasticity), which is
    false here: it would let a 400-vote county and a 400,000-vote county
    pull the fit equally hard, even though the small county's share is
    much less informative.

    The Gauss-Markov theorem's generalization (sometimes called Aitken's
    theorem) says that when Var(error_c) = sigma^2 / w_c for some known
    weights w_c, the BLUE (best linear unbiased estimator -- i.e. the
    minimum-variance unbiased estimator among linear estimators) is
    Weighted Least Squares using exactly those weights w_c. Since
    Var(share_c) ~ p(1-p)/n_c, and p(1-p) is bounded and doesn't vary
    enormously across counties (it's at most 0.25, typically not near its
    max or min), the standard simplification is to treat p(1-p) as
    roughly constant and weight by w_c = n_c = total_two_party_votes.
    That's exactly the `weights=` argument passed to smf.wls below.

    NOW THAT THE OUTCOME IS A SWING, NOT A LEVEL: this single-proportion
    formula is no longer quite right, because dem_swing is a DIFFERENCE of
    two proportions (2026 minus 2024), each with its own sample size --
    see compute_swing_weight() above for the full derivation. The
    `weight_col` parameter exists for exactly this: pass a column already
    holding compute_swing_weight(n_2024, n_2026) per county (that's what
    toy_data.py now does) rather than a single year's raw vote count.

    IMPORTANT CAVEAT (worth internalizing, not just accepting): this
    weighting scheme assumes the *only* source of error is binomial
    sampling noise from aggregating individual votes. In reality, a chunk
    of the residual is structural -- omitted variables, local effects not
    captured by our five covariates -- not sampling noise at all. The
    weighting is a reasonable, standard approximation, not a literal
    description of reality. It will usually still beat unweighted OLS
    because real demographic data IS noisier for small counties, but don't
    treat the n_c-weighting as some kind of proven-correct model of the
    error term.

    Note also: WLS does NOT make biased estimates less biased. If the
    underlying model is right, both OLS and WLS point estimates will be
    unbiased and similar on average. The benefit of WLS here is
    EFFICIENCY (smaller standard errors) and correctly-sized confidence
    intervals, not bias correction.

    THEORY -- cluster_se: why the plain WLS standard errors above are
    still wrong, and clustering (not more data) is the fix.
    --------------------------------------------------------------------
    Everything above assumes each COUNTY's error is an independent draw.
    But this project's own design puts a shared, state-level shock into
    every county in the same state (STATE_SWING_DEVIATION in toy_data.py
    -- and in reality, whatever drives a state's environment/candidate
    effect this cycle hits every county in that state together, not
    independently). When errors are correlated WITHIN a group and only
    assumed-independent ACROSS groups, that's exactly the setup clustered
    standard errors are built for.

    Empirically confirmed on this project's own toy data (conversation
    history): running the fit on hundreds of fresh random draws and
    comparing the TRUE spread of the recovered pct_college coefficient
    across draws to the AVERAGE plain (unclustered) standard error the
    model itself reported:

        n_states=5:  empirical SD=0.074 vs. reported SE=0.037 (half!)
        n_states=15: empirical SD=0.036 vs. reported SE=0.020 (~55%)
        n_states=32: empirical SD=0.021 vs. reported SE=0.013 (~65%)

    At every scale, the plain SE understates the true uncertainty by
    roughly 1.5-2x. Critically, THIS DOESN'T GO AWAY WITH MORE COUNTIES --
    going from 5 to 32 states helped (the effective sample size for a
    state-level shock is the STATE count, not the county count: 2,000+
    counties is still only ~30-some independent draws of that shock), but
    the understatement ratio barely budged. More counties within the same
    states doesn't fix it either, for the same reason: they all share the
    same state-level shock, so they aren't new independent information
    about it.

    The mechanics: statsmodels' cov_type="cluster" computes a different
    variance-covariance matrix for the coefficients -- instead of
    assuming a single scalar error variance shared by every observation,
    it allows errors to be correlated in an unrestricted way WITHIN each
    cluster (here, each state) while still assuming independence ACROSS
    clusters. This produces the same point estimates (clustering doesn't
    change what's being estimated, only how uncertain we say we are about
    it) but correctly WIDER standard errors and confidence intervals.
    Rule of thumb worth knowing: clustered SEs are themselves a
    large-number-of-clusters approximation, and get unreliable with very
    few clusters (rough guideline: at least ~20-30); with 32-33 states,
    this project is right at the edge of comfortable, not deep into
    safe territory.
    """
    predictors = predictors or BASE_PREDICTORS

    # Drop rows flagged as non-standard multiparty splits (the
    # Montana-type case discussed earlier) -- summing a D and a spoiler
    # independent's votes together would conflate two different choices
    # voters actually faced, so for this baseline pass we exclude them
    # rather than guess at a combination rule.
    fit_df = df.loc[df["excluded_multiparty_split"] == 0].copy()

    # NOTE on is_coalition_independent: right now this flag is only used
    # upstream, to decide how dem_swing itself was constructed (whether an
    # independent's votes were folded into the "D" numerator). It is
    # intentionally NOT yet included as a regressor here. Once there's
    # more than one or two such races, it's worth testing as its own
    # covariate -- i.e. does running as "I" instead of "D" carry a brand
    # effect net of demographics? For now there's nowhere near enough
    # variation in this flag to estimate that separately.

    formula = _build_formula(predictors, include_state_fe=include_state_fe, include_religiosity=include_religiosity)
    weights = fit_df[weight_col] if use_weights else np.ones(len(fit_df))

    model = smf.wls(formula, data=fit_df, weights=weights)
    if cluster_se:
        # groups=fit_df["state"] -- see the THEORY note above for why
        # clustering by state (not just using more data) is the actual fix.
        return model.fit(cov_type="cluster", cov_kwds={"groups": fit_df["state"]})
    return model.fit()


def compute_vif(df: pd.DataFrame, predictors=None) -> pd.DataFrame:
    """
    Variance Inflation Factors for the continuous/share predictors
    (deliberately excluding the state fixed-effect dummies -- see note
    below).

    THEORY -- what VIF actually measures:
    For predictor j, run an auxiliary regression of x_j on ALL the other
    predictors, and get its R-squared (call it R_j^2). Then:

        VIF_j = 1 / (1 - R_j^2)

    Intuition: if x_j is well explained by the other predictors (R_j^2
    close to 1), VIF_j blows up -- the data can't tell x_j's effect apart
    from some combination of the other predictors' effects, so its
    coefficient estimate becomes unstable (large standard error) even
    though the overall model might fit the data fine. A commonly used (if
    somewhat arbitrary) rule of thumb: VIF > 5 deserves a look, VIF > 10
    is a real problem. There's nothing magic about those thresholds --
    they're conventions, not theorems.

    Why exclude the state dummies here: with ~49 state dummy columns, VIF
    against "all other predictors" becomes dominated by collinearity among
    the dummies themselves (which is structural and expected -- they're
    mutually exclusive categories), and isn't the multicollinearity
    question we actually care about, which is whether pct_college,
    median_income_k, etc. are too entangled WITH EACH OTHER to separate.
    """
    predictors = predictors or BASE_PREDICTORS
    X = df[predictors].copy()
    X.insert(0, "const", 1.0)  # VIF formula requires an intercept column

    vifs = {
        col: variance_inflation_factor(X.values, i)
        for i, col in enumerate(X.columns)
        if col != "const"
    }
    return pd.Series(vifs, name="VIF").sort_values(ascending=False).to_frame()


def compare_interaction_model(df: pd.DataFrame, predictors=None, include_state_fe: bool = False, include_religiosity: bool = False, weight_col: str = "swing_weight", cluster_se: bool = True):
    """
    Test whether pairwise interactions among the demographic predictors
    improve on the main-effects-only model, using nested-model comparisons
    rather than eyeballing R-squared.

    THEORY -- why not just compare R-squared?
    Adding ANY extra column to a regression can only increase (never
    decrease) R-squared, even if that column is pure noise -- more
    parameters mechanically give the model more ways to fit the training
    data. So "the interaction model has higher R-squared" tells you
    nothing by itself. Two better tools:

      1. An F-test for nested models: formally tests whether the
         reduction in (weighted) residual sum of squares from adding the
         interaction terms is larger than you'd expect by chance alone,
         given how many extra parameters you added. statsmodels exposes
         this via `results.compare_f_test(restricted_results)`.
      2. AIC/BIC: both penalize additional parameters directly (BIC
         penalizes more heavily than AIC), so a lower AIC/BIC after adding
         interactions is a more honest signal of genuine improvement than
         R-squared.

    THEORY -- why center the variables before forming interaction terms?
    An interaction term like pct_college * median_income_k is
    mechanically correlated with its own main effects (pct_college and
    median_income_k) even in data with NO real collinearity problem --
    this is sometimes called "structural" or "non-essential"
    multicollinearity, and it's an artifact of how the product is formed,
    not a real data problem. Centering each variable (subtracting its
    mean) before multiplying substantially reduces this artificial
    inflation, so that VIF on the interaction term reflects genuine
    collinearity rather than this mechanical effect.

    CAVEAT -- this function is a SCREENING tool, not a confirmatory test,
    and its joint F-test can miss a single real interaction even when one
    is there. With 6 predictors there are 15 pairwise interactions; if
    only one is genuinely non-zero, the joint test has to detect a signal
    diluted across 14 pure-noise terms, which costs real power. Worse,
    some of those 14 noise terms will look "significant" on their own
    just by chance (with 15 tests at the usual 0.05 threshold, you'd
    expect ~0.75 false positives on average even if NOTHING were real).
    Confirmed on this project's own toy data: with a true
    college-x-black interaction of -0.15 baked into the DGP, the 15-term
    joint test came back non-significant (p=0.094) in one run, and a
    completely spurious black-x-asian interaction scored a "better"
    individual p-value (0.021) than the real one (0.087) purely by
    chance. If you have a SPECIFIC, theory-motivated interaction in mind
    (as opposed to "let's see what turns up"), test it on its own instead
    -- see test_single_interaction() below, which recovered the same true
    effect far more precisely and significantly (p=0.008) by not paying
    the multiple-comparisons tax for 14 interactions nobody hypothesized.

    CLUSTERING GAP (see fit_baseline_model's cluster_se docstring for the
    full theory): the underlying fits below DO use cluster_se, so
    main_results/full_results' individual coefficient SEs and p-values are
    properly state-clustered. The joint F-TEST is NOT -- compare_f_test()
    computes its statistic from a residual-sum-of-squares comparison that
    doesn't incorporate the clustered covariance matrix at all, regardless
    of what cov_type the two fits used. A statistically correct
    cluster-robust joint test would need a Wald test against the clustered
    covariance matrix (statsmodels' results.wald_test() with an explicit
    restriction matrix for "all 15 interaction coefficients = 0"), which
    is enough extra work that it wasn't implemented here -- treat
    f_test's p-value with the same "probably overconfident" caveat as an
    unclustered SE, on top of the multiple-comparisons caveat above.
    """
    predictors = predictors or BASE_PREDICTORS
    fit_df = df.loc[df["excluded_multiparty_split"] == 0].copy()

    # Center continuous predictors for the reasons above.
    centered_cols = []
    for col in predictors:
        centered_name = f"{col}_c"
        fit_df[centered_name] = fit_df[col] - fit_df[col].mean()
        centered_cols.append(centered_name)

    main_formula = _build_formula(centered_cols, include_state_fe=include_state_fe, include_religiosity=include_religiosity)
    weights = fit_df[weight_col]
    cluster_kwds = {"cov_type": "cluster", "cov_kwds": {"groups": fit_df["state"]}} if cluster_se else {}
    main_results = smf.wls(main_formula, data=fit_df, weights=weights).fit(**cluster_kwds)

    # All pairwise interactions among the centered demographic predictors.
    interaction_terms = [
        f"{a}:{b}"
        for i, a in enumerate(centered_cols)
        for b in centered_cols[i + 1 :]
    ]
    full_formula = _build_formula(centered_cols + interaction_terms, include_state_fe=include_state_fe, include_religiosity=include_religiosity)
    full_results = smf.wls(full_formula, data=fit_df, weights=weights).fit(**cluster_kwds)

    f_test = full_results.compare_f_test(main_results)
    # compare_f_test returns (F statistic, p-value, df_diff) -- NOT
    # cluster-aware even when cluster_se=True; see CLUSTERING GAP above.

    summary = pd.DataFrame(
        {
            "model": ["main_effects_only", "with_interactions"],
            "n_params": [main_results.df_model + 1, full_results.df_model + 1],
            "aic": [main_results.aic, full_results.aic],
            "bic": [main_results.bic, full_results.bic],
            "r_squared": [main_results.rsquared, full_results.rsquared],
        }
    )

    return {
        "main_results": main_results,
        "full_results": full_results,
        "f_test": {"F": f_test[0], "p_value": f_test[1], "df_diff": f_test[2]},
        "summary": summary,
    }


def test_single_interaction(df: pd.DataFrame, var_a: str, var_b: str, predictors=None, include_state_fe: bool = False, include_religiosity: bool = False, weight_col: str = "swing_weight", cluster_se: bool = True):
    """
    Test ONE specific, theory-motivated interaction (var_a x var_b) against
    the main-effects model, instead of scanning all pairwise interactions
    at once (see the CAVEAT in compare_interaction_model's docstring for
    why that matters: a real interaction can get buried, and a noise
    interaction can look "significant," once you're testing 15 of them
    together).

    This is the right tool when you have a specific hypothesis in hand --
    e.g. "college's effect on vote share should be smaller where pct_black
    is high" (the "diploma divide" is concentrated among white voters) --
    rather than a generic "let's see what interactions exist" scan.
    Pre-registering which interaction you're testing, before looking at
    the data, is what gives you the right to interpret a single p-value
    at face value; if you instead picked var_a/var_b AFTER eyeballing
    which pairwise interaction came back smallest in
    compare_interaction_model(), you're back to the multiple-comparisons
    problem, just with an extra step.

    Unlike compare_interaction_model's joint F-test, cluster_se here is
    fully valid, not just a partial fix: coefficient/std_err/p_value all
    come straight from one fitted model's own (optionally clustered)
    covariance matrix -- there's no separate RSS-based test statistic
    that could silently ignore it, the way compare_f_test does.
    """
    predictors = predictors or BASE_PREDICTORS
    fit_df = df.loc[df["excluded_multiparty_split"] == 0].copy()

    centered_cols = []
    for col in predictors:
        centered_name = f"{col}_c"
        fit_df[centered_name] = fit_df[col] - fit_df[col].mean()
        centered_cols.append(centered_name)

    formula = _build_formula(centered_cols + [f"{var_a}_c:{var_b}_c"], include_state_fe=include_state_fe, include_religiosity=include_religiosity)
    cluster_kwds = {"cov_type": "cluster", "cov_kwds": {"groups": fit_df["state"]}} if cluster_se else {}
    results = smf.wls(formula, data=fit_df, weights=fit_df[weight_col]).fit(**cluster_kwds)

    term = f"{var_a}_c:{var_b}_c"
    return {
        "results": results,
        "coefficient": results.params[term],
        "std_err": results.bse[term],
        "p_value": results.pvalues[term],
    }


if __name__ == "__main__":
    from toy_data import simulate_counties, TRUE_COEFS

    data = simulate_counties()
    true_demo = pd.Series(TRUE_COEFS).drop(["intercept", "college_x_black"])

    print("=" * 70)
    print("Fitting baseline WLS model (demographics only, no state FE) on synthetic (toy) county data...")
    print("=" * 70)
    results = fit_baseline_model(data, use_weights=True)

    print("\n--- Recovered vs. true coefficients ---")
    comparison = pd.DataFrame(
        {
            "true": true_demo,
            "recovered": results.params.reindex(true_demo.index),
            "std_err": results.bse.reindex(true_demo.index),
        }
    )
    print(comparison)

    print("\n--- OLS (unweighted) for comparison: point estimates should be similar, ")
    print("    but standard errors below are NOT directly comparable to WLS's (different ")
    print("    weighting of the same data) -- the real lesson is in the residual diagnostics, ")
    print("    not this side-by-side. ---")
    ols_results = fit_baseline_model(data, use_weights=False)
    print(ols_results.params.reindex(true_demo.index))

    print("\n--- Variance Inflation Factors ---")
    print(compute_vif(data))

    print("\n--- Interaction model comparison ---")
    interaction_results = compare_interaction_model(data)
    print(interaction_results["summary"])
    print(
        f"\nF-test (interactions vs. main effects only): "
        f"F={interaction_results['f_test']['F']:.3f}, "
        f"p={interaction_results['f_test']['p_value']:.4f}, "
        f"df_diff={interaction_results['f_test']['df_diff']}"
    )
