# Pipeline walkthrough: data, toy model, baseline model

What every file does, what's real vs. simulated at each step, the
assumptions baked in, and concrete refactoring opportunities for the
current implementation.

## 1. The shape of the pipeline

```
Census API ──────────────► fetch_census_demographics.py ──┐
                                                            │
tonmcg GitHub mirror ────► build_pres_2024_county.py ──────┤── data/raw/*.csv
                                                            │
NBC/DDHQ __NEXT_DATA__ ──► (manual/agent-fetched, raw) ─────┘
          │
          ▼
real_counties.py  (loads + cleans the data/raw/*_senate_2026.csv files,
                    merges everything together)
          │
          ▼
toy_data.py        (simulates a KNOWN outcome on top of real-or-fake
                     covariates, for validating the fitting code)
          │
          ▼
baseline_model.py   (the actual statistical model: fit, VIF, interaction
                      tests, clustering)
```

Four separate data-cleaning scripts feed one loading/merging module
(`real_counties.py`), which feeds a simulation module (`toy_data.py`), which
feeds the model (`baseline_model.py`). That chain is the main thing Section
4's refactoring notes are about.

## 2. Data layer — what's real, and how it got cleaned

### `fetch_census_demographics.py` — ACS demographics (100% real)

Pulls county- and state-level American Community Survey estimates for all 33
Class-2 Senate states via the Census API (one script, ~33 HTTP calls, no
agent dispatch — this was deliberately built as deterministic code instead of
research-agent exploration specifically to keep token cost low).

Variables: `pct_college` (B15003, bachelor's+), `pct_black`/`pct_hispanic`/
`pct_asian` (B03002, mutually exclusive with white as the implicit omitted
category), `median_age` (B01002), `median_income_k` (B19013), plus
`total_population` (B03002 total, used later as a turnout-proxy input).

County-name cleanup has to handle non-uniform county-equivalent naming
across states. Louisiana parishes and Alaska boroughs/census
areas/municipalities use different vocabulary than "County", so the code
splits on the last comma (removing the trailing state name regardless of
what precedes it) rather than matching one specific suffix word, then
strips a list of known suffix words (County/Parish/Borough/Census
Area/Municipality/City and Borough/city).

Virginia's independent cities use a *lowercase* `"city"` suffix in Census's
own `NAME` field (e.g. `"Suffolk city, Virginia"`), distinct from the
capitalized suffix words above. A complication: four Virginia places
(Fairfax, Franklin, Richmond, Roanoke) exist as *both* an independent city
and a differently-bordered same-named county, so stripping "city"
unconditionally would collide them onto one key. This is handled with a
general collision check (if stripping would produce a duplicate
`county_name` within a state, the city's suffix is restored instead of
stripped) rather than a hardcoded 4-name list, so it should self-correct if
another state or a future Census vintage introduces a similar case.

### `build_pres_2024_county.py` — 2024 presidential results (real, with one state dropped)

Cleans `tonmcg/US_County_Level_Election_Results_08-24` (an open GitHub mirror
of MEDSL's official county presidential returns — the real Harvard Dataverse
copy is gated behind a guestbook form that isn't scriptable, so this mirror
was used instead). **Alaska is dropped**: this source reports Alaska by State
House District (40 of them), not by borough/census area (~30) — a genuinely
different geography, confirmed by comparing row counts against the Census
pull before assuming anything, not just a naming quirk.

Output: `dem_votes_2024`, `rep_votes_2024`, `total_two_party_votes_2024`,
`dem_share_2024` per county, keyed by `(state, county_key)`.

### `real_counties.py` — 2026 Senate results (real, with real-world complications)

Loads `data/raw/{STATE}_senate_2026.csv` (gathered state-by-state: 5 states
via individual research-agent dispatch early in the project, the remaining
27 via one reusable NBC/DDHQ-JSON-scraping script dispatched as a single
subagent — a 10x+ difference in token cost between those two approaches for
the same kind of data). Classifies each row's party string into D/R/OTHER,
sums to two-party vote totals per county.

This is where the project's "Osborne flag" / "Montana-split" design concept
(from the very first model-spec conversation) actually had to be implemented
for real data, and it needed to be candidate-name-based, not party-string-based,
because Idaho has two independents sharing the identical `"ind"` label with
different coalition status:

| State | Situation | Handling |
|---|---|---|
| NE | No Democrat on ballot (withdrew post-primary). Dan Osborn (labeled `"other"`, not even `"ind"`) is the de facto opposition — the project's original prototype case. | `COALITION_INDEPENDENTS`: Osborn's votes → "D" |
| SD | No Democrat on ballot. Brian Bengs (SD's 2022 Dem nominee) running as `"ind"`. | Same: Bengs's votes → "D" |
| ID | No Democrat on ballot (withdrew in July). Two independents: Todd Achilles (33.5%, well-funded, the real challenger) and Natalie Fleming (6.9%, no FEC filings, a repeat minor candidate) — same party label, different coalition status. | Achilles → "D"; Fleming stays OTHER |
| MT | **Has** a real Democrat (Alani Bankhead) *and* a competing independent (Seth Bodnar) splitting the anti-R vote. | `EXCLUDED_MULTIPARTY_STATES`: whole race flagged `excluded_multiparty_split=1`, dropped from the main fit by default rather than guessing a combination rule |

A loud safety check in `load_real_county_skeleton()` raises if any state has
zero votes classified as "D" or "R" — this is what makes a missing-Democrat
case like Idaho's surface as an explicit error rather than silently
producing a wrong two-party total. It's the mechanism doing real work here,
not a formality.

Two loader functions exist with overlapping but inconsistent purposes (see
Section 4c):
- `load_real_counties_with_census()` — real election skeleton + Census
  demographics. Does **not** merge in 2024 presidential data, so on its own
  it can't produce a swing outcome (gap, see below).
- `load_all_class2_states_county_frame()` — Census demographics + a
  population-proportional *simulated* 2026 vote-total proxy + real 2024
  data. Written when only 5/33 states had real 2026 results; now that 32/33
  do, it's using a simulated proxy where real data already exists (gap, see
  below).

### Known unresolved gap: Maine/Massachusetts/New Hampshire/Rhode Island

These four states report election results by town/city, not county — they
don't administer elections by county at all. Census demographics are
county-keyed, so `county_key` values never match for these four (1,100+ rows
fail to merge). No crosswalk exists in this project yet. Currently handled
by exclusion at the call site, not inside `real_counties.py` itself — another
consistency gap (Section 4c).

## 3. The toy model (`toy_data.py`) — what's simulated, what's real, and why

**Purpose**: before trusting any coefficient the model produces on real data,
check whether the fitting code can recover a *known* answer. We pick the
true coefficients ourselves, generate data consistent with them, fit, and
compare recovered-vs-true. If that fails, the code is wrong — no need to
even look at real data yet.

### What's simulated (always)
- `TRUE_COEFS` — the ground-truth relationship between demographics and
  **swing** (not level — see below), picked to be large enough to spot
  easily in a recovery check. Includes one genuine interaction term
  (`college_x_black`, modeling the real "diploma divide" pattern: college's
  effect on swing is smaller where `pct_black` is high) layered on top of
  pure covariation (age↔college↔income correlation, separately modeled in
  `_draw_fake_demographics()` — covariation is a property of the X's joint
  distribution, interaction is a property of the true Y-generating formula,
  and the two are independent design choices, demonstrated by having one
  without implying the other).
- `STATE_SWING_DEVIATION` — a random per-state true effect (std=0.04,
  calibrated to the spread actually observed across the 5 real states with
  real 2026 results, +1.9 to +10.6 points). This represents exactly what the
  project wants left in the model's residual — state environment and
  candidate quality, not something to be fit as a covariate. Drawn randomly
  rather than from the real 5 states' observed means on purpose: those means
  already mix demographics + state effect + noise, so using them directly
  would double-count demographic signal into a term that's supposed to be
  demographics-independent.
- The outcome noise, scaled via `compute_swing_weight()` (see Section 4) so
  the simulated heteroskedasticity pattern matches exactly what the WLS fit
  assumes.

### What's real (when available)
- `_draw_fake_demographics()` is **only** called by `simulate_counties()`
  (the fully-synthetic 50-state path). `simulate_demographics_for_real_counties()`
  uses actual Census values for every covariate — this is a "plasmode"
  design: 100% real X, simulated Y only, which tests the fitting code
  against *real* collinearity structure instead of an invented
  approximation, while still having a known truth to validate against.
- `STATE_LEAN_2024` (real 2024 presidential results by state) and
  `RELIGIOSITY` (real Pew data) are loaded but **not used in the DGP math**
  — kept as reference/diagnostic data for a later stage (does the fitted
  model's residual correlate with religiosity?), not wired into either the
  true generating process or the fitted baseline. This reflects a
  deliberate design choice (see `state_effect_options.md`): including a
  state-level term in the *fitted* model — even a modest one like
  religiosity — recreates the problem it was meant to solve, by explaining
  away state-level signal that's supposed to stay in the residual for later
  analysis.

### Why "swing," not "level"

The outcome (`dem_swing`) is 2026 Senate dem-coalition share minus 2024
presidential dem two-party share, not a raw level. This was a deliberate
mid-session pivot: predicting a level requires *something* to absorb each
state's huge persistent baseline (Mississippi's baseline D share vs.
Vermont's isn't explainable by six demographic covariates alone), which is
exactly the state-FE-vs-residual tension above. Switching to swing
differences out persistent state culture automatically — you don't need a
state term for that purpose at all, because whatever caused a state's 2024
baseline mostly cancels out in the 2026-minus-2024 subtraction. The
remaining state-level variation *in the swing itself* is much closer to
"state environment + candidate quality this cycle," which is the actual
object of interest.

One structural limitation this doesn't fix: comparing a presidential
electorate (2024) to a midterm Senate electorate (2026) means part of every
county's swing reflects a generic, non-demographic, non-candidate
president-to-midterm dropoff pattern. It gets absorbed into the model's
intercept, indistinguishable from the genuine national mood shift — not
fixable without a second data source (e.g. a generic-ballot series), noted
as an accepted limitation, not a bug.

### `compute_swing_weight()` and the noise model

Because the outcome is now a *difference* of two proportions, its variance
is the sum of both years' sampling variance, not either one alone:
`Var(swing) = p(1-p) × (1/n_2024 + 1/n_2026)`, giving a weight of
`n_2024 × n_2026 / (n_2024 + n_2026)` — half the harmonic mean, dominated by
whichever year had fewer votes (a chain is as strong as its weakest link).
`toy_data.py`'s simulated noise uses this exact formula so the simulated
heteroskedasticity pattern matches what the WLS fit assumes — this is also
why the toy model doubles as a demonstration of why WLS beats OLS, not just
a validation tool.

### Flag passthrough

`simulate_demographics_for_real_counties()` passes `is_coalition_independent`
and `excluded_multiparty_split` through from the input frame when present
(e.g. from `real_counties.load_real_county_skeleton()`, which sets them for
NE/SD/ID/MT), defaulting to 0 only when the column is genuinely absent
(e.g. `load_all_class2_states_county_frame()`'s simulated-vote-total frame,
which carries no election-outcome flags at all). This matters because the
two loader functions in `real_counties.py` don't share a schema (see
Section 6c) — any code consuming either one needs to tolerate the flag
columns sometimes not being there.

## 4. The model (`baseline_model.py`)

### Specification

`dem_swing ~ pct_college + pct_black + pct_hispanic + pct_asian + median_age
+ median_income_k`, fit by WLS, **no state fixed effect and no religiosity
term by default** (`include_state_fe=False, include_religiosity=False`).
`pct_black`/`pct_hispanic`/`pct_asian` are county race/ethnicity shares with
white (+ other) as the implicit omitted reference category — including all
race shares plus an intercept would be the dummy-variable trap (perfect
multicollinearity, since the shares sum to ~1).

### Weighting

Default weight column is `swing_weight` (see `compute_swing_weight()`
above), not a raw vote count — this changed when the outcome changed from
level to swing, and the module docstring explicitly documents the
derivation so it doesn't silently drift back to the wrong formula.

### Clustering

`cluster_se=True` by default. The model's errors have a shared state-level
component by construction (`STATE_SWING_DEVIATION` hits every county in a
state together) — this is exactly what cluster-robust standard errors are
for. Empirically confirmed by repeated-draw simulation (documented in the
`fit_baseline_model` docstring): plain WLS standard errors understate true
sampling variability by 1.5–2x at every state count tried (5/15/32), and
this doesn't improve with more *counties* — only with more *states*, since
the effective sample size for a state-level shock is the state count, not
the county count. Clustering closes most of this gap once there are enough
clusters (~15+); with only 5 states it helps but is itself unreliable
(cluster-robust SEs are a large-number-of-clusters approximation).

`compare_interaction_model()`'s joint F-test is **not** cluster-aware even
when `cluster_se=True` — `compare_f_test()` computes its statistic from a
residual-sum-of-squares comparison that doesn't touch the clustered
covariance matrix at all (confirmed directly: statsmodels itself raises
`InvalidTestWarning: F test for comparison is likely invalid with robust
covariance` when you try). `test_single_interaction()`'s single-coefficient
test doesn't have this problem — its p-value comes straight from one
model's own (correctly clustered) covariance matrix, with no separate
RSS-based statistic to silently ignore it.

### Multiple comparisons

`compare_interaction_model()` scans all 15 pairwise interactions at once;
`test_single_interaction()` tests one pre-specified interaction.
Empirically, on this project's own toy data, the *true* `college_x_black`
interaction came back non-significant in the 15-term joint scan (diluted
power across 14 null terms) while a *spurious* interaction scored a better
p-value purely by chance — and the same true effect came back clearly
significant when tested alone. The lesson generalizes: decide which
interaction you're testing before looking at the data; don't fish through
all pairs and report whichever looks smallest.

## 5. Real vs. dummy, at a glance

| Data | Status |
|---|---|
| County demographics (college/race/age/income) | Real (ACS), for all 33 states except the 4 town-reporting ones can't merge yet |
| 2024 presidential county results | Real, 32/33 states (Alaska excluded — geography mismatch) |
| 2026 Senate county results | Real, 32/33 states (Alaska not attempted; MT flagged for exclusion, not missing) |
| State religiosity | Real (Pew), loaded but currently inert — diagnostic-only |
| `TRUE_COEFS` (demographic effects on swing) | **Fully invented** — picked to be recoverable, not realistic magnitudes |
| `STATE_SWING_DEVIATION` | **Fully invented** (random), calibrated in *scale* to real observed swings but not anchored to them |
| 2026 vote totals, for the states real returns exist for | Real |
| 2026 vote totals, for any remaining gap | Simulated (population-proportional proxy, not a turnout model) |
| The outcome `dem_swing`, in the toy table (`toy_data.py`) | Simulated — Y generated from `TRUE_COEFS`, for validating the fitting code |
| The outcome `dem_swing`, in the real table (`real_counties.load_real_swing_table()`) | **Real** — observed 2026 Senate minus 2024 presidential, per county. Fit for the first time this project: R²=0.43, n=2,241 counties/28 states, all demographic coefficients significant (p<0.05) except `median_income_k`. |

Update (see `class_architecture.md` Section 0 for the fuller status):
`load_real_swing_table()` closed the gap this section used to describe —
`simulate_demographics_for_real_counties()` is still accurately named for
what it does (simulates demographics *for* real counties, for the toy
table), but it's no longer true that the project has never fit a real
outcome. `main.py` now runs both the toy and real fits end to end.

## 6. Refactoring opportunities

Ranked roughly by how much current pain they're causing, not alphabetically.

### a. Model-fitting logic is duplicated three ways, and that's specifically what's blocking "models replaceable, don't affect analysis"

`fit_baseline_model()`, `compare_interaction_model()`, and
`test_single_interaction()` each independently: build a formula via
`_build_formula()`, assemble `cluster_kwds`, and call
`smf.wls(formula, data, weights=...).fit(**cluster_kwds)`. The formula
assembly is already shared (good); the actual *fitting call* is copy-pasted
three times. Concretely:

- Extract a single `_fit_wls(formula, df, weight_col, cluster_se) -> results`
  helper used by all three. Mechanical, low-risk, removes the duplication
  immediately.
- More ambitiously, to actually get "models replaceable without affecting
  analysis": right now, swapping WLS for anything else (a GLM/logit variant,
  a mixed-effects model, eventually something Bayesian) means editing all
  three functions, because they're not calling an abstraction — they're
  calling `smf.wls` directly, three times. Define a small model interface
  (even just a function signature contract: `fit(df, formula) -> an object
  exposing .params, .bse, .pvalues, .resid, .rsquared, .aic, .bic`) and have
  `compute_vif`, `compare_interaction_model`, `test_single_interaction` all
  consume that contract instead of statsmodels specifics. `compute_vif()` is
  already model-agnostic (it only touches raw predictor columns, never a
  fitted model) — worth noting as the one function in this file that
  already has the property the other three are missing.

### b. County/state-name normalization is duplicated and inconsistent across three files

`fetch_census_demographics.py` strips `County|Parish|Borough|Census
Area|Municipality|City and Borough|city` with a collision-detector for
same-named city/county pairs. `build_pres_2024_county.py` independently
strips only `County|Parish` with its own regex (works today only because
its source already distinguishes VA cities by name, not because of any
shared logic — the next state-level quirk in either source won't
automatically get the other's fix). `real_counties.py` does the simplest
version, bare `.str.lower().str.strip()`, relying entirely on upstream
sources already being clean.

Pull the suffix-stripping regex, the collision-detection logic, and the
`county_key` convention into one shared utility (e.g. `geo_utils.py`), used
by all three. Right now a new state-naming quirk discovered in one source
has to be independently rediscovered in the others.

### c. Two real-county loader functions with inconsistent, overlapping schemas

`load_real_counties_with_census()` and `load_all_class2_states_county_frame()`
both exist in `real_counties.py`, both produce "a county frame with
demographics," but:
- Only the second merges in 2024 presidential data (needed for swing) — the
  first can't currently feed `simulate_demographics_for_real_counties()` at
  all without a manual extra merge, since it never joins in 2024 data on
  its own.
- The second always uses the simulated turnout-proxy for 2026 votes, even
  for the 32 states that now have real 2026 returns on file — it doesn't
  check `data/raw/{STATE}_senate_2026.csv` before falling back to the proxy.
- Neither handles the ME/MA/NH/RI town-vs-county exclusion internally;
  callers have to filter those four states out manually before merging.

These should be unified into one function that: merges real 2026 returns
wherever they exist, falls back to the simulated proxy only where they
don't (soon to be nowhere, once ME/MA/NH/RI get a crosswalk), always merges
2024 data, and excludes/flags the known-incompatible states itself rather
than leaving that to every caller.

### d. Party-classification domain rules are mixed into the loading/merging mechanics

`DEM_PARTY_LABELS`, `REP_PARTY_LABELS`, `COALITION_INDEPENDENTS`,
`EXCLUDED_MULTIPARTY_STATES`, and `_classify_party()` are all in
`real_counties.py`, interleaved with the actual file-loading and
pivoting code. This is the piece most likely to need updates as more
states' real results come in (every new state can introduce a new label
spelling or a new coalition-independent case) — separating it into its own
module (e.g. `party_classification.py`) would isolate the
frequently-churning domain knowledge from the comparatively stable
load/merge mechanics, and make it the obvious single place to look when the
next Idaho-shaped surprise shows up.

### e. Scattered state reference data

FIPS codes (`fetch_census_demographics.py`'s `CLASS2_SENATE_STATE_FIPS`),
full-name lookups (`build_pres_2024_county.py`'s `NAME_TO_ABBR`, and a
separate `STATE_FULL_NAMES` inside `fetch_census_demographics.py`'s
`__main__` block), and the Class-2 state list itself (re-derived as
`STATE_ABBRS` in `toy_data.py`, independently excluding Alaska there too)
are three independent representations of overlapping information. A single
canonical table (abbr, full name, FIPS, Class-2 membership, swing-eligible
flag) would remove the risk of these silently drifting apart — e.g. if a
34th state's Senate seat flips to Class 2 in a future redistricting-adjacent
event, right now that would need updating in at least four places.

### f. `toy_data.py` mixes "real reference data" with "simulation logic"

`PRES_2024_VOTES`, `STATE_LEAN_2024`, and `RELIGIOSITY` are real external
data, independently useful outside the toy DGP (e.g. for the Section 3
diagnostic-on-residual step, or any future analysis). `TRUE_COEFS`,
`STATE_SWING_DEVIATION`, `_draw_fake_demographics()`, `_simulate_outcome()`
are simulation-specific and meaningless outside a validation context.
Splitting these into a `reference_data.py` (real, reusable) and a
`toy_dgp.py` (simulation-only, imports from `reference_data.py` for its real
anchors) would make it obvious at a glance which parts of this file anyone
should trust as "real" versus "made up for testing."

### g. Suggested directory layout, pulling the above together

```
ingest/
    fetch_census_demographics.py
    build_pres_2024_county.py
    real_counties.py           (split per (c)/(d) above)
    party_classification.py
    geo_utils.py
reference_data.py               (real, reusable: religiosity, 2024 results)
model/
    baseline_model.py           (formula + a swappable fit interface)
    toy_dgp.py                  (simulation-only)
analysis/
    diagnostics.py              (VIF, interaction tests — consume the fit
                                  interface, never statsmodels directly)
```

None of this is urgent — the current flat layout hasn't broken anything —
but it directly addresses the stated goal: data cleaning isolated from
modeling, and the model swappable without the analysis tools needing to
change at all.
