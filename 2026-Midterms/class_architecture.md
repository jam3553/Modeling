# Class architecture for the ingest / model / analysis refactor

Covers the folder layout, every new class, how data flows through them,
and judgment calls made along the way. **Partially implemented** --
Section 0 tracks exactly what's real code today vs. still just design, so
a future agent reading this doesn't have to guess or re-derive it.

## 0. Implementation status

Implemented (real files, verified working):
- `county_utils.py` -- the shared county-name normalization utility from
  Section 3/6b, as plain functions (`strip_county_suffix`, `county_key`),
  not yet the full `County` dataclass shown below. Used by both
  `fetch_census_demographics.py` and `build_pres_2024_county.py`, which
  previously had independently-duplicated copies of this logic.
- `candidate.py` -- `Party` (enum) and `Candidate` (frozen dataclass),
  matching Section 3's design closely, including `effective_party`,
  `is_coalition_independent`, and a `coalition_states()` classmethod.
  Includes the Alaska ranked-choice-voting caveat as a module docstring
  (see Section 10 below, which this supersedes/cross-references rather
  than duplicates). Wired into `real_counties.py`, replacing its former
  inline `DEM_PARTY_LABELS`/`_classify_party`.
- `real_counties.py` gained `load_real_swing_table()` -- not in the
  original design below. It's the first function in this project to
  compute a REAL (not simulated) `dem_swing`: merges the real 2026
  skeleton, Census demographics, and 2024 presidential data, excluding
  Maine/Massachusetts/New Hampshire/Rhode Island (town-level mismatch,
  Section 10) and Alaska (never fetched). Currently covers 2,241 counties
  across 28 states.
- `main.py` -- a pragmatic orchestration script (download → toy data →
  real data → fit both → diagnostics), NOT the `ingest`/`model`/`analysis`
  package split below. It's today's actual entry point; the package split
  is still the target, not yet started.
- **The real model has been fit for the first time this project**: R² =
  0.43 on the 2,241/28 real table, all six demographic coefficients
  significant (p < 0.05) except `median_income_k`. `pipeline_walkthrough.md`
  Section 5 is stale on this point and needs updating alongside this doc.

Still design only, no code yet: `State`, `Cycle`/`ElectionOffice`,
`SenateRace`, the `schema/`/`ingest/`/`model/`/`analysis/` folder split,
`ModelSpec`, `Estimator`/`WLSEstimator`, `SwingSimulator`. Everything below
describes the target design for these; treat class names/file paths for
them as proposed, not existing, until this section says otherwise.

## 1. Design goals, as stated

- `ingest/`, `model/`, `analysis/`, and a `data/` folder for the files
  themselves (code and data physically separated).
- Every ingestion script: download → clean → save a CSV with consistent
  columns. One combining script merges all per-source CSVs into one big
  county-level table.
- Census is the canonical source of *which counties exist* — every other
  source's county list is matched against Census's, not treated as
  independently authoritative.
- The NBC live-vote-count fetch becomes a deterministic, permanent
  ingestion script, not a one-off agent task.
- Toy and real model tables share **one schema**. The toy table is the
  combined table with `dem_swing`/`swing_weight` replaced by simulated
  values; the real table is the combined table restricted to counties
  where those columns are genuinely observed. Same baseline-model code
  runs on both; both feed the same diagnostics.
- Fitting should be swappable without touching the analysis/diagnostic
  code (the (a) decision below).
- `County` and `Party` are first-class types, not bare strings/columns.

## 2. Folder layout

```
2026-Midterms/
    data/
        external/                 raw downloads, unprocessed (as today)
        raw/                      per-source cleaned CSVs (as today, but
                                   every file now conforms to one of the
                                   Row schemas in schema/rows.py)
        combined/                 NEW: combined_counties.csv, one row per
                                   county, output of ingest/combine.py
        model_tables/              NEW: real_counties.csv and
                                   toy_counties.csv, both conforming to
                                   the same CombinedCountyRow schema

    schema/                        NEW: classes from Sections 3-4, shared
                                   by ingest/ and model/, depends on neither
        state.py                   State
        county.py                   County
        party.py                    Party, Candidate
        senate_race.py              Cycle, ElectionOffice, SenateRace
        rows.py                     CensusDemographicsRow, Pres2024Row,
                                   Senate2026Row, CombinedCountyRow

    ingest/
        census_demographics.py      was fetch_census_demographics.py
        pres_2024.py                 was build_pres_2024_county.py
        senate_2026_nbc.py           NEW: generalized, deterministic NBC
                                   scraper (formalizes the one-off
                                   subagent script from this project's
                                   data-gathering phase)
        combine.py                   NEW: merges data/raw/*.csv into
                                   data/combined/combined_counties.csv

    model/
        spec.py                      ModelSpec
        estimator.py                  Estimator (ABC), WLSEstimator,
                                   compute_swing_weight
        toy_dgp.py                     TRUE_COEFS, SwingSimulator

    analysis/
        vif.py
        interactions.py                compare_interaction_model,
                                   test_single_interaction — rewritten
                                   against Estimator instead of
                                   statsmodels directly
```

`state_effect_options.md` and `pipeline_walkthrough.md` stay where they are
(documentation, not code).

## 3. Value/identity classes

These represent *things*, not tables. They carry identity and behavior
(normalization, classification), and are used during ingestion/combination;
the on-disk CSVs and the DataFrames the model actually fits against stay
plain strings/numbers throughout, for compatibility with pandas and
statsmodels. (See Section 6 for why this split, rather than storing object
instances inside DataFrame cells.)

### `State`

```python
@dataclass(frozen=True)
class State:
    abbr: str                    # "VA"
    full_name: str                # "Virginia"
    fips: str                     # "51"
    is_class2_senate_2026: bool
    religiosity: float             # Pew "% highly religious"
```

This directly answers decision (e): FIPS codes, full names, and Class-2
Senate membership currently exist as three independently-maintained dicts
across `fetch_census_demographics.py`, `build_pres_2024_county.py`, and
`toy_data.py`. All three collapse into one registry, defined as a class
attribute on `State` itself:

```python
class State:
    ...
    _REGISTRY: ClassVar[dict[str, "State"]] = {
        "AL": State("AL", "Alabama", "01", True, religiosity=40),
        "AK": State("AK", "Alaska", "02", True, religiosity=26),
        ...
    }

    @classmethod
    def by_abbr(cls, abbr: str) -> "State":
        return cls._REGISTRY[abbr]

    @classmethod
    def class2_senate_states(cls) -> list["State"]:
        return [s for s in cls._REGISTRY.values() if s.is_class2_senate_2026]
```

Resolving decision (f) here too: `religiosity` is a genuine, enduring fact
about a state — the same kind of fact as its FIPS code — not
simulation-specific reference data. It belongs on `State`, which means it
flows into `CombinedCountyRow` as an ordinary column (broadcast from state
to every county in it), exactly like `pct_college` flows in from Census.
The baseline model's existing `include_religiosity` flag is *already* the
right mechanism for deciding whether to use it — nothing about the model
needs to change, only where the value lives. The toy DGP doesn't need to
know about it at all, since it isn't using it as a true predictor (see
Section 7).

`STATE_LEAN_2024` (state-level 2024 presidential lean) is deliberately
**not** a `State` field: it's a rollup of `Pres2024Row` county data, cleanly
computable on demand (vote-weighted average of `dem_share_2024` grouped by
state) rather than a second hardcoded fact that could drift out of sync
with the county-level numbers it's derived from.

### `County`

```python
@dataclass(frozen=True)
class County:
    state: State
    name: str      # display form, e.g. "Fairfax city"
    key: str        # normalized join key, e.g. "FAIRFAX CITY"

    @staticmethod
    def normalize_key(raw_name: str) -> str:
        # ONE utility, used by every ingestion script. Decision (b): uppercase.
        ...

    @classmethod
    def from_census_name(cls, state: State, raw_name: str) -> "County":
        # Handles the comma-split + suffix-strip + city/county collision
        # logic currently duplicated across fetch_census_demographics.py
        # and build_pres_2024_county.py.
        ...
```

Per decision (b): one normalization utility, used everywhere a raw county
name needs to become a join key, uppercase rather than lowercase
(`"MCLEOD"` instead of `"mcleod"` — same collision-resolving effect, just a
different convention). `County.from_census_name()` is where the
comma-splitting, suffix-stripping, and Fairfax/Franklin/Richmond/Roanoke
collision handling (documented in `pipeline_walkthrough.md`) lives — one
canonical place instead of two independently-maintained regexes.

`__eq__`/`__hash__` are derived from `(state.abbr, key)`, so `County`
instances are usable as dict keys / for deduplication during ingestion,
even though the final stored tables use plain `state`/`county_key` string
columns (Section 6).

**Census is canonical**: `County` instances are only ever *constructed* by
`ingest/census_demographics.py` (via `County.from_census_name`). Every
other ingestion script (2024 president, 2026 Senate) produces rows keyed by
`(state_abbr, county_key)` and gets matched *against* the set of `County`
objects Census already defined — it never defines a new county that Census
doesn't already know about. If a source's county doesn't match anything in
the Census-derived set, that's a loud error (as `load_real_county_skeleton`
already does today), not a silent new entry.

### `Party`

```python
class Party(Enum):
    DEM = "D"
    REP = "R"
    OTHER = "OTHER"

    @classmethod
    def from_label(cls, raw_label: str) -> "Party":
        # "democrat", "dem", "d", "democratic-farmer-labor", ... -> DEM
        # "republican", "gop", "r", ... -> REP
        # anything else -> OTHER
        ...
```

Decision (d). This is deliberately a *small* class — raw-label-string to
canonical-party-enum, nothing else. It does **not** know about
coalition-independent overrides (Dan Osborn, Brian Bengs, Todd Achilles);
that's a per-candidate fact, not a per-label fact, and conflating the two is
exactly what made Idaho's two independents (same `"ind"` label, different
coalition status) awkward in the current code. That logic lives one level
up.

### `Cycle` and `ElectionOffice`

Resolves the Section 9 follow-up: "Montana's anti-R vote was split" and
"Idaho's Democrat withdrew" are facts about the **2026** Senate race
specifically, not enduring facts about Montana or Idaho, and not even
facts that are only ever about Senate races. `Cycle` is the value object
that scopes election-specific facts to the election they belong to, so
they don't accumulate as permanent-looking exceptions on `State`:

```python
class ElectionOffice(Enum):
    PRESIDENT = "president"
    SENATE = "senate"
    HOUSE = "house"
    GOVERNOR = "governor"
    STATE_HOUSE = "state_house"
    STATE_SENATE = "state_senate"

@dataclass(frozen=True)
class Cycle:
    year: int

    @property
    def is_presidential(self) -> bool:
        return self.year % 4 == 0

    @property
    def is_midterm(self) -> bool:
        return not self.is_presidential

    @property
    def senate_class_up(self) -> int:
        # Senate classes rotate on a fixed 6-year pattern: Class 3 up in
        # years divisible by 6 (2016, 2022, 2028...), Class 1 in years
        # 2 mod 6 (2018, 2024, 2030...), Class 2 in years 4 mod 6 (2020,
        # 2026, 2032...). Only meaningful when office == SENATE.
        return {0: 3, 2: 1, 4: 2}[self.year % 6]

CYCLE_2026 = Cycle(year=2026)   # is_midterm=True, senate_class_up=2
CYCLE_2024 = Cycle(year=2024)   # is_presidential=True, senate_class_up=1
```

### `Candidate`

```python
@dataclass(frozen=True)
class Candidate:
    name: str
    state: State
    cycle: Cycle
    raw_party: Party

    _COALITION_OVERRIDES: ClassVar[set[tuple[int, str, str]]] = {
        (2026, "NE", "dan osborn"),
        (2026, "SD", "brian bengs"),
        (2026, "ID", "todd achilles"),
    }

    @property
    def is_coalition_independent(self) -> bool:
        return (self.cycle.year, self.state.abbr, self.name.strip().lower()) in self._COALITION_OVERRIDES

    @property
    def effective_party(self) -> Party:
        return Party.DEM if self.is_coalition_independent else self.raw_party
```

### `SenateRace`

One per `(cycle, state)`. Owns the exclusion decision that's currently a
module-level `EXCLUDED_MULTIPARTY_STATES` set, and the vote-aggregation
logic that's currently a free-floating groupby in `real_counties.py`:

```python
@dataclass
class SenateRace:
    cycle: Cycle
    state: State
    candidates: list[Candidate]

    _EXCLUDED: ClassVar[set[tuple[int, str]]] = {(2026, "MT")}  # Bankhead(D) + Bodnar(I) split

    @property
    def is_excluded(self) -> bool:
        return (self.cycle.year, self.state.abbr) in self._EXCLUDED

    def two_party_votes(self, votes_by_candidate: dict[str, int]) -> tuple[int, int]:
        """Returns (dem_votes, rep_votes) for one county, given that
        county's {candidate_name: votes}, applying effective_party."""
        ...
```

Keying both override sets by `(cycle.year, state.abbr, ...)` rather than
just `(state.abbr, ...)` means a future cycle's Montana or Idaho starts
from a clean slate — nothing about 2026 carries forward implicitly.

## 4. Row schemas (the "consistent columns" contracts)

Each is a `dataclass` defining exactly what one ingestion script outputs,
or what the combined table contains. Used both as documentation and as a
`columns()`/`validate(df)` contract each script can assert against before
writing its CSV — this is the actual mechanism behind "save as a csv with
consistent columns."

```python
@dataclass
class CensusDemographicsRow:
    state: str
    county_key: str
    county_name: str
    pct_college: float
    pct_black: float
    pct_hispanic: float
    pct_asian: float
    median_age: float
    median_income_k: float
    total_population: int

@dataclass
class Pres2024Row:
    state: str
    county_key: str
    dem_votes_2024: int
    rep_votes_2024: int
    total_two_party_votes_2024: int
    dem_share_2024: float

@dataclass
class Senate2026Row:
    state: str
    county_key: str
    dem_votes: int
    rep_votes: int
    total_two_party_votes: int
    is_coalition_independent: bool
    excluded_multiparty_split: bool

@dataclass
class CombinedCountyRow:
    # identity
    state: str
    county_key: str
    county_name: str
    # demographics (always present -- Census is canonical, Section 3)
    pct_college: float
    pct_black: float
    pct_hispanic: float
    pct_asian: float
    median_age: float
    median_income_k: float
    # state-level broadcast
    religiosity: float
    # 2024 (present for every county Census and the 2024 source agree on --
    # i.e. not Alaska, see Section 5)
    total_two_party_votes_2024: Optional[int]
    dem_share_2024: Optional[float]
    # 2026 (present only where real returns exist on file -- NaN elsewhere)
    total_two_party_votes: Optional[int]
    dem_votes: Optional[int]
    rep_votes: Optional[int]
    is_coalition_independent: Optional[bool]
    excluded_multiparty_split: Optional[bool]
    # derived (present only where both 2024 and 2026 are present)
    dem_swing: Optional[float]
    swing_weight: Optional[float]
```

`CombinedCountyRow` is the **same schema** used for the combined table, the
real model table, and the toy model table — this is the literal
implementation of "toy and real tables should be the same except for
whether dem_swing is real or fake." The toy and real tables are just two
different *filters/transforms* over rows of this one type, not two
different types.

## 5. The combine step (`ingest/combine.py`)

Starts from Census's county list (canonical, per Section 3) and left-joins
everything else onto it:

1. Load all `CensusDemographicsRow`s → base table, one row per county that
   exists.
2. Broadcast `State.religiosity` onto every row by `state`.
3. Left-join `Pres2024Row` on `(state, county_key)`. Alaska has no rows here
   (different geography, per `pipeline_walkthrough.md`) — those columns are
   NaN for every Alaska county, which is the honest representation, not a
   separate exclusion list to maintain.
4. Left-join `Senate2026Row` on `(state, county_key)`, for whichever states
   currently have a `data/raw/{state}_senate_2026.csv`. Maine,
   Massachusetts, New Hampshire, Rhode Island currently produce no rows
   here (town-level source data can't key against county-level Census
   rows) — same treatment: NaN, not a special-cased filter. Once a
   town→county crosswalk exists, this step is the *only* place that needs
   to change.
5. Compute `dem_swing`/`swing_weight` wherever both the 2024 and 2026
   columns are non-null.
6. Write `data/combined/combined_counties.csv`.

This is also where "Census is canonical" pays off concretely: nothing
downstream needs to know that Alaska or the New England four are special
cases. They just have missing columns, and every later step (building the
real table, building the toy table) filters on "does this row have what I
need" rather than on a hardcoded state-abbreviation list.

## 6. Why value classes, but flat CSVs

`County` and `Party` are real classes with real behavior (Section 3), but
`combined_counties.csv` and the model tables store plain strings/numbers,
not serialized objects. Two reasons:

- Pandas/statsmodels operate on columns of primitives — vectorized
  `groupby`, `merge`, and `WLS` all expect numeric/string dtypes, not
  objects, in the hot path. Thousands of `County` *instances* inside a
  DataFrame would work but would be slow and would require every
  downstream consumer (including statsmodels internals) to know how to
  handle them.
- Round-tripping through CSV requires serialization either way; a plain
  `state,county_key` pair of strings already *is* that serialization,
  cleanly.

So the pattern throughout ingestion is: construct `County`/`Candidate`
objects to get the normalization/classification *behavior*, then flatten
(`dataclasses.asdict`-style) into a dict/row for the DataFrame. The classes
own the logic; the DataFrame owns the bulk storage. This is also why
`County.__eq__`/`__hash__` matter even though they're never stored directly
— they make the *construction/matching* step (does this 2026 result's
county match a Census county?) correct and testable in isolation.

## 7. Model layer

### `ModelSpec`

Replaces the scattered `predictors=`, `include_state_fe=`,
`include_religiosity=` keyword arguments repeated across three functions
today:

```python
@dataclass(frozen=True)
class ModelSpec:
    predictors: tuple[str, ...] = BASE_PREDICTORS
    include_state_fe: bool = False
    include_religiosity: bool = False
    interaction_terms: tuple[tuple[str, str], ...] = ()

    def to_formula(self) -> str:
        ...
```

### `Estimator` (decision (a) — the ambitious one)

```python
class Estimator(Protocol):
    def fit(self, df: pd.DataFrame, spec: ModelSpec, weight_col: str, cluster_col: str | None) -> "FitResult": ...
```

`FitResult` is deliberately left as a **protocol, not a wrapper class**, for
now: statsmodels' WLS results objects already expose `.params`, `.bse`,
`.pvalues`, `.resid`, `.rsquared`, `.aic`, `.bic` natively, which is exactly
the surface `analysis/` needs. Writing a wrapper class today would be
premature abstraction with only one real implementation
(`WLSEstimator`) to abstract over. The point where a wrapper becomes
justified: the day a second estimator's native result object *doesn't*
expose that same surface (e.g. a future Bayesian model built on PyMC, whose
results object looks nothing like statsmodels'). Build the wrapper then,
against two concrete needs, not now against a hypothetical one.

```python
class WLSEstimator:
    def fit(self, df, spec, weight_col="swing_weight", cluster_col="state"):
        formula = spec.to_formula()
        model = smf.wls(formula, data=df, weights=df[weight_col])
        if cluster_col:
            return model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
        return model.fit()
```

This single method replaces the three independent `smf.wls(...).fit(...)`
call sites in the current `baseline_model.py`. `analysis/` functions take
an `Estimator` instance as an argument and never import `statsmodels`
directly.

### `toy_dgp.py` — `SwingSimulator`

```python
@dataclass
class SwingSimulator:
    true_coefs: dict[str, float]       # was module-level TRUE_COEFS
    state_swing_deviation_sd: float = 0.04

    def simulate(self, combined: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Takes the combined table (ignores any real dem_swing/swing_weight
        present), returns a same-schema table with simulated dem_swing and
        swing_weight for every row."""
        ...
```

This addresses the toy/real symmetry directly: `SwingSimulator.simulate()`
takes a `CombinedCountyRow`-shaped DataFrame, drops/ignores whatever
`dem_swing` it finds, and returns the **same shape** with a fake one. The
per-state random deviation it draws internally (`STATE_SWING_DEVIATION` in
the current code) is *not* a `CombinedCountyRow` column and never will be —
it's an artifact of the simulation, generated fresh each run, with no real
counterpart. It stays entirely inside this class.

Building the two model tables becomes two short scripts/calls rather than
two divergent code paths:

```python
real_table = combined[combined["dem_swing"].notna()]
real_table.to_csv("data/model_tables/real_counties.csv")

toy_table = SwingSimulator(TRUE_COEFS).simulate(combined, rng)
toy_table.to_csv("data/model_tables/toy_counties.csv")
```

Both are then fit identically: `WLSEstimator().fit(real_table, spec, ...)`
and `WLSEstimator().fit(toy_table, spec, ...)`, and both results go through
the same `analysis/` functions. The old `simulate_counties()` (fully fake
geography, used for synthetic 50-state stress tests) has no remaining role
once the combined table already covers real geography for every Class-2
state at full scale — it's dropped, not ported.

## 8. Analysis layer

`compute_vif`, `compare_interaction_model`, `test_single_interaction` move
to `analysis/` largely unchanged in logic, but take an `Estimator` +
`ModelSpec` instead of building formulas and calling `smf.wls` themselves.
`compute_vif` in particular was already model-agnostic (it only touches raw
predictor columns) — it doesn't need an `Estimator` at all, just the
DataFrame and `spec.predictors`.

## 9. Decisions

- **`State` registry source**: a literal Python class attribute (Section
  3), not a backing CSV. No reference-table layer — `State` is the single
  place this data lives, as code.
- **Cycle-scoping**: resolved by introducing `Cycle`/`ElectionOffice`
  (Section 3) — `Candidate` and `SenateRace` key their override sets by
  `(cycle.year, state.abbr, ...)`, so a future cycle's Montana or Idaho
  starts clean rather than inheriting 2026's exceptions, and `Cycle` is
  also where midterm-vs-presidential and which Senate class is up get
  derived, rather than re-deriving them ad hoc wherever needed.
- **`ingest/senate_2026_nbc.py` scope**: NBC-only, no official-SOS
  fallback layer. It worked for all 27 states tried, including Idaho's
  multi-independent race and Nebraska's missing-Democrat race — there's no
  evidence yet of a state it can't handle. A state where it fails
  surfaces as a loud, specific error for manual follow-up rather than
  triggering more automated fallback code to maintain for a path that
  hasn't actually been needed.

## 10. Can the currently-excluded states be added?

Four states (Maine, Massachusetts, New Hampshire, Rhode Island) and one
state (Alaska) are currently excluded from the swing model entirely. They
fail for two structurally different reasons, and the two groups have very
different prognoses.

### Maine / Massachusetts / New Hampshire / Rhode Island — yes, cleanly

These four report both Senate and (per-town) presidential results by
town/city, not county — they don't use counties as functional election
units. But two things make this fixable without any estimation:

1. **2024 presidential data is already fine.** The `tonmcg` mirror that
   `build_pres_2024_county.py` reads from reports these four states at
   proper county granularity already (confirmed: county *counts* for all
   four matched Census's exactly, back when the 2024-vs-Census merge was
   first checked) — something standard in curated election-data products,
   even though the states' own administrative reporting is by town. No
   rollup needed on the 2024 side at all.
2. **Census itself defines a town→county crosswalk, queryable right now.**
   County subdivisions ("county subdivision" in Census API terms — towns,
   in New England) carry their parent county's FIPS code directly:

   ```
   "Auburn city, Androscoggin County, Maine", state=23, county=001, county subdivision=02060
   "Barrington town, Bristol County, Rhode Island", state=44, county=001, county subdivision=05140
   ```

   Confirmed live against the Census API using the exact same
   infrastructure `ingest/census_demographics.py` already has — no
   external crosswalk source needed. The only piece that actually needs
   the rollup is the already-downloaded 2026 NBC town-level data
   (`ME_senate_2026.csv` etc.): group each town's `dem_votes`/`rep_votes`
   by its county (via this crosswalk) and sum.

Recommend: add a `County.from_town(state, town_name, county_fips)`
construction path (or a small `ingest/town_to_county.py` step that
produces the crosswalk once from the Census API and is consulted by
`combine.py`) and roll these four states' 2026 data up to county level.
No caveats needed in the output — this isn't an estimate, it's an exact
aggregation of a real vote total.

### Alaska — possible, but genuinely costlier and not recommended yet

Alaska's problem is different in kind, not just degree: its results are
*reported* by State House District (40 of them), not by borough/census
area (~30), and this isn't one source's quirk — it's how Alaska itself
distributes results. Confirmed directly against NBC's own Alaska Senate
page (the same source `ingest/senate_2026_nbc.py` would use for every
other state): its area selector lists "State House District 1" through
"40," no borough names anywhere. The 2024 presidential mirror has the
same shape, for the same reason.

Two separate problems, not one:

1. **2024 side**: a borough-level breakdown exists, but it's explicitly
   labeled an *estimate*, not an official count — Wikipedia's
   "2024 United States presidential election in Alaska" article carries a
   "By borough and census area (estimates)" section, sourced to a
   third-party aggregation of precinct-level data, with its own note that
   "official results by borough are not available." Summing that table's
   Harris/Trump columns gives 139,543 / 184,179 against certified
   statewide totals of 140,026 / 184,458 — within 0.3%/0.15%, plausible
   for an estimate, but not exact. Usable, with a clear "estimated, not
   official" label carried through to the model.
2. **2026 side**: no equivalent estimate exists anywhere checked. Building
   one would require Alaska's actual precinct-level 2026 Senate returns
   (not currently sourced anywhere in this project) plus a
   precinct-to-borough crosswalk (not confirmed to exist in an accessible
   form) — meaningfully more data-engineering than the New England fix,
   and introducing real estimation error into what's supposed to be the
   "real" half of the toy/real table pair. On top of that, the race itself
   was "too early to call" as of this check (86.2% reporting, a
   ranked-choice tabulation possible, determination expected ~Nov 18) — so
   even a perfect geographic fix would currently be modeling a
   provisional, not-yet-final result, unlike every other state's
   (unofficial-but-complete) count.

Recommend: do **not** add Alaska now. Revisit after Nov 18 once the race
is actually decided — at that point the 2024 side can use the labeled
Wikipedia estimate, but the 2026 side still needs either a genuine
borough-level source to surface (worth a quick check then, since a settled
race is more likely to get retrospective rollups published) or a decision
that Alaska enters the model with an explicitly estimated outcome, which
should be a visible flag on that row (e.g. `is_estimated_geography=True`
on `CombinedCountyRow`), not something silently blended in with the other
32 states' counts.
