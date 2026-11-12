# Replacing the state fixed effect: options and findings

## The problem

Gemini's critique: a saturated `C(state)` fixed effect, fit freely on a single
cross-section, can't distinguish "this state's enduring political culture"
from "this cycle's specific candidate quality" — with one Senate race per
state, there's nothing in the data to separate them. Since this project's
whole premise is separating candidate effects from environmental effects,
baking that confound into the baseline model directly undermines the goal.

Risk of just dropping it: real regional variation (especially how white
voters behave differently by region) that income/college/race alone won't
capture.

## Three options considered, plus the untouched baseline

1. **State religiosity** (Pew's 2023-24 Religious Landscape Study, "%
   highly religious," `data/raw/state_religiosity.csv`) — continuous,
   doesn't need a discrete regional boundary call, and doesn't move because
   of who's on the ballot this cycle.
2. **Census region** (Northeast/Midwest/South/West, 4-category) — a coarser
   grouping than full state FE, but still a somewhat arbitrary statistical
   boundary (Texas as "South" is the flagged edge case).
3. **Income/college/race alone, no regional control** — simplest, accept
   whatever regional signal isn't captured as noise.
4. *(Reference)* **Full state FE** — the thing we're trying to move away
   from, included as the upper-bound comparison.

## Toy-model design

The DGP's state-level term was decomposed losslessly: each state's real 2024
presidential lean (`STATE_LEAN_2024`) = a religiosity-explained part
(`TRUE_COEFS["religiosity"] = -0.006` × that state's real religiosity value)
+ an "unexplained" residual (`UNEXPLAINED_STATE_LEAN`). Verified exactly
lossless (max reconstruction error: 0.0000000000) before trusting any result
built on it. This means the new baseline's job is to recover the
religiosity-explained part as a proper covariate, while the unexplained part
honestly falls into the residual — by design, not as a flaw.

## Finding 1: dropping state FE has a real, measured cost

Fit on all 33 Class-2 states (2,316 counties), comparing recovered
coefficients against the known true values:

| spec | pct_college | pct_black | pct_hispanic | pct_asian | median_age | median_income_k | R² |
|---|---|---|---|---|---|---|---|
| **TRUE** | 0.300 | 0.250 | 0.150 | 0.100 | -0.0040 | 0.0015 | — |
| state FE | 0.249 | 0.212 | 0.131 | 0.023 | -0.0036 | 0.0017 | 0.824 |
| religiosity | 0.270 | 0.396 | 0.218 | 0.016 | 0.0004 | 0.0026 | 0.691 |
| Census region | 0.259 | 0.360 | 0.247 | 0.007 | 0.0002 | 0.0029 | 0.663 |
| religiosity + region | 0.260 | 0.405 | 0.231 | -0.001 | 0.0002 | 0.0027 | 0.696 |
| demographics only | 0.329 | 0.243 | 0.222 | 0.075 | 0.0018 | 0.0030 | 0.577 |

State FE recovers every coefficient closest to true, which makes sense by
construction in this toy world: religiosity and the unexplained residual are
both *constant within a state*, so a saturated per-state dummy perfectly
absorbs both, leaving clean within-state variation for the demographic
coefficients. Every non-FE option leaves real bias behind — `pct_black`
overshoots by as much as 60% (religiosity: 0.396 vs. true 0.250), and
`median_age`'s sign flips to positive in every non-FE spec. Oddly,
`demographics_only` (no regional control at all) does *better* than
religiosity/region on `pct_black` and `pct_asian` specifically — adding a
regional covariate that's correlated with those shares (Mississippi is both
the most religious state and has a high Black population share) can make
things worse, not better, if the covariate soaks up variance unevenly.

## Finding 2: in a single cross-section, state FE doesn't uniquely "launder" candidate quality

This is the test that actually matters for Gemini's concern, and the toy
model as built couldn't run it until a true per-state "candidate quality"
term (random, independent of demographics/religiosity — standing in for
what real 2026 candidates would add) was injected into the DGP specifically
for this check.

Result: correlation between **fitted state-FE coefficient** and true
candidate quality = **0.513**. Correlation between **religiosity-model's
state-mean residual** and true candidate quality = **0.519**. Essentially
identical.

Why: with only one election cycle, both approaches mix the same two
unmodeled sources (enduring state culture + this cycle's candidate quality)
together — state FE packages the mix into a parameter, no-FE packages it
into a residual. Neither one disentangles them. **The real advantage of a
slow-moving covariate like religiosity only shows up once you have multiple
election cycles**: its effect is assumed stable across cycles, so you can
pool cycles to estimate it once, then compare each cycle's leftover
deviation to the others — which a state FE, refit independently each cycle
with no shared baseline, can't give you. That's the mechanism this project
would actually need to extract candidate effects cleanly, and it isn't
available yet with one cycle of data.

## Recommendation for now (revised)

Finding 2 was initially (wrongly) read as "it doesn't matter which spec you
pick, so use the one with less-biased demographic coefficients" — i.e.
state FE. That's backwards. A SATURATED state FE forces the within-state
average residual to ~0 by construction: whatever state-level signal exists
gets absorbed into the FE parameter, not left in the residual. Since this
project's whole point is to use the residual as the state-environment +
candidate-quality signal for later analysis, that's fatal regardless of
Finding 2's correlation numbers being similar — the information being
"contained" somewhere isn't the same as it being usable where it's needed.

**Corrected recommendation: state FE OFF, religiosity OFF as a baseline
regressor too.** Pulling religiosity out as a covariate has the same
problem on a smaller scale — it explains away part of the state-level
signal before the residual is computed, when the point of this baseline is
to preserve that signal, not minimize it. The baseline model is
demographics only (`pct_college`, `pct_black`, `pct_hispanic`, `pct_asian`,
`median_age`, `median_income_k`), no state-level term at all. Religiosity's
role becomes a DIAGNOSTIC applied AFTER fitting: regress the state-level
residual on religiosity to see how much of the leftover signal is
explainable by enduring culture vs. genuinely candidate/short-term-specific
— a second-stage analysis question, not a baseline-model design choice.
`data/raw/state_religiosity.csv` and the decomposition code in
`toy_data.py` stay useful for exactly that second stage, and for the
eventual multi-cycle pooling discussed in Finding 2, once more cycles of
real data exist.
