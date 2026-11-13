# 2026 Midterms Model

The goal of this model is to attempt to separate candidate effects from
environmental effects in the US 2026 midterms.

## Research questions

1. How are races correlated nationally across states?
2. How are races correlated within states? (Senate, if present; House;
   Governor, if present; State House; State Senate)
3. How do these correlations hold at the county level?
4. How do the demographics of states and counties correlate with levels
   of correlation?
5. Does exit polling data match swings based on county demographics?

## WAR scoring

The objective of this is to give candidates WAR scores. WAR is made up of
Senate WAR, House WAR, and Governor WAR, based on a candidate's modeled
impact across all races.

An important consideration is that H2H persuasion depends on the opposing
candidate, as does turnout to an extent. It may be necessary to express
WAR as a shared attribute of both candidates.

## Analysis questions (once the baseline model is fit)

- Which states had the highest residual? Is there a way of estimating how
  much random noise there would be in the residual?
- What are the coefficients?
- Can we create a map of the Senate map using the baseline model?
  - Would need a new map-maker class.

## Project documentation

- [`pipeline_walkthrough.md`](pipeline_walkthrough.md) — what every script
  does today, what's real vs. simulated at each step, and the assumptions
  baked into the toy model and baseline model.
- [`class_architecture.md`](class_architecture.md) — the target
  ingest/model/analysis class design, and (Section 0) exactly what's been
  implemented so far vs. still just planned.
- [`state_effect_options.md`](state_effect_options.md) — the
  investigation into state fixed effects vs. a religiosity covariate vs.
  leaving state signal in the residual, and why the model dropped the
  state-level term entirely.
