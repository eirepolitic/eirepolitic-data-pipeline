# Context-filtered voting metrics

## Status

Investigation completed on 2026-09-04 using production batch:

- `division-context-20260903-1`

No production metric, denominator or schema changed during this investigation.

The existing production voting calculators can be applied safely inside each certified division context by filtering divisions first and then reusing the existing formulas unchanged.

This supports a future additive contextual-voting output, but that requires a separate production implementation plan.

## Goal

Determine whether existing voting metrics remain interpretable and sufficiently populated when restricted to:

- `bill_or_legislation`
- `motion_proceeding`
- `procedural_business`
- `other`

The two formulas under review were:

1. member recorded-vote participation;
2. party recorded-vote cohesion/agreement.

## Evidence

Temporary investigation branch:

- `ops/investigate-contextual-vote-metrics-20260903`

Runs:

- `33848623450` — first read-only attempt; failed during the diagnostic because the temporary script used a non-existent membership key.
- `33848716141` — diagnostic retry with traceback capture; confirmed the source-key issue and made no production change.
- `33848800973` — corrected contextual-voting metric profile; successful.

Artifact:

- `analysis/contextual_vote_metrics_digest.json`

Current production counts:

- 401 divisions
- 59,325 recorded member-vote rows
- 401 `division_context` rows

## Method used

For each division context independently:

1. select the divisions in that context;
2. select member-vote rows for those divisions;
3. recompute eligible member × division pairs using the existing membership-interval logic;
4. call the existing `member_vote_participation` production calculator unchanged;
5. call the existing `party_vote_metrics` production calculator unchanged.

This means context changes only the population of divisions being analysed. It does not change the denominator definitions or reliability rules.

## Member recorded-vote participation

Definition remains:

`distinct divisions with a recorded vote / eligible member × division opportunities`

The denominator is restricted to divisions inside the selected context.

### Bill divisions

- 168 divisions
- 24,427 recorded member-vote rows
- 29,087 eligible member × division pairs
- 176 members with eligible opportunities
- median member participation: 89.3%
- 25th percentile: 79.8%
- 75th percentile: 94.6%

All 176 members have at least 25 eligible Bill-division opportunities in the current source snapshot.

### Motion proceedings

- 153 divisions
- 22,777 recorded member-vote rows
- 26,501 eligible member × division pairs
- 176 members
- median member participation: 90.8%
- 25th percentile: 81.7%
- 75th percentile: 95.4%

Sample-size distribution:

- 174 members: at least 25 eligible opportunities
- 2 members: 10–24 opportunities

### Procedural business

- 53 divisions
- 7,979 recorded member-vote rows
- 9,184 eligible member × division pairs
- 176 members
- median member participation: 90.6%
- 25th percentile: 83.0%
- 75th percentile: 96.2%

Sample-size distribution:

- 174 members: at least 25 eligible opportunities
- 2 members: 5–9 opportunities

### Other

- 27 divisions
- 4,142 recorded member-vote rows
- 4,671 eligible member × division pairs
- 176 members
- median member participation: 92.6%
- 25th percentile: 85.2%
- 75th percentile: 100.0%

Sample-size distribution:

- 172 members: at least 25 eligible opportunities
- 1 member: 10–24 opportunities
- 1 member: 5–9 opportunities
- 2 members: fewer than 5 opportunities

### Participation interpretation

The current contextual participation denominators are large enough for most members in all four contexts.

However, the existing production metric does not currently assign a formal reliability status to member participation based on denominator size.

**Decision:** if contextual member participation is materialized publicly, carry the eligible-division denominator explicitly and add a small-sample presentation rule rather than presenting all percentages as equally stable.

A conservative display rule could be:

- 25+ eligible divisions: normal display;
- 10–24: display with caution;
- 5–9: small-sample caution;
- fewer than 5: suppress comparison or mark insufficient.

This is a proposed presentation/reliability rule, not yet a production metric change.

## Party recorded-vote cohesion

The existing production calculation remains:

1. assign each recorded vote to the member's historical party at the division date;
2. a party/division qualifies only when at least 2 recorded party-member votes exist;
3. aligned votes are the modal vote count within that party/division;
4. aggregate aligned votes / total qualifying votes across divisions;
5. reliability by number of qualifying divisions:
   - 10+ = `reliable`
   - 5–9 = `caution`
   - fewer than 5 = `insufficient_for_comparison`

No contextual rule changed this method.

## Party reliability by context

Each context currently contains 11 historical party/group URIs.

In every context:

- 9 groupings are `reliable` under the existing 10+ qualifying-division rule;
- 2 are `insufficient_for_comparison` because they never have at least two recorded members in a division;
- no current grouping falls in the 5–9 `caution` band.

The two insufficient groupings in the current Dáil source snapshot are:

- Green Party
- 100% Redress

Their recorded-vote participation can still be described, but a party cohesion percentage is not certified because the minimum two-member-per-division requirement is not met.

## Context-specific party examples

The following are descriptive examples from the current snapshot, not performance rankings.

### Bill divisions

For large party samples:

- Fianna Fáil: 168 qualifying cohesion divisions; recorded-vote agreement about 99.8%; participation about 85.1%.
- Sinn Féin: 168 qualifying divisions; recorded-vote agreement 100%; participation about 88.9%.
- Fine Gael: 168 qualifying divisions; recorded-vote agreement about 99.8%; participation about 87.1%.

### Motion proceedings

- Fianna Fáil: 153 qualifying divisions; recorded-vote agreement 100%; participation about 87.5%.
- Sinn Féin: 153 qualifying divisions; recorded-vote agreement 100%; participation about 90.7%.
- Fine Gael: 153 qualifying divisions; recorded-vote agreement 100%; participation about 86.9%.

### Procedural business

The major party samples all have 53 qualifying cohesion divisions and remain reliable under the existing rule.

### Other divisions

Even though there are only 27 divisions in this context, the larger party groupings still exceed the current 10-division reliability threshold.

This does not mean the `other` context is substantively homogeneous. It only means the recorded-vote agreement calculation has enough qualifying divisions under the existing sample rule.

## Important Independent-group caveat

The source includes an `Independent` historical grouping URI.

The existing calculator can produce a recorded-vote agreement proportion for members assigned to this grouping because multiple Independent members vote in the same divisions.

However, this must **not** be interpreted as party discipline or party cohesion in the organizational sense.

For example, current agreement levels for the Independent grouping are materially lower than the major organized parties across contexts, but the safe interpretation is:

- **recorded-vote agreement among members recorded as Independent**

not:

- party discipline;
- whipping effectiveness;
- organizational unity;
- political quality.

**Decision:** future public contextual-cohesion outputs should either exclude the Independent grouping from party-discipline-style charts or label it explicitly as an agreement measure among Independents.

## Why contextual comparisons require caution

Differences between contexts are descriptive and can reflect:

- different parliamentary purposes;
- different mixes of government/opposition motions;
- different types of amendments or questions put;
- different time periods;
- different membership opportunity counts;
- changing party membership over time.

Therefore a higher participation or agreement percentage in one context must not be described as caused by that context.

Do not infer:

- stronger political effectiveness;
- better representation;
- greater discipline quality;
- more important voting;
- higher attendance generally.

## Safe public measures

The following are supported if numerator, denominator and reliability metadata are preserved:

### Member

- recorded-vote participation by division context;
- votes cast count by context;
- eligible division count by context.

For small member denominators, display caution/suppression metadata.

### Party/group

- recorded-member-vote participation by context;
- qualifying cohesion division count by context;
- recorded-vote agreement proportion by context where existing reliability is `reliable` or `caution`;
- aligned and total qualifying vote counts.

For Independents, use “recorded-vote agreement” language rather than “party discipline.”

### Bill-linked views

Because `division_context` carries certified Bill IDs for Bill divisions, downstream consumers can safely show:

- a Bill's recorded divisions;
- party/group vote distributions within those divisions;
- member recorded-vote histories for that Bill;
- contextual participation/cohesion summaries over sets of certified Bill divisions.

Do not add Bill stage unless an exact stage relationship is separately certified.

## Production architecture implication

Contextual metrics are feasible and do not require new voting formulas.

The main production need is an additive context dimension in the voting foundations/results so consumers do not need to repeatedly rejoin and recalculate raw eligibility.

Two possible implementation paths are safe:

### Option A — extend additive voting foundations with `division_context`

Add context to event/daily voting components so arbitrary-period calculations can filter before aggregation.

Advantages:

- preserves arbitrary-period recomputation;
- keeps numerator/denominator components additive;
- avoids proliferating separate metric IDs for each context.

### Option B — materialize context-dimensional monthly result rows only

Add `dimension_name = division_context` and `dimension_value` to selected completed-month voting metrics.

Advantages:

- simpler first consumer surface.

Disadvantages:

- weaker support for arbitrary ranges unless context-aware foundations are also present.

**Recommendation:** Option A first, with monthly contextual rows derived from it. This is consistent with the existing materialization design, which stores additive foundations and derives completed-month presentation results.

This is an architecture proposal only; no production implementation was made in this investigation.

## Required implementation guardrails

Any future contextual-voting production change should require:

1. filter divisions by context before calculating eligible member × division pairs;
2. preserve existing historical party temporal joins;
3. preserve existing party cohesion thresholds exactly unless separately reviewed;
4. carry eligible-division counts alongside member participation;
5. add a member small-sample reliability/presentation rule if contextual member comparisons are public;
6. preserve aligned-vote and total-vote components for party agreement;
7. never aggregate monthly percentages to create longer-period percentages;
8. recompute longer periods from additive components;
9. keep Independent agreement semantics distinct from organized-party discipline;
10. audit that contextual totals reconcile to the unfiltered voting population where categories are exhaustive.

## Living next-step plan

1. Prepare a production implementation plan for context-aware voting foundations, using `division_context` as the certified dimension.
2. Prefer additive context-aware components over hard-coded context-specific metric IDs.
3. Design member participation components so arbitrary periods can recompute:
   - recorded votes;
   - eligible member × division opportunities;
   - division context.
4. Extend party division-vote components with division context while preserving party-at-vote attribution.
5. Define and document a small-sample reliability rule for contextual member participation before public comparison views are enabled.
6. Keep current party cohesion reliability thresholds unchanged initially.
7. Treat Independent agreement separately in consumer wording and public chart design.
8. Add monthly contextual result rows only after the additive foundations are available and audited.
9. Validate reconciliation: summing context-specific additive counts across the four exhaustive contexts must equal the existing unfiltered counts for the same period/population.
10. Continue to defer Bill stage attribution until an exact section/division-stage relationship is certified.
