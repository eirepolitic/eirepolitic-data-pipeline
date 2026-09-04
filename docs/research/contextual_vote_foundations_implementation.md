# Context-aware voting foundations implementation

## Status

Production deployment completed successfully on 2026-09-04.

Foundation production batch:

- `contextual-vote-foundations-20260904-1`

Implementation PR:

- PR `#96`

Validation runs:

- `33893753187` — initial focused validation found a test/aggregation issue; no production change.
- `33893911894` — corrected focused validation; compilation and all reconciliation/contract tests passed.

Successful foundation deployment and post-promotion reconciliation:

- `33894036435`

Read-only foundation footprint verification:

- `33894221886`

The additive foundations remain deployed and are now consumed by the completed-month contextual voting layer deployed in production batch:

- `contextual-monthly-voting-20260904-1`

Monthly contextual implementation details are recorded in:

- `docs/research/contextual_monthly_voting_implementation.md`

## Production datasets

### `daily_context_vote_components`

Purpose:

- additive daily voting numerators and denominators by certified `division_context`;
- supports arbitrary-period contextual participation recalculation.

Production grain:

- `activity_date`
- `division_context`
- `grain`
- `entity_id`
- `component_id`

Allowed grains:

- member
- party
- constituency

Allowed components:

- `recorded_vote_count`
- `eligible_member_division_count`

Live row count at deployment:

- **65,593**

### `context_division_party_vote_components`

Purpose:

- division-level historical party vote distributions carrying certified division context;
- supports arbitrary-period contextual party recorded-vote agreement using the existing production rules.

Production grain:

- `division_id`
- `division_context`
- `party_uri`
- `vote_code`

Live row count at deployment:

- **4,575**

## Reconciliation guarantees

The foundation deployment confirmed:

- allowed context values only;
- unique daily primary key;
- unique context-party primary key;
- **0 daily reconciliation mismatches** against existing `daily_activity_components` voting rows;
- **0 party reconciliation mismatches** against existing `division_party_vote_components`;
- collapsing `division_context` reproduces the established unfiltered foundations exactly.

This reconciliation remains part of standard candidate materialization.

## Existing formulas unchanged

The foundations did **not** change:

- membership eligibility logic;
- historical party-at-vote attribution;
- recorded-vote participation formulas;
- party recorded-vote agreement formula;
- minimum two recorded party-member votes per qualifying party/division;
- 10+ qualifying divisions = `reliable`;
- 5–9 = `caution`;
- fewer than 5 = `insufficient_for_comparison`.

The foundations add only the certified `division_context` dimension to additive voting components.

## Monthly contextual handoff now completed

The previously planned completed-month layer is now deployed.

Current production includes `division_context` monthly result rows for:

- `member_vote_participation_pct`
- `party_vote_cohesion_pct`

The monthly layer:

- keeps the same metric IDs and formulas;
- uses the established 0–1 proportion scale;
- carries numerator and denominator fields;
- formalizes member contextual sample reliability;
- preserves existing party qualifying-division thresholds;
- carries Independent-group wording warnings;
- is generated automatically by future full candidate materialization.

Live contextual monthly footprint:

- 11,844 contextual rows
- 20 completed months
- 0 duplicate monthly primary keys

See `contextual_monthly_voting_implementation.md` for the full live reliability and warning distribution.

## Consumer semantics

### Arbitrary-period participation

For a selected context and period:

- numerator = sum of `recorded_vote_count`;
- denominator = sum of `eligible_member_division_count`;
- calculate the ratio after filtering to the required context, period, grain and entity.

Never average monthly participation percentages to create a longer-period value.

### Arbitrary-period party recorded-vote agreement

Use `context_division_party_vote_components` and the existing production party-vote calculation:

1. group recorded party votes within each qualifying division;
2. require at least two recorded members for that party/division;
3. use modal vote count as aligned votes;
4. aggregate aligned/total qualifying votes across the selected context and period;
5. retain the existing reliability threshold based on qualifying divisions.

For the Independent grouping, describe the output as **recorded-vote agreement among Independents**, not party discipline.

## Methodological guardrails

1. `division_context` is descriptive parliamentary context, not a performance measure.
2. Context filtering must happen before numerator/denominator aggregation.
3. Historical party attribution must remain date-correct.
4. Missing recorded votes do not prove absence.
5. Do not average percentages/proportions across periods.
6. Do not describe participation or agreement as effectiveness, quality or political performance.
7. `motion_proceeding` is parliamentary form, not one substantive policy topic.
8. Bill stage remains separate and must not be inferred from Bill ID plus division date.
9. Context-specific additive totals must continue to reconcile exactly to existing unfiltered foundations when context is collapsed.
10. Consumer code should use these foundations rather than recomputing context joins from speech-level data.

## Living next-step plan

The additive foundations and first completed-month contextual voting layer are both deployed.

Next research priority:

1. Run a **consumer-readiness audit** for Appsmith, Power BI and API use of the new contextual monthly rows.
2. Define default suppression and ranking behavior so `not_certified` rows are not compared as if equally stable.
3. Verify that consumer views always retain numerator, denominator, reliability, public-use and warning metadata.
4. Verify Independent-group wording in public party tables/charts.
5. Investigate whether `party_vote_participation_pct` has a sufficiently clear use case to add `division_context` monthly rows; the additive foundation already supports it.
6. Investigate constituency contextual participation only if a concrete consumer view benefits from it.
7. Keep arbitrary-period contextual calculations foundation-based rather than aggregating monthly proportions.
8. Develop certified Bill-specific voting histories as an event-level consumer view using existing Bill linkage.
9. Continue to defer Bill stage attribution until an exact stage relationship is certified.
10. Keep all voting outputs descriptive and politically neutral.
