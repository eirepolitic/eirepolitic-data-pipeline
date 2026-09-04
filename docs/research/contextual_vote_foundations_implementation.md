# Context-aware voting foundations implementation

## Status

Production deployment completed successfully on 2026-09-04.

Current production batch:

- `contextual-vote-foundations-20260904-1`

Implementation PR:

- PR `#96`

Validation runs:

- `33893753187` — initial focused validation found a test/aggregation issue; no production change.
- `33893911894` — corrected focused validation; compilation and all reconciliation/contract tests passed.

Successful production deployment and post-promotion reconciliation:

- `33894036435`

Read-only live footprint verification:

- `33894221886`

The deployment is additive. Existing unfiltered voting foundations and existing monthly voting metrics remain unchanged.

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

Live row count:

- **65,593**

### `context_division_party_vote_components`

Purpose:

- division-level historical party vote distributions carrying certified division context;
- supports arbitrary-period contextual party recorded-vote agreement/cohesion using the existing production rules.

Production grain:

- `division_id`
- `division_context`
- `party_uri`
- `vote_code`

Live row count:

- **4,575**

## Live context footprint

Daily contextual component rows:

- `bill_or_legislation`: 17,694
- `motion_proceeding`: 18,936
- `procedural_business`: 22,022
- `other`: 6,941

Party division-vote component rows:

- `bill_or_legislation`: 1,881
- `motion_proceeding`: 1,750
- `procedural_business`: 620
- `other`: 324

Recorded party-member vote totals by context:

- Bill divisions: 24,427
- motion proceedings: 22,777
- procedural business: 7,979
- other: 4,142

These reconcile to the existing 59,325 recorded member-vote rows.

## Daily additive totals

Because daily contextual components are materialized for member, party and constituency grains, their aggregate component totals are intentionally larger than the raw member-vote row count.

Across all three grains, the live totals are:

### Bill divisions

- eligible member-division components: 87,261
- recorded-vote components: 73,281

### Motion proceedings

- eligible member-division components: 79,503
- recorded-vote components: 68,331

### Procedural business

- eligible member-division components: 27,552
- recorded-vote components: 23,937

### Other

- eligible member-division components: 14,013
- recorded-vote components: 12,426

These values are additive components by grain and must not be interpreted as unique people or unique votes when summed across grains.

## Reconciliation guarantees

The live post-promotion audit confirmed:

- allowed context values only;
- unique daily primary key;
- unique context-party primary key;
- **0 daily reconciliation mismatches** against existing `daily_activity_components` voting rows;
- **0 party reconciliation mismatches** against existing `division_party_vote_components`;
- collapsing `division_context` reproduces the established unfiltered foundations exactly.

This reconciliation is now part of the standard candidate materialization gate.

## Existing formulas unchanged

This deployment did **not** change:

- membership eligibility logic;
- historical party-at-vote attribution;
- recorded-vote participation formulas;
- party cohesion/agreement formula;
- minimum two recorded party-member votes per qualifying party/division;
- 10+ qualifying divisions = `reliable`;
- 5–9 = `caution`;
- fewer than 5 = `insufficient_for_comparison`;
- existing completed-month metric rows.

The new foundations only add the certified `division_context` dimension to additive voting components.

## Consumer semantics

### Participation

For a selected context and period:

- numerator = sum of `recorded_vote_count`;
- denominator = sum of `eligible_member_division_count`;
- calculate the ratio after filtering to the required context, period, grain and entity.

Never average monthly participation percentages to create a longer-period value.

### Party recorded-vote agreement

Use `context_division_party_vote_components` and the existing production party-vote calculation:

1. group recorded party votes within each qualifying division;
2. require at least two recorded members for that party/division;
3. use modal vote count as aligned votes;
4. aggregate aligned/total qualifying votes across the selected context and period;
5. retain the existing reliability threshold based on qualifying divisions.

For the Independent grouping, describe the output as **recorded-vote agreement among Independents**, not party discipline.

## Methodological guardrails

1. `division_context` is a descriptive parliamentary-form/context dimension, not a performance measure.
2. Context filtering must happen before numerator/denominator aggregation.
3. Historical party attribution must remain date-correct.
4. Missing recorded votes do not prove absence.
5. Do not average percentages across periods.
6. Do not describe participation or agreement as effectiveness, quality or political performance.
7. `motion_proceeding` is parliamentary form, not one substantive policy topic.
8. Bill stage remains separate and must not be inferred from Bill ID plus division date.
9. Context-specific additive totals must continue to reconcile exactly to the existing unfiltered foundations when context is collapsed.
10. Consumer code should use these foundations rather than recomputing context joins directly from speech-level data.

## Living next-step plan

The context-aware additive voting foundations are now deployed and audited.

Next research/implementation priority:

1. Design completed-month contextual voting result rows derived from the new additive foundations.
2. Add `dimension_name = division_context` and the four certified context values to selected existing voting metric families rather than creating separate metric IDs per context.
3. Preserve existing metric IDs/formulas where possible and introduce a version change only if required by result-contract semantics.
4. Define a formal small-sample reliability rule for **member contextual participation** before public member comparison views are enabled. The research-supported candidate rule remains:
   - 25+ eligible divisions: normal display;
   - 10–24: caution;
   - 5–9: small-sample caution;
   - fewer than 5: insufficient/suppress comparison.
5. Keep party cohesion reliability thresholds unchanged initially.
6. Treat Independent outputs as recorded-vote agreement, not organizational party discipline.
7. Add reconciliation audits showing contextual monthly numerators/denominators sum back to the corresponding unfiltered monthly counts where appropriate.
8. Keep arbitrary-range consumers based on additive foundations rather than summing monthly percentages.
9. Defer Bill stage context until an exact stage relationship is separately certified.
10. Keep all voting outputs descriptive and denominator-explicit; do not present them as effectiveness, quality or performance.
