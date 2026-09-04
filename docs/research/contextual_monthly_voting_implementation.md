# Contextual monthly voting results implementation

## Status

Production deployment completed successfully on 2026-09-04.

Current production batch:

- `contextual-monthly-voting-20260904-1`

Implementation PR:

- PR `#99`

Deployment-fix PR:

- PR `#101`

Focused validation:

- `33896205686` — initial implementation compilation and regression tests passed.

Successful production deployment and post-promotion audit:

- `33896920206`

Read-only live footprint verification:

- `33897107153`

The deployment is additive. Existing unfiltered monthly voting rows remain in `monthly_metric_results`; new rows use `dimension_name = division_context`.

## Production result rows

Two existing metric IDs now also have contextual completed-month rows:

- `member_vote_participation_pct`
- `party_vote_cohesion_pct`

The metric IDs and formulas were not changed.

Contextual result identity adds:

- `dimension_name = division_context`
- `dimension_value` in:
  - `bill_or_legislation`
  - `motion_proceeding`
  - `procedural_business`
  - `other`

Values continue to use the existing **0–1 proportion scale**, despite the `_pct` suffix in the metric IDs.

## Member contextual participation

Formula remains:

`recorded_vote_count / eligible_member_division_count`

The division context is filtered first, then the additive numerator and denominator are aggregated for the completed month.

A formal contextual small-sample rule is now materialized:

- 25+ eligible divisions: `reliable`, normal public display;
- 10–24: `caution`, suitable with context;
- 5–9: `caution`, suitable with context;
- fewer than 5: `insufficient_for_comparison`, `not_certified`.

Warning codes:

- `none`
- `small_context_sample`
- `insufficient_context_sample`

The denominator remains visible in the result row and must be preserved by consumers.

## Party contextual recorded-vote agreement

The existing `party_vote_cohesion_pct` calculation is applied after filtering to the selected division context.

Rules remain unchanged:

1. historical party attribution is evaluated on the division date;
2. a party/division qualifies only with at least two recorded party-member votes;
3. aligned votes are the modal vote count in that party/division;
4. monthly value = aligned qualifying votes / all qualifying recorded party votes;
5. reliability remains:
   - 10+ qualifying divisions: `reliable`;
   - 5–9: `caution`;
   - fewer than 5: `insufficient_for_comparison`.

For the Independent grouping, contextual rows carry the `independent_group_agreement` warning where applicable. Public wording should describe this as **recorded-vote agreement among Independents**, not party discipline.

## Live footprint

Current `monthly_metric_results` row count:

- **191,356**

Contextual monthly voting rows:

- **11,844**

Completed months represented:

- **20**

Metric counts:

- member contextual participation: **11,266** rows
- party contextual recorded-vote agreement: **578** rows

Context counts across both metrics:

- `bill_or_legislation`: 2,916
- `motion_proceeding`: 3,099
- `procedural_business`: 3,464
- `other`: 2,365

Detailed metric/context counts:

### Member participation

- Bill: 2,773
- motion: 2,947
- procedural business: 3,295
- other: 2,251

### Party recorded-vote agreement

- Bill: 143
- motion: 152
- procedural business: 169
- other: 114

## Reliability footprint

Across all 11,844 contextual rows:

- `reliable`: 306
- `caution`: 4,788
- `insufficient_for_comparison`: 6,750

Public-use status:

- `suitable`: 172
- `suitable_with_context`: 4,922
- `not_certified`: 6,750

The high number of non-certified rows is expected and desirable: monthly context slices are often small, especially for individual members, and the new rule prevents unstable percentages from being presented as equivalent to large-denominator comparisons.

## Warning footprint

Current warning counts:

- `none`: 277
- `small_context_sample`: 4,675
- `insufficient_context_sample`: 6,419
- `small_division_sample`: 86
- `insufficient_division_sample`: 257
- `independent_group_agreement`: 29
- `small_division_sample;independent_group_agreement`: 27
- `insufficient_division_sample;independent_group_agreement`: 74

## Production audits

The successful deployment confirmed:

- contextual result primary keys are unique;
- all contextual rows use `dimension_name = division_context`;
- only the four certified division contexts appear;
- all contextual values remain on the 0–1 proportion scale;
- source batch is consistent across the republished monthly dataset;
- the full live monthly dataset has **0 duplicate primary keys**;
- post-promotion audit passed.

## Deployment failure and fix record

The first deployment attempt, run `33896315263`, failed safely before promotion.

### Failure 1 — mixed source batch provenance

The narrow candidate updater cloned existing monthly rows that still carried the previous production `source_batch_id`, then appended new contextual rows carrying the new candidate batch ID.

Diagnostic run:

- `33896480727`

Fix:

- restamp the entire republished monthly dataset to the new immutable candidate batch.

### Failure 2 — mixed CSV/Parquet column types

After provenance was fixed, the cloned CSV rows still had string-typed fields while the newly calculated rows had numeric Python types. PyArrow rejected the mixed object column during Parquet serialization.

Validation/diagnostic runs:

- `33896626356`
- `33896707234`

Fix:

- normalize string, integer and numeric result columns before candidate publication.

Final candidate-only validation:

- `33896818878` — focused tests passed and the real candidate monthly dataset published successfully without changing the production pointer.

Final production deployment:

- `33896920206` — candidate append, manifest assembly, atomic promotion and live audit all passed.

## Future full-batch behavior

`process/political_metrics_materialize_candidate.py` now builds contextual monthly rows automatically after the context-aware additive voting foundations are created.

Future full candidate batches therefore rebuild:

1. certified `division_context`;
2. context-aware daily voting numerators/denominators;
3. context-aware party division-vote components;
4. ordinary monthly metrics;
5. contextual monthly member participation and party agreement rows;
6. contextual monthly audit;
7. combined monthly primary-key audit.

The narrow updater exists only for controlled additive deployment to an already-produced production snapshot.

## Methodological guardrails

1. Contextual participation is recorded-vote participation, not physical attendance.
2. A higher contextual participation rate is not evidence of political effectiveness or representation quality.
3. Party recorded-vote agreement is descriptive; it is not a judgement about policy quality, discipline quality or political performance.
4. Independent-group results must not be described as party discipline.
5. Contexts describe parliamentary form/source relationship, not political importance.
6. Do not average monthly proportions to produce longer-period values; arbitrary ranges must be recalculated from additive foundations.
7. Preserve numerator, denominator, reliability and warning fields in public surfaces.
8. `not_certified` rows should not be silently ranked against reliable rows.
9. Bill stage must not be inferred from Bill ID plus division date.
10. Existing unfiltered voting rows remain valid and should not be replaced by contextual rows.

## Living next-step plan

The contextual monthly member-participation and party-agreement layer is now deployed.

Next research priority:

1. **Run a consumer-readiness audit of the new contextual monthly rows.**
   - identify which member/context/month combinations are reliable enough for default public comparison;
   - verify that Appsmith/Power BI/API filtering preserves denominator and warning metadata;
   - define default suppression/ranking behavior for `not_certified` rows;
   - verify Independent wording in party charts/tables.
2. Investigate whether `party_vote_participation_pct` should also receive `division_context` monthly rows. The additive foundation already supports this, but the public use case should be demonstrated before expanding the result surface.
3. Investigate whether constituency contextual participation is useful enough to justify monthly materialization; again, the additive denominator components already exist.
4. Keep arbitrary-period contextual views based on additive foundations rather than monthly percentage aggregation.
5. Develop certified Bill-specific voting histories through `division_context`/Bill linkage as an event-level consumer view, separate from monthly context summaries.
6. Continue to defer Bill-stage attribution until an exact stage relationship is certified.
7. Preserve politically neutral terminology: recorded voting participation, recorded-vote agreement, parliamentary context and sample reliability.
