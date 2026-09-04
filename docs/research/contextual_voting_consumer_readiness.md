# Contextual voting consumer-readiness audit

## Status

Read-only consumer-readiness audit completed on 2026-09-04 against production batch:

- `contextual-monthly-voting-20260904-1`

No production data, schema, metric formula or consumer application was changed.

Audit run:

- `33898417419`

Audit artifact:

- `analysis/contextual_voting_consumer_audit.json`

## Scope

The goal was to determine whether the newly deployed contextual monthly voting rows are safe for use by Appsmith, Power BI and API consumers.

This repository contains the production datasets, contracts and consumer guidance, but it does **not** contain the actual Appsmith application, Power BI model or API implementation.

Therefore this audit can certify:

- the live data shape;
- CSV/Parquet consistency;
- required metadata availability;
- safe filtering/ranking rules;
- contract requirements consumers should implement.

It cannot truthfully certify that external Appsmith, Power BI or API consumers already apply those rules unless their source/model definitions are provided separately.

## Live data checked

Contextual monthly voting rows:

- 11,844

Metric families:

- `member_vote_participation_pct`
- `party_vote_cohesion_pct`

Certified dimension:

- `dimension_name = division_context`

Context values:

- `bill_or_legislation`
- `motion_proceeding`
- `procedural_business`
- `other`

## Data-contract readiness

All 11,844 contextual rows have non-empty values for:

- `value`
- `numerator`
- `denominator`
- `reliability_status`
- `public_use_status`
- `warning_code`
- `source_batch_id`

This means consumers have the metadata required to make safe display decisions without recomputing reliability themselves.

CSV and Parquet live outputs have:

- equal row counts;
- identical primary-key sets.

This supports the existing recommendation:

- Appsmith/simple consumers may use CSV;
- Power BI/API models may use Parquet;
- both formats should produce the same contextual result identity.

## Critical ranking finding

There are:

- **6,750 `not_certified` contextual rows**.

All 6,750 still carry numeric metric values.

This is intentional: the value is mathematically calculable, but the denominator/sample size is too small for certified comparison.

A simulation of a naïve ranking flow—sorting by metric value without applying `public_use_status` first—showed:

- **703 `not_certified` rows** would appear in top-10 result groups.

A safe ranking flow that first removes `not_certified` rows produced:

- **0 `not_certified` rows** in top-10 result groups.

### Required public ranking order

Consumers should apply filters in this order:

1. select metric;
2. select completed month/date period;
3. select `division_context`;
4. exclude `public_use_status = not_certified` from default public comparisons/rankings;
5. only then sort/rank by `value`.

Do **not** sort first and hide warning metadata afterward.

## Default public-use rule

Recommended default public filter:

`public_use_status != 'not_certified'`

Interpretation:

- `suitable` — normal public display;
- `suitable_with_context` — public display allowed, but denominator/reliability/warning should remain visible or available in the same view;
- `not_certified` — exclude from default comparison/ranking tables and charts.

`not_certified` rows may still be available in explicitly diagnostic/internal views if they are clearly labelled and not ranked as equivalent evidence.

## Metadata that must travel with the value

Every consumer should retain:

- `numerator`
- `denominator`
- `reliability_status`
- `public_use_status`
- `warning_code`

For member participation, the denominator is the number of eligible member-division opportunities in that context/month.

For party recorded-vote agreement, numerator/denominator represent aligned and total qualifying recorded party votes; the reliability decision also depends on qualifying division count in the underlying calculation.

Consumers should not display only the proportion without its reliability/public-use metadata.

## Independent-group wording

The live result set contains:

- 130 contextual rows carrying `independent_group_agreement` in the warning code.

For those rows, public wording must be:

- **recorded-vote agreement among Independents**

Do not label this as:

- party discipline;
- party unity in an organizational sense;
- whipping effectiveness.

## Human-readable entity labels

Audit finding:

- all 11,844 contextual rows currently have `entity_name` equal to `entity_id`.

This is safe for identity but not ideal for public presentation because member IDs and party URIs are not human-friendly labels.

### Consumer implication

Appsmith, Power BI and API presentation layers should resolve display labels from the canonical member/party dimension sources rather than assuming `monthly_metric_results.entity_name` is presentation-ready.

This should be done as a lookup/join using stable entity IDs; it should not modify the metric identity or denominator logic.

A future pipeline enhancement could populate public entity names directly in `monthly_metric_results`, but that is a separate design decision because it affects the result contract and historical naming semantics.

## Appsmith readiness rules

The repository-side data is suitable for Appsmith if the app query/view applies these rules:

1. filter `metric_id` explicitly;
2. filter `period_start`/`period_end` explicitly;
3. filter `dimension_name = division_context` and desired `dimension_value`;
4. default to `public_use_status != not_certified` for comparison/ranking widgets;
5. display or expose denominator and reliability information;
6. show warning text for `suitable_with_context` rows;
7. resolve human member/party names using canonical dimensions;
8. use Independent-specific agreement wording.

The actual Appsmith app is not stored in this repository, so enforcement is **not yet verified**.

## Power BI readiness rules

The Parquet result is structurally suitable for Power BI.

Recommended model behavior:

1. treat metric identity as `metric_id + metric_version`;
2. retain `dimension_name` and `dimension_value` as part of filter context;
3. use a public-use dimension/filter so `not_certified` rows are excluded from default ranking visuals;
4. keep numerator and denominator fields in the model;
5. never aggregate `value` by sum or average across months;
6. for longer periods, use context-aware additive foundations rather than monthly proportions;
7. join human-readable member/party dimensions by stable entity ID;
8. surface `warning_code`/reliability in tooltips or contextual text.

The actual Power BI model is not stored in this repository, so enforcement is **not yet verified**.

## API readiness rules

A public API endpoint should not simply return every numeric row without interpretation metadata.

Recommended default endpoint behavior:

- return `value`, `numerator`, `denominator`, `reliability_status`, `public_use_status` and `warning_code` together;
- offer explicit inclusion of `not_certified` rows rather than including them in default comparison/ranking responses;
- never rank before applying the public-use filter;
- expose stable entity IDs plus resolved display labels;
- preserve `division_context` explicitly;
- use additive foundations for arbitrary-period calculations.

The actual API implementation is not stored in this repository, so enforcement is **not yet verified**.

## Safe default consumer policy

Recommended common policy for all public consumers:

1. **Filter first, rank second.**
2. Exclude `not_certified` rows from default public rankings/comparison charts.
3. Never hide denominator/reliability metadata from contextual percentages.
4. Preserve `suitable_with_context` rows, but show their caution/warning state.
5. Resolve human-readable names from canonical entity dimensions.
6. Treat Independent results as agreement among Independents.
7. Never average monthly proportions to build longer-period values.
8. Use additive foundations for arbitrary periods.
9. Do not present participation/agreement/context as effectiveness, quality or political performance.

## Architecture conclusion

No new production metric foundation is required for consumer safety.

The main remaining work is consumer-layer enforcement and presentation design.

Because the actual Appsmith, Power BI and API implementations are outside this repository, the safest next step is **not** another data-pipeline schema change.

Instead, consumer implementations should be audited or updated individually when their source/model definitions are available.

## Living next-step plan

1. Obtain or identify the actual Appsmith query/widget definitions that consume political metrics.
2. Audit Appsmith for filter-before-rank behavior, denominator display, warning handling and entity-label resolution.
3. Obtain or identify the Power BI model/query definitions.
4. Audit Power BI relationships, measures and visual-level filters so monthly proportions are not averaged and `not_certified` rows are excluded from default rankings.
5. Obtain or identify the API code/schema serving political metrics.
6. Audit API defaults so reliability/public-use/warning metadata travels with every contextual result.
7. Only after concrete consumer evidence exists, decide whether pipeline-side human-readable `entity_name` population is worth a separate contract change.
8. Investigate contextual `party_vote_participation_pct` and constituency participation only after a demonstrated consumer use case exists.
9. Keep arbitrary-range contextual calculations on additive foundations.
10. Preserve politically neutral language and existing denominator definitions.
