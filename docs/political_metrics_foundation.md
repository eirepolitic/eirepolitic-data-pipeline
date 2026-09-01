# Political metrics foundation

## Purpose

This layer turns canonical political data into reusable, explainable measures for the public, dashboards, APIs, analysis, and future content systems.

It sits between canonical Oireachtas data and presentation layers:

`source facts -> canonical political data -> political metrics -> presentation`

Each metric keeps both a plain-English public explanation and a precise technical definition.

## Public-first design

Every public metric should make four things clear:

1. What is being measured?
2. What does a higher or lower value mean?
3. What is the comparison or denominator?
4. What should the number not be interpreted as?

Technical measures can exist for validation and analysis without exposing statistical jargon to the public.

## Historical correctness

TD, party, and constituency measures use the political context valid on the event date. Current affiliation or representation must never be applied retrospectively.

The Oireachtas history data is interpreted as start-inclusive and end-exclusive:

`start_date <= event_date < end_date`

This supports same-day transitions where one relationship ends on the same date the next begins. True overlaps fail validation rather than being resolved silently.

### Party affiliation wording

Party measures use the **Oireachtas-listed affiliation for the relevant date**. This is the reproducible parliamentary source available to the pipeline.

Public wording must not imply that this independently proves a party's internal membership, suspension, disciplinary, or organisational status.

## TD and non-TD scope

Broad Dáil speech measures can include recorded contributions by identified people who are not active TDs on that date.

TD, party, and constituency measures require active Dáil membership on the event date. An identified non-TD contribution remains part of broad Dáil activity but is not forced into TD/party/constituency statistics.

The same active-membership rule is reused for speeches, divisions/votes, and parliamentary questions.

## Period handling

The foundation currently resolves:

- calendar month (`YYYY-MM`)
- calendar year (`YYYY`)
- quarter (`YYYY-Q1` to `YYYY-Q4`)
- `last_completed_month`
- `rolling_7d`
- `rolling_30d`
- `rolling_90d`
- explicit inclusive date ranges

Political periods use Europe/Dublin calendar dates.

Dáil-term and formal sitting/session resolution remain planned until the required canonical term/calendar source is formally wired into this layer.

## Speech measures

The commissioned speech catalogue includes public measures such as:

- **Speeches**
- **Speaking days**
- **Speeches per available debate day**
- **Share of TD speeches**
- **Party speeches**
- **Constituency speeches**
- **Dáil speeches**
- **Speeches per debate day**
- **Speeches per TD**
- **Speeches per TD representing the constituency**

Debate-day-normalized measures use the authoritative debate-date universe from `silver_debate_records`, not only dates that happen to contain speech rows.

An eligible debate day is not evidence of attendance. True attendance remains unsupported until an authoritative attendance source is available.

## Issue measures

Issue metrics are gated by a one-to-one classification quality check. For every speech in scope, the gate verifies:

- one classification row per `speech_id`
- matching source speech-text hash
- approved issue label
- final classification status
- populated label provenance
- model name where a model produced the label

Public issue measures include:

- **Speeches about an issue**
- **Share of policy speeches about an issue**
- **Share of a TD's policy speeches about an issue**
- **Party focus on an issue**
- **Party emphasis compared with all TDs**
- **Party emphasis compared with the average party**
- **Constituency focus on an issue**

`NONE` is excluded from policy-share denominators but remains part of classifier coverage reporting.

The all-TD baseline and average-party baseline are intentionally different. The average-party baseline is an unweighted mean across eligible parties; parties with fewer than 20 policy-labelled speeches are excluded from that baseline, and the synthetic Independent grouping is excluded from the average-party comparison population.

## Voting measures

Voting metrics use **eligible divisions**, not sitting days, as the denominator. A TD is eligible for a division when the division date falls within their active Dáil membership period.

Public measures include:

- **Recorded voting participation**
- **Party recorded voting participation**
- **Party voting unity**
- **Constituency representatives' recorded voting participation**
- **Dáil divisions**

A missing recorded vote is not proof of physical absence. These are voting-record measures, not attendance measures.

Party voting unity describes how often recorded party votes matched the party's most common recorded vote in qualifying divisions. Divisions with fewer than two recorded party voters are excluded. Reliability is flagged as:

- reliable: at least 10 qualifying divisions
- caution: 5-9 qualifying divisions
- insufficient for comparison: fewer than 5

## Parliamentary-question measures

The first question measures are deliberately limited to concepts supported directly by `silver_questions`:

- **Parliamentary questions submitted**
- **Party parliamentary questions**
- **Parliamentary questions from constituency TDs**
- **Share of questions by type**
- **Ministers or departments questioned**
- **Share of questions sent to a minister or department**

Question totals use active Dáil membership and event-date party/constituency context.

A size-adjusted **questions per TD** measure has deliberately not been defined yet. Choosing calendar-day, active-month, question-day, or another exposure basis would materially change the meaning, so that denominator should be approved separately before implementation.

## Historical audit and commissioning

`process/political_metrics_audit.py` is a read-only audit of the promoted Oireachtas batch. It checks history ranges, TD eligibility, historical attribution, and reconciliation.

`.github/workflows/political_metrics_historical_audit.yml` runs the historical audit manually and writes only workflow artifacts, not S3 metric outputs.

Separate non-publishing commissioning runners now exist for:

- core speech metrics
- issue metrics
- voting metrics
- parliamentary-question metrics

Commissioning runs calculate real results from promoted canonical data, perform source and reconciliation checks, and write only temporary workflow artifacts for review.

Historical certification applies only to the date range actually represented and covered in the promoted batch. A structurally correct history model does not prove that older Dáil terms have already been backfilled.

## Reliability and result context

Public/comparative metric results should carry enough context to avoid misleading claims. Depending on metric type, outputs should include:

- numerator
- denominator
- eligible population
- source coverage
- reliability status
- warning codes
- metric version
- source batch ID
- period start/end

The public label should remain simple even when the backend calculation is technical.

## Materialization status

The foundation does **not** yet write metric outputs to S3 or alter production refresh workflows.

This is deliberate. Materialization affects storage, refresh behaviour, historical reproducibility, and downstream contracts, so it is a separate architecture decision.

The recommended next design is:

1. materialize small additive daily foundations inside the same validated Oireachtas candidate batch;
2. materialize standard completed-month aggregates for common downstream use;
3. calculate arbitrary-range comparisons, ranks, percentages and indexes from those foundations rather than rescanning raw speeches;
4. promote metric outputs only when their source canonical batch and metric validations pass together;
5. store `source_batch_id`, `metric_id`, `metric_version`, period, numerator/denominator and reliability metadata with every materialized result.

No production S3 or refresh wiring should be added until this materialization contract is approved.
