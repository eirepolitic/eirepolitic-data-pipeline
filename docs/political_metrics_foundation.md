# Political metrics foundation

## Purpose

This layer turns canonical political data into reusable, explainable measures for the public, dashboards, APIs, analysis, and future content systems.

It deliberately sits between canonical Oireachtas data and presentation layers:

`source facts -> canonical political data -> political metrics -> presentation`

Public wording does not define the calculation. Each metric keeps both a plain-English explanation and a precise technical definition.

## Public-first design

Every public metric should answer four questions clearly:

1. What is being measured?
2. What does a higher or lower value mean?
3. What is the fair comparison or denominator?
4. What should the number not be interpreted as?

For example, `member_speaking_day_count` is presented as **Speaking days** and explicitly states that it is not an attendance measure.

## TD and non-TD speech scope

The canonical Dáil speech data can include recorded contributions by identified people who are not active TDs on that date.

Those speeches remain part of broad measures such as **Dáil speeches**, because they are genuine recorded contributions to the debate. They are not forced into TD, party, or constituency statistics.

For TD-level measures, an identified speaker must have an active Dáil membership covering the speech date. If a known Dáil member has no membership interval covering that date, the metric audit treats that as a history gap rather than silently including or excluding the record.

## Historical correctness

Party and constituency attribution use the political relationship valid on the event date. Current party or constituency values must not be applied retrospectively to historical speeches.

The Oireachtas history data uses transitions where one relationship can end on the same date the next one starts. The metrics layer therefore interprets these source ranges as **start-inclusive and end-exclusive**:

`start_date <= event_date < end_date`

An open end date means the relationship continues. A same-day handover is a valid transition, while a true overlap still fails validation rather than being resolved silently.

### Party affiliation wording

Party metrics use the **Oireachtas-listed affiliation for the relevant date**. This is the most reproducible basis available in the canonical parliamentary data.

Public descriptions should not claim that this always proves a party's separate internal membership, suspension, disciplinary, or organisational status. The catalogue uses the provenance field `affiliation_basis: oireachtas_listed_affiliation` for party measures where relevant.

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

Political periods are defined in Europe/Dublin calendar dates.

Dáil-term and formal sitting/session resolution remain planned until the required canonical term/calendar source is formally wired into this layer.

## Fair member comparisons

A member who joins or leaves during a period should not automatically be compared as if they had been active for the full period.

The exposure helpers therefore count **eligible debate days**: debate dates that fall within the TD's active Dáil membership interval. Party and constituency exposure is also split across historical changes on the correct dates.

This supports measures such as **Speeches per available debate day**, **Speeches per TD**, and **Speeches per TD representing the constituency**.

An eligible debate day is not evidence of attendance. True attendance remains unsupported until an authoritative attendance source is available.

## First metric set

The initial catalogue contains:

- Speeches
- Speaking days
- Speeches per available debate day
- Share of TD speeches
- Party speeches
- Constituency speeches
- Dáil speeches
- Speeches per debate day
- Speeches per TD
- Speeches per TD representing the constituency

These are intentionally simple foundation measures. Issue, voting, question, ranking, percentile, diversity, and comparative metrics should build on the same temporal, eligibility, and period rules.

## Historical audit

`process/political_metrics_audit.py` is a read-only audit of the currently promoted Oireachtas batch. It checks:

- membership, party, and constituency history date ranges
- true temporal overlaps
- TD eligibility on speech dates
- period-correct party attribution
- period-correct constituency attribution
- national/identified speech reconciliation
- examples of unmatched or out-of-scope records for diagnosis

`.github/workflows/political_metrics_historical_audit.yml` runs the audit manually and uploads both a machine-readable JSON report and a plain-language Markdown summary. The workflow does not publish data or write metric outputs to S3.

Historical certification applies only to the date range actually present and covered in the promoted batch. A structurally correct schema must not be interpreted as proof that older Dáil terms have already been backfilled.

## Reliability and future public metrics

Future comparative metrics should carry enough context to avoid misleading small-sample claims. Public outputs should eventually include, where relevant:

- numerator
- denominator
- eligible population
- source coverage
- reliability status
- warning codes
- metric version
- source batch ID

Technical diagnostics may be used internally without exposing statistical jargon to the public.

## Materialization

This foundation does not yet write metric outputs to S3 or alter production refresh workflows.

That is deliberate. The calculation semantics, live historical audit, and tests should be reviewed first. After approval, materialized metric outputs should be attached to the existing Oireachtas candidate -> validation -> promotion lifecycle so source data and metrics cannot drift across batches.

## Next implementation steps

1. Keep the historical audit as a gate for public historical speech measures.
2. Wire canonical speech/debate inputs to a non-writing metric commissioning run.
3. Reconcile sample member, party, constituency, and national results against source facts.
4. Add source/join coverage and reliability fields to metric result contracts.
5. Add issue metrics only after classifier coverage/version gates are available in the same promoted batch.
6. Add materialized outputs and refresh orchestration after commissioned metric results pass reconciliation.
