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

## Historical correctness

Party and constituency attribution must use the political relationship valid on the event date. Current party or constituency values must not be applied retrospectively to historical speeches.

Temporal joins use inclusive validity intervals and fail when more than one history row matches the same member/event date. Ambiguity is treated as a data-quality error rather than resolved silently.

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

The first exposure helper therefore counts **eligible debate days**: debate dates that fall within the member's active Dáil membership interval.

This supports measures such as **Speeches per available debate day**.

An eligible debate day is not evidence of attendance. True attendance remains unsupported until an authoritative attendance source is available.

## First metric set

The initial catalogue contains:

- Speeches
- Speaking days
- Speeches per available debate day
- Share of Dáil speeches
- Party speeches
- Constituency speeches
- Dáil speeches
- Speeches per debate day

These are intentionally simple foundation measures. Issue, voting, question, ranking, percentile, diversity, and comparative metrics should build on the same temporal and period rules.

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

That is deliberate. The calculation semantics and tests should be reviewed first. After approval, materialized metric outputs should be attached to the existing Oireachtas candidate -> validation -> promotion lifecycle so source data and metrics cannot drift across batches.

## Next implementation steps

1. Run the historical coverage audit for membership, party, and constituency histories.
2. Wire canonical speech/debate inputs to the foundation calculators.
3. Add party and constituency exposure denominators.
4. Add source/join coverage outputs and reliability flags.
5. Add issue metrics only after classifier coverage/version gates are available in the same promoted batch.
6. Add materialized outputs and refresh orchestration after the metric results reconcile against canonical tables.
