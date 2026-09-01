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

Separate non-publishing commissioning runners exist for core speech, issue, voting, parliamentary-question, and materialization validation.

Historical certification applies only to the date range actually represented and covered in the promoted batch. A structurally correct history model does not prove that older Dáil terms have already been backfilled.

## Approved materialization design — Option A

The approved design keeps political metrics inside the same immutable Oireachtas batch as the canonical data that produced them.

`configs/political_metrics/materialization.yml` defines the contract.

### Daily/event foundations

The materialization layer supports:

- `daily_activity_components`
- `daily_issue_activity`
- `division_party_vote_components`
- `daily_question_dimensions`

These store additive components used to rebuild arbitrary date-range measures without rescanning raw source facts or averaging already-calculated percentages.

National issue populations are deliberately separated:

- `entity_id=dail` = all recorded Dáil speakers
- `entity_id=eligible_tds` = speeches/questions belonging to active TDs and suitable for TD/party comparison baselines

This prevents broad Dáil totals from being accidentally mixed with TD-only comparison populations.

### Completed-month results

`monthly_metric_results` is a long-form consumer dataset. It stores one row per metric/entity/dimension/month with:

- `metric_id`
- `metric_version`
- period start/end
- grain and entity
- optional dimension such as issue, question type or question recipient
- value
- numerator
- denominator
- output unit
- reliability/public-use status
- warning code
- source batch ID
- calculation timestamp
- contract version

Metrics without a dimension use the explicit sentinel values `dimension_name=none` and `dimension_value=none`; nulls are not used in the primary key.

Only fully completed calendar months are included in the monthly consumer dataset. The current partial month is never emitted beside completed months as if it were directly comparable.

### Candidate-batch safety

`political_metrics/candidate_publish.py` writes metric files only to immutable candidate-batch paths and records them as batch entries. It cannot update `production.json` or `previous.json`.

The Oireachtas batch key mapper supports logical metric keys under:

`processed/oireachtas_unified/latest/metrics/...`

which map to:

`processed/oireachtas_unified/batches/<batch_id>/metrics/...`

`process/political_metrics_materialize_candidate.py` reads canonical data from the candidate itself, validates issue-classifier completeness, rebuilds the daily/event foundations across available history, rebuilds all completed-month result rows, writes the five metric datasets into the same candidate, and records each as a validated batch entry.

The candidate manifest requires all five metric entries before promotion can succeed:

- `political_metrics_daily_activity_components`
- `political_metrics_daily_issue_activity`
- `political_metrics_division_party_vote_components`
- `political_metrics_daily_question_dimensions`
- `political_metrics_monthly_metric_results`

## Approved classifier policy — A1

A1 is implemented in `.github/workflows/oireachtas_refresh_reusable.yml`.

The normalized refresh inputs now expose whether the actual refresh table set changes `silver_speeches`.

The candidate lifecycle is:

1. seed the candidate from the current promoted batch;
2. rebuild the requested canonical tables;
3. if `silver_speeches` changed, run the speech issue classifier before metrics;
4. rebuild political metrics from that candidate;
5. require all political-metric entries during candidate manifest assembly;
6. validate the complete candidate;
7. only then allow the existing promotion workflow to update the production pointer.

The manual `classify_speeches` input remains available as an explicit force-run, but a speech-changing candidate no longer depends on that flag being set correctly.

With the current default refresh table sets, weekly refreshes change `silver_speeches` and therefore classify automatically. Monthly and yearly refreshes do not currently rebuild `silver_speeches`, so they reuse the candidate's already-classified seeded speech snapshot and do not incur unnecessary classifier calls. If a future monthly/yearly/custom refresh adds `silver_speeches`, A1 automatically requires classification for that candidate.

## Integration validation

Candidate-only integration run `33469008937` passed after regression fixes for completed-month date handling and zero-division months.

The successful test:

- passed the political-metrics unit suite;
- seeded a throwaway candidate from production;
- rebuilt full-history political metric foundations and completed-month results;
- wrote the metric datasets only inside the throwaway candidate batch;
- assembled a validated candidate manifest with all five metric datasets required; and
- verified that the production pointer did not point to the throwaway candidate.

The temporary integration harness was removed after the successful run. PR #42 remains the review boundary; no merge or production promotion was performed as part of this work.
