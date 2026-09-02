# Oral-question exchange participant implementation

Status: **Live in production and audited**  
Date: **2 September 2026**  
Parent research: [Oral-question exchange participant metrics](parliamentary_questions_exchange_metrics.md)

This note records the implementation, candidate validation, production deployment and current next-step plan for the deterministic Oral Parliamentary Question exchange participant architecture.

## Implemented structure

### Expanded `oral_question_sections`

The existing one-row-per-exchange foundation now includes deterministic exchange components for:

- recorded submitting-member participation;
- grouped vs single-question status;
- ordinary non-submitting TD participation;
- total transcript intervention and word volume;
- ministerial intervention/word components and word share;
- chair intervention/word components;
- ordinary-member intervention/word components;
- collective/unidentified intervention/word components.

Primary key:

`debate_section_id`

### `oral_question_exchange_participants`

The participant foundation records observed participation at:

`debate_section_id + participant_key + participant_role`

Fields include:

- `debate_section_id`;
- `debate_date`;
- `participant_key`;
- `member_code` when identified;
- `speaker_name`;
- `participant_role`;
- `is_recorded_submitter`;
- `intervention_count`;
- `word_count`;
- standard provenance/version fields.

Certified roles:

- `ministerial`;
- `chair`;
- `ordinary_member`;
- `collective_or_unidentified`.

Role remains part of the primary key because one TD can genuinely contribute as an ordinary member and later act as chair in the same exchange.

### Question-taker attribution remains separate

This foundation records **observed transcript participation only**.

It does not materialize `taken_by_member_code` and does not infer who formally took a submitted question.

The separately researched substitute/question-taker evidence remains bounded to explicit evidence and is not required for exchange participation metrics.

## Materialization and consumer contracts

The political-metrics manifest now requires **eight datasets**:

1. `daily_activity_components`
2. `daily_issue_activity`
3. `division_party_vote_components`
4. `daily_question_dimensions`
5. `oral_question_sections`
6. `oral_question_exchange_participants`
7. `speech_question_context`
8. `monthly_metric_results`

CSV and Parquet logical paths exist for the participant dataset.

Consumer contracts state explicitly that exchange participants represent observed participation and cannot be used as a question-taker inference.

## Regression and permanent audit coverage

Tests and audits protect the following invariants:

- grouped oral questions count each exchange speech once;
- exchange intervention/word totals reconcile to source `silver_speeches`;
- participant intervention/word totals reconcile to the same source rows;
- participant PK is unique at exchange + participant + role grain;
- the same member may hold ordinary-member and chair roles within one exchange;
- collective/unidentified speakers retain no identified `member_code`;
- written questions do not create oral-question exchanges;
- `speech_question_context` remains one row per source speech;
- role partitions reconcile to total interventions and words;
- question-taker attribution is not materialized;
- question-classifier calls remain zero.

`process/political_metrics_question_context_audit.py` permanently validates these rules against production.

## Pre-merge validation evidence

Feature audit **33655165749** passed:

- political-metrics unit/regression tests;
- historical metrics audit;
- production parliamentary-question/speech relationship audit.

Disposable feature-branch candidate:

`oral-exchange-participants-33655165749-1`

Run **33655415884** passed:

- production snapshot seed;
- political-metrics materialization;
- exact exchange/participant reconciliation;
- eight-dataset manifest assembly;
- candidate-local auxiliary enrichments;
- downstream staging;
- compatibility adapters;
- downstream contracts;
- strict compatibility/mismatch checks;
- year-aware member metrics;
- consumer smoke;
- final manifest reassembly.

Enhanced permanent audit **33656105298** also passed before merge.

PR **#53** merged the implementation to `main`.

No source refresh or question classifier was used in these validation runs.

## Merged-main deployment validation

A fresh candidate was created after merge from the then-current production snapshot.

Candidate batch:

`structure-oral-exchange-participants-20260902-1`

Merged-main validation run:

**33678027849** — **SUCCESS**

The first temporary wrapper attempt, run **33677930521**, failed at workflow startup before any job or S3 operation. It was an orchestration-wrapper issue only; no candidate or production data was changed. The validation steps were then run directly in the temporary wrapper.

### Merged-main candidate checks passed

Run 33678027849 passed:

- exact production snapshot seed;
- merged-main political-metrics materialization;
- exact exchange/participant reconciliation;
- participant PK uniqueness;
- allowed participant-role validation;
- collective/unidentified attribution guardrail;
- eight-dataset manifest assembly;
- candidate-local auxiliary enrichments;
- downstream staging;
- compatibility adapters;
- downstream contracts;
- strict compatibility/mismatch validation;
- year-aware member metrics;
- candidate-only Instagram consumer smoke;
- final manifest reassembly.

Materialization completed in approximately **1 minute 55 seconds** in this deployment candidate.

No source refresh was performed.

No speech classifier was run.

No question classifier was run.

## Production promotion

Promotion run:

**33678430937** — **SUCCESS**

The exact validated candidate was promoted:

`structure-oral-exchange-participants-20260902-1`

Post-promotion verification in the same run confirmed:

- the production pointer resolves to the exact validated candidate;
- all **eight** political-metrics logical paths resolve inside that production batch;
- participant PK remains unique;
- participant role values remain limited to the certified set;
- collective/unidentified participant rows do not carry identified TD member codes;
- question-classifier calls remain zero.

## Live production counts

Read-only live-count verification run:

**33678761232** — **SUCCESS**

Current production counts for the new question-exchange structures:

- `oral_question_sections`: **2,127 rows**;
- `oral_question_exchange_participants`: **6,133 rows**;
- `speech_question_context`: **66,192 rows**;
- participant primary key unique: **yes**.

Participant-role rows:

| Role | Rows |
| --- | ---: |
| ordinary_member | **3,432** |
| ministerial | **2,180** |
| chair | **517** |
| collective_or_unidentified | **4** |

The participant-role row count differs from raw intervention count because each row aggregates one participant-role combination within one exchange.

## Post-promotion production audits

### Production inventory

Run **33678504760** — **SUCCESS**.

### Political metrics historical + question relationship audit

Run **33678513654** — **SUCCESS**.

This passed:

- political-metrics tests;
- read-only historical metrics audit;
- live parliamentary-question / speech relationship audit;
- new exchange-participant reconciliation against production;
- audit summary/artifact creation.

The production deployment is therefore considered certified and complete.

## Decisions preserved

1. **Observed participation is separate from question-taking.**
2. **Role is part of participant grain.**
3. **Anonymous/collective transcript speakers remain in totals but outside identified-TD counts.**
4. **Word/intervention components are additive; shares must be recomputed from components.**
5. **Grouped questions never multiply section speech rows.**
6. **Question issue classification remains deferred.**
7. Production deployment must continue to use seed → candidate → reconciliation → downstream validation → promotion → production audit rather than writing metrics directly to live paths.

## Living next-steps plan

### 1. Investigate section-heading normalization

This is now the immediate research task.

Goal: determine whether Oireachtas debate-section headings provide a stable, useful **deterministic topical layer for Oral question exchanges** without pretending to be the EirePolitic issue taxonomy.

Measure:

- total heading uniqueness and reuse;
- heading frequency distribution;
- spelling/capitalization/punctuation variants;
- near-equivalent headings that can be normalized safely;
- headings that are procedural rather than topical;
- relationship between headings and question recipient/department;
- heading stability across time;
- grouped versus single-question behaviour by heading;
- whether recurring headings support useful public filters and post formats.

Do not use AI classification in this phase.

### 2. Decide whether a deterministic heading dimension is worthwhile

If normalization is stable, consider a small foundation/dimension with:

- raw section heading;
- normalized heading;
- normalization rule/version;
- optional broader source-derived family only where deterministic;
- provenance.

Do not force headings into the existing speech issue taxonomy.

### 3. Compare Oral and Written scrutiny

Once the Oral exchange layer is settled, compare:

- recipient mix;
- TD use of Oral versus Written questions;
- portfolio specialization;
- grouped Oral participation;
- departments receiving high Written PQ volume but comparatively little Oral exchange activity, and vice versa;
- member/party/constituency channel profiles.

### 4. Continue broader speech-context work

Return to deterministic context categories only where section/source metadata can prove them safely, such as:

- Leaders' Questions;
- legislation/Bills;
- motions;
- statements;
- procedural/business;
- other.

### 5. Question issue classification remains deferred

The deterministic question structure continues to produce useful analytical value without classifier cost.

Do not classify the full ~121k question history until a concrete unmet use case justifies the backfill and ongoing cost.
