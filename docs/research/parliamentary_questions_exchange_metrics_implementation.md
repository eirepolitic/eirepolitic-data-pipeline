# Oral-question exchange participant implementation

Status: **Implemented and candidate-validated; pending merge and production promotion**  
Date: **2 September 2026**  
Parent research: [Oral-question exchange participant metrics](parliamentary_questions_exchange_metrics.md)

This note records the implementation and isolated-candidate validation of the exchange participant architecture certified in the preceding research.

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

The primary key remains:

`debate_section_id`

### New `oral_question_exchange_participants`

A new deterministic foundation records observed participation at:

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

Certified role values:

- `ministerial`;
- `chair`;
- `ordinary_member`;
- `collective_or_unidentified`.

The role is part of the primary key because an identified TD can genuinely contribute as an ordinary member and later act as chair in the same exchange.

### Explicit separation from question-taker attribution

This implementation describes **observed transcript participation only**.

It does not materialize `taken_by_member_code` and does not infer who formally took a submitted question.

The previously researched question-taker evidence remains separate and bounded.

## Materialization and consumer contracts

The materialization contract now includes the new participant foundation.

The candidate political-metrics manifest therefore requires **eight datasets**:

1. `daily_activity_components`
2. `daily_issue_activity`
3. `division_party_vote_components`
4. `daily_question_dimensions`
5. `oral_question_sections`
6. `oral_question_exchange_participants`
7. `speech_question_context`
8. `monthly_metric_results`

Downstream logical paths were added for CSV and Parquet participant outputs.

Consumer rules explicitly state that the participant foundation describes observed exchange participation and must not be used to infer question-taker attribution.

## Regression coverage

Unit/regression tests now cover:

- grouped oral questions count exchange speeches once;
- exchange word/intervention components reconcile;
- participating submitter count/share;
- ordinary non-submitting TD counts;
- ministerial, chair, ordinary-member and collective/unidentified partitions;
- participant primary-key uniqueness at exchange + participant + role grain;
- the same member may occupy both ordinary-member and chair roles in one exchange;
- collective/unidentified transcript speakers retain no identified `member_code`;
- written questions do not create oral-question exchanges;
- speech-question context remains one row per source speech.

## Feature validation

Feature audit run **33655165749** passed:

- political-metrics unit/regression tests;
- historical metrics audit;
- production parliamentary-question/speech relationship audit.

A later permanent-audit enhancement adds the participant foundation itself to the production relationship audit, including exact intervention/word reconciliation and role validation.

## Disposable production-seeded candidate

Candidate batch:

`oral-exchange-participants-33655165749-1`

Workflow run:

**33655415884**

The candidate was seeded from the exact current production snapshot.

No source refresh was performed.

No speech classifier was run.

No question classifier was built or run.

The production pointer was not changed.

### Candidate build checks

The following all passed:

- production snapshot seed;
- revised political-metrics materialization;
- exact exchange/participant reconciliation against candidate source speeches;
- participant primary-key uniqueness;
- allowed participant-role validation;
- collective/unidentified member-code validation;
- eight-dataset political-metrics manifest assembly.

The full political-metrics materializer completed in approximately **2 minutes 12 seconds** on the production-sized candidate snapshot. This is acceptable for the current pipeline cadence and does not require optimization before merge.

### Exact reconciliation checks

The candidate validation asserted that:

- sum of `oral_question_sections.related_speech_count` equals unique source speech IDs in oral-question sections;
- sum of participant `intervention_count` equals the same unique source speech count;
- sum of exchange `related_speech_word_count` equals source oral-exchange speech words;
- sum of participant `word_count` equals the same source word total;
- participant roles are limited to the certified set;
- collective/unidentified rows do not carry an identified member code.

All assertions passed.

### Full downstream validation

The same candidate then passed the normal downstream validation stack in run **33655415884**:

- candidate-local auxiliary enrichments;
- downstream staging;
- compatibility adapters;
- downstream contract checks;
- strict compatibility and mismatch validation;
- year-aware member metrics;
- candidate-only Instagram consumer smoke test;
- final candidate manifest reassembly.

This provides evidence that the new metric foundation does not break existing downstream consumers.

## Permanent audit requirement

Before merge, `process/political_metrics_question_context_audit.py` was extended so future production audits also validate:

- `oral_question_exchange_participants` against its materialization contract;
- participant primary-key uniqueness;
- participant role values;
- participant intervention totals against source oral-exchange speech IDs;
- participant word totals against source oral-exchange word counts;
- exchange intervention/word totals against the same source data;
- exchange role partitions;
- collective/unidentified rows remaining outside identified-member attribution;
- zero question-classifier calls;
- no question-taker attribution materialization.

This prevents the candidate-only reconciliation from becoming a one-off check.

## Implementation decisions preserved

1. **Observed participation is separate from question-taking.**
2. **Role is part of participant grain.** A member can change role inside one exchange.
3. **Anonymous/collective transcript speakers stay in totals but not TD counts.**
4. **Word and intervention components are additive; shares must be recomputed from components.**
5. **Grouped questions never multiply section speech rows.**
6. **Question classification remains deferred.**

## Revised next-steps plan

### 1. Merge the validated implementation to `main`

Only after the enhanced permanent audit passes on the feature branch.

The merge should include:

- code;
- contracts;
- regression tests;
- eight-dataset manifest requirement;
- enhanced permanent production audit;
- this implementation evidence.

### 2. Run a fresh structure-only candidate from merged `main`

Do not promote the feature-branch disposable candidate.

After merge:

- seed a new candidate from the then-current production batch;
- run the merged-main materializer;
- require all eight political-metrics datasets;
- run exact participant/exchange reconciliation;
- run the full downstream validation/consumer stack;
- verify no classifier calls and no source refresh.

This ensures the deployable candidate is produced by the exact merged code.

### 3. Promote only that merged-main candidate

If the merged-main candidate passes all checks:

- promote the exact candidate;
- verify the production pointer;
- verify all eight logical metric paths resolve into the promoted batch;
- verify participant PK uniqueness and allowed role values;
- rerun production inventory and political-metrics relationship audits.

### 4. Update research after promotion

Record:

- merged PR/commit;
- validation run;
- promotion run;
- new production batch ID;
- final production audit runs;
- confirmed live row counts for sections and participant-role rows.

### 5. Continue investigation with section-heading normalization

Once the participant foundation is live, investigate whether Oireachtas section headings can provide a stable deterministic topical hierarchy for Oral questions.

The investigation should measure:

- heading uniqueness/reuse;
- spelling and formatting variants;
- normalization opportunities;
- relationship to recipients;
- stability over time;
- whether headings support useful public filters without pretending to be the EirePolitic issue taxonomy.

### 6. Question issue classification remains deferred

The deterministic structure continues to produce useful analytical value. Do not classify the full question history unless a concrete unmet use case later justifies the cost.
