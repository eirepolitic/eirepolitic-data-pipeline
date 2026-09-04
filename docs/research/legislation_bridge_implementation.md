# Certified Bill-section bridge implementation

## Status

Production deployment completed successfully on 2026-09-03.

The additive `bill_debate_sections` foundation is now live.

Current production batch:

- `certified-bill-sections-20260903-1`

Successful deployment run:

- `33825032483`

Implementation PRs:

- PR `#64` — production implementation, contracts, tests and deployment workflow
- PR `#65` — deployment-workflow import-path fix

The first deployment attempt, run `33824958231`, failed during the preflight test step before candidate seeding or production promotion. It made no production change.

## Purpose

Materialize a conservative deterministic relationship between Bills and canonical parliamentary debate sections so speeches and divisions can inherit Bill context without whole-debate over-attribution or row multiplication.

This implementation follows the research and plan recorded in:

- `docs/research/legislation_investigation.md`
- `docs/research/legislation_bridge_implementation_plan.md`

## Production dataset

Dataset:

- `bill_debate_sections`

Logical locations:

- `processed/oireachtas_unified/latest/metrics/event/bill_debate_sections/csv/bill_debate_sections.csv`
- `processed/oireachtas_unified/latest/metrics/event/bill_debate_sections/parquet/bill_debate_sections.parquet`

Production grain:

- one row per `(bill_id, debate_section_id)`

The production bridge is certified-only. Unmatched, conflicting and multi-Bill source cases are excluded rather than guessed.

## Certification rule implemented

A Bill-debate source record is eligible only where all of the following are true:

1. `(bill_debate.debate_id, bill_debate.debate_section_id)` resolves to one canonical debate section through `(debate_section.debate_id, debate_section.section_eid)`.
2. The Bill source `debate_show_as` value resolves to one exact section heading within that same debate.
3. The source-section check and exact-heading check identify the same canonical `debate_section_id`.
4. The canonical section is associated with only one distinct Bill ID across eligible rows.

Eligible raw Bill-debate rows are then collapsed to one `(bill_id, debate_section_id)` record.

Multiple source `bill_debate_id` values for the same certified Bill-section pair are retained as provenance rather than emitted as duplicate bridge rows.

## Live production footprint

The successful candidate and post-promotion audit confirmed the expected initial footprint:

- 371 `bill_debate_sections` rows
- 168 distinct Bills
- 371 distinct debate sections
- 7,352 speeches linked through exact `debate_section_id`
- 168 divisions linked through exact `debate_section_id`

The 371 production rows are the collapsed form of 396 certified source Bill-debate records. Twenty-five duplicate source records above the production grain do not multiply speeches or divisions.

These counts are a snapshot of the current production source batch, not permanent constants.

## Implementation components

Production implementation added:

- `political_metrics/legislation_context.py`
  - deterministic bridge builder
  - bridge audit function
- `process/political_metrics_bill_debate_sections_candidate.py`
  - additive candidate materializer for the bridge
- `process/political_metrics_materialize_candidate.py`
  - standard candidate materialization now also rebuilds `bill_debate_sections`
- `configs/political_metrics/materialization.yml`
  - materialization contract and primary key
- `configs/political_metrics/downstream_contracts.yml`
  - consumer semantics and exact join rules
- `tests/political_metrics/test_legislation_context.py`
  - regression tests for composite section keys, duplicate collapse, conflicts, multi-Bill exclusion and join multiplicity
- `.github/workflows/bill_debate_sections_deploy.yml`
  - controlled seed/build/validate/promote/post-audit workflow

The deployment is additive. Existing question and speech foundations were not replaced.

## Validation evidence

Focused pre-merge validation:

- run `33824880291`
- compilation passed
- legislation-context regression tests passed
- downstream-contract tests passed
- candidate-materialization tests passed

First production deployment attempt:

- run `33824958231`
- stopped at the workflow test step because the repository root was not on the test runner import path
- candidate seeding was skipped
- production promotion was skipped
- no production data changed

Workflow correction:

- PR `#65`
- set repository-root `PYTHONPATH`
- invoked tests through `python -m pytest`

Successful deployment:

- run `33825032483`
- preflight tests passed
- current validated production batch was cloned to candidate batch `certified-bill-sections-20260903-1`
- certified Bill-section foundation built successfully
- candidate manifest reassembled and validated successfully
- candidate promoted atomically
- post-promotion audit passed

## Permanent audit behaviour

The bridge audit checks:

- output is non-empty;
- `(bill_id, debate_section_id)` is unique;
- each certified section maps to at most one Bill;
- speech joins do not multiply rows;
- division joins do not multiply rows.

The builder itself enforces the source-section/heading agreement and multi-Bill exclusion before rows are emitted.

The deployment workflow runs the audit before publication and again after promotion against the live production pointer.

## Confirmed production semantics

### Bill context

A speech or division is Bill-linked only through exact `debate_section_id` membership in `bill_debate_sections`.

Shared `debate_id`, `debate_uri` or debate date is not sufficient evidence of Bill context.

### Source-row duplication

Raw Bill-debate records are not the public grain. Duplicate source records for one Bill-section pair are collapsed and retained only as provenance.

### Unresolved cases

The production bridge deliberately does not emit:

- source section/heading conflicts;
- sections associated with multiple Bills;
- Bill-debate records outside currently available canonical debate-section coverage.

These remain unresolved research/coverage cases rather than negative matches.

### Sponsors and lifecycle stages

Bill sponsors and Bill lifecycle/stage history remain separate source-backed structures.

The bridge does not infer a sponsor from names and does not force ministerial-office sponsors onto a named office-holder.

The bridge also does not claim that every Bill stage-history row corresponds one-to-one with one transcript section.

## Rejected approaches retained as guardrails

The production implementation continues to reject:

- global joins on `dbsect_*` / `section_eid` without `debate_id`;
- whole-debate Bill attribution by `debate_id` or `debate_uri`;
- heading-only Bill classification;
- speech-text similarity for Bill linkage;
- division-subject text as the primary Bill relationship;
- sponsor name matching as the primary identity relationship;
- assigning office sponsors to a current office-holder without date-aware evidence.

## Production implications

The deployed foundation makes two previously unsafe analyses deterministic for the certified subset:

1. Bill-linked speech analysis at exact debate-section grain.
2. Bill-linked division analysis at exact debate-section grain.

It does **not** provide whole-Oireachtas legislation coverage yet. Current unresolved coverage remains concentrated in Seanad, committee and older historical debate records outside the present canonical section/speech foundation.

It also does not replace `speech_question_context` and does not yet create a broad top-level `speech_context` dataset.

## Living next-step plan

1. Return to the broader deterministic speech-context investigation using the new production `bill_debate_sections` foundation as the certified `bill_or_legislation` rule.
2. Recompute actual overlaps and precedence among:
   - `oral_question_exchange`;
   - `bill_or_legislation`;
   - exact certified Leaders' Questions headings;
   - candidate statement/procedural/motion families.
3. Finish exact source-heading certification for statements and parliamentary/procedural business.
4. Decide the public meaning and scope of `motions` before adding a broad motion context.
5. Only if the resulting deterministic coverage is strong enough, propose the final broader `speech_context` architecture before changing production again.
6. Revisit voting analysis using certified Bill context so divisions can be described by linked legislation and, where source-supported, stage/proceeding context rather than only party-unity aggregates.
7. Investigate the small unresolved legislation anomalies when useful:
   - source section/heading conflicts;
   - multi-Bill sections;
   - `Cream List` stage semantics.
8. Expand canonical debate-section/speech coverage to Seanad and committee proceedings before describing Bill-linked speech coverage as whole-Oireachtas.
9. Keep Parliamentary Question full issue classification deferred unless a concrete downstream use case shows that deterministic recipient, heading, speech context and legislation context remain insufficient.
