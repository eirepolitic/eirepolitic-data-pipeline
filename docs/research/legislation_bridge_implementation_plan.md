# Certified Bill-section bridge implementation plan

## Status

Plan prepared on 2026-09-03 and **fulfilled in production on 2026-09-03**.

The deployed implementation record is:

- `docs/research/legislation_bridge_implementation.md`

Production batch:

- `certified-bill-sections-20260903-1`

Successful deployment run:

- `33825032483`

This file remains the durable record of the architecture decision that preceded implementation.

## Goal

Materialize a conservative, deterministic bridge between Bills and parliamentary debate sections so speeches and divisions can inherit Bill context without debate-day over-attribution or row multiplication.

## Evidence base

Research record:

- `docs/research/legislation_investigation.md`

Additional grain check:

- run `33824117083`
- artifact `analysis/legislation_bridge_grain_digest.json`

The grain check confirmed:

- 396 certified source rows
- 371 unique `(bill_id, debate_section_id)` pairs
- 371 unique debate sections
- 25 duplicate source rows above the intended pair grain
- 7,352 speeches if each certified Bill-section pair is joined once
- 168 divisions if each certified Bill-section pair is joined once

Therefore the production grain was set to one row per:

`(bill_id, debate_section_id)`

Raw `bill_debate_id` is treated as provenance, not as the output grain.

## Approved production foundation

Dataset:

`bill_debate_sections`

Grain:

one row per certified `(bill_id, debate_section_id)` pair.

Implemented fields:

- `bill_id`
- `debate_section_id`
- `debate_id`
- `debate_date`
- `source_section_eid`
- `debate_show_as`
- `evidence_method`
- `source_bill_debate_count`
- `source_bill_debate_ids_json`
- `certification_version`
- `source_batch_id`
- `calculated_at_utc`
- `contract_version`

Sponsor attribution and Bill lifecycle stage history remain separate source-backed concepts.

## Certification rule

A source Bill-debate record is eligible only when all of the following hold:

1. `(bill_debate.debate_id, bill_debate.debate_section_id)` resolves to exactly one debate section via `(debate_section.debate_id, debate_section.section_eid)`.
2. The Bill record's exact `debate_show_as` resolves to exactly one section heading within that same debate.
3. Both checks identify the same canonical `debate_section_id`.
4. The canonical debate section is associated with only one distinct Bill ID across eligible rows.

Eligible source rows are collapsed to one `(bill_id, debate_section_id)` row, preserving contributing `bill_debate_id` values as provenance.

Anything failing these checks remains unresolved and is not emitted as a certified bridge row.

## Unresolved rows

The production bridge intentionally excludes:

- Seanad and committee debate rows outside present debate-section coverage;
- older historical debate rows outside current speech coverage;
- source section/heading conflicts;
- multi-Bill section anomalies.

These remain unresolved coverage/research cases rather than negative matches.

## Required audits

The approved deployment required checks for:

- duplicate `(bill_id, debate_section_id)` output rows;
- more than one Bill per certified `debate_section_id`;
- source section-ID / exact-heading disagreement;
- missing canonical debate sections;
- speech joins multiplying rows;
- division joins multiplying rows;
- provenance inconsistency;
- empty output;
- material structural coverage changes.

The deployed builder and post-promotion audit implement the core production-safety checks. Deployment evidence is recorded in `legislation_bridge_implementation.md`.

## Expected and observed initial footprint

Expected before deployment:

- 371 `bill_debate_sections` rows
- 168 distinct Bills
- 371 distinct sections
- 7,352 linked speeches
- 168 linked divisions

The successful production deployment matched this footprint and passed the live audit.

These values are regression expectations for that source snapshot, not permanent constants.

## Downstream rules

### Speeches

A speech is Bill-linked only when its exact `debate_section_id` appears in `bill_debate_sections`.

### Divisions

A division is Bill-linked only through exact `debate_section_id` membership in the bridge.

### Prohibited shortcuts

Do not infer Bill context from:

- shared `debate_id`;
- shared `debate_uri`;
- shared debate date;
- heading-only text matching;
- speech similarity;
- division subject text alone.

## Compatibility decision

The deployment is additive.

Existing datasets remain in place, including:

- `speech_question_context`
- `oral_question_sections`
- `oral_question_exchange_participants`

No existing downstream consumer was required to migrate as part of this deployment.

## Fulfilled deployment sequence

1. builder + tests implemented;
2. materialization and downstream contracts added;
3. candidate deployment workflow added;
4. focused validation passed in run `33824880291`;
5. first production attempt `33824958231` stopped safely during preflight tests before candidate seeding;
6. workflow import path fixed in PR `#65`;
7. successful candidate built and audited in run `33825032483`;
8. batch `certified-bill-sections-20260903-1` promoted atomically;
9. post-promotion audit passed;
10. production implementation record added under `docs/research/`.

## Current next step

The implementation plan is complete. The living next-step plan now lives in:

- `docs/research/legislation_bridge_implementation.md`

The immediate research priority is to return to broader deterministic speech context using the live `bill_debate_sections` foundation as the certified legislation rule, then revisit voting analysis with Bill context.
