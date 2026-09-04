# Certified Bill-section bridge implementation plan

## Status

Plan prepared on 2026-09-03.

This document proposes a production architecture change. No production implementation has been made yet.

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

Therefore the production grain must be one row per:

`(bill_id, debate_section_id)`

Raw `bill_debate_id` must be treated as provenance, not as the output grain.

## Proposed production foundation

Recommended dataset name:

`bill_debate_sections`

Recommended grain:

one row per certified `(bill_id, debate_section_id)` pair.

Recommended fields:

- `bill_id`
- `debate_section_id`
- `debate_id`
- `debate_date`
- `source_section_eid`
- `debate_show_as`
- `evidence_method`
- `source_bill_debate_count`
- `source_bill_debate_ids`
- `certification_version`
- `source_batch_id`

Optional later fields, only if source semantics are separately validated:

- normalized/public stage label
- chamber
- sitting/committee type

Do not put sponsor attribution or Bill lifecycle stage history into this bridge. Those remain separate source-backed concepts.

## Certification rule

A source Bill-debate record is eligible only when all of the following hold:

1. `(bill_debate.debate_id, bill_debate.debate_section_id)` resolves to exactly one debate section via `(debate_section.debate_id, debate_section.section_eid)`.
2. The Bill record's exact `debate_show_as` resolves to exactly one section heading within that same debate.
3. Both checks identify the same canonical `debate_section_id`.
4. The canonical debate section is associated with only one distinct Bill ID across eligible rows.

Eligible source rows are then collapsed to one `(bill_id, debate_section_id)` row, preserving all contributing `bill_debate_id` values as provenance.

Anything failing these checks remains unresolved and is not emitted as a certified bridge row.

## Why unresolved rows should not be included in the production bridge yet

The current unmatched/conflicting set includes:

- Seanad and committee debates outside present debate-section coverage;
- older historical debate rows outside current speech coverage;
- a few source section/heading conflicts;
- multi-Bill section anomalies.

Mixing these into one bridge with status values would make the dataset appear broader than its certified coverage and would complicate downstream joins.

Recommended first implementation: materialize **certified rows only** and keep unresolved counts in audit output. Revisit an all-status research table later only if a concrete operational need arises.

## Proposed implementation steps

1. Add a deterministic builder for `bill_debate_sections`.
2. Read only existing production source tables:
   - `silver_bill_debates`
   - `silver_debate_sections`
3. Apply the certification rule above.
4. Collapse duplicate source rows to unique `(bill_id, debate_section_id)` grain.
5. Preserve raw source-record provenance in aggregated fields.
6. Register the dataset in political-metrics materialization configuration.
7. Add a downstream contract describing the exact grain and certified-only meaning.
8. Add candidate-materialization support without changing existing `speech_question_context`.
9. Add permanent audits before promotion.
10. Deploy as a new additive foundation; do not replace any current dataset.

## Required audits

The deployment should fail if any of these conditions occur:

- duplicate `(bill_id, debate_section_id)` output rows;
- one certified `debate_section_id` maps to more than one Bill ID;
- source section-ID and exact heading checks disagree in an emitted row;
- emitted `debate_section_id` does not exist in `silver_debate_sections`;
- joining the bridge to speeches multiplies any speech row;
- joining the bridge to divisions multiplies any division row;
- provenance list/count is inconsistent with collapsed source rows;
- certification output unexpectedly becomes empty;
- current certified coverage changes materially without an explicit audit report.

Audit output should separately report:

- certified source rows;
- certified unique Bill-section pairs;
- duplicate raw source rows collapsed;
- conflicts;
- multi-Bill section exclusions;
- unmatched rows by chamber/year/debate type where available;
- linked speech count;
- linked division count.

## Expected initial production footprint

Using the current production batch and current certification rule, expected values are approximately:

- 371 `bill_debate_sections` rows
- 168 distinct Bills represented
- 371 distinct sections
- 7,352 linked speeches
- 168 linked divisions across 94 sections

These are regression expectations, not permanent constants. Audits should tolerate legitimate source growth while flagging abrupt structural changes.

## Downstream use

### Speeches

A speech is Bill-linked only when its exact `debate_section_id` appears in `bill_debate_sections`.

Because the bridge is one row per Bill-section and multi-Bill sections are excluded, each currently certified speech can inherit at most one Bill ID from this bridge.

### Divisions

A division is Bill-linked only through exact `debate_section_id` membership in the bridge.

Do not infer Bill context from division subject text or shared debate date.

### Broader speech context

Do not immediately replace `speech_question_context`.

After the Bill-section bridge is deployed and audited, a later broader `speech_context` design can safely consider:

1. `oral_question_exchange`
2. `bill_or_legislation`
3. exact certified Leaders' Questions heading allowlist
4. other categories only after separate certification
5. `other`

Precedence must be tested against actual overlaps before deployment.

## Compatibility

This should be an additive deployment.

Keep existing datasets and contracts unchanged, including:

- `speech_question_context`
- `oral_question_sections`
- `oral_question_exchange_participants`

No current downstream consumer should be required to migrate as part of the first Bill-section deployment.

## Deployment sequence

Recommended sequence:

1. implement builder + unit/regression tests;
2. add materialization and downstream contract entries;
3. materialize a candidate batch;
4. run Bill-section audits;
5. verify exact expected grain and no speech/division multiplication;
6. review coverage/conflict diagnostics;
7. promote only if all audits pass;
8. run post-promotion audit against the live pointer;
9. update the research implementation record with batch/run IDs and final live counts.

## Production-change decision

The architecture change recommended by the research is:

**Add a new certified `bill_debate_sections` foundation at one row per `(bill_id, debate_section_id)` pair.**

It should be additive, certified-only, and should not yet create a broader all-purpose `speech_context` dataset.

## Next step after approval

Implement and deploy the additive `bill_debate_sections` foundation with the audits above. After successful promotion, return to broader speech-context research and voting analysis using the new certified legislation relationship.
