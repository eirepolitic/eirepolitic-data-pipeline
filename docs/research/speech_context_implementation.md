# Broader deterministic speech context implementation

## Status

Production deployment completed successfully on 2026-09-03.

Current production batch:

- `broader-speech-context-20260903-1`

Implementation PR:

- PR `#80`

Validation run:

- `33827005250`

Successful deployment and post-promotion audit run:

- `33827057307`

The deployment is additive. Existing `speech_question_context`, `bill_debate_sections`, Oral-question exchange foundations and all existing metric outputs remain in place.

## Production dataset

Dataset:

- `speech_context`

Logical locations:

- `processed/oireachtas_unified/latest/metrics/event/speech_context/csv/speech_context.csv`
- `processed/oireachtas_unified/latest/metrics/event/speech_context/parquet/speech_context.parquet`

Production grain:

- exactly one row per source `speech_id`

The production audit confirmed complete one-to-one source coverage and no duplicate or missing speeches.

## Production context rules

Explicit precedence is encoded in production as:

1. `oral_question_exchange`
2. `bill_or_legislation`
3. `leaders_questions`
4. `statements`
5. `procedural_business`
6. `motion_proceeding`
7. `other`

The first matching certified rule wins. `other` remains an explicit valid fallback and is not filled through semantic guessing.

### `oral_question_exchange`

Inherited exactly from the existing certified `speech_question_context` relationship.

The production audit verifies exact agreement between the two datasets.

### `bill_or_legislation`

Inherited only from exact `debate_section_id` membership in the certified `bill_debate_sections` foundation.

For these rows:

- `linked_entity_type = bill`
- `linked_entity_id = bill_id`

The production audit verifies that every Bill-linked speech resolves to the certified Bill bridge.

### `leaders_questions`

Assigned only from the two certified exact Oireachtas section headings:

- `Ceisteanna ó Cheannairí - Leaders' Questions`
- `Ceisteanna ó Cheannairí (Atógáil) - Leaders' Questions (Resumed)`

### `statements`

Assigned only from certified source-heading forms ending in:

- `: Statements`
- `: Statements (Resumed)`
- `: Ráitis`
- `: Ráitis (Atógáil)`

This is a parliamentary proceeding label, not semantic interpretation of speech text.

### `procedural_business`

Assigned only from the certified exact allowlist covering:

- Order of Business;
- resumed Order of Business variants;
- Questions on Promised Legislation;
- Business of Dáil.

### `motion_proceeding`

Assigned only from certified source-heading forms ending in formal Motion/Motions variants.

This means the Oireachtas source identifies the section as a motion proceeding. It does **not** mean that all motions are one substantive political category or are comparable in political significance.

### `other`

Explicit fallback for any speech not covered by a certified deterministic rule.

No AI/classifier call is used to reduce the `other` category.

## Live footprint

Current production source speeches:

- 66,192

Production context distribution:

- `oral_question_exchange`: 18,485
- `bill_or_legislation`: 7,352
- `leaders_questions`: 10,821
- `statements`: 4,906
- `procedural_business`: 5,067
- `motion_proceeding`: 6,821
- `other`: 12,740

Non-`other` deterministic coverage:

- 53,452 speeches
- approximately 80.75%

The remaining 12,740 speeches are intentionally preserved as `other`.

## Production fields

The deployed schema contains:

- `speech_id`
- `debate_date`
- `debate_section_id`
- `speech_context`
- `evidence_method`
- `linked_entity_type`
- `linked_entity_id`
- `context_version`
- `source_batch_id`
- `calculated_at_utc`
- `contract_version`

## Implementation components

Production implementation added:

- `political_metrics/speech_context.py`
  - deterministic builder
  - precedence rules
  - complete-source audit
- `process/political_metrics_speech_context_candidate.py`
  - additive candidate materializer
- `process/political_metrics_materialize_candidate.py`
  - standard candidate materialization now rebuilds `speech_context`
- `configs/political_metrics/materialization.yml`
  - production contract and allowed context values
- `configs/political_metrics/downstream_contracts.yml`
  - consumer semantics and interpretation guardrails
- `tests/political_metrics/test_speech_context.py`
  - rule, precedence, completeness and compatibility regression coverage
- `.github/workflows/speech_context_deploy.yml`
  - controlled seed/build/validate/promote/post-audit workflow

## Validation and deployment evidence

### Pre-merge validation

Run:

- `33827005250`

Passed:

- Python compilation for changed modules;
- broader speech-context regression tests;
- downstream contract tests;
- candidate-materialization tests.

### Production deployment

Run:

- `33827057307`

Passed in order:

1. preflight tests;
2. clone of the current production batch into candidate `broader-speech-context-20260903-1`;
3. broader `speech_context` build;
4. candidate audit;
5. complete candidate manifest validation;
6. atomic promotion;
7. post-promotion audit against the live production pointer.

The workflow completed successfully.

## Permanent audit behavior

The production audit requires:

- exactly one output row per source speech;
- unique `speech_id`;
- no missing source speeches;
- no extra output speeches;
- allowed context values only;
- exact Oral-question compatibility with `speech_question_context`;
- every `bill_or_legislation` row resolving to the certified Bill bridge;
- no Bill entity attached to non-Bill context rows.

Unexpected future source overlap remains controlled by explicit precedence rather than row ordering.

## Methodological guardrails

1. `speech_context` is a parliamentary-context foundation, not a political-effectiveness measure.
2. `motion_proceeding` describes proceeding form, not substantive political topic.
3. `bill_or_legislation` must never be inferred from debate-day co-occurrence.
4. `other` remains valid and should not be filled through AI or vague text interpretation without a specific approved use case.
5. Keep source section heading, Bill ID, recipient and other dimensions separate from the top-level context label.
6. Keep `speech_question_context` for dedicated Oral-question compatibility and downstream consumers.
7. Do not multiply speech records when joining contextual relationships.
8. Current coverage should not be described as whole-Oireachtas historical coverage because Seanad/committee and older canonical speech coverage remain separate limitations.

## Living next-step plan

Broader deterministic speech context is now settled and deployed.

Next research priority:

1. **Revisit voting analysis using the new legislation and parliamentary-context foundations.**
   - link divisions to certified Bills where section-supported;
   - distinguish Bill-related votes from other motion/procedural divisions where source structure permits;
   - identify useful stage/proceeding context without relying on division subject text alone;
   - recompute party/member voting summaries with substantive context and safe denominators;
   - avoid describing unity/activity measures as political effectiveness or quality.
2. Investigate whether certified Bill stage/proceeding information can be added to division context without creating false one-to-one stage assumptions.
3. Use `motion_proceeding` only as a parliamentary-form filter unless a narrower source-supported motion taxonomy is separately certified.
4. Continue cross-metric analysis only where historical party/constituency attribution and denominators remain safe.
5. Investigate the remaining `other` speech contexts only when a concrete downstream use case requires additional deterministic categories.
6. Expand canonical Seanad/committee speech coverage separately before making whole-Oireachtas claims.
7. Keep full Parliamentary Question issue classification deferred unless deterministic recipient, heading, legislation and speech-context dimensions prove insufficient for a specific use case.
