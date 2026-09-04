# Division context implementation

## Status

Production deployment completed successfully on 2026-09-03.

Current production batch:

- `division-context-20260903-1`

Implementation PR:

- PR `#90`

Pre-merge validation run:

- `33836444030`

Successful deployment and post-promotion audit run:

- `33836502291`

The deployment is additive. Existing vote, Bill and speech-context foundations remain in place and existing participation/cohesion calculations are unchanged.

## Production dataset

Dataset:

- `division_context`

Logical locations:

- `processed/oireachtas_unified/latest/metrics/event/division_context/csv/division_context.csv`
- `processed/oireachtas_unified/latest/metrics/event/division_context/parquet/division_context.parquet`

Production grain:

- exactly one row per source `division_id`

The live audit confirmed one-to-one coverage of all current divisions and no member-vote row multiplication.

## Production context rules

Production contexts are:

- `bill_or_legislation`
- `motion_proceeding`
- `procedural_business`
- `other`

### Bill context

A division is assigned `bill_or_legislation` only when its exact `debate_section_id` appears in the certified `bill_debate_sections` foundation.

For these rows:

- `linked_entity_type = bill`
- `linked_entity_id = bill_id`

Bill context is never inferred from division subject text, debate date or whole-debate co-occurrence.

### Motion context

A non-Bill division is assigned `motion_proceeding` only when its debate section has one certified non-`other` speech context of `motion_proceeding`.

This describes parliamentary form, not substantive political topic or importance.

### Procedural-business context

A non-Bill division is assigned `procedural_business` only when its debate section has one certified non-`other` speech context of `procedural_business`.

### Other

Any division not covered by the certified Bill, motion or procedural-business relationships remains `other`.

No AI/classifier call is used to reduce `other`.

## Live footprint

Current production divisions:

- 401

Context distribution:

- `bill_or_legislation`: 168
- `motion_proceeding`: 153
- `procedural_business`: 53
- `other`: 27

Current member-vote rows:

- 59,325

Post-promotion audit confirmed:

- 401 source divisions → 401 context rows;
- unique `division_id`;
- no missing divisions;
- no extra divisions;
- all Bill-linked divisions resolve to one certified Bill;
- non-Bill context rows carry no Bill ID;
- 59,325 member-vote rows remain exactly 59,325 after joining `division_context` by `division_id`.

## Production fields

The deployed schema contains:

- `division_id`
- `division_date`
- `debate_section_id`
- `division_context`
- `evidence_method`
- `linked_entity_type`
- `linked_entity_id`
- `context_version`
- `source_batch_id`
- `calculated_at_utc`
- `contract_version`

## Implementation components

Production implementation added:

- `political_metrics/division_context.py`
  - deterministic builder
  - section-context projection
  - Bill precedence
  - completeness and vote-join audits
- `process/political_metrics_division_context_candidate.py`
  - additive candidate materializer
- `process/political_metrics_materialize_candidate.py`
  - standard candidate materialization now rebuilds `division_context`
- `configs/political_metrics/materialization.yml`
  - production dataset contract
- `configs/political_metrics/downstream_contracts.yml`
  - consumer semantics and denominator guardrails
- `tests/political_metrics/test_division_context.py`
  - context, precedence, conflict and no-multiplication regression coverage
- `.github/workflows/division_context_deploy.yml`
  - controlled seed/build/validate/promote/post-audit workflow

## Validation and deployment evidence

### Pre-merge validation

Run:

- `33836444030`

Passed:

- Python compilation;
- division-context regression tests;
- downstream-contract tests;
- candidate-materialization tests.

### Production deployment

Run:

- `33836502291`

Passed in order:

1. preflight regression/contract tests;
2. clone of current production into candidate `division-context-20260903-1`;
3. `division_context` build;
4. candidate completeness and member-vote no-multiplication audit;
5. candidate manifest reassembly;
6. atomic production promotion;
7. live post-promotion audit;
8. deployment artifact upload.

## Methodological guardrails

1. `division_context` is a descriptive parliamentary-context dimension, not an effectiveness or quality metric.
2. Existing member participation denominators remain eligible member × division opportunities.
3. Existing party cohesion calculations remain recorded-vote agreement measures with their current minimum-sample safeguards.
4. Context filters must restrict the set of divisions first, then reuse the existing denominator logic inside that restricted set.
5. Missing recorded votes must not be interpreted as proven absence.
6. `motion_proceeding` must not be treated as one substantive political topic.
7. Bill stage must not be inferred from Bill ID plus division date.
8. Bill context must come only from `bill_debate_sections`.
9. Join member votes to `division_context` by `division_id`; do not join divisions to speech-level context directly.
10. Close vote margins are descriptive outcomes and do not by themselves establish political importance, rebellion, effectiveness or instability.

## Contextual voting follow-up

The contextual voting metric investigation is now complete and documented in:

- `docs/research/contextual_voting_metrics.md`

It confirmed that the existing voting formulas work safely when divisions are filtered by `division_context` first.

Key results:

- all four contexts support denominator-correct member participation calculations;
- most member/context denominators are large, but small-sample presentation rules are still needed for a few members;
- 9 of 11 party/group URIs are `reliable` for cohesion in every current context under the existing 10+ qualifying-division rule;
- Green Party and 100% Redress are `insufficient_for_comparison` for cohesion because they do not have at least two recorded members in qualifying divisions;
- the Independent grouping requires agreement-language rather than party-discipline interpretation.

## Living next-step plan

1. Prepare a production implementation plan for **context-aware additive voting foundations** rather than changing existing metrics in place.
2. Prefer adding `division_context` to additive event/daily vote components so arbitrary date ranges can be recomputed safely.
3. Preserve the existing voting formulas and historical party-at-vote temporal joins.
4. Carry member eligible-division counts with contextual participation and define a public small-sample rule before enabling member comparison views.
5. Keep the existing party cohesion reliability thresholds unchanged initially.
6. Treat Independent results as recorded-vote agreement among Independents, not party discipline.
7. Derive completed-month contextual result rows only after context-aware additive foundations are available and reconciled.
8. Require reconciliation audits showing context-specific additive counts sum back to the existing unfiltered population for the same period.
9. Continue to defer Bill stage attribution until an exact stage relationship is certified.
10. Keep all contextual voting measures descriptive; do not present participation, agreement or context as effectiveness, quality or performance.
