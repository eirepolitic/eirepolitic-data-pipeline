# Oireachtas release readiness summary

## Candidate

- Candidate batch: `batch6-validation-29519369155-1`
- End-to-end candidate workflow: `29519369155`
- End-to-end candidate conclusion: `success`
- Source branch: `repair/oireachtas-production-hardening`
- Candidate source commit: `dabde43b5d8886f6db7d6f550abb5174690373e4`

## Candidate validation evidence

The candidate passed:

- complete weekly refresh
- table-level DQ and immutable object checks
- questions pagination beyond the API offset ceiling
- gold dependency ordering
- enrichment freshness and staging
- downstream schema contracts
- 176/176 member roster compatibility
- bounded incremental vote member-key drift
- mismatch review
- year-aware metrics
- Instagram HTML consumer smoke
- final batch-manifest reassembly
- production-pointer equality before and after validation

## Release-readiness gate

- Workflow: `Oireachtas release readiness`
- Run: `29521918215`
- Conclusion: `success`
- Workflow source commit: `eff8c7ad47d028c9a713b879d2628cf9c9d0aea9`
- Promotion performed: `false`

The workflow:

1. captured current production and previous pointer state;
2. reassembled the candidate using its preserved required-table set;
3. verified no missing tables, failed tables, missing objects, or duplicate table entries;
4. determined the current rollback target;
5. produced a 180-day release-readiness artifact.

## Rollback readiness

First-cutover promotion now records `legacy_direct` as the previous target when no production batch pointer exists. Operators can restore it with either:

- `rollback-previous`; or
- `rollback --batch-id legacy_direct`.

Later cutovers can roll back to the recorded previous immutable batch.

## Approval boundary

The system is ready for an explicit cutover decision, but no production promotion has occurred.

`OIREACHTAS_PUBLISH_ENABLED` must remain `false` until a separate promotion approval is given.

## Operator reference

Use `docs/oireachtas_production_cutover_runbook.md` for the promote, verify, rollback, observation, and closeout procedure.
