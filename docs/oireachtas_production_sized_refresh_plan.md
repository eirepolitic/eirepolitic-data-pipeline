# Oireachtas production-sized refresh plan

**Status:** controlled refresh plan  
**Last updated:** 2026-06-16  
**No downstream consumer cutover is approved by this plan.**

## Goal

Refresh the unified latest outputs needed by downstream compatibility adapters with a larger non-test run, then rerun adapters, comparisons, and consumer smoke tests.

## Target tables

The first production-sized refresh should be limited to the tables needed for the current consumer trial path:

```text
silver_members
silver_member_memberships
silver_member_parties
silver_member_constituencies
silver_member_offices
gold_current_members
silver_member_votes
```

## Settings

Use non-test mode so latest pointer publishing is intentional and visible:

```text
mode=full
publish_latest=auto
chamber=dail
house_no=34
date_start=2025-01-01
date_end=2025-12-31
limit=200
sample_rows=10
```

`publish_latest=auto` publishes latest pointers because `mode=full`. This updates only unified Oireachtas latest paths under `processed/oireachtas_unified/latest/...`; it does not repoint downstream consumers.

## Expected impact

| Area | Expected impact |
|---|---|
| Unified partitioned outputs | New snapshot/run outputs written. |
| Unified latest pointers | Updated for the target tables. |
| Compatibility adapters | Must be rerun after refresh. |
| Legacy S3 keys | No overwrite expected. |
| Instagram/consumer defaults | No change. |
| Cutover approval | Still required separately. |

## Validation sequence

1. Run the production-sized refresh dry-run workflow.
2. Confirm DQ pass for every target table.
3. Confirm larger row counts for `gold_current_members` and `silver_member_votes`.
4. Rerun downstream compatibility adapters.
5. Rerun member profile metrics trial.
6. Rerun compatibility adapter comparison.
7. Rerun Instagram consumer smoke test.
8. Update cutover request package with refreshed counts.

## Stop conditions

Stop and do not proceed to downstream reruns if any target refresh table fails DQ or if latest manifests show unexpected row-count regression.

## Approval boundary

This refresh is allowed to update unified latest outputs. It is not approval to alter any production consumer workflow, Instagram workflow default, legacy key, or publishing path.
