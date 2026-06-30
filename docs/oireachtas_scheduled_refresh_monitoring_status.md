# Oireachtas scheduled refresh monitoring status

**Status:** active monitoring  
**Last updated:** 2026-06-30

## Current workflow states

| Refresh | Workflow ID | Workflow state | Latest run | Latest result |
|---|---:|---|---:|---|
| Weekly | `294426406` | active | `28421557467` | success, manual safe validation |
| Monthly | `294432002` | active | `27397121321` | success, manual validation |
| Yearly | `294432103` | active | `27397123885` | success, manual validation |

## Weekly refresh history

Recent weekly runs:

| Run | Event | Result | Note |
|---:|---|---|---|
| `27492577174` | schedule | success | earlier scheduled success |
| `27898282130` | schedule | failure | failed before PDF-link DQ patch |
| `28314747505` | schedule | failure | failed before PDF-link DQ patch |
| `28421557467` | workflow_dispatch | success | safe/manual validation after patch |

## Current weekly status

The immediate blocker is fixed and validated in manual safe mode.

Patch:

```text
extract/oireachtas/table_debate_records.py
```

Validation:

```text
Workflow ID: 294426406
Run ID: 28421557467
Result: success
Artifact ID: 7971444843
```

## What still needs monitoring

The next scheduled weekly run should be checked after it runs because scheduled mode uses:

```text
mode=incremental
publish_latest=auto
limit=100
```

Manual default validation uses safer workflow-dispatch defaults:

```text
mode=test
publish_latest=auto
limit=10
```

## Monthly and yearly status

Monthly and yearly refresh workflows are active and have passed manual validation. They have not yet produced a scheduled run since creation.

## Recommended monitoring checklist

After the next scheduled weekly run:

1. Confirm workflow conclusion is `success`.
2. Confirm `silver_debate_records` DQ is `pass`.
3. Confirm review branch publish succeeds.
4. Run compatibility adapters.
5. Run compatibility comparison.
6. Run mismatch review.
7. Run consumer validation if member/vote/profile outputs changed.
