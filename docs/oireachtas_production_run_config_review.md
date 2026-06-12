# Oireachtas production run configuration review

**Status:** review only  
**Last updated:** 2026-06-12  
**No workflow limits or schedules were changed in this packet.**

This document reviews the active unified Oireachtas refresh workflows and lists production-safe changes to consider before using unified outputs as downstream replacements.

## Current workflow configuration

| Workflow | File | Schedule | Scheduled mode | Scheduled date window | Scheduled limit | Manual default mode | Manual default limit | Last validation |
|---|---|---:|---|---|---:|---|---:|---|
| Weekly | `.github/workflows/oireachtas_weekly_refresh.yml` | `20 3 * * 0` | `incremental` | `2025-01-01` to `2025-01-31` | `100` | `test` | `10` | run `27396638715`, success |
| Monthly | `.github/workflows/oireachtas_monthly_refresh.yml` | `35 4 1 * *` | `incremental` | `2025-01-01` to `2025-01-31` | `100` | `test` | `10` | run `27397121321`, success |
| Yearly | `.github/workflows/oireachtas_yearly_refresh.yml` | `15 5 2 1 *` | `full` | `2025-01-01` to `2025-12-31` | `500` | `test` | `10` | run `27397123885`, success |
| Cutover comparison | `.github/workflows/oireachtas_cutover_comparison.yml` | none | manual only | n/a | n/a | n/a | n/a | run `27397256307`, success |

## Main production risks

### 1. Scheduled weekly/monthly date windows are fixed to January 2025

Weekly and monthly scheduled runs currently use:

```text
date_start=2025-01-01
date_end=2025-01-31
```

This is safe for validation but not suitable for ongoing current-data refresh unless intentionally backfilling that period.

Recommended production change:

- add dynamic date window support in the workflow shell step or CLI;
- weekly: previous 14 to 30 days;
- monthly: previous full calendar month plus a small overlap;
- yearly: previous/current full year depending on schedule purpose.

### 2. Manual defaults are intentionally tiny

Manual workflow defaults use:

```text
mode=test
limit=10
```

This is good for smoke tests, but manual operators may mistake these outputs for production-ready latest pointers.

Recommended production change:

- keep manual `test` defaults for low-cost validation;
- add explicit named manual inputs or docs for production runs, for example:
  - `mode=incremental`, `limit=500`, current 30-day window;
  - `mode=full`, `limit=5000`, full-year window;
- avoid silent production-sized runs by default.

### 3. Latest pointers can be overwritten by limited test runs

All unified builders publish latest CSV/Parquet pointers. A manual test run with `limit=10` can replace latest outputs with a small sample.

Recommended production change options:

| Option | Safety | Work required | Notes |
|---|---|---:|---|
| Keep current behavior and document clearly | medium | low | Current state. Simple but easy to misuse. |
| Add `--publish-latest` flag default true only for non-test modes | high | medium | Best long-term control. Test runs still write partitioned outputs/review but do not replace latest. |
| Separate `latest_test/` and `latest/` prefixes | high | medium | Cleaner S3 separation, but more downstream path changes. |

Recommended first implementation: add `--publish-latest` to the CLI and make workflows pass it only when `mode != test`.

### 4. Control tables read manifests and can grow over time

`control_pipeline_runs`, `control_table_manifests`, and `control_data_quality_results` scan manifest prefixes. This is fine now, but may grow as scheduled runs accumulate.

Recommended production change:

- keep current implementation for now;
- add prefix/date filtering later if manifest scans become slow;
- preserve `limit` for review outputs but consider separate full control outputs when needed.

### 5. API/rate-cost risk is still unknown for high limits

The Oireachtas API extraction has been validated with small samples and moderate workflow table sets. Production limits should be increased gradually.

Recommended ramp:

1. Run weekly workflow manually with `mode=incremental`, `limit=100`, current 30-day window after dynamic windows exist.
2. Run monthly workflow manually with `mode=incremental`, `limit=250`, previous month window.
3. Run yearly workflow manually with `mode=full`, `limit=500`, one year.
4. Inspect run duration, API errors, row counts, DQ, and S3 object counts.
5. Increase limits only after stable runs.

## Recommended production settings

These are proposed settings, not applied changes.

### Weekly

| Setting | Proposed value |
|---|---|
| Mode | `incremental` |
| Window | previous 30 days or rolling 35-day overlap |
| Limit | start `100`, ramp to `500` |
| Tables | current weekly list is acceptable |
| Latest publishing | only when mode is not `test` |

### Monthly

| Setting | Proposed value |
|---|---|
| Mode | `incremental` |
| Window | previous full calendar month plus 7-day overlap |
| Limit | start `250`, ramp to `1000` |
| Tables | current monthly list is acceptable |
| Latest publishing | only when mode is not `test` |

### Yearly

| Setting | Proposed value |
|---|---|
| Mode | `full` |
| Window | previous calendar year for scheduled January run |
| Limit | start `500`, ramp to full expected volume |
| Tables | current yearly list is acceptable; consider adding remaining bill child tables for full annual reconciliation |
| Latest publishing | only when mode is not `test` |

## Recommended next implementation packets

| Packet | Change | Risk |
|---|---|---|
| P01 | Add `--publish-latest` CLI switch and workflow logic. | medium |
| P02 | Add dynamic date window helper for weekly/monthly/yearly workflows. | medium |
| P03 | Add compatibility adapters for downstream trial. | low-medium |
| P04 | Run side-by-side production-like validation without changing consumers. | low |

## Current recommendation

Keep scheduled workflows active only as validation until dynamic windows and latest-publishing controls are added. Do not use current latest pointers as production downstream replacements yet because manual/test runs can overwrite them with limited samples.
