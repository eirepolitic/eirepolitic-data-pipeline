# Oireachtas scheduled refresh readiness review

**Status:** ready with monitoring caveats  
**Last updated:** 2026-06-30

## Reviewed workflows

| Workflow | Path | Schedule | Scheduled mode | Latest publishing |
|---|---|---|---|---|
| Weekly refresh | `.github/workflows/oireachtas_weekly_refresh.yml` | `20 3 * * 0` | `incremental` | `auto` |
| Monthly refresh | `.github/workflows/oireachtas_monthly_refresh.yml` | `35 4 1 * *` | `incremental` | `auto` |
| Yearly refresh | `.github/workflows/oireachtas_yearly_refresh.yml` | `15 5 2 1 *` | `full` | `auto` |

## Current readiness

Scheduled refreshes are ready to run, but should be followed by adapter and consumer checks because downstream consumers now depend on unified compatibility outputs.

## Recommended post-refresh sequence

After any scheduled refresh completes successfully, run:

1. `Oireachtas Downstream Compatibility Adapters (Manual)`
2. `Oireachtas Compatibility Adapter Comparison (Manual)`
3. `Oireachtas Member Mismatch Review (Manual)`
4. `Generate Instagram Constituency Test Post (Manual)`
5. `Build Member Profile Metrics 2025 (Manual)` if member/vote tables changed
6. `Instagram Campaign Render (Manual)` if member metrics changed

## Important behavior

- Manual dispatch defaults remain safe: `mode=test`, low limits, and `publish_latest=auto` which suppresses latest writes in test mode.
- Scheduled runs use non-test modes, so `publish_latest=auto` writes unified latest pointers.
- Compatibility adapters must be rerun after latest pointers are updated.
- Existing legacy S3 keys are not deleted by unified refresh workflows.

## Known caveat

Several workflows publish review output to the shared `oireachtas-review-output` branch. If multiple review-publishing workflows run at the same time, a branch publish conflict can occur. The clean rerun pattern is validated: rerun the failed review workflow after the other publishing workflow finishes.

Observed example:

```text
Mismatch review run 28414820972: build success, review-branch publish failure.
Mismatch review run 28414847238: clean rerun success.
```

## Recommendation

Do not run review-publishing workflows concurrently when avoidable. For scheduled refreshes, let the refresh finish, then run adapters/comparison/mismatch review sequentially.
