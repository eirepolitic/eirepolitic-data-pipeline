# Oireachtas review publish hardening

**Status:** patched  
**Last updated:** 2026-06-30

## Problem

Several review workflows publish generated review files to the shared branch:

```text
oireachtas-review-output
```

When two workflows publish at nearly the same time, one workflow can build successfully but fail on branch push.

Observed example:

```text
Compatibility comparison run 28414819264: success.
Mismatch review run 28414820972: build success, review-branch publish failure.
Mismatch review rerun 28414847238: success.
```

## Patch pattern

The publish step retries the review branch push after pulling/rebasing the latest branch state.

```bash
for attempt in 1 2 3; do
  git -C /tmp/oireachtas-review-worktree pull --rebase origin "${REVIEW_BRANCH}" || true
  if git -C /tmp/oireachtas-review-worktree push origin "${REVIEW_BRANCH}"; then
    exit 0
  fi
  if [[ "${attempt}" == "3" ]]; then
    echo "Review branch push failed after ${attempt} attempts."
    exit 1
  fi
  sleep $((attempt * 5))
done
```

## Patched and validated workflows

| Workflow | Workflow ID | Validation run | Result |
|---|---:|---:|---|
| `.github/workflows/oireachtas_compat_comparison.yml` | `294874693` | `28416432150` | success |
| `.github/workflows/oireachtas_mismatch_review.yml` | `297343766` | `28416434690` | success |

Success criteria met:

1. Both workflows completed successfully.
2. Both published review output to `oireachtas-review-output`.
3. Both uploaded artifacts successfully.

## Patched broader refresh publishers

| Workflow | Status | Validation |
|---|---|---|
| `.github/workflows/oireachtas_weekly_refresh.yml` | patched | not manually run after patch |
| `.github/workflows/oireachtas_monthly_refresh.yml` | patched | not manually run after patch |
| `.github/workflows/oireachtas_yearly_refresh.yml` | patched | not manually run after patch |

These were not manually dispatched after patching because they are broad table-refresh jobs. The patch is the same shell pattern already validated on the comparison and mismatch review workflows.

## Remaining risk

The retry/rebase patch reduces branch push conflicts but does not eliminate all conflict risk. Review-publishing workflows should still be run sequentially when possible.

## Recommendation

For routine operations:

1. Run refresh workflow.
2. Run compatibility adapters.
3. Run comparison.
4. Run mismatch review.
5. Run consumer validations.

Avoid starting multiple review-publishing workflows at the same time unless necessary.
