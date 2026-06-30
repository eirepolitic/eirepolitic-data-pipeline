# Oireachtas review publish hardening

**Status:** patched and pending validation  
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

## Patch applied

The publish step now retries the review branch push after pulling/rebasing the latest branch state.

Patched workflows:

```text
.github/workflows/oireachtas_compat_comparison.yml
.github/workflows/oireachtas_mismatch_review.yml
```

New publish behavior:

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

## Scope

This patch targets the two workflows involved in the observed conflict. Weekly, monthly, yearly, and other review publishers should be patched next if conflicts recur there.

## Validation plan

Run both patched workflows:

```text
Oireachtas Compatibility Adapter Comparison (Manual)
Oireachtas Member Mismatch Review (Manual)
```

Success criteria:

1. Both workflows complete successfully.
2. Both publish review output to `oireachtas-review-output`.
3. Artifacts upload successfully.
4. Latest review files remain readable.
