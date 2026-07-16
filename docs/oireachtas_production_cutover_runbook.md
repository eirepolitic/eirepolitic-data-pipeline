# Oireachtas production cutover runbook

## Scope

This runbook promotes one validated immutable Oireachtas batch by changing a single S3 pointer. It does not copy table objects during cutover.

## Preconditions

- Repair branch CI is green.
- Candidate batch manifest status is `validated`.
- Candidate refresh, enrichment, compatibility, metrics, and Instagram smoke checks passed.
- Production pointer status was captured before cutover.
- `OIREACHTAS_PUBLISH_ENABLED` remains `false` until explicit approval.
- No other Oireachtas promotion or refresh is running.

## Candidate evidence

Record these values before approval:

- Branch and commit SHA
- Candidate batch ID
- Refresh workflow run ID
- Validation workflow run ID
- Final manifest key
- Production pointer before cutover
- Required-table count
- Failed/missing table and object counts

## Approval boundary

Do not promote based only on a successful build. Promotion requires a separate explicit approval after reviewing the candidate manifest and consumer evidence.

## Promote

1. In GitHub, open **Actions**.
2. Open **Oireachtas batch control**.
3. Select **Run workflow**.
4. Choose the repaired production branch.
5. Set `operation` to `promote`.
6. Enter the validated candidate `batch_id`.
7. Leave `required_tables` blank.
8. Enable repository variable `OIREACHTAS_PUBLISH_ENABLED=true` only for the approved cutover window.
9. Run the workflow.
10. Save the workflow artifact and resulting production pointer JSON.

The workflow also requires the per-run pointer switch. A promotion cannot occur with only one switch enabled.

## Immediate verification

After promotion:

1. Run **Oireachtas batch control** with `operation=status`.
2. Confirm `production.mode=batch`.
3. Confirm `production.batch_id` equals the approved candidate.
4. Confirm `previous.mode=legacy_direct` on first cutover, or the prior batch ID on later cutovers.
5. Run validation-only orchestration against the production pointer.
6. Verify roster count, primary-key uniqueness, compatibility checks, metrics build, and Instagram HTML smoke.
7. Confirm no table resolves outside the promoted batch except explicitly documented legacy raw inputs.

## First-cutover rollback

If the first cutover fails:

1. Open **Oireachtas batch control**.
2. Set `operation` to `rollback-previous`.
3. Run the workflow while the production switch is still enabled.
4. Confirm `production.mode=legacy_direct`.
5. Run `status` and verify logical keys resolve to legacy direct objects.
6. Disable `OIREACHTAS_PUBLISH_ENABLED` immediately after rollback.

Equivalent explicit target: set `operation=rollback` and `batch_id=legacy_direct`.

## Later rollback

For later cutovers, prefer `rollback-previous`. To select a specific known-good batch, use `operation=rollback` with its validated batch ID.

## Observation window

For the first 24 hours:

- Check every orchestrator run.
- Verify table row counts and DQ status.
- Verify production pointer batch ID has not changed unexpectedly.
- Check compatibility and enrichment freshness.
- Check metrics and Instagram consumer smoke results.
- Treat missing scheduled evidence as a failure requiring investigation.

For the following six scheduled runs:

- Record run ID, batch ID, conclusion, failed table count, and consumer status.
- Roll back on repeated completeness, schema, or key-consistency failures.

## Closeout

After a stable observation window:

- Disable temporary cutover permissions.
- Confirm `OIREACHTAS_PUBLISH_ENABLED=false` unless automated promotion is separately approved.
- Store the promotion and verification artifacts.
- Record the rollback target and final production batch ID.
