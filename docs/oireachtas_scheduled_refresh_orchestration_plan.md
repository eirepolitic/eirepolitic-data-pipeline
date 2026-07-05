# Oireachtas scheduled refresh orchestration plan

**Status:** plan complete  
**Last updated:** 2026-07-05

## Purpose

Design one manual/scheduled orchestration workflow that runs the refresh and downstream validations in the correct order.

## Problem

Current workflows work individually, but the operator must run them in sequence:

```text
refresh
compat adapters
compat comparison
mismatch review
member profile validation
Instagram validation
```

This makes scheduled refresh readiness harder to prove because a scheduled refresh can pass or fail without automatically triggering downstream checks.

## Proposed workflow

File:

```text
.github/workflows/oireachtas_refresh_validation_orchestrator.yml
```

Name:

```text
Oireachtas Refresh Validation Orchestrator
```

Triggers:

```yaml
workflow_dispatch:
  inputs:
    refresh_type:
      type: choice
      options: [weekly, monthly, yearly, none]
      default: weekly
    run_consumers:
      type: choice
      options: [true, false]
      default: true
schedule:
  - cron: "45 6 * * 0"
```

The scheduled trigger should run after weekly refresh, leaving enough time for upstream refresh completion.

## Recommended execution order

### 1. Optional refresh

Run one refresh workflow based on `refresh_type`:

```text
weekly  -> .github/workflows/oireachtas_weekly_refresh.yml
monthly -> .github/workflows/oireachtas_monthly_refresh.yml
yearly  -> .github/workflows/oireachtas_yearly_refresh.yml
none    -> skip refresh and validate current latest outputs
```

### 2. Compatibility adapters

Run:

```text
.github/workflows/oireachtas_compat_adapters.yml
```

Purpose:

```text
Rebuild downstream compatibility CSV/parquet outputs from latest unified outputs.
```

### 3. Compatibility comparison

Run:

```text
.github/workflows/oireachtas_compat_comparison.yml
```

Expected result:

```text
success
members_roster_compat: pass
member_votes_compat: pass
```

### 4. Mismatch review

Run:

```text
.github/workflows/oireachtas_mismatch_review.yml
```

Expected result:

```text
success
Only known roster caveats unless new data changes appear.
```

### 5. Member-profile validation

Run:

```text
.github/workflows/build_member_profile_metrics_2025.yml
```

Current default inputs already use unified compatibility outputs:

```text
members
member votes
member photos
classified issue labels
```

### 6. Instagram consumer validation

Run if `run_consumers=true`:

```text
.github/workflows/instagram_constituency_test.yml
.github/workflows/instagram_campaign_render.yml
```

Expected result:

```text
success and artifact uploaded
```

## Implementation options

### Option A — reusable workflow calls

Use `workflow_call` in child workflows and call them from one orchestrator.

Pros:

```text
Clean dependency graph
Native job needs/dependencies
Centralized summary
```

Cons:

```text
Requires modifying each child workflow to support workflow_call
```

### Option B — GitHub CLI dispatch loop

Use one orchestrator workflow that dispatches child workflows with `gh workflow run`, then polls status.

Pros:

```text
Minimal child workflow changes
Can be added quickly
```

Cons:

```text
More shell logic
Needs careful timeout/retry handling
```

## Recommended first implementation

Use **Option B** first, because current child workflows are already validated and can remain mostly unchanged.

Core approach:

```bash
gh workflow run <workflow-file> --ref main
# poll gh run list / gh run view until completed
# fail orchestrator if child run fails
```

Required permissions:

```yaml
permissions:
  actions: write
  contents: read
```

## Failure handling

The orchestrator should stop on first failed required step and write a summary with:

```text
workflow name
workflow ID/path
run ID
conclusion
artifact name if available
```

For optional consumer validations, the initial recommendation is still to fail the orchestrator if rendering fails, because those checks prove downstream compatibility.

## Output summary

The orchestrator should produce a GitHub step summary containing:

```text
refresh run ID and result
compat adapter run ID and result
compat comparison run ID and result
mismatch review run ID and result
member-profile run ID and result
Instagram validation run IDs and artifacts
final status
```

## Rollout plan

1. Add manual-only orchestrator workflow.
2. Validate with `refresh_type=none`.
3. Validate with `refresh_type=monthly` after the monthly DQ fixes.
4. Add scheduled weekly trigger after stable manual runs.
5. Later convert child workflows to `workflow_call` if more structure is needed.

## Current prerequisite state

Latest known validated runs:

```text
Monthly production-like validation: 294432002 / 28726922946 / success
Compatibility comparison: 294874693 / 28691936308 / success
Mismatch review: 297343766 / 28691938402 / success
Member profile production validation: 266755732 / 28684033733 / success
Instagram constituency validation: 261945698 / 28672901108 / success
```

## Recommendation

Proceed next with an orchestrator implementation using `gh workflow run` and polling. Keep it manual-only for the first validation packet.
