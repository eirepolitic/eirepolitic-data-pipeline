# Oireachtas orchestrator schedule gate decision

**Status:** schedule deferred  
**Last updated:** 2026-07-12 America/Vancouver

## Decision

Do not add a scheduled trigger to the orchestrator yet.

## Why

The orchestrator has now passed two manual validations:

```text
refresh_type=none: run 28727286429 / success
refresh_type=monthly: run 29218772777 / success
```

However, the refresh-enabled validation used the monthly workflow's `workflow_dispatch` defaults. That proves the orchestration chain works, but it does not yet prove that the orchestrator can run production-like refresh inputs from a scheduled event.

## Current evidence

### Current-latest downstream validation

```text
Orchestrator run: 28727286429
Result: success
Artifact ID: 8087672457
```

### Refresh-enabled validation

```text
Orchestrator run: 29218772777
Result: success
Artifact ID: 8267558251
```

Child refresh run:

```text
Monthly refresh: 294432002 / 29218778131 / success / artifact 8267502791
```

Separate production-like monthly validation remains:

```text
Monthly refresh: 294432002 / 28726922946 / success / artifact 8087552815
```

## Current scheduled refresh state

Weekly scheduled refresh is now green after the debate-record DQ patch:

```text
2026-07-05 scheduled weekly: 28732507909 / success
2026-07-12 scheduled weekly: 29182334702 / success
```

Monthly scheduled refresh still needs the next real scheduled observation after the bill-stage/sponsor/event DQ patches:

```text
Latest scheduled monthly: 28504651002 / failure / before fix
Latest manual monthly validation: 29218778131 / success
```

Yearly refresh has no new scheduled run yet:

```text
Latest yearly validation: 27397123885 / workflow_dispatch / success
```

## Required before adding a scheduled orchestrator trigger

Add one of these implementation improvements:

```text
Option A: child workflows support workflow_call with explicit inputs
Option B: orchestrator dispatches child refresh workflows with explicit production-like -f inputs
```

Then validate an orchestrator run that uses production-like refresh inputs, not only default workflow_dispatch values.

## Candidate future schedule

When ready, use a weekly-safe schedule after the weekly refresh window:

```yaml
schedule:
  - cron: "45 6 * * 0"
```

For that schedule, the orchestrator should explicitly set:

```text
refresh_type=weekly
run_consumers=true
```

## Conclusion

The orchestrator is validated for manual use. The scheduled trigger is intentionally deferred until production-like scheduled inputs are implemented and validated.
