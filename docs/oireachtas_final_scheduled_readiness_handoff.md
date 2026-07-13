# Oireachtas final scheduled-readiness handoff

**Status:** manual orchestration validated; scheduled orchestrator deferred  
**Last updated:** 2026-07-12 America/Vancouver

## Executive summary

The unified Oireachtas pipeline is validated for manual and controlled pre-production operation.

The following are green:

```text
silver/gold/control table builds
monthly refresh DQ fixes
weekly scheduled refresh after DQ fix
compatibility adapters
compatibility comparison
mismatch review
member profile metrics
Instagram constituency render
Instagram campaign render
manual validation orchestrator
refresh-enabled validation orchestrator
```

The orchestrator scheduled trigger is not yet added. The gate decision is to defer scheduling until the orchestrator can pass explicit production-like refresh inputs to child workflows or the child workflows support `workflow_call`.

## Latest refresh evidence

### Weekly scheduled refresh

Latest scheduled runs:

```text
2026-07-05 UTC: run 28732507909 / success
2026-07-12 UTC: run 29182334702 / success
```

Conclusion:

```text
Weekly scheduled refresh is green after the debate-record DQ patch.
```

### Monthly refresh

Latest failed scheduled run before fix:

```text
2026-07-01 UTC: run 28504651002 / failure
```

Production-like manual validation after fix:

```text
2026-07-05 UTC: run 28726922946 / success / artifact 8087552815
```

Refresh-enabled orchestrator child validation:

```text
2026-07-13 UTC: run 29218778131 / success / artifact 8267502791
```

Conclusion:

```text
Monthly fixes are validated manually. The next real scheduled monthly run still needs observation.
```

### Yearly refresh

Latest validation:

```text
Run 27397123885 / workflow_dispatch / success
```

Conclusion:

```text
No new scheduled yearly run has occurred yet.
```

## Latest orchestrator evidence

### Current-latest downstream validation

```text
Orchestrator workflow ID: 307332237
Run ID: 28727286429
Result: success
Artifact ID: 8087672457
```

### Refresh-enabled validation

```text
Orchestrator workflow ID: 307332237
Run ID: 29218772777
Result: success
Artifact ID: 8267558251
```

Child runs from refresh-enabled validation:

```text
monthly refresh: 294432002 / 29218778131 / success / artifact 8267502791
compat adapters: 294866317 / 29218841290 / success / artifact 8267509616
compat comparison: 294874693 / 29218866832 / success / artifact 8267516514
mismatch review: 297343766 / 29218901115 / success / artifact 8267524757
member profile metrics: 266755732 / 29218925171 / success
Instagram constituency render: 261945698 / 29218960864 / success / artifact 8267546600
Instagram campaign render: 271160957 / 29219005969 / success / artifact 8267555282
```

## Current orchestrator state

Workflow:

```text
.github/workflows/oireachtas_refresh_validation_orchestrator.yml
```

Trigger state:

```text
workflow_dispatch only
no schedule trigger
```

Manual defaults:

```text
refresh_type=none
run_consumers=true
```

Reason:

```text
Defaulting to none keeps manual runs safe unless a refresh-enabled run is intentionally requested.
```

## Gate before scheduled orchestrator

Before adding a scheduled trigger, implement one of these:

```text
Option A: convert child workflows to reusable workflow_call jobs with explicit inputs
Option B: update orchestrator gh workflow run calls to pass explicit production-like -f inputs
```

Then validate:

```text
weekly production-like orchestrator run
monthly production-like orchestrator run
```

Only then add a schedule such as:

```yaml
schedule:
  - cron: "45 6 * * 0"
```

## Current caveats

```text
Next real scheduled monthly run still needs observation.
Scheduled orchestrator trigger is intentionally not enabled.
Production Instagram publishing remains artifact-only unless explicitly enabled.
Legacy enrichment keys remain preserved for rollback.
Two known legacy-only roster records remain: Catherine Connolly and Paschal Donohoe.
```

## Recommended next packets

```text
P69 — production-input orchestrator design
P70 — production-input orchestrator implementation
P71 — production-input orchestrator validation
```

## Final readiness statement

The unified Oireachtas data pipeline and downstream consumer validations are ready for controlled pre-production use. Weekly scheduled refresh is now green. Monthly refresh fixes are manually validated and need the next real scheduled monthly observation. The validation orchestrator is ready for manual use, but its scheduled trigger should remain disabled until production-like child workflow inputs are implemented and validated.
