# Oireachtas refresh validation orchestrator validation summary

**Status:** complete  
**Last updated:** 2026-07-05

## Purpose

Validate the new refresh validation orchestrator using `refresh_type=none`, so it checks the current latest outputs without running a new refresh first.

## Orchestrator workflow

```text
.github/workflows/oireachtas_refresh_validation_orchestrator.yml
```

Workflow ID:

```text
307332237
```

Validation run:

```text
Run ID: 28727286429
Run number: 1
Event: workflow_dispatch
Head SHA: 594b1c099405b116693aa11624b5c761c9c80539
Result: success
Artifact ID: 8087672457
```

Inputs used:

```text
refresh_type: none
run_consumers: true
```

## Child workflow results

### Refresh

```text
Skipped because refresh_type=none.
```

### Compatibility adapters

```text
Workflow ID: 294866317
Run ID: 28727289983
Result: success
Artifact ID: 8087650930
```

### Compatibility comparison

```text
Workflow ID: 294874693
Run ID: 28727303180
Result: success
Artifact ID: 8087654840
```

### Mismatch review

```text
Workflow ID: 297343766
Run ID: 28727318126
Result: success
Artifact ID: 8087658973
```

### Member profile metrics

```text
Workflow ID: 266755732
Run ID: 28727335843
Result: success
```

### Instagram constituency render

```text
Workflow ID: 261945698
Run ID: 28727349094
Result: success
Artifact ID: 8087668776
```

### Instagram campaign render

```text
Workflow ID: 271160957
Run ID: 28727366673
Result: success
Artifact ID: 8087671962
```

## What this validates

The orchestrator can run the full downstream validation chain in order:

```text
compat adapters
compat comparison
mismatch review
member profile metrics
Instagram constituency render
Instagram campaign render
```

It also proves the orchestrator can dispatch and monitor child workflows using the repository GitHub token with `actions: write` permissions.

## Current limitations

This validation did not run a refresh first. It validated the current latest outputs only.

The next validation should use:

```text
refresh_type=monthly
```

or:

```text
refresh_type=weekly
```

That will prove refresh plus downstream validations in one orchestrated run.

## Conclusion

The manual orchestrator is validated for `refresh_type=none` and is ready for a production-like refresh validation run before adding any scheduled trigger.
