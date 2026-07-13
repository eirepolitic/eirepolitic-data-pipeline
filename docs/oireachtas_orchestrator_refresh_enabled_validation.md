# Oireachtas orchestrator refresh-enabled validation

**Status:** complete  
**Last updated:** 2026-07-12 America/Vancouver

## Purpose

Validate that the refresh validation orchestrator can run with a refresh step enabled and then continue through the downstream validation chain.

## Temporary setup

The repository dispatch integration cannot pass workflow inputs. To validate a refresh-enabled run, the orchestrator default was temporarily changed from:

```text
refresh_type=none
```

to:

```text
refresh_type=monthly
```

After validation, the default was restored to:

```text
refresh_type=none
```

## Orchestrator validation run

```text
Workflow: Oireachtas Refresh Validation Orchestrator
Workflow ID: 307332237
Run ID: 29218772777
Run number: 2
Event: workflow_dispatch
Head SHA: 57ca53f3105b1c0e616496197d29df831eb29150
Result: success
Artifact ID: 8267558251
```

Inputs used by default:

```text
refresh_type: monthly
run_consumers: true
```

## Child workflow results

### Monthly refresh

```text
Workflow ID: 294432002
Run ID: 29218778131
Result: success
Artifact ID: 8267502791
```

Note:

```text
This used the monthly workflow_dispatch defaults. It validates refresh-enabled orchestration mechanics. The separate production-like monthly validation remains run 28726922946.
```

### Compatibility adapters

```text
Workflow ID: 294866317
Run ID: 29218841290
Result: success
Artifact ID: 8267509616
```

### Compatibility comparison

```text
Workflow ID: 294874693
Run ID: 29218866832
Result: success
Artifact ID: 8267516514
```

### Mismatch review

```text
Workflow ID: 297343766
Run ID: 29218901115
Result: success
Artifact ID: 8267524757
```

### Member profile metrics

```text
Workflow ID: 266755732
Run ID: 29218925171
Result: success
```

### Instagram constituency render

```text
Workflow ID: 261945698
Run ID: 29218960864
Result: success
Artifact ID: 8267546600
```

### Instagram campaign render

```text
Workflow ID: 271160957
Run ID: 29219005969
Result: success
Artifact ID: 8267555282
```

## Conclusion

The orchestrator successfully ran a refresh-enabled chain:

```text
monthly refresh
compat adapters
compat comparison
mismatch review
member profile metrics
Instagram constituency render
Instagram campaign render
```

The manual default has been restored to `refresh_type=none` for safety.
