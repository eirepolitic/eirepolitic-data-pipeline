# Oireachtas scheduled refresh follow-up check

**Status:** follow-up required  
**Last updated:** 2026-07-04

## Purpose

Check scheduled refresh workflow state after the `silver_debate_records` DQ patch and after enrichment/consumer cutovers.

## Weekly refresh

Workflow:

```text
.github/workflows/oireachtas_weekly_refresh.yml
```

Workflow ID:

```text
294426406
```

Latest run:

```text
Run ID: 28421557467
Event: workflow_dispatch
Result: success
Created: 2026-06-30T05:03:07Z
Head SHA: af6ce9c856d523e2f396cb099fac3b3bf73b3268
```

Scheduled state:

```text
No newer scheduled run after the manual validation was available in the checked run list.
Previous scheduled failures: 28314747505, 27898282130
Previous scheduled success before those failures: 27492577174
```

Conclusion:

```text
Manual post-patch validation passed. Next scheduled weekly run still needs observation.
```

## Monthly refresh

Workflow:

```text
.github/workflows/oireachtas_monthly_refresh.yml
```

Workflow ID:

```text
294432002
```

Latest run:

```text
Run ID: 28504651002
Event: schedule
Result: failure
Created: 2026-07-01T08:36:19Z
Head SHA: 0d6fdf3b1f1872a76a45f6077c6437a379537b90
Artifact ID: 8004493556
```

Failed job step:

```text
Run monthly table set
```

Available evidence:

```text
Install dependencies: success
Run monthly table set: failure
Publish review output branch: success
Upload refresh artifact: success
```

Conclusion:

```text
Monthly scheduled refresh needs investigation. The root cause was not determined in this follow-up check because the available API evidence exposed the failed step and artifact metadata, but not the detailed table-level log contents.
```

## Yearly refresh

Workflow:

```text
.github/workflows/oireachtas_yearly_refresh.yml
```

Workflow ID:

```text
294432103
```

Latest run:

```text
Run ID: 27397123885
Event: workflow_dispatch
Result: success
Created: 2026-06-12T05:43:25Z
```

Scheduled state:

```text
No scheduled yearly run has occurred since the manual validation.
```

## Required follow-up

1. Investigate monthly scheduled run `28504651002` using the uploaded artifact/log.
2. Confirm whether the failure is table-specific or caused by scheduled-mode parameters.
3. Rerun monthly refresh manually with production-like inputs after the fix.
4. Observe the next scheduled weekly run after the debate-record DQ patch.

## Current readiness impact

Unified deterministic and enrichment compatibility cutovers are validated. Scheduled refresh automation is not fully green because monthly scheduled run `28504651002` failed and still needs investigation.
