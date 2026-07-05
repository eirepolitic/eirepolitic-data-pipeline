# Oireachtas orchestrator scheduled-readiness update

**Status:** manual orchestrator validated; scheduled trigger deferred  
**Last updated:** 2026-07-05

## Decision

Do not add a scheduled trigger yet.

The refresh validation orchestrator passed the first manual validation with:

```text
refresh_type=none
run_consumers=true
```

This proves the orchestrator can run downstream validations in sequence, but it does not yet prove refresh plus downstream validation in one run.

## Evidence

Orchestrator validation:

```text
Workflow ID: 307332237
Run ID: 28727286429
Result: success
Artifact ID: 8087672457
```

Child validations:

```text
compat adapters: 294866317 / 28727289983 / success / artifact 8087650930
compat comparison: 294874693 / 28727303180 / success / artifact 8087654840
mismatch review: 297343766 / 28727318126 / success / artifact 8087658973
member profile metrics: 266755732 / 28727335843 / success
Instagram constituency render: 261945698 / 28727349094 / success / artifact 8087668776
Instagram campaign render: 271160957 / 28727366673 / success / artifact 8087671962
```

## Current readiness position

Ready:

```text
manual downstream validation orchestration
manual current-latest validation
GitHub token child-workflow dispatch and polling
```

Not yet proven:

```text
orchestrated refresh_type=weekly
orchestrated refresh_type=monthly
scheduled orchestrator trigger
```

## Recommended next validation

Run the orchestrator with:

```text
refresh_type=monthly
run_consumers=true
```

This is the strongest next test because monthly refresh was the latest scheduled failure mode and has now been patched/validated separately.

## Scheduled trigger recommendation

Only add the scheduled trigger after one successful production-like orchestrator run with refresh enabled.

Candidate future schedule:

```yaml
schedule:
  - cron: "45 6 * * 0"
```

This would run after weekly refresh. Monthly refresh should still be handled manually or by a separate monthly orchestration schedule until stable.

## Conclusion

The orchestrator implementation is valid for manual downstream validation. Scheduled readiness is improved, but the scheduled trigger should remain deferred until the orchestrator successfully validates a refresh-enabled run.
