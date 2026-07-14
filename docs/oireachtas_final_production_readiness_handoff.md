# Oireachtas final production-readiness handoff

**Status:** controlled pre-production ready; scheduled orchestrator enabled for first observation  
**Last updated:** 2026-07-13 America/Vancouver

## Executive summary

The unified Oireachtas pipeline is now built, validated, cut over for controlled downstream consumers, and scheduled for weekly orchestrated validation.

The remaining work is observation, not core build-out:

```text
observe first scheduled orchestrator run
observe next real scheduled monthly refresh after fixes
choose steady-state weekly scheduling shape
```

## Current readiness state

Ready for controlled pre-production use:

```text
silver tables
gold tables
control tables
weekly/monthly/yearly refresh workflows
compatibility adapters
compatibility comparison
mismatch review
member profile metrics consumer
Instagram constituency consumer
Instagram campaign render consumer
manual refresh validation orchestrator
production-input refresh validation orchestrator
weekly scheduled refresh
weekly scheduled orchestrator trigger
```

Not yet final steady-state:

```text
first scheduled orchestrator run has not occurred yet
next real scheduled monthly refresh after fixes has not occurred yet
production Instagram publishing remains artifact-only unless explicitly enabled
```

## Orchestrator schedule

Workflow:

```text
.github/workflows/oireachtas_refresh_validation_orchestrator.yml
```

Workflow ID:

```text
307332237
```

Schedule added:

```yaml
schedule:
  - cron: "45 6 * * 0"
```

Scheduled behavior:

```text
refresh_type=weekly
run_consumers=true
```

Manual behavior remains safe:

```text
refresh_type=none
run_consumers=true
```

Schedule implementation commit:

```text
17e140f238b06fb1e667d42dedf22dc64ac914c7
```

First expected scheduled orchestrator run:

```text
2026-07-19 06:45 UTC
2026-07-18 23:45 America/Vancouver
```

## Latest orchestrator validations

### Current-output validation

```text
Run ID: 28727286429
Result: success
Artifact ID: 8087672457
```

### Refresh-enabled validation using child defaults

```text
Run ID: 29218772777
Result: success
Artifact ID: 8267558251
```

### Production-input validation

```text
Run ID: 29299431600
Result: success
Artifact ID: 8298102893
```

Production-input child results:

```text
monthly refresh: 294432002 / 29299437311 / success / artifact 8298023987
compat adapters: 294866317 / 29299539592 / success / artifact 8298034081
compat comparison: 294874693 / 29299580373 / success / artifact 8298049611
mismatch review: 297343766 / 29299619179 / success / artifact 8298060582
member profile metrics: 266755732 / 29299647855 / success
Instagram constituency render: 261945698 / 29299676372 / success / artifact 8298085395
Instagram campaign render: 271160957 / 29299727612 / success / artifact 8298101606
```

## Scheduled refresh state

### Weekly

Existing weekly refresh schedule:

```text
.github/workflows/oireachtas_weekly_refresh.yml
cron: 20 3 * * 0
```

Latest observed scheduled weekly runs:

```text
2026-07-05 scheduled weekly: 28732507909 / success
2026-07-12 scheduled weekly: 29182334702 / success
```

New orchestrator schedule runs later in the same weekly window:

```text
.github/workflows/oireachtas_refresh_validation_orchestrator.yml
cron: 45 6 * * 0
```

### Monthly

Original failed scheduled monthly run before fixes:

```text
Workflow ID: 294432002
Run ID: 28504651002
Result: failure
```

Validated after fixes:

```text
Manual production-like validation: 28726922946 / success / artifact 8087552815
Explicit-input orchestrated validation: 29299437311 / success / artifact 8298023987
```

Next real scheduled monthly run still needs observation:

```text
2026-08-01 04:35 UTC
2026-07-31 21:35 America/Vancouver
```

### Yearly

Latest validation:

```text
Workflow ID: 294432103
Run ID: 27397123885
Result: success
```

No new scheduled yearly run has occurred yet.

## Current downstream consumer state

### Member profile metrics

Workflow:

```text
.github/workflows/build_member_profile_metrics_2025.yml
```

Current unified compat inputs:

```text
members compat
member votes compat
member photo URLs compat
classified debate issue labels compat
```

Latest validation:

```text
Run ID: 29299647855
Result: success
```

### Instagram constituency renderer

Workflow:

```text
.github/workflows/instagram_constituency_test.yml
```

Current unified compat inputs:

```text
members compat
member summaries compat
constituency images compat
```

Latest validation:

```text
Run ID: 29299676372
Result: success
Artifact ID: 8298085395
```

### Instagram campaign renderer

Workflow:

```text
.github/workflows/instagram_campaign_render.yml
```

Current default:

```text
spec_file=render_spec.yml
upload_preview=false
```

Latest validation:

```text
Run ID: 29299727612
Result: success
Artifact ID: 8298101606
```

## Known caveats

```text
First scheduled orchestrator run needs observation.
Next real scheduled monthly refresh needs observation.
Existing weekly refresh and scheduled orchestrator are both enabled, so first orchestrator schedule is additive.
Production Instagram publishing remains artifact-only unless explicitly enabled.
Legacy enrichment keys remain preserved for rollback.
Known legacy-only roster caveats remain Catherine Connolly and Paschal Donohoe.
```

## Recommended next packets

```text
P75 — observe first scheduled orchestrator run
P76 — observe next scheduled monthly refresh after fixes
P77 — steady-state scheduling decision and final handoff
```

## Final readiness statement

The unified Oireachtas data model and supporting pipeline infrastructure are complete for controlled pre-production use. The first weekly scheduled orchestrator trigger has been enabled. Final production steady-state should wait for the first scheduled orchestrator observation and the next real scheduled monthly refresh observation.
