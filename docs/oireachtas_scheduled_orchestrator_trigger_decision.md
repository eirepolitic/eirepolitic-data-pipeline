# Oireachtas scheduled orchestrator trigger decision

**Status:** enabled  
**Last updated:** 2026-07-13 America/Vancouver

## Decision

Enable a weekly scheduled trigger for the Oireachtas Refresh Validation Orchestrator.

Workflow:

```text
.github/workflows/oireachtas_refresh_validation_orchestrator.yml
```

Workflow ID:

```text
307332237
```

Implementation commit:

```text
17e140f238b06fb1e667d42dedf22dc64ac914c7
```

## Schedule

The orchestrator now runs weekly:

```yaml
schedule:
  - cron: "45 6 * * 0"
```

That is:

```text
Sunday 06:45 UTC
Saturday 23:45 America/Vancouver during daylight saving time
```

The first expected scheduled run after this change is:

```text
2026-07-19 06:45 UTC
2026-07-18 23:45 America/Vancouver
```

## Scheduled behavior

Scheduled events force:

```text
refresh_type=weekly
run_consumers=true
```

Manual events still default to:

```text
refresh_type=none
run_consumers=true
```

## Why this is now safe enough

The scheduled trigger was previously deferred because the orchestrator did not pass explicit child refresh inputs. That gap has now been closed and validated.

Latest production-input orchestrator validation:

```text
Workflow ID: 307332237
Run ID: 29299431600
Result: success
Artifact ID: 8298102893
```

That validation dispatched a child refresh with explicit production-like inputs and then completed downstream checks:

```text
monthly refresh
compat adapters
compat comparison
mismatch review
member profile metrics
Instagram constituency render
Instagram campaign render
```

## Additive first schedule

The existing weekly refresh workflow remains enabled:

```text
.github/workflows/oireachtas_weekly_refresh.yml
cron: 20 3 * * 0
```

The orchestrator schedule runs after that window:

```text
orchestrator cron: 45 6 * * 0
```

This means the first scheduled orchestrator is additive. It may run a second weekly refresh on the same weekend. That is acceptable for the first controlled scheduled validation because the workflows write versioned outputs and publish latest pointers intentionally.

## Follow-up decision after observation

After the first scheduled orchestrator succeeds, choose one of these steady-state options:

```text
Option A: keep both weekly refresh and weekly orchestrator enabled
Option B: keep the orchestrator schedule and disable the standalone weekly refresh
Option C: change scheduled orchestrator to refresh_type=none and let standalone weekly refresh remain primary
```

Recommended next observation:

```text
P75 — observe first scheduled orchestrator run
```

## Remaining scheduled caveat

The monthly refresh fixes are manually validated, but the next real scheduled monthly run still needs observation.

Next expected monthly scheduled refresh:

```text
2026-08-01 04:35 UTC
2026-07-31 21:35 America/Vancouver
```
