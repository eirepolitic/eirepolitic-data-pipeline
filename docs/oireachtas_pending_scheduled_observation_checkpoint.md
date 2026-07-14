# Oireachtas pending scheduled observation checkpoint

**Status:** pending scheduled observations  
**Last checked:** 2026-07-13 America/Vancouver

## Purpose

Record the current observation state for the scheduled orchestrator and monthly refresh after the scheduled orchestrator trigger was enabled.

## P75 — first scheduled orchestrator observation

Status:

```text
pending
```

Latest orchestrator runs checked:

```text
29299431600 / workflow_dispatch / success / created 2026-07-14T01:47:13Z
29218772777 / workflow_dispatch / success / created 2026-07-13T02:05:39Z
28727286429 / workflow_dispatch / success / created 2026-07-05T02:44:16Z
```

No `schedule` event has occurred yet for:

```text
.github/workflows/oireachtas_refresh_validation_orchestrator.yml
```

Expected first scheduled orchestrator run:

```text
2026-07-19 06:45 UTC
2026-07-18 23:45 America/Vancouver
```

Current conclusion:

```text
P75 cannot be marked complete yet because the first scheduled orchestrator event has not occurred.
```

## P76 — next scheduled monthly refresh observation

Status:

```text
pending
```

Latest monthly refresh runs checked:

```text
29299437311 / workflow_dispatch / success / created 2026-07-14T01:47:21Z
29218778131 / workflow_dispatch / success / created 2026-07-13T02:05:46Z
28726922946 / workflow_dispatch / success / created 2026-07-05T02:26:14Z
28504651002 / schedule / failure / created 2026-07-01T08:36:19Z / before fix
```

No post-fix scheduled monthly run has occurred yet.

Expected next scheduled monthly refresh:

```text
2026-08-01 04:35 UTC
2026-07-31 21:35 America/Vancouver
```

Current conclusion:

```text
P76 cannot be marked complete yet because the next real scheduled monthly event has not occurred.
```

## P77 — steady-state scheduling decision

Status:

```text
blocked pending P75 and P76 observations
```

Reason:

```text
The first scheduled orchestrator run and next post-fix scheduled monthly refresh need observation before choosing the final steady-state schedule.
```

Candidate steady-state options remain:

```text
Option A: keep both standalone weekly refresh and scheduled orchestrator enabled
Option B: keep the orchestrator schedule and disable standalone weekly refresh
Option C: change scheduled orchestrator to refresh_type=none and leave standalone weekly refresh as primary
```

Recommended next action:

```text
After 2026-07-19 06:45 UTC, check workflow 307332237 for a schedule event.
After 2026-08-01 04:35 UTC, check workflow 294432002 for a schedule event.
```
