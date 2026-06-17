# Oireachtas cutover approval decision

**Status:** waiting for explicit approval  
**Last updated:** 2026-06-17  
**No production cutover was applied.**

## Decision state

The technical readiness evidence is complete for the first two candidate consumers:

```text
Instagram constituency renderer
Member profile metrics
```

The latest user instruction was:

```text
Continue
```

That is not the required consumer-specific approval phrase, so P17 and P18 remain blocked.

## Required approval phrase

A cutover patch may only be applied after one of these explicit approvals:

```text
Approved: cut over Instagram constituency renderer from legacy Oireachtas keys to unified compatibility outputs.
```

```text
Approved: cut over member profile metrics from legacy Oireachtas keys to unified compatibility outputs.
```

## Evidence summary

| Evidence | Run | Result |
|---|---:|---|
| Production-sized refresh | `27661934424` | success |
| Compatibility adapters | `27661982505` | success |
| Member profile trial | `27661985049` | success |
| Compatibility comparison | `27661990358` | success |
| Instagram consumer smoke | `27661992188` | success |
| Mismatch review | `27662884471` | success |

## Blocked packets

| Packet | Status | Reason |
|---|---|---|
| P17 approved Instagram cutover patch | blocked | Missing explicit Instagram approval phrase. |
| P18 approved member-profile metrics cutover patch | blocked | Missing explicit member-profile approval phrase. |

## Ready patch plan

The exact reversible cutover patches are documented in:

```text
docs/oireachtas_approved_cutover_patch_plan.md
```

## Current safe action

Review the evidence and choose whether to approve one consumer at a time. Until approval exists, keep all legacy consumer defaults active.
