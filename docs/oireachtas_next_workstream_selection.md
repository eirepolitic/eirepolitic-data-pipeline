# Oireachtas next workstream selection

**Status:** selected  
**Last updated:** 2026-06-30

## Decision

The next workstream is:

```text
Enrichment replacement planning and refresh hardening
```

## Why this workstream

The deterministic unified Oireachtas table model and the first downstream consumer cutovers are now validated.

Completed consumer validations:

| Consumer | Run | Result |
|---|---:|---|
| Instagram constituency renderer | `28414647932` | success |
| Member profile metrics | `28414678714` | success |
| Instagram campaign renderer | `28415050102` | success |

The remaining known gaps are not core Oireachtas table extraction gaps. They are enrichment or workflow-hardening gaps:

```text
classified debate issues
member photo URLs
member summaries/backgrounds
constituency image indexes
review-branch publish conflicts
```

## Workstream priorities

1. Document enrichment dependencies and decide which can become deterministic pipeline outputs.
2. Keep non-deterministic enrichment outputs separate from deterministic Oireachtas source tables.
3. Harden review-output publishing to reduce branch push conflicts.
4. Continue validating consumers through artifacts before enabling publishing or public preview uploads.

## Out of scope for this workstream

- Deleting legacy S3 keys.
- Disabling legacy workflows.
- Replacing LLM/classified outputs without a separate enrichment design.
- Publishing Instagram content automatically.

## Current recommendation

Proceed with enrichment planning and review-publishing hardening before adding more consumer cutovers.
