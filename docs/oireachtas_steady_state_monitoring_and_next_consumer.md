# Oireachtas steady-state monitoring and next consumer selection

**Status:** active  
**Last updated:** 2026-06-30

## Current cutover state

The following pre-production consumers now use unified Oireachtas compatibility outputs directly or indirectly:

| Consumer | Status | Validation |
|---|---|---|
| Instagram constituency renderer | cut over to unified compat roster | run `28414647932`, success |
| Member profile metrics | cut over to unified compat roster and vote records | run `28414678714`, success |
| Instagram campaign renderer | selected as next validation consumer | run `28415050102` dispatched |

## Next consumer selected

The next downstream consumer is:

```text
Instagram Campaign Render (Manual)
```

Workflow:

```text
.github/workflows/instagram_campaign_render.yml
```

Reason:

- It consumes `processed/members/member_profile_metrics_2025.csv` through `instagram/campaigns/member_profile_batch_v1/render_spec.yml`.
- That metrics file is now produced by the cut-over member-profile metrics workflow using unified compatibility inputs.
- No new legacy-shaped adapter is required for this consumer.

## Change applied

The campaign render workflow default spec file was changed from the synthetic fixture to the real campaign spec:

```yaml
      spec_file:
        default: "render_spec.yml"
```

The workflow remains artifact-only by default:

```yaml
      upload_preview:
        default: "false"
```

It does not publish, schedule, or approve Instagram content.

## Monitoring baseline

Latest post-cutover validation summary:

```text
docs/oireachtas_post_cutover_validation_summary.md
```

Key baseline:

| Dataset | Legacy members | Unified members | Matched | Legacy-only | Unified-only |
|---|---:|---:|---:|---:|---:|
| roster | 176 | 174 | 174 | 2 | 0 |
| member_profile_metrics | 174 | 174 | 174 | 0 | 0 |

Remaining roster-only mismatches:

```text
Catherine Connolly — Independent — Galway West
Paschal Donohoe — Fine Gael — Dublin Central
```

## Monitoring checks

After each refresh or consumer run:

1. Confirm workflow conclusion is `success`.
2. Confirm expected artifact exists.
3. Rerun compatibility comparison if unified latest outputs changed.
4. Rerun mismatch review if roster or member metrics changed.
5. Check member-profile metrics remains at 174 matched member codes unless source data changes.
6. Keep rollback instructions visible in `docs/oireachtas_post_cutover_monitoring_plan.md`.
