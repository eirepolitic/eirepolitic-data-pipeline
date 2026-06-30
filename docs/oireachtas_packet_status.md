# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-30  
**Current packet:** P26 — next workstream selection

This is the compact operational handoff for `docs/oireachtas_unified_data_model_plan.md`. Continue from `main`.

## Shared implementation state

- CLI: `python -m extract.oireachtas.build_table`
- Registry: `configs/oireachtas/tables.yml`
- Manual test workflow: `.github/workflows/oireachtas_table_test.yml`
- Weekly refresh workflow: `.github/workflows/oireachtas_weekly_refresh.yml`
- Monthly refresh workflow: `.github/workflows/oireachtas_monthly_refresh.yml`
- Yearly refresh workflow: `.github/workflows/oireachtas_yearly_refresh.yml`
- Production-sized refresh dry-run workflow: `.github/workflows/oireachtas_production_refresh_dry_run.yml`
- Compatibility adapter workflow: `.github/workflows/oireachtas_compat_adapters.yml`
- Member profile trial workflow: `.github/workflows/oireachtas_member_profile_trial.yml`
- Compatibility comparison workflow: `.github/workflows/oireachtas_compat_comparison.yml`
- Instagram consumer smoke workflow: `.github/workflows/oireachtas_instagram_consumer_smoke.yml`
- Mismatch review workflow: `.github/workflows/oireachtas_mismatch_review.yml`
- AWS region: `ca-central-1`
- S3 bucket: `eirepolitic-data`
- Review branch: `oireachtas-review-output`
- Runtime rule: `mode=test` suppresses writes to `processed/oireachtas_unified/latest/*` unless explicitly overridden.

## Confirmed foundation, table, gold, and control packets

- **F01-F03** complete.
- **T01-T23** silver tables complete with DQ pass.
- **G01-G05** gold tables complete with DQ pass.
- **C01-C03** control tables complete with DQ pass.
- `configs/oireachtas/tables.yml` marks all validated silver/gold/control tables as `confirmed`.

## Confirmed production-hardening and consumer packets

- **P01-P09** complete through initial cutover package.
- **P10-P12** complete through production-sized refresh and post-refresh validation.
- **P13-P15** complete through mismatch review, readiness checklist, and patch plan.
- **P16** approval gate documented; user later clarified these are pre-production systems and explicit approval phrase is not required.
- **P17** Instagram constituency renderer cutover applied and validated.
- **P18** member-profile metrics cutover applied and validated.
- **P19** post-cutover monitoring plan added.
- **P20-P22** post-cutover evidence verified, comparison/mismatch rerun, docs updated.

## Applied pre-production cutovers

### Instagram constituency renderer

- File: `.github/workflows/instagram_constituency_test.yml`
- Default roster input:

```yaml
      INSTAGRAM_MEMBERS_DATASET_KEYS: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
```

- Validation:
  - Workflow ID `261945698`
  - Run `28414647932`
  - Status success
  - Artifact ID `7968986389`

### Member profile metrics

- File: `.github/workflows/build_member_profile_metrics_2025.yml`
- Default inputs:

```yaml
      MEMBERS_INPUT_KEY: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
      MEMBER_VOTES_INPUT_KEY: "processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv"
```

- Legacy vote-record rebuild step removed from this workflow because the metrics build now reads compat vote records directly.
- Validation:
  - Failed pre-correction run `28414649704`: old legacy vote rebuild failed before metrics step.
  - Corrected run `28414678714`: success.

## Post-cutover validation

- **P20** generated outputs verified:
  - Instagram run `28414647932`: success; artifact ID `7968986389`.
  - Member profile metrics run `28414678714`: success.
- **P21** comparison/mismatch rerun:
  - Compatibility comparison run `28414819264`: success.
  - Mismatch first rerun `28414820972`: build success, review-branch publish failure due concurrent branch update.
  - Mismatch clean rerun `28414847238`: success, DQ pass.
- **P22** operational docs updated:
  - `docs/oireachtas_post_cutover_validation_summary.md`
  - `docs/oireachtas_production_readiness_checklist.md`

## Confirmed steady-state packets

### P23 — steady-state monitoring / next consumer selection

- File added/updated: `docs/oireachtas_steady_state_monitoring_and_next_consumer.md`
- Next consumer selected: `Instagram Campaign Render (Manual)`.
- Workflow changed: `.github/workflows/instagram_campaign_render.yml`
- Change: default `spec_file` now uses `render_spec.yml` instead of the synthetic fixture.
- Workflow remains artifact-only by default: `upload_preview=false`.
- Validation:
  - Workflow ID `271160957`
  - Run `28415050102`
  - Run number 6
  - Status success
  - Artifact `instagram-campaign-render-output`
  - Artifact ID `7969146127`

### P24 — scheduled refresh readiness review

- File added: `docs/oireachtas_scheduled_refresh_readiness_review.md`
- Result: weekly/monthly/yearly refreshes are ready with monitoring caveats.
- Recommendation: do not run review-publishing workflows concurrently; after any scheduled refresh, run adapters, comparison, mismatch review, and consumer validations sequentially.

### P25 — optional next compatibility adapter

- File added: `docs/oireachtas_next_compatibility_adapter_review.md`
- Result: no new compatibility adapter required for the campaign renderer.
- Reason: campaign renderer consumes `processed/members/member_profile_metrics_2025.csv`, which is already produced from unified compatibility inputs after P18.

## Current caveats

- Roster comparison has 2 legacy-only member codes:
  - Catherine Connolly — Independent — Galway West
  - Paschal Donohoe — Fine Gael — Dublin Central
- Member profile metrics now have 0 member-code mismatches after the cutover build.
- Deterministic unified outputs still do not replace classified debate issues, photo URL indexes, member summaries, or constituency image indexes.
- Review branch publish conflicts can happen if multiple review-publishing workflows run concurrently. Rerun after the other publish completes.

## Next packet batch

### P26 — next workstream selection

Goal:

- choose the next workstream: enrichment replacement, scheduled refresh hardening, or additional consumer cutovers.

### P27 — enrichment replacement planning

Goal:

- map non-deterministic or enrichment-only dependencies that still use legacy outputs: classified issues, photo URLs, member summaries, constituency images.

### P28 — refresh hardening patch review

Goal:

- decide whether to patch weekly/monthly/yearly review-branch publishing to reduce concurrent push conflicts.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P26 next workstream selection, then P27 enrichment replacement planning, then P28 refresh hardening patch review.
Latest successful validations: Instagram constituency run 28414647932, member-profile metrics run 28414678714, comparison run 28414819264, mismatch review run 28414847238, campaign render run 28415050102.
```
