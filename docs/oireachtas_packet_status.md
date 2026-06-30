# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-30  
**Current packet:** P20 — continue post-cutover validation

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

- **P01-P09** complete through initial cutover package; no consumer defaults were changed at that stage.
- **P10-P12** complete through production-sized refresh and post-refresh validation.
- **P13-P15** complete through mismatch review, readiness checklist, and patch plan.
- **P16** approval gate documented; later user clarified these are pre-production systems and explicit approval phrase is not required.

## Applied pre-production cutovers

### P17 — Instagram constituency renderer cutover

- File changed: `.github/workflows/instagram_constituency_test.yml`
- Change:
  - Added `INSTAGRAM_MEMBERS_DATASET_KEYS=processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv`
- Validation:
  - Workflow ID `261945698`
  - Run `28414647932`
  - Run number 5
  - Status success
  - Artifact `instagram-constituency-test`
  - Artifact ID `7968986389`

### P18 — member profile metrics cutover

- File changed: `.github/workflows/build_member_profile_metrics_2025.yml`
- Change:
  - Added `MEMBERS_INPUT_KEY=processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv`
  - Added `MEMBER_VOTES_INPUT_KEY=processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv`
  - Removed the legacy vote-record rebuild step from this workflow because the metrics build now reads compat vote records directly.
- Validation:
  - First run `28414649704` failed before metrics build in the removed legacy vote-record rebuild step.
  - Corrected run `28414678714`
  - Run number 4
  - Status success

### P19 — post-cutover monitoring plan

- File added: `docs/oireachtas_post_cutover_monitoring_plan.md`
- Result: monitoring and rollback instructions documented.

## Current caveats

- Roster comparison has 2 legacy-only member codes: Catherine Connolly and Paschal Donohoe.
- Member profile metrics comparison has 2 legacy-only member codes, Catherine Connolly and Paschal Donohoe, and 2 trial-only member codes, Daniel Ennis and Seán Kyne.
- Deterministic unified outputs still do not replace classified debate issues, photo URL indexes, member summaries, or constituency image indexes.

## Next packet batch

### P20 — verify post-cutover generated outputs

Goal:

- inspect workflow artifacts/output evidence from P17/P18 where available;
- confirm generated post artifact exists and member metrics workflow completed successfully.

### P21 — rerun comparison after cutover

Goal:

- rerun adapter comparison and mismatch review after cutover;
- verify no unexpected drift from the P13 baseline.

### P22 — update operational docs and final status

Goal:

- update readiness/checklist docs with applied cutover status;
- keep rollback instructions visible.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P20 verify post-cutover generated outputs, then P21 rerun comparison after cutover, then P22 update operational docs and final status.
Latest successful cutover validation: Instagram run 28414647932 and member-profile metrics run 28414678714.
```
