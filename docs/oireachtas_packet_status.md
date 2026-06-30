# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-30  
**Current packet:** P23 — steady-state monitoring / next consumer selection

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
  - Run number 5
  - Status success
  - Artifact `instagram-constituency-test`
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

### P20 — verify generated outputs

- Instagram run `28414647932`: success; artifact `instagram-constituency-test`, artifact ID `7968986389`.
- Member profile metrics run `28414678714`: success; no artifact expected.

### P21 — rerun comparison after cutover

- Compatibility comparison workflow ID `294874693`, run `28414819264`, success.
- Sample result:
  - Roster: 176 legacy rows vs 174 compat rows; 174 matched keys, 2 legacy-only, 0 compat-only.
  - Member votes: 30,968 legacy rows vs 29,805 compat rows; 173 matched member-code keys, 0 legacy-only, 0 compat-only.
- Mismatch review workflow ID `297343766`:
  - First rerun `28414820972`: build success, review-branch publish failure due concurrent branch update.
  - Clean rerun `28414847238`: success, DQ pass.
- Latest mismatch summary:
  - Roster: 176 legacy members, 174 unified members, 174 matched, 2 legacy-only, 0 unified-only.
  - Member profile metrics: 174 legacy members, 174 unified members, 174 matched, 0 legacy-only, 0 unified-only.

### P22 — operational docs and final status

- Added `docs/oireachtas_post_cutover_validation_summary.md`.
- Updated `docs/oireachtas_production_readiness_checklist.md`.
- Updated `docs/oireachtas_final_cutover_request_package.md` previously after P17/P18.
- Rollback remains workflow-config only and is documented in `docs/oireachtas_post_cutover_monitoring_plan.md`.

## Current caveats

- Roster comparison has 2 legacy-only member codes:
  - Catherine Connolly — Independent — Galway West
  - Paschal Donohoe — Fine Gael — Dublin Central
- Member profile metrics now have 0 member-code mismatches after the cutover build.
- Deterministic unified outputs still do not replace classified debate issues, photo URL indexes, member summaries, or constituency image indexes.

## Next packet batch

### P23 — steady-state monitoring / next consumer selection

Goal:

- choose whether to monitor current cutovers only or identify the next downstream consumer for unified compat outputs.

### P24 — scheduled refresh readiness review

Goal:

- review whether weekly/monthly/yearly Oireachtas refresh workflows should be run manually after cutover.

### P25 — optional next compatibility adapter

Goal:

- if another consumer needs a legacy-shaped input, add a compatibility adapter rather than repointing raw unified silver/gold tables directly.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P23 steady-state monitoring / next consumer selection, then P24 scheduled refresh readiness review, then P25 optional next compatibility adapter if a next consumer is known.
Latest successful validations: Instagram cutover run 28414647932, member-profile metrics run 28414678714, comparison run 28414819264, mismatch review run 28414847238.
```
