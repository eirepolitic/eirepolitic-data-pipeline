# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-17  
**Current packet:** P16 — approval-dependent cutover decision

This is the compact operational handoff for `docs/oireachtas_unified_data_model_plan.md`. Continue from `main`. Existing legacy pipelines remain untouched while unified replacements are built and validated table-by-table.

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

## Confirmed workflow/planning packets

- **W01** weekly refresh workflow: ID `294426406`; validation run `27396638715`; success.
- **W02** monthly refresh workflow: ID `294432002`; validation run `27397121321`; success.
- **W03** yearly refresh workflow: ID `294432103`; validation run `27397123885`; success.
- **X01** cutover comparison report: ID `294432488`; validation run `27397256307`; success; DQ pass.
- **X02** downstream cutover planning: `docs/oireachtas_downstream_cutover_plan.md`; complete; no consumers changed.
- **X03** production run configuration review: `docs/oireachtas_production_run_config_review.md`; complete.
- **X04** registry/status cleanup: `configs/oireachtas/tables.yml`; complete.

## Confirmed production-hardening and consumer packets

- **P01** latest publishing control: workflow ID `287859326`, run `27431598142`, success; `mode=test` suppressed latest writes.
- **P02** dynamic date windows: weekly/monthly/yearly scheduled windows are dynamic; manual date overrides remain available.
- **P03** downstream compatibility adapters: workflow ID `294866317`, run `27431601110`, success; initial compat roster 10 rows and votes 512 rows.
- **P04** side-by-side member profile trial: workflow ID `294874303`, run `27432417013`, success; initial trial metrics 10 rows.
- **P05** compatibility adapter comparison report: workflow ID `294874693`, run `27432358137`, success; initial row gaps expected from limited latest outputs.
- **P06** production readiness checklist: `docs/oireachtas_production_readiness_checklist.md`; complete.
- **P07** consumer smoke test planning: `docs/oireachtas_consumer_smoke_test_plan.md`; complete.
- **P08** Instagram consumer smoke workflow: workflow ID `297114820`, run `27636367782`, success; artifact-only output; no defaults changed.
- **P09** final cutover request package: `docs/oireachtas_final_cutover_request_package.md`; complete; no cutover approved.

## Confirmed production-sized refresh packets

- **P10** production-sized refresh plan: `docs/oireachtas_production_sized_refresh_plan.md`; complete.
- **P11** production-sized refresh dry run:
  - Workflow ID `297334648`, run `27661934424`, success.
  - Settings: `mode=full`, `publish_latest=auto`, `date_start=2025-01-01`, `date_end=2025-12-31`, `limit=200`.
  - `gold_current_members`: 174 rows, DQ pass, latest publish enabled.
  - `silver_member_votes`: 29,805 rows, 200 divisions, DQ pass.
  - No downstream consumers were repointed.
- **P12** post-refresh reruns:
  - Adapters: workflow ID `294866317`, run `27661982505`, success; roster compat 174 rows, member votes compat 29,805 rows.
  - Member profile trial: workflow ID `294874303`, run `27661985049`, success; trial metric rows 174; matched legacy member codes 172.
  - Comparison: workflow ID `294874693`, run `27661990358`, success; roster 176 vs 174, member votes 30,968 vs 29,805.
  - Instagram smoke: workflow ID `297114820`, run `27661992188`, success; artifact ID `7684743075`.

## Confirmed approval-readiness packets

### P13 — remaining mismatch review

- Files changed:
  - `extract/oireachtas/mismatch_review.py`
  - `.github/workflows/oireachtas_mismatch_review.yml`
- Workflow ID: `297343766`
- Validation run: `27662884471`; run number 1; success; DQ pass.
- Review: `review/member_code_mismatch_review/latest/{manifest.json,sample.csv,dq.json,report.md}`.
- Remaining mismatch details:
  - roster legacy-only: Catherine Connolly, Paschal Donohoe.
  - member profile legacy-only: Catherine Connolly, Paschal Donohoe.
  - member profile trial-only: Daniel Ennis, Seán Kyne.
- Interpretation: source freshness/member lifecycle differences rather than a deterministic build failure.

### P14 — cutover approval checklist update

- File updated: `docs/oireachtas_production_readiness_checklist.md`
- Result: post-refresh evidence and mismatch review evidence added.
- Status: all technical gates complete except explicit consumer-specific approval.

### P15 — optional approved cutover patch preparation

- File added: `docs/oireachtas_approved_cutover_patch_plan.md`
- Result: exact reversible patch plan documented for:
  - Instagram constituency renderer.
  - Member profile metrics.
- No production cutover patch was applied.

## Current caveats

- Roster comparison has 2 legacy-only member codes: Catherine Connolly and Paschal Donohoe.
- Member profile metrics comparison has 2 legacy-only member codes, Catherine Connolly and Paschal Donohoe, and 2 trial-only member codes, Daniel Ennis and Seán Kyne.
- Deterministic unified outputs still do not replace classified debate issues, photo URL indexes, member summaries, or constituency image indexes.
- Explicit user approval is required before any downstream cutover.

## Next packet batch

### P16 — approval-dependent cutover decision

Goal:

- stop and request explicit consumer-specific approval before any production cutover;
- no code changes unless approval phrase is provided.

### P17 — approved Instagram cutover patch, only if approved

Goal:

- if explicitly approved, apply the documented Instagram environment-variable patch;
- rerun Instagram workflow and confirm rollback instructions.

### P18 — approved member-profile metrics cutover patch, only if approved

Goal:

- if explicitly approved, apply the documented member-profile input-key patch;
- rerun member-profile workflow and confirm rollback instructions.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P16 approval-dependent cutover decision.
Do not apply P17 or P18 unless the user provides the exact explicit approval phrase for that consumer.
Latest validated run: P13 mismatch review 27662884471.
All technical readiness evidence is documented, but no downstream cutover is approved yet.
```
