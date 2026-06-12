# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-12  
**Current packet:** P07 — consumer smoke test planning

This is the compact operational handoff for `docs/oireachtas_unified_data_model_plan.md`. Continue from `main`. Existing legacy pipelines remain untouched while unified replacements are built and validated table-by-table.

## Shared implementation state

- CLI: `python -m extract.oireachtas.build_table`
- Registry: `configs/oireachtas/tables.yml`
- Manual test workflow: `.github/workflows/oireachtas_table_test.yml`
- Weekly refresh workflow: `.github/workflows/oireachtas_weekly_refresh.yml`
- Monthly refresh workflow: `.github/workflows/oireachtas_monthly_refresh.yml`
- Yearly refresh workflow: `.github/workflows/oireachtas_yearly_refresh.yml`
- Cutover comparison workflow: `.github/workflows/oireachtas_cutover_comparison.yml`
- Compatibility adapter workflow: `.github/workflows/oireachtas_compat_adapters.yml`
- Member profile trial workflow: `.github/workflows/oireachtas_member_profile_trial.yml`
- Compatibility comparison workflow: `.github/workflows/oireachtas_compat_comparison.yml`
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

## Confirmed production-hardening packets

### P01 — latest publishing control

- Files changed:
  - `extract/oireachtas/io_s3.py`
  - `extract/oireachtas/build_table.py`
  - `.github/workflows/oireachtas_table_test.yml`
  - `.github/workflows/oireachtas_weekly_refresh.yml`
  - `.github/workflows/oireachtas_monthly_refresh.yml`
  - `.github/workflows/oireachtas_yearly_refresh.yml`
- Behavior:
  - CLI option: `--publish-latest auto|true|false`.
  - `auto` disables latest writes for `mode=test` and enables them for non-test modes.
  - Shared S3 IO guard suppresses writes only under `processed/oireachtas_unified/latest/` when disabled.
- Validation: workflow ID `287859326`, run `27431598142`, success, `publish_latest=false`, `latest_write_policy=suppressed`, DQ pass.

### P02 — dynamic date windows

- Weekly scheduled runs use rolling UTC `35 days ago` to today.
- Monthly scheduled runs use previous full month plus 7-day overlap.
- Yearly scheduled runs use previous calendar year.
- Manual workflow dispatch still accepts explicit `date_start` and `date_end` overrides.
- Workflows remain active after YAML updates.

### P03 — downstream compatibility adapters

- Files changed:
  - `extract/oireachtas/downstream_compat.py`
  - `.github/workflows/oireachtas_compat_adapters.yml`
- Workflow ID: `294866317`
- Validation run: `27431601110`; success; DQ pass.
- Outputs:
  - `processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv`
  - `processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv`
- Latest validation row counts:
  - members roster compat: 10 rows
  - member votes compat: 512 rows
- No legacy S3 keys were overwritten.

### P04 — side-by-side member profile trial

- Files changed:
  - `extract/oireachtas/member_profile_trial_report.py`
  - `.github/workflows/oireachtas_member_profile_trial.yml`
- Workflow ID: `294874303`
- First run `27432354981` built trial/report successfully but failed during review-branch publish due concurrent branch update.
- Patched workflow publish step with `git pull --rebase` before push.
- Final validation run: `27432417013`; run number 2; success; DQ pass.
- Trial outputs:
  - `processed/oireachtas_unified/compat/members/member_profile_metrics_2025_trial.csv`
  - `processed/oireachtas_unified/compat/members/parquets/member_profile_metrics_2025_trial.parquet`
- Review: `review/member_profile_metrics_trial/latest/{manifest.json,sample.csv,dq.json,report.md}`.
- Latest trial summary: legacy rows 174, trial rows 10, matched legacy member codes 10, common columns 12.
- No legacy metric keys were overwritten.

### P05 — compatibility adapter comparison report

- Files changed:
  - `extract/oireachtas/compat_comparison.py`
  - `.github/workflows/oireachtas_compat_comparison.yml`
- Workflow ID: `294874693`
- Validation run: `27432358137`; run number 1; success; DQ pass.
- Review: `review/compat_adapter_comparison/latest/{manifest.json,sample.csv,dq.json,report.md}`.
- Latest comparison summary:
  - roster compat: 176 legacy rows, 10 compat rows, 10 matched keys, 166 legacy-only keys.
  - member-votes compat: 30,968 legacy rows, 512 compat rows, 172 matched member codes, 1 legacy-only member code.
- Row gaps are expected until production-sized unified latest outputs exist.

### P06 — production readiness checklist

- File: `docs/oireachtas_production_readiness_checklist.md`
- Result: complete and updated after P04/P05 validation.
- Still not approved for cutover.
- Remaining required gate: consumer smoke test using trial/compat keys and explicit user approval.

## Next packet batch

### P07 — consumer smoke test planning

Goal:

- identify the safest downstream consumer smoke test path using trial/compat keys;
- prefer environment-variable overrides or a dedicated trial workflow;
- do not change production consumer defaults.

### P08 — Instagram/member-profile consumer trial workflow

Goal:

- run a downstream preview/render or member-profile consumer workflow against trial/compat keys;
- write outputs to trial locations or artifacts only;
- do not publish or overwrite production outputs.

### P09 — final cutover request package

Goal:

- summarize validated evidence and known row-count gaps;
- list exact consumer-specific changes that require approval;
- stop before applying any cutover unless the user explicitly approves.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P07 consumer smoke test planning, then P08 Instagram/member-profile consumer trial workflow, then P09 final cutover request package.
Do not repoint downstream consumers or disable old workflows without explicit user approval.
Latest validated runs: P04 member profile trial 27432417013, P05 compat comparison 27432358137.
Current recommendation: do not cut over yet; run consumer smoke test first.
```
