# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-12  
**Current packet:** P04 — side-by-side member profile trial

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
- AWS region: `ca-central-1`
- S3 bucket: `eirepolitic-data`
- Review branch: `oireachtas-review-output`
- Review publishing preserves existing table folders and runs after table/DQ failure when local review output exists.
- Standard confirmed outputs: raw API/source files, partitioned CSV, partitioned Parquet, latest CSV/Parquet pointers, run manifest, review sample/schema/manifest/DQ.
- Runtime rule: `mode=test` now suppresses writes to `processed/oireachtas_unified/latest/*` unless explicitly overridden.

## Confirmed foundation, table, gold, and control packets

- **F01-F03** complete.
- **T01-T23** silver tables complete with DQ pass.
- **G01-G05** gold tables complete with DQ pass.
- **C01-C03** control tables complete with DQ pass.
- `configs/oireachtas/tables.yml` marks all validated silver/gold/control tables as `confirmed`.

## Confirmed workflow packets

- **W01 — weekly refresh workflow**: workflow ID `294426406`; validation run `27396638715`; success; schedule `20 3 * * 0`.
- **W02 — monthly refresh workflow**: workflow ID `294432002`; validation run `27397121321`; success; schedule `35 4 1 * *`.
- **W03 — yearly refresh workflow**: workflow ID `294432103`; validation run `27397123885`; success; schedule `15 5 2 1 *`.

## Confirmed comparison/planning packets

- **X01 — cutover comparison report**: workflow ID `294432488`; final validation run `27397256307`; success; 5 comparison rows; DQ pass.
- **X02 — downstream cutover planning**: `docs/oireachtas_downstream_cutover_plan.md`; complete; no consumers changed.
- **X03 — production run configuration review**: `docs/oireachtas_production_run_config_review.md`; complete.
- **X04 — registry/status cleanup**: `configs/oireachtas/tables.yml`; complete.

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
- Validation:
  - Workflow: `Oireachtas Table Test (Manual)`, ID `287859326`.
  - Run: `27431598142`; run number 49; success.
  - Review manifest: `publish_latest=false`, `latest_write_policy=suppressed`, DQ pass.

### P02 — dynamic date windows

- Files changed:
  - `.github/workflows/oireachtas_weekly_refresh.yml`
  - `.github/workflows/oireachtas_monthly_refresh.yml`
  - `.github/workflows/oireachtas_yearly_refresh.yml`
- Behavior:
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
- Validation run: `27431601110`; run number 1; success.
- DQ: pass
- Output summary rows: 2
- Compatibility outputs:
  - `processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv`
  - `processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv`
- Latest validation row counts:
  - members roster compat: 10 rows from `gold_current_members`
  - member votes compat: 512 rows from `silver_member_votes`
- Review: `review/compat_downstream_adapters/latest/{manifest.json,sample.csv,dq.json}`.
- No legacy S3 keys were overwritten.

## Next packet batch

### P04 — side-by-side member profile trial

Goal:

- run `process/build_member_profile_metrics_2025.py` against compatibility input keys;
- write trial outputs under `processed/oireachtas_unified/compat/members/...`;
- do not overwrite legacy metric outputs;
- publish a review summary comparing row counts and key fields.

### P05 — compatibility adapter comparison report

Goal:

- compare legacy member roster/vote records to compatibility outputs;
- report row counts, populated-key coverage, and key overlap;
- publish review-only report.

### P06 — production readiness checklist

Goal:

- create final checklist for approval before any downstream cutover;
- include latest publishing policy, scheduled windows, adapter outputs, rollback, and explicit approval gate;
- keep it documentation-only.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P04 side-by-side member profile trial, then P05 compatibility adapter comparison report, then P06 production readiness checklist.
Do not repoint downstream consumers or disable old workflows without explicit user approval.
Latest validated runs: P01 table test 27431598142, P03 compat adapters 27431601110.
P02 workflow YAML is active; scheduled windows are dynamic but scheduled events were not manually simulated.
```
