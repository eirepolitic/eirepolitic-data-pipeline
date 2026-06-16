# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-16  
**Current packet:** P10 — production-sized unified refresh planning

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
- Instagram consumer smoke workflow: `.github/workflows/oireachtas_instagram_consumer_smoke.yml`
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

- **P01 — latest publishing control**: workflow ID `287859326`, run `27431598142`, success, `mode=test` suppressed latest writes.
- **P02 — dynamic date windows**: weekly/monthly/yearly scheduled windows are dynamic; manual date overrides remain available.
- **P03 — downstream compatibility adapters**: workflow ID `294866317`, run `27431601110`, success, DQ pass; compat roster/vote outputs written under `processed/oireachtas_unified/compat/...`.
- **P04 — side-by-side member profile trial**: workflow ID `294874303`, final run `27432417013`, success, DQ pass; trial metrics written under unified compat paths.
- **P05 — compatibility adapter comparison report**: workflow ID `294874693`, run `27432358137`, success, DQ pass; row gaps expected from limited latest outputs.
- **P06 — production readiness checklist**: `docs/oireachtas_production_readiness_checklist.md`; complete; cutover still not approved.

## Confirmed consumer smoke packets

### P07 — consumer smoke test planning

- File: `docs/oireachtas_consumer_smoke_test_plan.md`
- Result: complete.
- Selected safest first test: Instagram constituency local renderer with only the members roster input overridden to the unified compat roster.
- No production defaults changed.

### P08 — Instagram consumer smoke workflow

- Files changed:
  - `process/instagram_render_post.py`
  - `.github/workflows/oireachtas_instagram_consumer_smoke.yml`
- Workflow ID: `297114820`
- Validation run: `27636367782`; run number 1; success.
- Artifact: `oireachtas-instagram-consumer-smoke-output`; artifact ID `7674904547`.
- Smoke configuration:
  - constituency `Wicklow-Wexford`
  - member `Brian Brennan`
  - `INSTAGRAM_MEMBERS_DATASET_KEYS=processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv`
- Validation confirmed generated context/slides and that the compat members S3 key was used.
- Output remained artifact-only under `generated_posts_oireachtas_compat_trial/`.
- Existing Instagram workflows and specs were not changed.

### P09 — final cutover request package

- File: `docs/oireachtas_final_cutover_request_package.md`
- Result: complete and updated after P08 validation.
- Still not approved for cutover.
- Exact approval phrase required before any cutover:

```text
Approved: cut over <consumer name> from legacy Oireachtas keys to unified compatibility outputs.
```

## Current caveats

- Current compat/latest outputs are still sample-sized.
- Latest validated roster compat count: 10 rows versus 176 legacy rows.
- Latest validated member-votes compat count: 512 rows versus 30,968 legacy rows.
- Deterministic unified outputs still do not replace classified debate issues, photo URL indexes, member summaries, or constituency image indexes.

## Next packet batch

### P10 — production-sized unified refresh planning

Goal:

- define safest production-sized refresh sequence for roster/votes/gold outputs;
- keep latest publishing explicit and controlled;
- avoid consumer cutover.

### P11 — production-sized refresh dry run

Goal:

- run a limited but larger non-test refresh for required consumer tables;
- publish latest only if using intended non-test mode;
- validate row counts and DQ.

### P12 — rerun adapters, comparisons, and consumer smoke after refresh

Goal:

- rebuild compatibility adapters from refreshed latest outputs;
- rerun P04/P05/P08 validation;
- update cutover package with refreshed evidence.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P10 production-sized unified refresh planning, then P11 production-sized refresh dry run, then P12 rerun adapters/comparisons/consumer smoke after refresh.
Do not repoint downstream consumers or disable old workflows without explicit user approval.
Latest validated run: P08 Instagram consumer smoke 27636367782.
Current recommendation: do not cut over yet; current compat outputs are still sample-sized.
```
