# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-17  
**Current packet:** P13 — remaining mismatch review

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

- **P01 — latest publishing control**: workflow ID `287859326`, run `27431598142`, success; `mode=test` suppressed latest writes.
- **P02 — dynamic date windows**: weekly/monthly/yearly scheduled windows are dynamic; manual date overrides remain available.
- **P03 — downstream compatibility adapters**: workflow ID `294866317`, run `27431601110`, success; initial compat roster 10 rows and votes 512 rows.
- **P04 — side-by-side member profile trial**: workflow ID `294874303`, run `27432417013`, success; initial trial metrics 10 rows.
- **P05 — compatibility adapter comparison report**: workflow ID `294874693`, run `27432358137`, success; initial row gaps expected from limited latest outputs.
- **P06 — production readiness checklist**: `docs/oireachtas_production_readiness_checklist.md`; complete; cutover not approved.
- **P07 — consumer smoke test planning**: `docs/oireachtas_consumer_smoke_test_plan.md`; complete.
- **P08 — Instagram consumer smoke workflow**: workflow ID `297114820`, run `27636367782`, success; artifact-only output; no defaults changed.
- **P09 — final cutover request package**: `docs/oireachtas_final_cutover_request_package.md`; complete; no cutover approved.

## Confirmed production-sized refresh packets

### P10 — production-sized unified refresh planning

- File: `docs/oireachtas_production_sized_refresh_plan.md`
- Result: complete.
- Target tables:
  - `silver_members`
  - `silver_member_memberships`
  - `silver_member_parties`
  - `silver_member_constituencies`
  - `silver_member_offices`
  - `gold_current_members`
  - `silver_member_votes`

### P11 — production-sized refresh dry run

- Workflow: `.github/workflows/oireachtas_production_refresh_dry_run.yml`
- Workflow ID: `297334648`
- Validation run: `27661934424`; run number 1; success.
- Settings:
  - `mode=full`
  - `publish_latest=auto`
  - `date_start=2025-01-01`
  - `date_end=2025-12-31`
  - `limit=200`
- Key refreshed outputs:
  - `gold_current_members`: 174 rows, DQ pass, latest publish enabled.
  - `silver_member_votes`: 29,805 rows, 200 divisions, DQ pass.
- No downstream consumers were repointed.

### P12 — rerun adapters, comparisons, and consumer smoke after refresh

- Adapter rerun:
  - Workflow ID `294866317`, run `27661982505`, success.
  - Compat roster: 174 rows.
  - Compat member votes: 29,805 rows.
- Member profile trial rerun:
  - Workflow ID `294874303`, run `27661985049`, success.
  - Trial metric rows: 174.
  - Matched legacy member codes: 172.
  - Legacy-only member codes: 2.
  - Trial-only member codes: 2.
- Compatibility comparison rerun:
  - Workflow ID `294874693`, run `27661990358`, success.
  - Roster: 176 legacy rows vs 174 compat rows; 174 matched keys, 2 legacy-only keys, 0 compat-only keys.
  - Member votes: 30,968 legacy rows vs 29,805 compat rows; 173 matched member-code keys, 0 legacy-only keys, 0 compat-only keys.
- Instagram consumer smoke rerun:
  - Workflow ID `297114820`, run `27661992188`, success.
  - Artifact: `oireachtas-instagram-consumer-smoke-output`, artifact ID `7684743075`.
  - Confirmed compat members key was used.
- Cutover package updated: `docs/oireachtas_final_cutover_request_package.md`.
- Cutover remains unapproved.

## Current caveats

- Roster has 2 legacy-only member codes after refresh.
- Member profile trial has 2 legacy-only and 2 trial-only member codes.
- Deterministic unified outputs still do not replace classified debate issues, photo URL indexes, member summaries, or constituency image indexes.
- Explicit user approval is still required before any downstream cutover.

## Next packet batch

### P13 — remaining mismatch review

Goal:

- identify the 2 roster/member-profile mismatches where possible;
- document whether they are expected member lifecycle differences, source freshness differences, or transformation issues;
- keep this review-only unless a bug is found.

### P14 — cutover approval checklist update

Goal:

- update production readiness checklist with post-refresh evidence;
- keep approval gate explicit and consumer-specific.

### P15 — optional approved cutover patch preparation

Goal:

- prepare the exact patch that would be applied after approval for Instagram or member-profile metrics;
- do not apply the patch unless explicit approval phrase is provided.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P13 remaining mismatch review, then P14 cutover approval checklist update, then P15 optional approved cutover patch preparation.
Do not repoint downstream consumers or disable old workflows without explicit user approval.
Latest validated runs: P11 refresh 27661934424, P12 adapters 27661982505, P12 member profile trial 27661985049, P12 comparison 27661990358, P12 Instagram smoke 27661992188.
Current recommendation: no cutover yet; review remaining 2-member mismatch and wait for explicit approval.
```
