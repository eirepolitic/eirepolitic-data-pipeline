# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-30  
**Current packet:** P29 — enrichment dependency audit

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
- **P23-P25** steady-state consumer selection, scheduled refresh review, and adapter review complete.

## Applied pre-production cutovers

### Instagram constituency renderer

- File: `.github/workflows/instagram_constituency_test.yml`
- Default roster input:

```yaml
      INSTAGRAM_MEMBERS_DATASET_KEYS: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
```

- Validation: workflow ID `261945698`, run `28414647932`, success, artifact ID `7968986389`.

### Member profile metrics

- File: `.github/workflows/build_member_profile_metrics_2025.yml`
- Default inputs:

```yaml
      MEMBERS_INPUT_KEY: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
      MEMBER_VOTES_INPUT_KEY: "processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv"
```

- Legacy vote-record rebuild step removed from this workflow because the metrics build now reads compat vote records directly.
- Validation: failed pre-correction run `28414649704`; corrected run `28414678714`, success.

### Instagram campaign renderer

- File: `.github/workflows/instagram_campaign_render.yml`
- Default `spec_file` now uses `render_spec.yml` instead of the synthetic fixture.
- Workflow remains artifact-only by default: `upload_preview=false`.
- Validation: workflow ID `271160957`, run `28415050102`, success, artifact ID `7969146127`.

## Post-cutover validation

- Compatibility comparison run `28414819264`: success.
- Mismatch clean rerun `28414847238`: success, DQ pass.
- Latest mismatch summary:
  - Roster: 176 legacy members, 174 unified members, 174 matched, 2 legacy-only, 0 unified-only.
  - Member profile metrics: 174 legacy members, 174 unified members, 174 matched, 0 legacy-only, 0 unified-only.

## Confirmed P26-P28 packets

### P26 — next workstream selection

- File added: `docs/oireachtas_next_workstream_selection.md`
- Decision: next workstream is enrichment replacement planning and refresh hardening.
- Rationale: deterministic Oireachtas tables and first consumer cutovers are validated; remaining gaps are enrichment/media/classification and workflow hardening.

### P27 — enrichment replacement planning

- File added: `docs/oireachtas_enrichment_replacement_plan.md`
- Remaining enrichment dependencies documented:
  - classified debate issues
  - member photo URLs
  - member summaries/backgrounds
  - constituency image indexes
- Recommended first enrichment packet: `E01 — classified issue dependency audit`.

### P28 — review publish hardening patch review

- Files changed:
  - `.github/workflows/oireachtas_compat_comparison.yml`
  - `.github/workflows/oireachtas_mismatch_review.yml`
  - `docs/oireachtas_review_publish_hardening.md`
- Patch: review branch publish now uses pull/rebase plus 3 push attempts.
- Validation:
  - Compatibility comparison workflow ID `294874693`, run `28416432150`, success.
  - Mismatch review workflow ID `297343766`, run `28416434690`, success.

## Current caveats

- Roster comparison has 2 legacy-only member codes:
  - Catherine Connolly — Independent — Galway West
  - Paschal Donohoe — Fine Gael — Dublin Central
- Member profile metrics now have 0 member-code mismatches after the cutover build.
- Deterministic unified outputs still do not replace classified debate issues, photo URL indexes, member summaries, or constituency image indexes.
- Review branch publish conflicts are reduced for compatibility comparison and mismatch review, but weekly/monthly/yearly publishers still use the older direct push pattern.

## Next packet batch

### P29 — enrichment dependency audit

Goal:

- inspect classified issue, photo URL, member summary, and constituency image workflows/scripts;
- document exact input/output keys and consumers.

### P30 — classified issue replacement design

Goal:

- design `enrichment_speech_issue_labels` or equivalent side-by-side output without overwriting existing classified debate keys.

### P31 — broader review-publisher hardening

Goal:

- patch weekly/monthly/yearly and other review-output publishers with the same retry/rebase pattern if needed.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P29 enrichment dependency audit, then P30 classified issue replacement design, then P31 broader review-publisher hardening.
Latest successful validations: campaign render run 28415050102, compatibility comparison run 28416432150, mismatch review run 28416434690.
```
