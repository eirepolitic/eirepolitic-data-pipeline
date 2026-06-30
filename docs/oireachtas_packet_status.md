# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-30  
**Current packet:** P32 — enrichment trial builder implementation

This is the compact operational handoff for `docs/oireachtas_unified_data_model_plan.md`. Continue from `main`.

## Shared implementation state

- CLI: `python -m extract.oireachtas.build_table`
- Registry: `configs/oireachtas/tables.yml`
- AWS region: `ca-central-1`
- S3 bucket: `eirepolitic-data`
- Review branch: `oireachtas-review-output`
- Runtime rule: `mode=test` suppresses writes to `processed/oireachtas_unified/latest/*` unless explicitly overridden.

## Confirmed core packets

- **F01-F03** complete.
- **T01-T23** silver tables complete with DQ pass.
- **G01-G05** gold tables complete with DQ pass.
- **C01-C03** control tables complete with DQ pass.
- `configs/oireachtas/tables.yml` marks all validated silver/gold/control tables as `confirmed`.

## Confirmed hardening and cutover packets

- **P01-P09** complete through initial cutover package.
- **P10-P12** complete through production-sized refresh and post-refresh validation.
- **P13-P15** complete through mismatch review, readiness checklist, and patch plan.
- **P16** approval gate documented; user later clarified these are pre-production systems and explicit approval phrase is not required.
- **P17** Instagram constituency renderer cutover applied and validated.
- **P18** member-profile metrics cutover applied and validated.
- **P19** post-cutover monitoring plan added.
- **P20-P22** post-cutover evidence verified, comparison/mismatch rerun, docs updated.
- **P23-P25** steady-state consumer selection, scheduled refresh review, and adapter review complete.
- **P26-P28** next workstream selected, enrichment plan documented, review publish hardening started.

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

## Latest post-cutover validation

- Compatibility comparison run `28416432150`: success.
- Mismatch review run `28416434690`: success.
- Latest mismatch summary:
  - Roster: 176 legacy members, 174 unified members, 174 matched, 2 legacy-only, 0 unified-only.
  - Member profile metrics: 174 legacy members, 174 unified members, 174 matched, 0 legacy-only, 0 unified-only.

## Confirmed P29-P31 packets

### P29 — enrichment dependency audit

- File added: `docs/oireachtas_enrichment_dependency_audit.md`
- Audited workflows/scripts:
  - `.github/workflows/speech_issue_classifier.yml` / `process/speech_issue_classifier.py`
  - `.github/workflows/member_photo_urls.yml` / `process/members_photo_urls.py`
  - `.github/workflows/members_background_summarizer.yml` / `process/members_background_summarizer.py`
  - `.github/workflows/constituency_images_index.yml` / `process/constituency_images_indexer.py`
- Conclusion: keep enrichment/media outputs separate from deterministic Oireachtas silver/gold tables.

### P30 — classified issue replacement design

- File added: `docs/oireachtas_classified_issue_replacement_design.md`
- Designed side-by-side output:

```text
processed/oireachtas_unified/enrichment/speech_issue_labels/speech_issue_labels_2025_trial.csv
processed/oireachtas_unified/enrichment/speech_issue_labels/parquets/speech_issue_labels_2025_trial.parquet
```

- Designed compat output:

```text
processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
```

- No production classified debate key was changed.

### P31 — broader review-publisher hardening

- Files patched:
  - `.github/workflows/oireachtas_weekly_refresh.yml`
  - `.github/workflows/oireachtas_monthly_refresh.yml`
  - `.github/workflows/oireachtas_yearly_refresh.yml`
  - `docs/oireachtas_review_publish_hardening.md`
- Patch: review branch publish now uses pull/rebase plus 3 push attempts.
- Validated earlier on:
  - `.github/workflows/oireachtas_compat_comparison.yml`, run `28416432150`, success.
  - `.github/workflows/oireachtas_mismatch_review.yml`, run `28416434690`, success.
- Weekly/monthly/yearly were not manually run after patch because they are broad refresh workflows.

## Current caveats

- Roster comparison has 2 legacy-only member codes:
  - Catherine Connolly — Independent — Galway West
  - Paschal Donohoe — Fine Gael — Dublin Central
- Member profile metrics have 0 member-code mismatches after the cutover build.
- Deterministic unified outputs still do not replace classified debate issues, photo URL indexes, member summaries, or constituency image indexes.
- Weekly/monthly/yearly publish hardening is patched but not separately runtime-validated after patch.

## Next packet batch

### P32 — enrichment trial builder implementation

Goal:

- build `extract/oireachtas/enrichment_speech_issue_labels.py` as a side-by-side trial builder.
- Do not overwrite `processed/debates/debate_speeches_classified.csv`.

### P33 — enrichment trial workflow

Goal:

- add `.github/workflows/oireachtas_enrichment_speech_issue_labels_trial.yml`.
- Run in limited/test mode by default.

### P34 — classified issue compat adapter and comparison plan

Goal:

- add or plan `processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv`.
- compare coverage against legacy classified debate output before any consumer repointing.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P32 enrichment trial builder implementation, then P33 enrichment trial workflow, then P34 classified issue compat adapter and comparison plan.
Do not overwrite processed/debates/debate_speeches_classified.csv.
Latest successful validations: campaign render run 28415050102, compatibility comparison run 28416432150, mismatch review run 28416434690.
```
