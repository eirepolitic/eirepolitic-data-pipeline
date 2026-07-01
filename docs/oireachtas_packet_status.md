# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-07-01  
**Current packet:** P48 — member photo production workflow patch

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

## Confirmed hardening, cutover, and enrichment packets

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
- **P29-P31** enrichment dependencies audited, classified issue design added, broader review-publisher hardening patched.
- **P32-P34** classified issue enrichment trial builder/workflow/compat plan complete.
- **P35** weekly refresh failure investigated, patched, and safe validation passed.
- **P36-P38** classified issue consumer trial passed, media enrichment namespace plan added, scheduled refresh monitoring status documented.
- **P39-P41** member photo enrichment trial passed and classified issue production cutover was deferred.
- **P42-P44** member photo consumer trial passed, constituency image enrichment design added, and full classified issue run plan added.
- **P45-P47** constituency image enrichment trial passed and member photo production cutover decision documented.

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
- Validation: corrected run `28414678714`, success.

### Instagram campaign renderer

- File: `.github/workflows/instagram_campaign_render.yml`
- Default `spec_file` now uses `render_spec.yml` instead of the synthetic fixture.
- Workflow remains artifact-only by default: `upload_preview=false`.
- Validation: workflow ID `271160957`, run `28415050102`, success, artifact ID `7969146127`.

## Classified issue enrichment status

Trial builder:

```text
extract/oireachtas/enrichment_speech_issue_labels.py
```

Trial workflow/run:

```text
304470256 / 28421444809
```

Result:

```text
success; DQ pass; artifact ID 7971387010
```

Consumer trial:

```text
Workflow ID: 294874303
Run ID: 28422192492
Result: success
Artifact ID: 7971637215
```

Production cutover decision:

```text
Deferred because the current classified issue compat file was built from 50 trial rows, not full 47,275-row source coverage.
```

Full run plan:

```text
docs/oireachtas_full_classified_issue_enrichment_run_plan.md
```

## Member photo enrichment status

Builder:

```text
extract/oireachtas/enrichment_member_photo_urls.py
```

Workflow/run:

```text
304478490 / 28422342745
```

Result:

```text
success; DQ pass; artifact ID 7971687268
```

Review manifest:

```text
source_key: processed/members/member_photos/members_photo_urls.csv
source_rows: 174
output_rows: 174
compat_rows: 174
photo_url_populated_count: 174
photo_url_missing_count: 0
```

Outputs:

```text
processed/oireachtas_unified/enrichment/media/member_photo_urls/member_photo_urls_trial.csv
processed/oireachtas_unified/enrichment/media/member_photo_urls/parquets/member_photo_urls_trial.parquet
processed/oireachtas_unified/compat/media/members_photo_urls_compat.csv
processed/oireachtas_unified/compat/media/parquets/members_photo_urls_compat.parquet
```

Consumer trial:

```text
Workflow ID: 294874303
Run ID: 28461617338
Run number: 5
Result: success
Artifact ID: 7987802357
```

Review sample:

```text
legacy_rows: 174
trial_rows: 174
matched_member_count: 174
trial_only_member_count: 0
legacy_only_member_count: 0
common_column_count: 12
DQ: pass
```

Cutover decision:

```text
docs/oireachtas_member_photo_cutover_decision.md
```

Decision: approved for controlled pre-production workflow patch, not yet applied.

## Constituency image enrichment status

Builder:

```text
extract/oireachtas/enrichment_constituency_images.py
```

Workflow/run:

```text
305600627 / 28547829924
```

Result:

```text
success; DQ pass; artifact ID 8022576293
```

Review manifest:

```text
source_key: processed/constituencies/constituency_images.csv
source_rows: 43
output_rows: 43
compat_rows: 43
image_locator_populated_count: 43
image_locator_missing_count: 0
```

Outputs:

```text
processed/oireachtas_unified/enrichment/media/constituency_images/constituency_images_trial.csv
processed/oireachtas_unified/enrichment/media/constituency_images/parquets/constituency_images_trial.parquet
processed/oireachtas_unified/compat/media/constituency_images_compat.csv
processed/oireachtas_unified/compat/media/parquets/constituency_images_compat.parquet
```

No legacy constituency image key was overwritten.

## Scheduled refresh status

Latest refresh state:

```text
Weekly: active; latest run 28421557467 success, manual safe validation
Monthly: active; latest run 27397121321 success, manual validation
Yearly: active; latest run 27397123885 success, manual validation
```

Weekly scheduled mode should still be monitored because manual validation used safe defaults while scheduled mode uses incremental mode.

## Current caveats

- Roster comparison has 2 legacy-only member codes:
  - Catherine Connolly — Independent — Galway West
  - Paschal Donohoe — Fine Gael — Dublin Central
- Member profile metrics have 0 member-code mismatches after the cutover build.
- Deterministic unified outputs still do not replace member summaries.
- Classified issue compat path is structurally valid but not production-ready until full-row build/comparison.
- Member photo compat path is structurally valid and consumer-validated; production workflow patch is approved but not yet applied.
- Constituency image compat path is structurally valid, but Instagram consumers have not yet been run against it.
- Weekly scheduled mode should still be monitored on the next schedule, even though safe/manual validation passed.

## Next packet batch

### P48 — member photo production workflow patch

Goal:

- add `MEMBER_PHOTOS_INPUT_KEY` to `.github/workflows/build_member_profile_metrics_2025.yml`.
- run production member-profile metrics manual validation.

### P49 — constituency image consumer trial

Goal:

- run Instagram constituency/campaign trial with `processed/oireachtas_unified/compat/media/constituency_images_compat.csv`.

### P50 — member summaries enrichment design

Goal:

- design side-by-side member summaries enrichment namespace and compatibility output.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P48 member photo production workflow patch, then P49 constituency image consumer trial, then P50 member summaries enrichment design.
Do not overwrite legacy constituency image keys, legacy photo URL keys, or processed/debates/debate_speeches_classified.csv.
Latest successful validations: constituency image enrichment run 28547829924, member photo consumer trial run 28461617338, member photo enrichment run 28422342745.
```
