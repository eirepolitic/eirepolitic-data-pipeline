# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-30  
**Current packet:** P39 — member photo enrichment trial builder

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

## Classified issue enrichment trial status

Trial builder:

```text
extract/oireachtas/enrichment_speech_issue_labels.py
```

Trial workflow:

```text
.github/workflows/oireachtas_enrichment_speech_issue_labels_trial.yml
```

Workflow ID/run:

```text
304470256 / 28421444809
```

Result:

```text
success; DQ pass; artifact ID 7971387010
```

Outputs:

```text
processed/oireachtas_unified/enrichment/speech_issue_labels/speech_issue_labels_2025_trial.csv
processed/oireachtas_unified/enrichment/speech_issue_labels/parquets/speech_issue_labels_2025_trial.parquet
processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
processed/oireachtas_unified/compat/debates/parquets/debate_speeches_classified_compat.parquet
```

No production classified-debate key was overwritten.

## P36 classified issue consumer trial

Workflow patched:

```text
.github/workflows/oireachtas_member_profile_trial.yml
```

Change:

- Added `debate_issues_input_key` workflow input.
- Default now uses:

```text
processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
```

Validation:

```text
Workflow ID: 294874303
Run ID: 28422192492
Run number: 4
Result: success
Artifact: oireachtas-member-profile-trial-output
Artifact ID: 7971637215
```

Review sample result:

```text
legacy_rows: 174
trial_rows: 174
matched_member_count: 174
trial_only_member_count: 0
legacy_only_member_count: 0
common_column_count: 12
DQ: pass
```

This validates that member-profile metrics can consume the classified issue compat file side-by-side.

## P37 media enrichment namespace follow-up

File added:

```text
docs/oireachtas_media_enrichment_namespace_plan.md
```

Decision:

- Keep photo URLs, member summaries, and constituency images out of deterministic silver/gold tables.
- Use separate enrichment namespaces:

```text
processed/oireachtas_unified/enrichment/media/member_photo_urls/
processed/oireachtas_unified/enrichment/text/member_summaries/
processed/oireachtas_unified/enrichment/media/constituency_images/
```

Recommended next implementation:

```text
enrichment_member_photo_urls
```

because it is lower risk than generated summaries.

## P38 scheduled refresh monitoring

File added:

```text
docs/oireachtas_scheduled_refresh_monitoring_status.md
```

Latest refresh state:

```text
Weekly: active; latest run 28421557467 success, manual safe validation
Monthly: active; latest run 27397121321 success, manual validation
Yearly: active; latest run 27397123885 success, manual validation
```

Next scheduled weekly run should still be monitored because manual validation used safe defaults while scheduled mode uses incremental mode.

## Weekly refresh failure investigation

Files added/changed:

```text
docs/oireachtas_weekly_refresh_failure_investigation.md
extract/oireachtas/table_debate_records.py
```

Failed scheduled weekly runs:

```text
27898282130
28314747505
```

Root cause:

- `silver_debate_records` required every row to have a PDF source link.
- Recent debate records had XML source links but no PDF links.

Patch:

- XML links remain required.
- PDF links are optional and tracked via `pdf_present_count` / `pdf_missing_count`.

Validation:

```text
Workflow ID: 294426406
Run ID: 28421557467
Result: success
Artifact ID: 7971444843
```

## Current caveats

- Roster comparison has 2 legacy-only member codes:
  - Catherine Connolly — Independent — Galway West
  - Paschal Donohoe — Fine Gael — Dublin Central
- Member profile metrics have 0 member-code mismatches after the cutover build.
- Deterministic unified outputs still do not replace photo URL indexes, member summaries, or constituency image indexes.
- The classified issue compat path has passed side-by-side consumer validation but is not yet repointed in production member-profile metrics.
- Weekly scheduled mode should still be monitored on the next schedule, even though safe/manual validation passed.

## Next packet batch

### P39 — member photo enrichment trial builder

Goal:

- build side-by-side `enrichment_member_photo_urls` output.
- Do not overwrite legacy photo URL keys.

### P40 — member photo enrichment workflow

Goal:

- add manual workflow for member photo enrichment trial.
- use a safe row limit by default.

### P41 — classified issue production cutover decision

Goal:

- decide whether to repoint production member-profile metrics `DEBATE_ISSUES_INPUT_KEY` to the classified issue compat output.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P39 member photo enrichment trial builder, then P40 member photo enrichment workflow, then P41 classified issue production cutover decision.
Do not overwrite legacy photo URL keys or processed/debates/debate_speeches_classified.csv.
Latest successful validations: classified issue consumer trial run 28422192492, enrichment trial run 28421444809, weekly validation run 28421557467.
```
