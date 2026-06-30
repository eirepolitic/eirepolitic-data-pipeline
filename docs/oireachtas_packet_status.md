# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-30  
**Current packet:** P36 — classified issue consumer trial

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

## Latest post-cutover validation

- Compatibility comparison run `28416432150`: success.
- Mismatch review run `28416434690`: success.
- Latest mismatch summary:
  - Roster: 176 legacy members, 174 unified members, 174 matched, 2 legacy-only, 0 unified-only.
  - Member profile metrics: 174 legacy members, 174 unified members, 174 matched, 0 legacy-only, 0 unified-only.

## P32-P34 enrichment trial status

### P32 — enrichment trial builder implementation

- File added: `extract/oireachtas/enrichment_speech_issue_labels.py`
- Behavior:
  - reads `processed/debates/debate_speeches_classified.csv`;
  - writes side-by-side unified enrichment output;
  - writes legacy-compatible classified debate output;
  - does not call OpenAI;
  - does not overwrite the production classified-debate key.

Trial outputs:

```text
processed/oireachtas_unified/enrichment/speech_issue_labels/speech_issue_labels_2025_trial.csv
processed/oireachtas_unified/enrichment/speech_issue_labels/parquets/speech_issue_labels_2025_trial.parquet
```

Compat outputs:

```text
processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
processed/oireachtas_unified/compat/debates/parquets/debate_speeches_classified_compat.parquet
```

### P33 — enrichment trial workflow

- File added: `.github/workflows/oireachtas_enrichment_speech_issue_labels_trial.yml`
- Workflow ID: `304470256`
- Validation run: `28421444809`
- Result: success
- Artifact: `oireachtas-speech-issue-labels-enrichment-trial-output`
- Artifact ID: `7971387010`
- Review manifest showed:
  - source rows: 47,275
  - row limit: 50
  - output rows: 50
  - compat rows: 50
  - DQ: pass

### P34 — classified issue compat adapter and comparison plan

- File added: `docs/oireachtas_classified_issue_compat_comparison_plan.md`
- First comparison is built into trial DQ.
- Production member-profile metrics has not been repointed to the classified issue compat file yet.

## P35 weekly refresh failure investigation

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

- The weekly run failed at `silver_debate_records`.
- Member tables before it completed successfully.
- `silver_debate_records` failed DQ because recent debate records had XML source links but no PDF source links.
- Previous DQ required every row to have a PDF URI.

Patch:

- XML links remain required.
- PDF links are optional.
- PDF coverage is now recorded through:

```text
pdf_present_count
pdf_missing_count
source_pdf_uri_optional
source_file_id_pdf_consistent_when_present
```

Validation:

```text
Workflow ID: 294426406
Run ID: 28421557467
Run number: 5
Event: workflow_dispatch
Result: success
Artifact: oireachtas-weekly-refresh-output
Artifact ID: 7971444843
```

The immediate weekly blocker is resolved in safe/manual validation mode. Monitor the next scheduled weekly incremental run to confirm scheduled-mode success.

## Current caveats

- Roster comparison has 2 legacy-only member codes:
  - Catherine Connolly — Independent — Galway West
  - Paschal Donohoe — Fine Gael — Dublin Central
- Member profile metrics have 0 member-code mismatches after the cutover build.
- Deterministic unified outputs still do not replace photo URL indexes, member summaries, or constituency image indexes.
- The classified issue enrichment trial exists, but production consumers are not repointed to it yet.
- Weekly scheduled mode should still be monitored on the next schedule, even though the safe/manual validation passed.

## Next packet batch

### P36 — classified issue consumer trial

Goal:

- run member-profile metrics trial using `processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv` as `DEBATE_ISSUES_INPUT_KEY`.

### P37 — enrichment media/index audit follow-up

Goal:

- decide whether photo URL, member summary, and constituency image indexes need unified enrichment namespaces.

### P38 — next scheduled refresh monitoring

Goal:

- monitor the next scheduled weekly run after the `silver_debate_records` DQ patch.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P36 classified issue consumer trial, then P37 enrichment media/index audit follow-up, then P38 next scheduled refresh monitoring.
Do not overwrite processed/debates/debate_speeches_classified.csv.
Latest successful validations: enrichment trial run 28421444809, weekly validation run 28421557467, compatibility comparison run 28416432150, mismatch review run 28416434690.
```
