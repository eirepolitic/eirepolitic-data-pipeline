# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-07-04  
**Current packet:** P60 — monthly scheduled refresh failure investigation

This is the compact operational handoff for `docs/oireachtas_unified_data_model_plan.md`. Continue from `main`.

## Shared implementation state

- CLI: `python -m extract.oireachtas.build_table`
- Registry: `configs/oireachtas/tables.yml`
- AWS region: `ca-central-1`
- S3 bucket: `eirepolitic-data`
- Review branch: `oireachtas-review-output`
- Runtime rule: `mode=test` suppresses writes to `processed/oireachtas_unified/latest/*` unless explicitly overridden.

## Completed packets

- **F01-F03** complete.
- **T01-T23** silver tables complete with DQ pass.
- **G01-G05** gold tables complete with DQ pass.
- **C01-C03** control tables complete with DQ pass.
- **P01-P09** initial publishing, dynamic windows, adapters, consumer smoke planning, and cutover package complete.
- **P10-P18** production-sized dry run and controlled pre-production cutovers complete.
- **P19-P25** monitoring, post-cutover validation, steady-state selection, scheduled review, adapter review complete.
- **P26-P35** enrichment planning, review publish hardening, classified issue design, weekly refresh failure patch complete.
- **P36-P44** classified issue consumer trial, member photo enrichment, constituency image design, classified issue full-run plan complete.
- **P45-P53** constituency image enrichment, member photo production patch, member summaries enrichment, and Instagram consumer trials complete.
- **P54-P56** full classified issue enrichment, full member-profile trial, and production classified issue cutover complete.
- **P57-P59** final validation sweep, scheduled refresh follow-up, and final readiness summary complete.

## Applied controlled pre-production cutovers

### Member profile metrics

Workflow:

```text
.github/workflows/build_member_profile_metrics_2025.yml
```

Current defaults:

```yaml
MEMBERS_INPUT_KEY: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
MEMBER_VOTES_INPUT_KEY: "processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv"
MEMBER_PHOTOS_INPUT_KEY: "processed/oireachtas_unified/compat/media/members_photo_urls_compat.csv"
DEBATE_ISSUES_INPUT_KEY: "processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv"
```

Latest production validation:

```text
Workflow ID: 266755732
Run ID: 28684033733
Result: success
```

### Instagram constituency renderer

Workflow:

```text
.github/workflows/instagram_constituency_test.yml
```

Current defaults:

```yaml
INSTAGRAM_MEMBERS_DATASET_KEYS: "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv"
INSTAGRAM_MEMBER_SUMMARIES_DATASET_KEYS: "processed/oireachtas_unified/compat/text/members_summaries_compat.csv"
INSTAGRAM_CONSTITUENCY_IMAGES_DATASET_KEYS: "processed/oireachtas_unified/compat/media/constituency_images_compat.csv"
```

Latest validation:

```text
Workflow ID: 261945698
Run ID: 28672901108
Result: success
Artifact ID: 8071309560
```

### Instagram campaign renderer

Workflow:

```text
.github/workflows/instagram_campaign_render.yml
```

Current default:

```text
spec_file=render_spec.yml
upload_preview=false
```

Validation:

```text
Workflow ID: 271160957
Run ID: 28415050102
Result: success
Artifact ID: 7969146127
```

## Enrichment compatibility status

### Classified issue labels

```text
Builder: extract/oireachtas/enrichment_speech_issue_labels.py
Workflow/run: 304470256 / 28683964925
Result: success; DQ pass; artifact ID 8074954501
Rows: source 47275, output 47275, compat 47275
Compat key: processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
```

Full member-profile trial:

```text
Workflow ID: 294874303
Run ID: 28683998319
Result: success
Artifact ID: 8074963772
```

### Member photo URLs

```text
Builder: extract/oireachtas/enrichment_member_photo_urls.py
Workflow/run: 304478490 / 28422342745
Result: success; DQ pass; artifact ID 7971687268
Rows: source 174, output 174, compat 174
Compat key: processed/oireachtas_unified/compat/media/members_photo_urls_compat.csv
```

### Constituency images

```text
Builder: extract/oireachtas/enrichment_constituency_images.py
Workflow/run: 305600627 / 28547829924
Result: success; DQ pass; artifact ID 8022576293
Rows: source 43, output 43, compat 43
Compat key: processed/oireachtas_unified/compat/media/constituency_images_compat.csv
```

### Member summaries

```text
Builder: extract/oireachtas/enrichment_member_summaries.py
Workflow/run: 306762190 / 28672859337
Result: success; DQ pass; artifact ID 8071290965
Rows: source 174, output 174, compat 174
Compat key: processed/oireachtas_unified/compat/text/members_summaries_compat.csv
```

## Final validation sweep

Compatibility comparison:

```text
Workflow ID: 294874693
Run ID: 28691936308
Result: success
Artifact ID: 8077413255
```

Mismatch review:

```text
Workflow ID: 297343766
Run ID: 28691938402
Result: success
Artifact ID: 8077412883
```

Known remaining roster caveat:

```text
Catherine Connolly — Independent — Galway West
Paschal Donohoe — Fine Gael — Dublin Central
```

These are legacy-only roster records. Current member-profile metrics have 174 matched members and no member-code mismatches.

## Scheduled refresh status

Weekly:

```text
Latest run: 28421557467
Event: workflow_dispatch
Result: success
```

Monthly:

```text
Latest run: 28504651002
Event: schedule
Result: failure
Failed step: Run monthly table set
Artifact ID: 8004493556
```

Yearly:

```text
Latest run: 27397123885
Event: workflow_dispatch
Result: success
```

## Current caveats

- Monthly scheduled refresh run `28504651002` failed and needs investigation.
- Next scheduled weekly run still needs observation after the debate-record DQ patch.
- Production Instagram publishing remains artifact-only unless a workflow explicitly enables upload/publish.
- Legacy enrichment keys remain preserved for rollback.

## Key docs

```text
docs/oireachtas_final_post_cutover_validation_sweep.md
docs/oireachtas_scheduled_refresh_followup_check.md
docs/oireachtas_final_handoff_readiness_summary.md
docs/oireachtas_classified_issue_cutover_decision.md
docs/oireachtas_member_photo_cutover_decision.md
```

## Next packet batch

### P60 — monthly scheduled refresh failure investigation

Goal:

- inspect run `28504651002` artifact/log evidence.
- identify the failed monthly table and root cause.

### P61 — monthly scheduled refresh fix/validation

Goal:

- patch the monthly refresh failure if needed.
- rerun monthly refresh with production-like inputs.

### P62 — scheduled refresh orchestration plan

Goal:

- design one workflow that runs refresh, adapters, comparison, mismatch review, and consumer validations in order.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P60 monthly scheduled refresh failure investigation, then P61 monthly scheduled refresh fix/validation, then P62 scheduled refresh orchestration plan.
Do not overwrite legacy enrichment keys.
Latest successful validations: compat comparison run 28691936308, mismatch review run 28691938402, production metrics run 28684033733.
Known blocker: monthly scheduled refresh run 28504651002 failed at Run monthly table set.
```
