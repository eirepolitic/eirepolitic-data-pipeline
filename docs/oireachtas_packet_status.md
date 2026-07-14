# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-07-13 America/Vancouver  
**Current packet:** P72 — scheduled orchestrator trigger decision

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
- **P01-P59** complete through final post-cutover validation, scheduled follow-up, and handoff summary.
- **P60-P62** monthly scheduled refresh failure investigated/fixed/validated, and orchestration plan added.
- **P63-P65** refresh validation orchestrator implemented, validated with `refresh_type=none`, and scheduled-readiness docs updated.
- **P66-P68** refresh-enabled orchestrator validation complete, schedule gate deferred, final scheduled-readiness handoff added.
- **P69-P71** production-input orchestrator designed, implemented, validated, and restored to safe default.

## Current controlled pre-production cutovers

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

Latest validation:

```text
Workflow ID: 266755732
Run ID: 29299647855
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
Run ID: 29299676372
Result: success
Artifact ID: 8298085395
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

Latest validation:

```text
Workflow ID: 271160957
Run ID: 29299727612
Result: success
Artifact ID: 8298101606
```

## Refresh validation orchestrator

Workflow:

```text
.github/workflows/oireachtas_refresh_validation_orchestrator.yml
```

Workflow ID:

```text
307332237
```

Current trigger state:

```text
workflow_dispatch only
no schedule trigger
```

Manual defaults restored after validation:

```text
refresh_type=none
run_consumers=true
```

Production-input implementation commit:

```text
e14711cb97d4c361ee3606fd0821539af0bff8c7
```

Safe-default restore commit:

```text
5126e5a2fb31f23ea6f12e446bdf2b185e23fa71
```

### Current-latest validation

```text
Run ID: 28727286429
Result: success
Artifact ID: 8087672457
```

### Refresh-enabled validation using child defaults

```text
Run ID: 29218772777
Result: success
Artifact ID: 8267558251
```

### Production-input refresh validation

```text
Run ID: 29299431600
Result: success
Artifact ID: 8298102893
```

Child runs from production-input validation:

```text
monthly refresh: 294432002 / 29299437311 / success / artifact 8298023987
compat adapters: 294866317 / 29299539592 / success / artifact 8298034081
compat comparison: 294874693 / 29299580373 / success / artifact 8298049611
mismatch review: 297343766 / 29299619179 / success / artifact 8298060582
member profile metrics: 266755732 / 29299647855 / success
Instagram constituency render: 261945698 / 29299676372 / success / artifact 8298085395
Instagram campaign render: 271160957 / 29299727612 / success / artifact 8298101606
```

Validated monthly child inputs:

```text
mode=incremental
publish_latest=auto
date_start=2026-05-25
date_end=2026-06-30
limit=250
sample_rows=10
```

## Monthly refresh investigation and fix

Original failed scheduled run:

```text
Workflow ID: 294432002
Run ID: 28504651002
Event: schedule
Result: failure
Failed step: Run monthly table set
Artifact ID: 8004493556
```

Patched files:

```text
extract/oireachtas/table_bill_stages.py
extract/oireachtas/table_bill_sponsors.py
extract/oireachtas/table_bill_events.py
```

Production-like validation:

```text
Workflow ID: 294432002
Run ID: 28726922946
Result: success
Artifact ID: 8087552815
```

Latest explicit production-input orchestrated monthly validation:

```text
Workflow ID: 294432002
Run ID: 29299437311
Result: success
Artifact ID: 8298023987
```

## Scheduled refresh status

Weekly:

```text
2026-07-05 scheduled weekly: 28732507909 / success
2026-07-12 scheduled weekly: 29182334702 / success
```

Monthly:

```text
Latest scheduled monthly: 28504651002 / failure / before fix
Latest explicit-input monthly validation: 29299437311 / success
Next real scheduled monthly run still needs observation.
```

Yearly:

```text
Latest yearly validation: 27397123885 / workflow_dispatch / success
No new scheduled yearly run yet.
```

## Known caveats

```text
Scheduled orchestrator trigger is not yet enabled.
Next real scheduled monthly refresh still needs observation.
Production Instagram publishing remains artifact-only unless explicitly enabled.
Legacy enrichment keys remain preserved for rollback.
Known legacy-only roster caveats remain Catherine Connolly and Paschal Donohoe.
```

## Enrichment compatibility status

```text
Classified issue labels: 304470256 / 28683964925 / success / 47275 rows
Member photo URLs: 304478490 / 28422342745 / success / 174 rows
Constituency images: 305600627 / 28547829924 / success / 43 rows
Member summaries: 306762190 / 28672859337 / success / 174 rows
```

## Key docs

```text
docs/oireachtas_production_input_orchestrator_design.md
docs/oireachtas_production_input_orchestrator_validation.md
docs/oireachtas_orchestrator_refresh_enabled_validation.md
docs/oireachtas_orchestrator_schedule_gate_decision.md
docs/oireachtas_final_scheduled_readiness_handoff.md
docs/oireachtas_monthly_refresh_failure_investigation.md
```

## Next packet batch

### P72 — scheduled orchestrator trigger decision

Goal:

- decide whether to add a scheduled trigger now that production-input orchestration has passed.
- safest option: weekly-only orchestrator schedule first, after existing weekly refresh window.

### P73 — scheduled orchestrator trigger implementation or deferral documentation

Goal:

- if enabling, patch `.github/workflows/oireachtas_refresh_validation_orchestrator.yml` with a schedule and event-aware defaults.
- if deferring, document why no schedule was added.

### P74 — final production-readiness handoff update

Goal:

- update final handoff docs with the schedule decision, production-input validation evidence, and remaining observation items.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P72 scheduled orchestrator trigger decision, then P73 schedule implementation or deferral, then P74 final production-readiness handoff update.
Do not overwrite legacy enrichment keys.
Latest successful validations: production-input orchestrator run 29299431600, monthly child run 29299437311, member profile metrics run 29299647855.
```
