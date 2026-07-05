# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-07-05  
**Current packet:** P63 — refresh validation orchestrator implementation

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

Root causes found:

```text
silver_bill_stages: strict house_uri/house_name DQ failed on optional house metadata
silver_bill_sponsors: strict sponsor_name/sponsor_uri DQ failed on role-only sponsors
silver_bill_events: strict chamber DQ failed on optional chamber metadata
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
Run number: 5
Result: success
Artifact ID: 8087552815
```

Validation window:

```text
mode: incremental
publish_latest: auto
date_start: 2026-05-25
date_end: 2026-06-30
limit: 250
```

Safe monthly manual defaults were restored after validation.

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

## Enrichment compatibility status

```text
Classified issue labels: 304470256 / 28683964925 / success / 47275 rows
Member photo URLs: 304478490 / 28422342745 / success / 174 rows
Constituency images: 305600627 / 28547829924 / success / 43 rows
Member summaries: 306762190 / 28672859337 / success / 174 rows
```

## Scheduled refresh status

Weekly:

```text
Latest run: 28421557467
Event: workflow_dispatch
Result: success
```

Monthly:

```text
Latest validated run: 28726922946
Event: workflow_dispatch
Result: success
Latest scheduled run: 28504651002 failed before fix
```

Yearly:

```text
Latest run: 27397123885
Event: workflow_dispatch
Result: success
```

## Current caveats

- Next scheduled weekly run still needs observation after the debate-record DQ patch.
- Next scheduled monthly run should be observed after the bill-stage/sponsor/event DQ patches.
- Production Instagram publishing remains artifact-only unless a workflow explicitly enables upload/publish.
- Legacy enrichment keys remain preserved for rollback.

## Key docs

```text
docs/oireachtas_monthly_refresh_failure_investigation.md
docs/oireachtas_scheduled_refresh_orchestration_plan.md
docs/oireachtas_final_post_cutover_validation_sweep.md
docs/oireachtas_scheduled_refresh_followup_check.md
docs/oireachtas_final_handoff_readiness_summary.md
docs/oireachtas_classified_issue_cutover_decision.md
docs/oireachtas_member_photo_cutover_decision.md
```

## Next packet batch

### P63 — refresh validation orchestrator implementation

Goal:

- add `.github/workflows/oireachtas_refresh_validation_orchestrator.yml`.
- start with manual-only workflow.
- dispatch child workflows with `gh workflow run` and poll status.

### P64 — orchestrator validation with refresh_type=none

Goal:

- validate current latest outputs without running a refresh.
- run adapters, comparison, mismatch review, member profile validation, and Instagram validation.

### P65 — orchestrator scheduled-readiness update

Goal:

- update readiness docs based on orchestrator validation.
- decide whether to add a scheduled trigger after stable manual validation.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P63 refresh validation orchestrator implementation, then P64 orchestrator validation with refresh_type=none, then P65 orchestrator scheduled-readiness update.
Do not overwrite legacy enrichment keys.
Latest successful validations: monthly production-like validation run 28726922946, compat comparison run 28691936308, mismatch review run 28691938402.
```
