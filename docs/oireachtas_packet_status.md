# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-07-05  
**Current packet:** P66 — orchestrator refresh-enabled validation

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
Run ID: 28727335843
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
Run ID: 28727349094
Result: success
Artifact ID: 8087668776
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
Run ID: 28727366673
Result: success
Artifact ID: 8087671962
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
no schedule yet
```

Default inputs:

```text
refresh_type=none
run_consumers=true
```

Validation:

```text
Run ID: 28727286429
Run number: 1
Result: success
Artifact ID: 8087672457
```

Child runs from validation:

```text
compat adapters: 294866317 / 28727289983 / success / artifact 8087650930
compat comparison: 294874693 / 28727303180 / success / artifact 8087654840
mismatch review: 297343766 / 28727318126 / success / artifact 8087658973
member profile metrics: 266755732 / 28727335843 / success
Instagram constituency render: 261945698 / 28727349094 / success / artifact 8087668776
Instagram campaign render: 271160957 / 28727366673 / success / artifact 8087671962
```

Decision:

```text
Manual orchestrator is validated. Scheduled trigger deferred until a refresh-enabled orchestrator run passes.
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
Run number: 5
Result: success
Artifact ID: 8087552815
```

Safe monthly manual defaults were restored after validation.

## Final validation sweep

Compatibility comparison:

```text
Workflow ID: 294874693
Latest orchestrated run ID: 28727303180
Result: success
Artifact ID: 8087654840
```

Mismatch review:

```text
Workflow ID: 297343766
Latest orchestrated run ID: 28727318126
Result: success
Artifact ID: 8087658973
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
Next scheduled weekly run still needs observation after the debate-record DQ patch.
```

Monthly:

```text
Latest validated run: 28726922946
Event: workflow_dispatch
Result: success
Latest scheduled run: 28504651002 failed before fix
Next scheduled monthly run should be observed after the bill-stage/sponsor/event DQ patches.
```

Yearly:

```text
Latest run: 27397123885
Event: workflow_dispatch
Result: success
```

## Current caveats

- Orchestrator has only been validated with `refresh_type=none`.
- Next orchestrator validation should use a refresh-enabled run, preferably `refresh_type=monthly`.
- Scheduled trigger is intentionally not added yet.
- Production Instagram publishing remains artifact-only unless a workflow explicitly enables upload/publish.
- Legacy enrichment keys remain preserved for rollback.

## Key docs

```text
docs/oireachtas_monthly_refresh_failure_investigation.md
docs/oireachtas_scheduled_refresh_orchestration_plan.md
docs/oireachtas_orchestrator_validation_summary.md
docs/oireachtas_orchestrator_scheduled_readiness_update.md
docs/oireachtas_final_post_cutover_validation_sweep.md
docs/oireachtas_final_handoff_readiness_summary.md
```

## Next packet batch

### P66 — orchestrator refresh-enabled validation

Goal:

- validate the orchestrator with a refresh step enabled.
- Preferred: `refresh_type=monthly`, because monthly was the latest scheduled failure mode.
- If the dispatch tool cannot pass inputs, temporarily set orchestrator default `refresh_type=monthly`, validate, then restore default to `none`.

### P67 — orchestrator schedule gate decision

Goal:

- decide whether to add a scheduled trigger after refresh-enabled orchestrator validation.
- If added, start with a weekly-safe schedule after the weekly refresh window.

### P68 — final scheduled-readiness handoff update

Goal:

- update all readiness/handoff docs with refresh-enabled orchestrator evidence and schedule decision.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P66 orchestrator refresh-enabled validation, then P67 orchestrator schedule gate decision, then P68 final scheduled-readiness handoff update.
Do not overwrite legacy enrichment keys.
Latest successful validations: orchestrator run 28727286429, monthly production-like validation run 28726922946, member profile metrics run 28727335843.
```
