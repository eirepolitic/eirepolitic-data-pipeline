# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-07-12 America/Vancouver  
**Current packet:** P69 — production-input orchestrator design

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
Run ID: 29218925171
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
Run ID: 29218960864
Result: success
Artifact ID: 8267546600
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
Run ID: 29219005969
Result: success
Artifact ID: 8267555282
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

### Current-latest validation

```text
Run ID: 28727286429
Result: success
Artifact ID: 8087672457
```

### Refresh-enabled validation

```text
Run ID: 29218772777
Result: success
Artifact ID: 8267558251
```

Child runs from refresh-enabled validation:

```text
monthly refresh: 294432002 / 29218778131 / success / artifact 8267502791
compat adapters: 294866317 / 29218841290 / success / artifact 8267509616
compat comparison: 294874693 / 29218866832 / success / artifact 8267516514
mismatch review: 297343766 / 29218901115 / success / artifact 8267524757
member profile metrics: 266755732 / 29218925171 / success
Instagram constituency render: 261945698 / 29218960864 / success / artifact 8267546600
Instagram campaign render: 271160957 / 29219005969 / success / artifact 8267555282
```

Schedule gate decision:

```text
Do not add a scheduled orchestrator trigger yet.
Reason: refresh-enabled validation used child workflow_dispatch defaults; production-like child refresh inputs still need explicit implementation.
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

Latest orchestrated monthly validation:

```text
Workflow ID: 294432002
Run ID: 29218778131
Result: success
Artifact ID: 8267502791
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
Latest manual monthly validation: 29218778131 / success
Next real scheduled monthly run still needs observation.
```

Yearly:

```text
Latest yearly validation: 27397123885 / workflow_dispatch / success
No new scheduled yearly run yet.
```

## Known caveats

```text
Scheduled orchestrator trigger is intentionally not enabled.
Production-like child workflow inputs are not yet implemented in the orchestrator.
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
docs/oireachtas_orchestrator_refresh_enabled_validation.md
docs/oireachtas_orchestrator_schedule_gate_decision.md
docs/oireachtas_final_scheduled_readiness_handoff.md
docs/oireachtas_orchestrator_validation_summary.md
docs/oireachtas_monthly_refresh_failure_investigation.md
docs/oireachtas_final_handoff_readiness_summary.md
```

## Next packet batch

### P69 — production-input orchestrator design

Goal:

- decide whether to use `workflow_call` or explicit `gh workflow run -f` inputs.
- specify production-like input sets for weekly and monthly child refresh workflows.

### P70 — production-input orchestrator implementation

Goal:

- patch the orchestrator so refresh-enabled scheduled/manual runs pass explicit child inputs.
- keep manual default safe unless intentionally changed.

### P71 — production-input orchestrator validation

Goal:

- validate the orchestrator with production-like refresh inputs.
- after that, reconsider the scheduled trigger.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P69 production-input orchestrator design, then P70 production-input orchestrator implementation, then P71 production-input orchestrator validation.
Do not overwrite legacy enrichment keys.
Latest successful validations: refresh-enabled orchestrator run 29218772777, monthly child run 29218778131, member profile metrics run 29218925171.
```
