# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-07-13 America/Vancouver  
**Current packet:** P75 — observe first scheduled orchestrator run

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
- **P72-P74** scheduled orchestrator trigger enabled, schedule decision documented, final production-readiness handoff added.

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

Schedule implementation commit:

```text
17e140f238b06fb1e667d42dedf22dc64ac914c7
```

Current trigger state:

```text
workflow_dispatch
schedule
```

Current schedule:

```yaml
schedule:
  - cron: "45 6 * * 0"
```

Scheduled behavior:

```text
refresh_type=weekly
run_consumers=true
```

Manual defaults:

```text
refresh_type=none
run_consumers=true
```

First expected scheduled orchestrator run:

```text
2026-07-19 06:45 UTC
2026-07-18 23:45 America/Vancouver
```

### Current-output validation

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

Weekly standalone refresh:

```text
2026-07-05 scheduled weekly: 28732507909 / success
2026-07-12 scheduled weekly: 29182334702 / success
```

Weekly orchestrator:

```text
Schedule enabled: 45 6 * * 0
First scheduled run pending observation.
```

Monthly:

```text
Latest scheduled monthly: 28504651002 / failure / before fix
Latest explicit-input monthly validation: 29299437311 / success
Next real scheduled monthly run still needs observation.
Expected next monthly scheduled run: 2026-08-01 04:35 UTC / 2026-07-31 21:35 America/Vancouver
```

Yearly:

```text
Latest yearly validation: 27397123885 / workflow_dispatch / success
No new scheduled yearly run yet.
```

## Known caveats

```text
First scheduled orchestrator run needs observation.
Next real scheduled monthly refresh still needs observation.
Standalone weekly refresh and scheduled orchestrator are both enabled, so the first orchestrator schedule is additive.
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
docs/oireachtas_scheduled_orchestrator_trigger_decision.md
docs/oireachtas_final_production_readiness_handoff.md
docs/oireachtas_production_input_orchestrator_design.md
docs/oireachtas_production_input_orchestrator_validation.md
docs/oireachtas_monthly_refresh_failure_investigation.md
```

## Next packet batch

### P75 — observe first scheduled orchestrator run

Goal:

- after the first scheduled run occurs, confirm orchestrator result and all child results.
- expected first scheduled run: 2026-07-19 06:45 UTC / 2026-07-18 23:45 America/Vancouver.

### P76 — observe next scheduled monthly refresh after fixes

Goal:

- confirm the next real scheduled monthly run after DQ fixes.
- expected next monthly run: 2026-08-01 04:35 UTC / 2026-07-31 21:35 America/Vancouver.

### P77 — steady-state scheduling decision and final handoff

Goal:

- decide whether to keep both standalone weekly refresh and scheduled orchestrator, disable standalone weekly, or change orchestrator scheduled refresh behavior.
- update final handoff once observations are complete.

Handoff instruction:

```text
Continue from main.
Process packets three at a time.
Start P75 first scheduled orchestrator observation, then P76 monthly scheduled observation if available, then P77 steady-state scheduling decision.
Do not overwrite legacy enrichment keys.
Latest successful validations: production-input orchestrator run 29299431600, monthly child run 29299437311, member profile metrics run 29299647855.
```
