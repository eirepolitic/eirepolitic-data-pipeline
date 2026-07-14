# Oireachtas production-input orchestrator validation

**Status:** complete  
**Last updated:** 2026-07-13 America/Vancouver

## Purpose

Validate that the refresh validation orchestrator can dispatch child refresh workflows with explicit production-like inputs instead of relying on child workflow defaults.

## Implementation validated

Workflow:

```text
.github/workflows/oireachtas_refresh_validation_orchestrator.yml
```

Implementation commit:

```text
e14711cb97d4c361ee3606fd0821539af0bff8c7
```

The orchestrator now has explicit refresh dispatch functions for:

```text
weekly
monthly
yearly
```

Each function calls the child refresh workflow with explicit `gh workflow run -f` inputs.

## Temporary validation setup

The repository dispatch integration cannot pass workflow inputs. For validation only, the orchestrator default was temporarily changed to:

```text
refresh_type=monthly
```

After validation, the default was restored to:

```text
refresh_type=none
```

Restore commit:

```text
5126e5a2fb31f23ea6f12e446bdf2b185e23fa71
```

## Production-like monthly inputs validated

The monthly child workflow was dispatched with explicit inputs equivalent to the scheduled monthly branch:

```text
mode=incremental
publish_latest=auto
tables=silver_constituencies,silver_parties,silver_source_files,silver_bills,silver_bill_versions,silver_bill_stages,silver_bill_related_docs,silver_bill_sponsors,silver_bill_debates,silver_bill_events,gold_constituency_activity_yearly,gold_content_fact_pool,control_pipeline_runs,control_table_manifests,control_data_quality_results
chamber=dail
house_no=34
date_start=2026-05-25
date_end=2026-06-30
limit=250
sample_rows=10
```

## Orchestrator validation run

```text
Workflow ID: 307332237
Run ID: 29299431600
Run number: 3
Head SHA: e14711cb97d4c361ee3606fd0821539af0bff8c7
Result: success
Artifact ID: 8298102893
```

## Child workflow results

### Monthly refresh

```text
Workflow ID: 294432002
Run ID: 29299437311
Run number: 7
Result: success
Artifact ID: 8298023987
```

### Compatibility adapters

```text
Workflow ID: 294866317
Run ID: 29299539592
Run number: 5
Result: success
Artifact ID: 8298034081
```

### Compatibility comparison

```text
Workflow ID: 294874693
Run ID: 29299580373
Run number: 8
Result: success
Artifact ID: 8298049611
```

### Mismatch review

```text
Workflow ID: 297343766
Run ID: 29299619179
Run number: 8
Result: success
Artifact ID: 8298060582
```

### Member profile metrics

```text
Workflow ID: 266755732
Run ID: 29299647855
Run number: 9
Result: success
```

### Instagram constituency render

```text
Workflow ID: 261945698
Run ID: 29299676372
Run number: 10
Result: success
Artifact ID: 8298085395
```

### Instagram campaign render

```text
Workflow ID: 271160957
Run ID: 29299727612
Run number: 9
Result: success
Artifact ID: 8298101606
```

## Conclusion

The production-input orchestrator validation passed.

The orchestrator can now run a monthly refresh with explicit production-like inputs, then complete all downstream validations.

The workflow remains manual-only and defaults to:

```text
refresh_type=none
run_consumers=true
```

## Next decision

The remaining scheduling decision is whether to add a weekly scheduled orchestrator trigger now that explicit child refresh inputs have been validated.

Recommended next packets:

```text
P72 — scheduled orchestrator trigger decision
P73 — scheduled orchestrator trigger implementation or deferral documentation
P74 — final production-readiness handoff update
```
