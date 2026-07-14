# Oireachtas production-input orchestrator design

**Status:** design approved for implementation  
**Last updated:** 2026-07-13 America/Vancouver

## Goal

Update the refresh validation orchestrator so refresh-enabled runs dispatch child refresh workflows with explicit production-like inputs instead of relying on child `workflow_dispatch` defaults.

## Decision

Use Option B:

```text
orchestrator calls gh workflow run with explicit -f inputs
```

Do not convert child workflows to `workflow_call` yet. The current child workflows already support the required manual inputs and are stable.

## Why Option B

Option B is the smallest safe change:

```text
no child workflow rewrite
no job-level interface migration
no old schedule behavior change
same child workflow files remain manually runnable
```

## Explicit production-like inputs

### Weekly refresh

Workflow:

```text
.github/workflows/oireachtas_weekly_refresh.yml
```

Inputs to pass:

```text
mode=incremental
publish_latest=auto
tables=silver_members,silver_member_memberships,silver_member_parties,silver_member_constituencies,silver_member_offices,silver_debate_records,silver_debate_sections,silver_speeches,silver_divisions,silver_division_tallies,silver_member_votes,silver_questions,gold_current_members,gold_member_activity_yearly,gold_member_activity_monthly,gold_content_fact_pool,control_pipeline_runs,control_table_manifests,control_data_quality_results
chamber=dail
house_no=34
date_start=<UTC today - 35 days>
date_end=<UTC today>
limit=100
sample_rows=10
```

This mirrors the weekly scheduled branch.

### Monthly refresh

Workflow:

```text
.github/workflows/oireachtas_monthly_refresh.yml
```

Inputs to pass:

```text
mode=incremental
publish_latest=auto
tables=silver_constituencies,silver_parties,silver_source_files,silver_bills,silver_bill_versions,silver_bill_stages,silver_bill_related_docs,silver_bill_sponsors,silver_bill_debates,silver_bill_events,gold_constituency_activity_yearly,gold_content_fact_pool,control_pipeline_runs,control_table_manifests,control_data_quality_results
chamber=dail
house_no=34
date_start=<UTC first day of current month - 1 month - 7 days>
date_end=<UTC first day of current month - 1 day>
limit=250
sample_rows=10
```

This mirrors the monthly scheduled branch.

### Yearly refresh

Workflow:

```text
.github/workflows/oireachtas_yearly_refresh.yml
```

Inputs to pass:

```text
mode=full
publish_latest=auto
tables=silver_houses,silver_constituencies,silver_parties,silver_members,silver_member_memberships,silver_member_parties,silver_member_constituencies,silver_member_offices,silver_bills,silver_bill_versions,silver_bill_stages,gold_current_members,gold_member_activity_yearly,gold_constituency_activity_yearly,gold_content_fact_pool,control_pipeline_runs,control_table_manifests,control_data_quality_results
chamber=dail
house_no=34
date_start=<previous UTC year>-01-01
date_end=<previous UTC year>-12-31
limit=500
sample_rows=10
```

This mirrors the yearly scheduled branch.

## Orchestrator behavior

The orchestrator should:

1. keep manual default `refresh_type=none` for safety;
2. when refresh type is `weekly`, `monthly`, or `yearly`, compute the production-like date window inside the orchestrator;
3. dispatch the child refresh workflow with explicit `-f` inputs;
4. wait for that child run to finish;
5. continue to downstream validations only if the refresh succeeds;
6. write the effective child inputs to the orchestrator summary artifact.

## Validation approach

Because the connected dispatch tool cannot pass orchestrator inputs, validation will temporarily set:

```text
refresh_type=monthly
```

Then run the orchestrator and verify that the monthly child run uses explicit production-like inputs.

After validation, restore:

```text
refresh_type=none
```

## Acceptance criteria

P70 implementation is accepted when:

```text
orchestrator has explicit refresh input functions
manual default remains restorable to none
child refresh dispatch includes -f mode/date/limit/table inputs
```

P71 validation is accepted when:

```text
orchestrator run succeeds
monthly child refresh succeeds with explicit production-like inputs
downstream validations succeed
orchestrator default is restored to none
packet status is updated
```
