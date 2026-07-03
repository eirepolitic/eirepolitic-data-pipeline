# Oireachtas member summaries enrichment design

**Status:** design complete  
**Last updated:** 2026-07-02

## Purpose

Design a side-by-side unified enrichment output for member summaries/background text without overwriting the current legacy summary output.

## Existing legacy workflow

Workflow:

```text
.github/workflows/members_background_summarizer.yml
```

Script:

```text
process/members_background_summarizer.py
```

Current outputs:

```text
processed/members/members_summaries.csv
processed/members/parquets/members_summaries.parquet
```

## Proposed unified enrichment table

Table name:

```text
enrichment_member_summaries
```

Proposed outputs:

```text
processed/oireachtas_unified/enrichment/text/member_summaries/member_summaries_trial.csv
processed/oireachtas_unified/enrichment/text/member_summaries/parquets/member_summaries_trial.parquet
```

Proposed compatibility outputs:

```text
processed/oireachtas_unified/compat/text/members_summaries_compat.csv
processed/oireachtas_unified/compat/text/parquets/members_summaries_compat.parquet
```

## Proposed columns

```text
record_id
member_code
full_name
summary_text
summary_source
model_name
source_key
source_hash
review_status
generated_at_utc
run_id
```

Optional columns if present in legacy source:

```text
party
constituency
profile_url
prompt_version
summary_style
```

## DQ checks

Required:

```text
row_count_gt_zero
record_id_unique
member_code_populated
summary_text_populated
row_count_expected
```

Informational:

```text
summary_text_missing_count
reviewed_count
model_name_populated_count
```

## Implementation approach

The first trial builder should not call OpenAI. It should reshape the existing legacy summary CSV into the unified enrichment shape, preserving the current text and adding provenance fields.

Recommended module:

```text
extract/oireachtas/enrichment_member_summaries.py
```

Recommended workflow:

```text
.github/workflows/oireachtas_member_summaries_enrichment_trial.yml
```

## Consumer rollout

Do not repoint Instagram renderers directly to the enrichment table first.

Use the compatibility output first:

```text
processed/oireachtas_unified/compat/text/members_summaries_compat.csv
```

Then run Instagram constituency/campaign render trials with:

```text
INSTAGRAM_MEMBER_SUMMARIES_DATASET_KEYS=processed/oireachtas_unified/compat/text/members_summaries_compat.csv
```

## Risk controls

- Do not overwrite `processed/members/members_summaries.csv`.
- Do not call OpenAI in the first compatibility trial.
- Keep `review_status` and provenance fields in the unified enrichment output.
- Require consumer artifact validation before any production workflow repointing.

## Current recommendation

Proceed with a legacy-to-unified summary trial builder next. Delay any new summary generation until the side-by-side compatibility output is validated.
