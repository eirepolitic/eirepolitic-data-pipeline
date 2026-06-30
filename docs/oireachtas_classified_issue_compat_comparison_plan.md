# Oireachtas classified issue compatibility and comparison plan

**Status:** implemented for trial output  
**Last updated:** 2026-06-30

## Purpose

Create a safe bridge between the new enrichment speech issue label shape and existing consumers that expect the legacy classified debate CSV shape.

## Trial builder

Module:

```text
extract/oireachtas/enrichment_speech_issue_labels.py
```

Workflow:

```text
.github/workflows/oireachtas_enrichment_speech_issue_labels_trial.yml
```

The builder reads the current legacy classified debate output and writes side-by-side unified enrichment and compatibility outputs.

## Source key

```text
processed/debates/debate_speeches_classified.csv
```

This source is read only. It is not overwritten.

## Unified enrichment outputs

```text
processed/oireachtas_unified/enrichment/speech_issue_labels/speech_issue_labels_2025_trial.csv
processed/oireachtas_unified/enrichment/speech_issue_labels/parquets/speech_issue_labels_2025_trial.parquet
```

## Legacy-compatible outputs

```text
processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
processed/oireachtas_unified/compat/debates/parquets/debate_speeches_classified_compat.parquet
```

## Review outputs

```text
review/enrichment_speech_issue_labels/latest/sample.csv
review/enrichment_speech_issue_labels/latest/manifest.json
review/enrichment_speech_issue_labels/latest/schema.json
review/enrichment_speech_issue_labels/latest/dq.json
review/enrichment_speech_issue_labels/latest/report.md
```

## Current comparison checks

The first comparison is built into the trial DQ:

| Check | Meaning |
|---|---|
| `row_count_gt_zero` | Trial output contains rows. |
| `record_id_unique` | Trial primary key is unique. |
| `speech_id_populated` | Every row has a speech identifier. |
| `approved_issue_labels` | Issue labels are approved categories or blank. |
| `row_count_expected` | Trial row count matches the configured row limit/source count. |

## Consumer repointing plan

Do not repoint production member profile metrics yet.

Future trial command should set:

```text
DEBATE_ISSUES_INPUT_KEY=processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
```

Then compare the member-profile metrics result against the current production metrics output.

## Rollback

No rollback is needed for the trial because no production key is overwritten. To stop using the trial, stop referencing these keys:

```text
processed/oireachtas_unified/enrichment/speech_issue_labels/speech_issue_labels_2025_trial.csv
processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
```
