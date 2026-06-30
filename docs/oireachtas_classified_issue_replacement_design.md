# Oireachtas classified issue replacement design

**Status:** design only  
**Last updated:** 2026-06-30

## Goal

Create a side-by-side enrichment output for classified speech issue labels without overwriting the current production classified debate key.

## Current production path

Input:

```text
raw/debates/debate_speeches_extracted.csv
```

Current output:

```text
processed/debates/debate_speeches_classified.csv
processed/debates/parquets/debate_speeches_classified.parquet
```

Current main consumer:

```text
process/build_member_profile_metrics_2025.py
```

Current consumer env var:

```text
DEBATE_ISSUES_INPUT_KEY=processed/debates/debate_speeches_classified.csv
```

## Proposed side-by-side output

New CSV:

```text
processed/oireachtas_unified/enrichment/speech_issue_labels/speech_issue_labels_2025_trial.csv
```

New parquet:

```text
processed/oireachtas_unified/enrichment/speech_issue_labels/parquets/speech_issue_labels_2025_trial.parquet
```

New review output:

```text
review/enrichment_speech_issue_labels/latest/{manifest.json,sample.csv,dq.json,report.md}
```

## Proposed table name

```text
enrichment_speech_issue_labels
```

## Proposed columns

Minimum columns:

```text
speech_id
member_code
speaker_name
debate_date
speech_order
source_speech_text_hash
issue_label
issue_label_source
model_name
classification_status
review_status
classified_at_utc
source_key
run_id
```

Recommended status values:

```text
classification_status: classified, none, skipped_short_text, failed
review_status: unreviewed, reviewed, rejected
```

## Compatibility output

A compatibility adapter can rebuild the legacy shape expected by `process/build_member_profile_metrics_2025.py`:

```text
processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
```

That compat output should include the legacy-compatible columns required by the metrics builder:

```text
Speaker Name
Debate Date
PoliticalIssues
speech_id
```

## Rollout sequence

1. Build `enrichment_speech_issue_labels` side-by-side from the current raw debate speeches input.
2. Write review sample, DQ JSON, and manifest.
3. Compare issue-label coverage against `processed/debates/debate_speeches_classified.csv`.
4. Build a legacy-compatible adapter file.
5. Run member-profile metrics trial with:

```text
DEBATE_ISSUES_INPUT_KEY=processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
```

6. Compare member profile metrics output against the current cut-over production metrics output.
7. Only then switch the production metrics workflow to the compat classified-issues key.

## DQ checks

Required checks:

- `speech_id` populated.
- `speech_id` unique if source grain is one row per speech.
- `issue_label` is one of the approved categories or `NONE`.
- `classification_status` is populated.
- `model_name` populated for model-classified rows.
- output row count equals source row count unless explicit test mode is used.

## Risk controls

- Do not overwrite `processed/debates/debate_speeches_classified.csv` during trial.
- Do not change member-profile production `DEBATE_ISSUES_INPUT_KEY` during initial build.
- Keep test row limits available because this workflow can call OpenAI.
- Include `source_speech_text_hash` to detect when old classifications are being reused against changed text.

## Recommended implementation packet

```text
E02 — enrichment_speech_issue_labels trial builder
```

Deliverables:

- `extract/oireachtas/enrichment_speech_issue_labels.py`
- `.github/workflows/oireachtas_enrichment_speech_issue_labels_trial.yml`
- review output under `review/enrichment_speech_issue_labels/latest/`
- no production consumer repointing
