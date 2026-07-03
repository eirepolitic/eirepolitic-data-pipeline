# Oireachtas classified issue production cutover decision

**Status:** approved and applied for controlled pre-production  
**Last updated:** 2026-07-03

## Decision

The production member-profile metrics workflow now uses the unified classified issue compatibility output.

This is a controlled pre-production workflow cutover. The legacy classified debate key is still preserved and was not overwritten.

## Production input now used

```text
processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
```

Production workflow patched:

```text
.github/workflows/build_member_profile_metrics_2025.yml
```

Applied env var:

```yaml
      DEBATE_ISSUES_INPUT_KEY: "processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv"
```

## Legacy key preserved

```text
processed/debates/debate_speeches_classified.csv
```

No workflow in this cutover overwrites the legacy classified debate key.

## Full classified issue enrichment evidence

Workflow:

```text
.github/workflows/oireachtas_enrichment_speech_issue_labels_trial.yml
```

Validation:

```text
Workflow ID: 304470256
Run ID: 28683964925
Result: success
Artifact ID: 8074954501
```

Review manifest:

```text
source_key: processed/debates/debate_speeches_classified.csv
source_rows: 47275
row_limit: 0
output_rows: 47275
compat_rows: 47275
DQ: pass
```

## Member-profile full-output trial evidence

Validation:

```text
Workflow ID: 294874303
Run ID: 28683998319
Result: success
Artifact ID: 8074963772
```

Review result:

```text
legacy_rows: 174
trial_rows: 174
matched_member_count: 174
trial_only_member_count: 0
legacy_only_member_count: 0
common_column_count: 12
DQ: pass
```

## Production workflow validation

Validation after production patch:

```text
Workflow ID: 266755732
Run ID: 28684033733
Result: success
```

## Rollback

Remove this env var from `.github/workflows/build_member_profile_metrics_2025.yml`:

```yaml
      DEBATE_ISSUES_INPUT_KEY: "processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv"
```

The script will fall back to:

```text
processed/debates/debate_speeches_classified.csv
```

## Final conclusion

The classified issue compatibility route has passed full-row enrichment, side-by-side member-profile validation, and production workflow validation. The controlled pre-production cutover is complete.
