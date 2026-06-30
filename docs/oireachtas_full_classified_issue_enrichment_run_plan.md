# Oireachtas full classified issue enrichment run plan

**Status:** planning complete  
**Last updated:** 2026-06-30

## Purpose

Plan the full classified issue enrichment compatibility build before any production cutover of member-profile metrics.

## Current state

The classified issue enrichment trial exists and passed validation, but the current compat output was built with the default trial limit:

```text
source_rows: 47275
row_limit: 50
output_rows: 50
```

Current compat key:

```text
processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
```

Current production classified debate key remains:

```text
processed/debates/debate_speeches_classified.csv
```

## Full run requirement

Run the enrichment workflow with:

```text
row_limit=0
```

Workflow:

```text
.github/workflows/oireachtas_enrichment_speech_issue_labels_trial.yml
```

Workflow ID:

```text
304470256
```

Expected full-row output:

```text
processed/oireachtas_unified/enrichment/speech_issue_labels/speech_issue_labels_2025_trial.csv
processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
```

## Required validations

After full run:

1. Confirm workflow result is `success`.
2. Confirm DQ is `pass`.
3. Confirm output rows match source rows.
4. Confirm approved issue labels check passes.
5. Confirm compat file contains these consumer-required columns:

```text
Speaker Name
Debate Date
PoliticalIssues
speech_id
```

6. Rerun member-profile trial using:

```text
DEBATE_ISSUES_INPUT_KEY=processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
```

7. Compare speech metric columns against current production metrics:

```text
speech_count_2025
speech_rank_2025
top_issue_2025
top_issue_count_2025
```

## Cutover gate

Production cutover is allowed only if:

```text
full classified issue run: success
full classified issue DQ: pass
member profile trial: success
member profile trial DQ: pass
member profile matched member count: 174
trial-only members: 0
legacy-only members: 0
```

## Production patch if approved later

Patch:

```text
.github/workflows/build_member_profile_metrics_2025.yml
```

Add:

```yaml
      DEBATE_ISSUES_INPUT_KEY: "processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv"
```

## Rollback

Remove the override or set it back to:

```text
processed/debates/debate_speeches_classified.csv
```

## Current recommendation

Do not cut over yet. First run the full classified issue enrichment workflow and then rerun the member-profile metrics trial comparison.
