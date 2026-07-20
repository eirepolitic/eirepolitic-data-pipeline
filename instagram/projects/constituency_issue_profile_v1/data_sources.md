# Data sources

## Current members

Logical key:

```text
processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv
```

Legacy fallback:

```text
raw/members/oireachtas_members_34th_dail.csv
```

Required fields are resolved from candidate lists for member name and constituency.

## Classified debate speeches

Logical key:

```text
processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv
```

Legacy fallback:

```text
processed/debates/debate_speeches_classified.csv
```

Required fields are resolved from candidate lists for speaker name and political issue.

## Join and aggregation

Speaker names are normalized and joined to the current member table. Matched speeches are grouped by constituency and issue. Unmatched speakers and empty/unclassified issue values remain recorded in the validation manifest and are not silently included.

## Latest validated source snapshot

Workflow run `29703335986` resolved both logical keys through production batch `current-government-backfill-20260716-1` in `ca-central-1`.

- member rows: 176
- speech rows: 47,275
- matched speeches: 29,233
- unmatched speeches: 107
- empty or unclassified issues ignored: 17,935
- constituencies with matched classified speeches: 43

Issue classification is a separately reviewed enrichment, not a source field supplied directly by the Houses of the Oireachtas.
