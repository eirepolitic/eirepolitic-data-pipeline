# Party issue profile decisions

## Purpose

Create one two-slide draft post per current party showing which classified parliamentary issues appear most often in speeches by that party's current TDs.

## Grain

- grain: `party`
- stable key: normalized `party_key`
- display label: `party`
- ordering: alphabetical by party name

## Slides

1. Cover card with party name, current TD count, and classified speech count.
2. Horizontal bar chart of the top seven classified issues.

## Interpretation rules

- Counts represent recorded classified speeches, not party policy positions or endorsements.
- Current party membership is used to attribute matched speeches.
- Empty and unclassified issue values are excluded.
- Unmatched speakers are reported in the join manifest.

## Validation

- Minimum and maximum scenarios are synthetic layout tests and cannot be published.
- The real example is selected by median complexity from live party records.
- Every generated item starts unreviewed and non-publishable.

## Operational policy

- Generation and readiness checks remain manual.
- No automatic approval, scheduling, or Instagram publishing.
