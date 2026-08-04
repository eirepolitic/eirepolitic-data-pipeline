# Party issue profile decisions

## Purpose

Create one two-slide draft post per current party showing which classified parliamentary issues appear most often in speeches by that party's current TDs.

## Grain

- grain: `party`
- stable key: normalized `party_key`
- display label: `party`
- ordering: alphabetical by party name

## Slides

1. Cover slide titled with the party name and a tall metric card showing current TDs, classified speeches, and displayed issue categories.
2. Horizontal bar chart titled `Top issues: <party>` showing the top seven classified issues.

The cover media does not repeat the party name and contains no small footer/status copy.

## Interpretation rules

- Counts represent recorded classified speeches, not party policy positions or endorsements.
- Current party membership is used to attribute matched speeches.
- Empty and unclassified issue values are excluded.
- Unmatched speakers are reported in the join manifest.

## Validation policy

- Validation is real-data-first.
- Every required visual scenario first searches the current production batch.
- A missing current scenario searches up to 12 recent historical unified-model batches.
- Synthetic contract-edge data is not generated for convenience.
- A scenario is waived only after current and searched historical production data contain no qualifying real case.
- Every scenario records its full search path: current real, historical real, synthetic contract edge, and waiver.
- Legacy `minimum` and `maximum` remain compatibility aliases only; the review matrix uses visual-specific scenario names.

## Review outputs

- `validation_contact_sheet.png` is the deduplicated approval summary.
- The cover appears once.
- Identical visual renders are grouped by hash and labelled with all scenarios they cover.
- Waivers appear in one compact block.
- `validation_audit_contact_sheet.png` preserves complete scenario-by-scenario evidence.
- Full provenance and quality measurements remain in JSON manifests.

## Automated quality gates

Validation fails before human review when declared thresholds are breached for:

- slide whitespace and occupied height
- media-slot vertical and area fill
- plot-area utilization
- category, value, and axis font sizes
- bar thickness
- label wrapping
- value-label right-edge headroom

## Historical validation result

Live validation searched eight historical batches. No current waiver was replaced by those batches, and three scenarios remained waived. The result confirms the fallback process rather than proving all possible historical edge cases exist.

## Operational policy

- Every generated item starts unreviewed and non-publishable.
- Generation and readiness checks remain manual.
- No automatic approval, scheduling, or Instagram publishing.
