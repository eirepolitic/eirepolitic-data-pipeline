# Oireachtas validation findings — fix plan

## Objective

Resolve the two confirmed data defects, refresh stale official-source data, rebuild stale control metadata, and prove the repaired candidate before any production promotion.

## Phase 1 — Repair member-history deduplication

Affected tables:

- `silver_member_parties`
- `silver_member_constituencies`

Actions:

1. Deduplicate normalized rows by stable business keys rather than generated primary keys.
2. Preserve the first identical row only when all compared values agree.
3. Record conflicting duplicate keys and fail DQ if values disagree.
4. Add DQ checks for business-key uniqueness.
5. Add regression tests for exact duplicate collapse and conflict detection.

Acceptance gates:

- No duplicate business keys.
- No unresolved conflicting keys.
- Existing primary keys remain non-null and unique.
- Current party and constituency values remain unchanged for every member.

## Phase 2 — Refresh recent official-source changes

Affected tables:

- `silver_debate_sections`
- `silver_questions`
- `silver_bill_versions`
- `silver_bill_debates`

Actions:

1. Run a non-production candidate refresh through the current date.
2. Confirm the newly observed official records are present.
3. Re-run comparable-scope API checks.
4. Confirm downstream gold tables recompute exactly from the refreshed silver tables.

Acceptance gates:

- Seven missing debate sections are captured or explained by a newer official response.
- The 1,212 newer questions are captured or reconciled to the latest official count.
- Bill 2026/1 English version C is present if still returned officially.
- Recent bill-debate links match the current official response by business key.

## Phase 3 — Correct control-table freshness

Affected table:

- `control_table_manifests`

Actions:

1. Build `control_pipeline_runs` and `control_data_quality_results` after all data tables.
2. Build `control_table_manifests` last, after every current manifest exists.
3. Validate each stored row count against the referenced CSV and Parquet objects.

Acceptance gates:

- All 31 manifest rows reference existing objects.
- Every stored row count equals both referenced object counts.
- Schema hashes and column counts match the registry.

## Phase 4 — Candidate validation

Actions:

1. Build a new immutable candidate batch.
2. Keep production publishing disabled.
3. Run all seven validation checkpoints.
4. Produce a new 31-table scorecard.

Acceptance gates:

- All files readable.
- All schemas pass.
- All primary keys pass.
- All CSV/Parquet pairs match.
- Zero `repair_required` tables.
- Zero `refresh_required` tables unless the official API changes during validation and the difference is documented.

## Phase 5 — Promotion preparation

Actions:

1. Open a pull request with code, tests, workflows, evidence, and rollback notes.
2. Keep the production switch disabled until the candidate and pull request are reviewed.
3. Promote only the exact validated candidate.
4. Verify live pointers and rerun smoke checks.
5. Disable the publish switch immediately after promotion.

## Rollback

Retain the current production batch pointer before promotion. On any post-promotion failure, restore that exact pointer and rerun live smoke checks.
