# Oireachtas 31-table validation scorecard

## Summary

- Tables checked: 31
- All schemas pass: True
- All primary keys pass: True
- All CSV/Parquet pairs pass: True
- Classifications: {"pass": 14, "pass_with_warning": 3, "refresh_required": 5, "repair_required": 2, "review_required": 7}

| Table | Rows | Schema | Primary key | CSV/Parquet | Overall | Finding |
|---|---:|---|---|---|---|---|
| control_data_quality_results | 1716 | pass | pass | pass | **pass** | No unresolved finding recorded. |
| control_pipeline_runs | 427 | pass | pass | pass | **pass** | No unresolved finding recorded. |
| control_table_manifests | 31 | pass | pass | pass | **refresh_required** | Stored row counts are stale for six tables: control DQ results, bill debates, three member-history tables, and source files. |
| gold_constituency_activity_yearly | 129 | pass | pass | pass | **pass** | No unresolved finding recorded. |
| gold_content_fact_pool | 8046 | pass | pass | pass | **pass_with_warning** | Some zero metrics serialize as 0.0 rather than 0; all numeric source reconciliations pass. |
| gold_current_members | 174 | pass | pass | pass | **pass** | No unresolved finding recorded. |
| gold_member_activity_monthly | 3363 | pass | pass | pass | **pass** | No unresolved finding recorded. |
| gold_member_activity_yearly | 531 | pass | pass | pass | **pass** | No unresolved finding recorded. |
| silver_bill_debates | 1215 | pass | pass | pass | **refresh_required** | A small number of July 16 debate links/titles changed in the current official response; historical ID drift is also present. |
| silver_bill_events | 692 | pass | pass | pass | **review_required** | The original checkpoint recorded 1 failed check(s); inspect the checkpoint report. |
| silver_bill_related_docs | 345 | pass | pass | pass | **review_required** | The original checkpoint recorded 1 failed check(s); inspect the checkpoint report. |
| silver_bill_sponsors | 1211 | pass | pass | pass | **review_required** | The original checkpoint recorded 2 failed check(s); inspect the checkpoint report. |
| silver_bill_stages | 1358 | pass | pass | pass | **review_required** | The original checkpoint recorded 1 failed check(s); inspect the checkpoint report. |
| silver_bill_versions | 506 | pass | pass | pass | **refresh_required** | One newer official bill version is absent: bill 2026/1, English version C. |
| silver_bills | 404 | pass | pass | pass | **review_required** | The original checkpoint recorded 2 failed check(s); inspect the checkpoint report. |
| silver_constituencies | 43 | pass | pass | pass | **pass** | No unresolved finding recorded. |
| silver_debate_records | 164 | pass | pass | pass | **pass** | No unresolved finding recorded. |
| silver_debate_sections | 5001 | pass | pass | pass | **refresh_required** | 7 official sections added for 2026-07-16 after the live snapshot. |
| silver_division_tallies | 1194 | pass | pass | pass | **pass_with_warning** | Official API sometimes omits zero tallies; stored zero values reconcile to individual vote rows. |
| silver_divisions | 398 | pass | pass | pass | **pass** | No unresolved finding recorded. |
| silver_houses | 68 | pass | pass | pass | **pass** | No unresolved finding recorded. |
| silver_member_constituencies | 276 | pass | pass | pass | **repair_required** | 196 duplicate business rows across 98 members; no conflicting current constituency values. |
| silver_member_memberships | 176 | pass | pass | pass | **review_required** | The original checkpoint recorded 5 failed check(s); inspect the checkpoint report. |
| silver_member_offices | 123 | pass | pass | pass | **review_required** | The original checkpoint recorded 6 failed check(s); inspect the checkpoint report. |
| silver_member_parties | 280 | pass | pass | pass | **repair_required** | 196 duplicate business rows across 98 members; no conflicting current party values. |
| silver_member_votes | 58868 | pass | pass | pass | **pass** | No unresolved finding recorded. |
| silver_members | 176 | pass | pass | pass | **pass** | No unresolved finding recorded. |
| silver_parties | 11 | pass | pass | pass | **pass** | No unresolved finding recorded. |
| silver_questions | 115830 | pass | pass | pass | **refresh_required** | 1,212 official questions added for 2026-07-14 through 2026-07-16 after the live snapshot. |
| silver_source_files | 3477 | pass | pass | pass | **pass** | No unresolved finding recorded. |
| silver_speeches | 66083 | pass | pass | pass | **pass_with_warning** | A valid Seanad chair speaking at a joint sitting is not present in the Dáil-only member dimension. |

## Recommended disposition

1. Run a current refresh to capture the identified July 14–16 proceedings and legislation additions, then rebuild control manifests.
2. Repair member-party and member-constituency business-key deduplication before the next history refresh.
3. Retain the zero-tally, joint-sitting speaker, and fact numeric-format items as documented warnings.
4. Re-run all checkpoints after repairs and refresh, then compare this scorecard to the new result.
