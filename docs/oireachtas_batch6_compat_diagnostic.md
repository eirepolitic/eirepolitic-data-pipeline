# Oireachtas Batch 6 compatibility diagnostic

- Run ID: 29519287299
- Batch ID: batch6-validation-29518530422-1
- Comparison exit code: 0
- Mismatch exit code: 0

## Adapter output

```text
{
  "dq_status": "pass",
  "rows": 2,
  "run_id": "compat_downstream_adapters_20260716T171953Z",
  "table": "compat_downstream_adapters"
}
```

## Comparison output

```text
{
  "dq": {
    "checks": [
      {
        "check_name": "row_count_gt_zero",
        "metric_value": 2,
        "status": "pass"
      },
      {
        "check_name": "primary_key_unique",
        "status": "pass"
      },
      {
        "check_name": "no_failed_comparisons",
        "failing_comparisons": [],
        "status": "pass"
      }
    ],
    "dq_status": "pass",
    "failing_comparisons": [],
    "primary_key": [
      "comparison_name"
    ],
    "primary_key_unique": true,
    "row_count": 2,
    "table": "compat_adapter_comparison"
  },
  "dq_status": "pass",
  "rows": [
    {
      "comparison_name": "members_roster_compat",
      "compat_columns": 7,
      "compat_join_column": "member_code",
      "compat_join_coverage_pct": 100.0,
      "compat_key": "processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv",
      "compat_only_key_count": 0,
      "compat_rows": 176,
      "failure_reasons": "",
      "legacy_columns": 8,
      "legacy_join_column": "member_code",
      "legacy_join_coverage_pct": 100.0,
      "legacy_key": "raw/members/oireachtas_members_34th_dail.csv",
      "legacy_only_key_count": 0,
      "legacy_rows": 176,
      "matched_key_count": 176,
      "max_compat_only_keys": 0,
      "max_legacy_only_keys": 0,
      "max_row_delta_pct": 2.0,
      "minimum_compat_join_coverage_pct": 100.0,
      "row_delta_pct": 0.0,
      "status": "pass"
    },
    {
      "comparison_name": "member_votes_compat",
      "compat_columns": 9,
      "compat_join_column": "memberCode",
      "compat_join_coverage_pct": 100.0,
      "compat_key": "processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv",
      "compat_only_key_count": 2,
      "compat_rows": 8973,
      "failure_reasons": "",
      "legacy_columns": 11,
      "legacy_join_column": "memberCode",
      "legacy_join_coverage_pct": 100.0,
      "legacy_key": "processed/votes/dail_vote_member_records.csv",
      "legacy_only_key_count": 2,
      "legacy_rows": 30968,
      "matched_key_count": 171,
      "max_compat_only_keys": 2,
      "max_legacy_only_keys": 2,
      "max_row_delta_pct": 100.0,
      "minimum_compat_join_coverage_pct": 99.0,
      "row_delta_pct": 71.02,
      "status": "pass"
    }
  ],
  "run_id": "compat_adapter_comparison_20260716T171957Z",
  "table": "compat_adapter_comparison"
}
```

## Mismatch output

```text
{
  "dq_status": "pass",
  "rows": 0,
  "run_id": "member_code_mismatch_review_20260716T172000Z",
  "summary": [
    {
      "dataset_name": "roster",
      "legacy_member_count": 176,
      "legacy_only_count": 0,
      "legacy_rows": 176,
      "matched_member_count": 176,
      "unified_member_count": 176,
      "unified_only_count": 0,
      "unified_rows": 176
    }
  ],
  "table": "member_code_mismatch_review"
}
```
