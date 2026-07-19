# Oireachtas validation — Checkpoint 6

| Table | Live rows | Tests | Passed | Failed |
|---|---:|---:|---:|---:|
| control_pipeline_runs | 427 | 13 | 13 | 0 |
| control_table_manifests | 31 | 11 | 10 | 1 |
| control_data_quality_results | 1716 | 9 | 9 | 0 |

## Findings

| Table | Test | Expected | Actual | Details |
|---|---|---|---|---|
| control_table_manifests | manifest_row_counts_equal_actual_objects | 0 | 12 | [{"table": "control_data_quality_results", "expected": 1592, "actual_csv": 1716}, {"table": "control_data_quality_results", "expected": 1592, "actual_parquet": 1716}, {"table": "silver_bill_debates", "expected": 1170, "actual_csv": 1215}, {"table": "silver_bill_debates", "expected": 1170, "actual_parquet": 1215}, {"table": "silver_member_constituencies", "expected": 176, "actual_csv": 276}, {"table": "silver_member_constituencies", "expected": 176, "actual_parquet": 276}, {"table": "silver_member_offices", "expected": 77, "actual_csv": 123}, {"table": "silver_member_offices", "expected": 77, "actual_parquet": 123}, {"table": "silver_member_parties", "expected": 178, "actual_csv": 280}, {"table": "silver_member_parties", "expected": 178, "actual_parquet": 280}, {"table": "silver_source_files", "expected": 3452, "actual_csv": 3477}, {"table": "silver_source_files", "expected": 3452, "actual_parquet": 3477}] |
