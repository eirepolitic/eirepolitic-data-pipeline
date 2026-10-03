# Compatibility adapter comparison

Run ID: `compat_adapter_comparison_20261003T213943Z`

Strict configured thresholds are applied to missing keys, row divergence, and join coverage.

| comparison_name | legacy_key | compat_key | legacy_rows | compat_rows | legacy_columns | compat_columns | legacy_join_column | compat_join_column | legacy_join_coverage_pct | compat_join_coverage_pct | matched_key_count | legacy_only_key_count | compat_only_key_count | status | failure_reasons | row_delta_pct | max_legacy_only_keys | max_compat_only_keys | max_row_delta_pct | minimum_compat_join_coverage_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| members_roster_compat | raw/members/oireachtas_members_34th_dail.csv | processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv | 176 | 176 | 8 | 7 | member_code | member_code | 100.0 | 100.0 | 176 | 0 | 0 | pass |  | 0.0 | 0 | 0 | 2.0 | 100.0 |
| member_votes_compat | processed/votes/dail_vote_member_records.csv | processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv | 30968 | 62447 | 11 | 9 | memberCode | memberCode | 100.0 | 100.0 | 173 | 0 | 2 | fail | row delta 101.65% exceeds 100.00% | 101.65 | 2 | 2 | 100.0 | 99.0 |
