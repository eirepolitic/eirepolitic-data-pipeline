# Compatibility adapter comparison

Run ID: `compat_adapter_comparison_20260617T023618Z`

This report compares legacy downstream inputs with non-destructive unified compatibility outputs.

| comparison_name | status | legacy_key | compat_key | legacy_rows | compat_rows | legacy_columns | compat_columns | legacy_join_column | compat_join_column | legacy_join_coverage_pct | compat_join_coverage_pct | matched_key_count | legacy_only_key_count | compat_only_key_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| members_roster_compat | pass | raw/members/oireachtas_members_34th_dail.csv | processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv | 176 | 174 | 8 | 7 | member_code | member_code | 100.00 | 100.00 | 174 | 2 | 0 |
| member_votes_compat | pass | processed/votes/dail_vote_member_records.csv | processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv | 30968 | 29805 | 11 | 9 | memberCode | memberCode | 100.00 | 100.00 | 173 | 0 | 0 |
