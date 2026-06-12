# `cutover_comparison_report` sample

| comparison_name | status | legacy_key | unified_key | legacy_exists | unified_exists | legacy_rows | unified_rows | legacy_columns | unified_columns | legacy_join_column | unified_join_column | legacy_join_coverage_pct | unified_join_coverage_pct | matched_key_count | legacy_only_key_count | unified_only_key_count | comparison_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| members_current_roster | pass | raw/members/oireachtas_members_34th_dail.csv | processed/oireachtas_unified/latest/csv/silver_members.csv | true | true | 176 | 10 | 8 | 15 | member_code | member_code | 100.00 | 100.00 | 10 | 166 | 0 | cmp:6d16fc60b72581f37444d7c4 |
| speeches | pass | raw/debates/debate_speeches_extracted.csv | processed/oireachtas_unified/latest/csv/silver_speeches.csv | true | true | 58540 | 357 | 6 | 18 | speech_id | speech_id |  | 100.00 |  |  |  | cmp:dd8dafb05f0997beb416561d |
| vote_divisions | pass | processed/votes/dail_vote_divisions.csv | processed/oireachtas_unified/latest/csv/silver_divisions.csv | true | true | 207 | 3 | 8 | 14 | division_id | division_id |  | 100.00 |  |  |  | cmp:b65f592c5ec814e8c2c9337f |
| member_votes | pass | processed/votes/dail_vote_member_records.csv | processed/oireachtas_unified/latest/csv/silver_member_votes.csv | true | true | 30968 | 512 | 11 | 11 | member_vote_id | member_vote_id |  | 100.00 |  |  |  | cmp:2882f68ac2daf3f25ced970e |
| member_profile_metrics_yearly | pass | processed/members/member_profile_metrics_2025.csv | processed/oireachtas_unified/latest/csv/gold_member_activity_yearly.csv | true | true | 174 | 10 | 12 | 13 | member_code | member_code | 100.00 | 100.00 | 10 | 164 | 0 | cmp:2339aa433c9a03117d49fe2c |
