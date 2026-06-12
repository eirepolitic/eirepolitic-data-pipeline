# `member_profile_metrics_trial` sample

| check_name | status | legacy_value | trial_value | message |
| --- | --- | --- | --- | --- |
| legacy_rows | info | 174 |  | processed/members/member_profile_metrics_2025.csv |
| trial_rows | pass |  | 10 | processed/oireachtas_unified/compat/members/member_profile_metrics_2025_trial.csv |
| legacy_member_count | info | 174 |  | distinct legacy member_code |
| trial_member_count | pass |  | 10 | distinct trial member_code |
| matched_member_count | pass | 174 | 10 | legacy/trial member_code overlap |
| trial_only_member_count | info |  | 0 | member_code only in trial |
| legacy_only_member_count | info | 164 |  | member_code only in legacy |
| common_column_count | pass | 12 | 12 | all_distinct_vote_ids_2025,constituency,distinct_votes_participated_2025,full_name,member_code,party,photo_url,speech_count_2025,speech_rank_2025,top_issue_2025,top_issue_count_2025,vote_participation_pct_2025 |
