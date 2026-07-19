# Oireachtas validation — Checkpoint 5

| Table | Live rows | Rebuilt rows | Tests | Passed | Failed |
|---|---:|---:|---:|---:|---:|
| gold_current_members | 174 | 174 | 9 | 9 | 0 |
| gold_member_activity_yearly | 531 | 531 | 9 | 9 | 0 |
| gold_member_activity_monthly | 3363 | 3363 | 9 | 9 | 0 |
| gold_constituency_activity_yearly | 129 | 129 | 9 | 9 | 0 |
| gold_content_fact_pool | 8046 | 8046 | 12 | 11 | 1 |
| cross_table | 0 | 0 | 2 | 2 | 0 |

## Findings

| Table | Test | Expected | Actual | Details |
|---|---|---|---|---|
| gold_content_fact_pool | recomputed_rows_equal | exact equality excluding snapshot_date | different | [{"key": {"fact_id": "fact:052f4eb36dd7e5be3813b71a"}, "merge": "both", "differences": {"metric_value": {"live": "0.0", "rebuilt": "0"}}}, {"key": {"fact_id": "fact:1204e234044a3ed09191202b"}, "merge": "both", "differences": {"metric_value": {"live": "0.0", "rebuilt": "0"}}}, {"key": {"fact_id": "fact:191903993537a99faf369d29"}, "merge": "both", "differences": {"metric_value": {"live": "0.0", "rebuilt": "0"}}}, {"key": {"fact_id": "fact:19de113104a678469b712b8a"}, "merge": "both", "differences": {"metric_value": {"live": "0.0", "rebuilt": "0"}}}, {"key": {"fact_id": "fact:4a30cdfd89a7935eed5b6a18"}, "merge": "both", "differences": {"metric_value": {"live": "0.0", "rebuilt": "0"}}}, {"key": {"fact_id": "fact:697fd3fdec90a9e7bb7dc36b"}, "merge": "both", "differences": {"metric_value": {"live": "0.0", "rebuilt": "0"}}}, {"key": {"fact_id": "fact:6b7c147a1a241d61d349c8ed"}, "merge": "both", "differences": {"metric_value": {"live": "0.0", "rebuilt": "0"}}}, {"key": {"fact_id": "fact:6ebf006cb75b6e8ca2257bef"}, "merge": "both", "differences": {"metric_value": {"live": "0.0", "rebuilt": "0"}}}, {"key": {"fact_id": "fact:74009b2067bc40eb52b574e8"}, "merge": "both", "differences": {"metric_ |
