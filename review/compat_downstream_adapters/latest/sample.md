# `compat_downstream_adapters` sample

| adapter_name | status | source_key | output_key | source_rows | output_rows | source_columns | output_columns | primary_key_column | primary_key_populated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| members_roster | pass | processed/oireachtas_unified/latest/csv/gold_current_members.csv | processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv | 10 | 10 | 7 | 7 | member_code | true |
| member_votes | pass | processed/oireachtas_unified/latest/csv/silver_member_votes.csv | processed/oireachtas_unified/compat/votes/dail_vote_member_records_compat.csv | 512 | 512 | 11 | 9 | memberCode | true |
