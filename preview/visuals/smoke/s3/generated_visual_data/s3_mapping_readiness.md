# S3 mapping readiness

Review-only readiness check for S3-backed Instagram visual mappings.

- Created at: `2026-06-30T05:21:49.524498+00:00`
- Overall ready: `True`
- Profile count: `2`
- Failure scope: readiness failures mark only the non-blocking S3 smoke status as failed.
- Publishing: this does not publish, schedule, or approve Instagram content.

## debate_issue_counts_s3

- Config: `instagram/visuals/data_mappings/debate_issue_counts_s3.yml`
- S3 key: `processed/debates/debate_speeches_classified.csv`
- Ready: `True`

### Transform checks

| Transform | Operation | Ready | Label matches | Value matches |
| --- | --- | --- | --- | --- |
| `issue_category_counts` | `count_by` | `True` | `PoliticalIssues` | _none_ |

## member_party_counts_s3

- Config: `instagram/visuals/data_mappings/member_party_counts_s3.yml`
- S3 key: `raw/members/oireachtas_members_34th_dail.csv`
- Ready: `True`

### Transform checks

| Transform | Operation | Ready | Label matches | Value matches |
| --- | --- | --- | --- | --- |
| `party_counts` | `count_by` | `True` | `party` | _none_ |
