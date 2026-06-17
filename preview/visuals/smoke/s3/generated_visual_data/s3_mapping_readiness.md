# S3 mapping readiness

Review-only readiness check for S3-backed Instagram visual mappings.

- Created at: `2026-06-17T03:04:07.732375+00:00`
- Overall ready: `False`
- Profile count: `2`
- Failure scope: readiness failures mark only the non-blocking S3 smoke status as failed.
- Publishing: this does not publish, schedule, or approve Instagram content.

## debate_issue_counts_s3

- Config: `instagram/visuals/data_mappings/debate_issue_counts_s3.yml`
- S3 key: `processed/debates/debate_speeches_classified.csv`
- Ready: `False`

### Errors

- issue_category_counts: No configured label field candidate matched the S3 schema.

### Transform checks

| Transform | Operation | Ready | Label matches | Value matches |
| --- | --- | --- | --- | --- |
| `issue_category_counts` | `count_by` | `False` | _none_ | _none_ |

## member_party_counts_s3

- Config: `instagram/visuals/data_mappings/member_party_counts_s3.yml`
- S3 key: `raw/members/oireachtas_members_34th_dail.csv`
- Ready: `True`

### Transform checks

| Transform | Operation | Ready | Label matches | Value matches |
| --- | --- | --- | --- | --- |
| `party_counts` | `count_by` | `True` | `party` | _none_ |
