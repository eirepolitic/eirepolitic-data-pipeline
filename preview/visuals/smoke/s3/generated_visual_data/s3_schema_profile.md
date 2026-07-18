# S3 schema profile summary

Review-only schema snapshot for Instagram visual smoke mappings.

- Created at: `2026-07-18T22:12:22.089653+00:00`
- Profile count: `2`
- Range bytes per file: `262144`
- Sample rows per file: `25`
- Sampled values included: `False`
- Download strategy: S3 prefix range read only; full datasets are not downloaded.
- Privacy: raw sampled field values are omitted by default.
- Publishing: this does not publish, schedule, or approve Instagram content.

## debate_issue_counts_s3

- Config: `instagram/visuals/data_mappings/debate_issue_counts_s3.yml`
- S3 key: `processed/oireachtas_unified/compat/debates/debate_speeches_classified_compat.csv`
- Bucket: `eirepolitic-data`
- Region: `ca-central-1`
- Last modified: `2026-07-16T20:25:53+00:00`
- Content range: `bytes 0-262143/61391796`
- Column count: `8`
- Sample rows inspected: `25`
- Sampled values included: `False`
- Range may be truncated: `True`

### Columns

`Debate Date`, `Debate Section`, `Debate Section Name`, `Speaker Name`, `Speech Text`, `Speech Order`, `speech_id`, `PoliticalIssues`

### Likely numeric columns

`Speech Order`

### Mapping candidate matches

- Transform `issue_category_counts` / `count_by`
  - Label matches: `PoliticalIssues`
  - Value matches: _none_

### Sample column coverage

Raw sampled field values are omitted from this public preview summary by default.

| Column | Non-empty sampled rows | Blank sampled rows |
| --- | ---: | ---: |
| `Debate Date` | 25 | 0 |
| `Debate Section` | 25 | 0 |
| `Debate Section Name` | 25 | 0 |
| `Speaker Name` | 25 | 0 |
| `Speech Text` | 25 | 0 |
| `Speech Order` | 25 | 0 |
| `speech_id` | 25 | 0 |
| `PoliticalIssues` | 25 | 0 |

## member_party_counts_s3

- Config: `instagram/visuals/data_mappings/member_party_counts_s3.yml`
- S3 key: `processed/oireachtas_unified/compat/members/oireachtas_members_34th_dail_compat.csv`
- Bucket: `eirepolitic-data`
- Region: `ca-central-1`
- Last modified: `2026-07-16T20:25:58+00:00`
- Content range: `bytes 0-18034/18035`
- Column count: `7`
- Sample rows inspected: `25`
- Sampled values included: `False`
- Range may be truncated: `False`

### Columns

`member_code`, `full_name`, `constituency`, `party`, `house_no`, `source`, `snapshot_date`

### Likely numeric columns

`house_no`

### Mapping candidate matches

- Transform `party_counts` / `count_by`
  - Label matches: `party`
  - Value matches: _none_

### Sample column coverage

Raw sampled field values are omitted from this public preview summary by default.

| Column | Non-empty sampled rows | Blank sampled rows |
| --- | ---: | ---: |
| `member_code` | 25 | 0 |
| `full_name` | 25 | 0 |
| `constituency` | 25 | 0 |
| `party` | 25 | 0 |
| `house_no` | 25 | 0 |
| `source` | 25 | 0 |
| `snapshot_date` | 25 | 0 |
