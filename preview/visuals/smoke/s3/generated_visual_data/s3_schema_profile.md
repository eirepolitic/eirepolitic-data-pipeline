# S3 schema profile summary

Review-only schema snapshot for Instagram visual smoke mappings.

- Created at: `2026-06-30T05:21:48.449591+00:00`
- Profile count: `2`
- Range bytes per file: `262144`
- Sample rows per file: `25`
- Sampled values included: `False`
- Download strategy: S3 prefix range read only; full datasets are not downloaded.
- Privacy: raw sampled field values are omitted by default.
- Publishing: this does not publish, schedule, or approve Instagram content.

## debate_issue_counts_s3

- Config: `instagram/visuals/data_mappings/debate_issue_counts_s3.yml`
- S3 key: `processed/debates/debate_speeches_classified.csv`
- Bucket: `eirepolitic-data`
- Region: `ca-central-1`
- Last modified: `2026-03-28T17:53:52+00:00`
- Content range: `bytes 0-262143/61391799`
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
- S3 key: `raw/members/oireachtas_members_34th_dail.csv`
- Bucket: `eirepolitic-data`
- Region: `ca-central-1`
- Last modified: `2026-06-01T15:37:47+00:00`
- Content range: `bytes 0-28963/28964`
- Column count: `8`
- Sample rows inspected: `25`
- Sampled values included: `False`
- Range may be truncated: `False`

### Columns

`full_name`, `first_name`, `last_name`, `constituency`, `party`, `gender`, `member_code`, `uri`

### Likely numeric columns

_none_

### Mapping candidate matches

- Transform `party_counts` / `count_by`
  - Label matches: `party`
  - Value matches: _none_

### Sample column coverage

Raw sampled field values are omitted from this public preview summary by default.

| Column | Non-empty sampled rows | Blank sampled rows |
| --- | ---: | ---: |
| `full_name` | 25 | 0 |
| `first_name` | 25 | 0 |
| `last_name` | 25 | 0 |
| `constituency` | 25 | 0 |
| `party` | 25 | 0 |
| `gender` | 0 | 25 |
| `member_code` | 25 | 0 |
| `uri` | 25 | 0 |
