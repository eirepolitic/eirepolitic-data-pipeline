# Oireachtas Batch 6 live refresh diagnostic

- Run ID: 29356769222
- Batch ID: batch6-diagnostic-29356769222-1
- Failed table: silver_questions
- Exit code: 1

```text
===== TABLE silver_members =====
TABLE=silver_members
MODE=incremental
ROWS=176
COLUMNS=15
PRIMARY_KEY=member_code
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29356769222-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_members/snapshot_date=2026-07-14/run_id=silver_members_20260714T181055Z/silver_members.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_members/snapshot_date=2026-07-14/run_id=silver_members_20260714T181055Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_members/run_id=silver_members_20260714T181055Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_members/runs/silver_members_20260714T181055Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_members/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_member_memberships =====
TABLE=silver_member_memberships
MODE=incremental
ROWS=176
COLUMNS=12
PRIMARY_KEY=membership_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29356769222-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_member_memberships/snapshot_date=2026-07-14/run_id=silver_member_memberships_20260714T181059Z/silver_member_memberships.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_member_memberships/snapshot_date=2026-07-14/run_id=silver_member_memberships_20260714T181059Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_member_memberships/run_id=silver_member_memberships_20260714T181059Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_member_memberships/runs/silver_member_memberships_20260714T181059Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_member_memberships/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_member_parties =====
TABLE=silver_member_parties
MODE=incremental
ROWS=178
COLUMNS=9
PRIMARY_KEY=member_party_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29356769222-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_member_parties/snapshot_date=2026-07-14/run_id=silver_member_parties_20260714T181103Z/silver_member_parties.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_member_parties/snapshot_date=2026-07-14/run_id=silver_member_parties_20260714T181103Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_member_parties/run_id=silver_member_parties_20260714T181103Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_member_parties/runs/silver_member_parties_20260714T181103Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_member_parties/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_member_constituencies =====
TABLE=silver_member_constituencies
MODE=incremental
ROWS=176
COLUMNS=9
PRIMARY_KEY=member_constituency_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29356769222-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_member_constituencies/snapshot_date=2026-07-14/run_id=silver_member_constituencies_20260714T181108Z/silver_member_constituencies.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_member_constituencies/snapshot_date=2026-07-14/run_id=silver_member_constituencies_20260714T181108Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_member_constituencies/run_id=silver_member_constituencies_20260714T181108Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_member_constituencies/runs/silver_member_constituencies_20260714T181108Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_member_constituencies/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_member_offices =====
TABLE=silver_member_offices
MODE=incremental
ROWS=77
COLUMNS=9
PRIMARY_KEY=member_office_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29356769222-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_member_offices/snapshot_date=2026-07-14/run_id=silver_member_offices_20260714T181111Z/silver_member_offices.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_member_offices/snapshot_date=2026-07-14/run_id=silver_member_offices_20260714T181111Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_member_offices/run_id=silver_member_offices_20260714T181111Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_member_offices/runs/silver_member_offices_20260714T181111Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_member_offices/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_debate_records =====
TABLE=silver_debate_records
MODE=incremental
ROWS=16
COLUMNS=17
PRIMARY_KEY=debate_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29356769222-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_debate_records/snapshot_date=2026-07-14/run_id=silver_debate_records_20260714T181116Z/silver_debate_records.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_debate_records/snapshot_date=2026-07-14/run_id=silver_debate_records_20260714T181116Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_debate_records/run_id=silver_debate_records_20260714T181116Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_debate_records/runs/silver_debate_records_20260714T181116Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_debate_records/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_debate_sections =====
TABLE=silver_debate_sections
MODE=incremental
ROWS=503
COLUMNS=9
PRIMARY_KEY=debate_section_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29356769222-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_debate_sections/snapshot_date=2026-07-14/run_id=silver_debate_sections_20260714T181121Z/silver_debate_sections.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_debate_sections/snapshot_date=2026-07-14/run_id=silver_debate_sections_20260714T181121Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_debate_sections/run_id=silver_debate_sections_20260714T181121Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_debate_sections/runs/silver_debate_sections_20260714T181121Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_debate_sections/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_speeches =====
TABLE=silver_speeches
MODE=incremental
ROWS=5618
COLUMNS=18
PRIMARY_KEY=speech_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29356769222-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_speeches/snapshot_date=2026-07-14/run_id=silver_speeches_20260714T181127Z/silver_speeches.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_speeches/snapshot_date=2026-07-14/run_id=silver_speeches_20260714T181127Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_speeches/run_id=silver_speeches_20260714T181127Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_speeches/runs/silver_speeches_20260714T181127Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_speeches/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_divisions =====
TABLE=silver_divisions
MODE=incremental
ROWS=57
COLUMNS=14
PRIMARY_KEY=division_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29356769222-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_divisions/snapshot_date=2026-07-14/run_id=silver_divisions_20260714T181138Z/silver_divisions.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_divisions/snapshot_date=2026-07-14/run_id=silver_divisions_20260714T181138Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_divisions/run_id=silver_divisions_20260714T181138Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_divisions/runs/silver_divisions_20260714T181138Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_divisions/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_division_tallies =====
TABLE=silver_division_tallies
MODE=incremental
ROWS=171
COLUMNS=7
PRIMARY_KEY=division_tally_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29356769222-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_division_tallies/snapshot_date=2026-07-14/run_id=silver_division_tallies_20260714T181142Z/silver_division_tallies.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_division_tallies/snapshot_date=2026-07-14/run_id=silver_division_tallies_20260714T181142Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_division_tallies/run_id=silver_division_tallies_20260714T181142Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_division_tallies/runs/silver_division_tallies_20260714T181142Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_division_tallies/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_member_votes =====
TABLE=silver_member_votes
MODE=incremental
ROWS=8402
COLUMNS=11
PRIMARY_KEY=member_vote_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29356769222-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_member_votes/snapshot_date=2026-07-14/run_id=silver_member_votes_20260714T181147Z/silver_member_votes.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_member_votes/snapshot_date=2026-07-14/run_id=silver_member_votes_20260714T181147Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_member_votes/run_id=silver_member_votes_20260714T181147Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_member_votes/runs/silver_member_votes_20260714T181147Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_member_votes/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_questions =====
ERROR: Failed to fetch /questions: Pagination failed on page 51: HTTPError: 422 Client Error: Unknown for url: https://api.oireachtas.ie/v1/questions?chamber=dail&house_no=34&date_start=2026-06-09&date_end=2026-07-14&limit=200&skip=10000
```
