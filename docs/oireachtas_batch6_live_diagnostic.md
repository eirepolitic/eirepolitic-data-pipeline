# Oireachtas Batch 6 live refresh diagnostic

- Run ID: 29518153776
- Batch ID: batch6-diagnostic-29518153776-1
- Failed table: gold_content_fact_pool
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
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_members/snapshot_date=2026-07-16/run_id=silver_members_20260716T170331Z/silver_members.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_members/snapshot_date=2026-07-16/run_id=silver_members_20260716T170331Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_members/run_id=silver_members_20260716T170331Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_members/runs/silver_members_20260716T170331Z
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
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_member_memberships/snapshot_date=2026-07-16/run_id=silver_member_memberships_20260716T170334Z/silver_member_memberships.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_member_memberships/snapshot_date=2026-07-16/run_id=silver_member_memberships_20260716T170334Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_member_memberships/run_id=silver_member_memberships_20260716T170334Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_member_memberships/runs/silver_member_memberships_20260716T170334Z
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
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_member_parties/snapshot_date=2026-07-16/run_id=silver_member_parties_20260716T170337Z/silver_member_parties.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_member_parties/snapshot_date=2026-07-16/run_id=silver_member_parties_20260716T170337Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_member_parties/run_id=silver_member_parties_20260716T170337Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_member_parties/runs/silver_member_parties_20260716T170337Z
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
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_member_constituencies/snapshot_date=2026-07-16/run_id=silver_member_constituencies_20260716T170339Z/silver_member_constituencies.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_member_constituencies/snapshot_date=2026-07-16/run_id=silver_member_constituencies_20260716T170339Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_member_constituencies/run_id=silver_member_constituencies_20260716T170339Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_member_constituencies/runs/silver_member_constituencies_20260716T170339Z
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
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_member_offices/snapshot_date=2026-07-16/run_id=silver_member_offices_20260716T170342Z/silver_member_offices.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_member_offices/snapshot_date=2026-07-16/run_id=silver_member_offices_20260716T170342Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_member_offices/run_id=silver_member_offices_20260716T170342Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_member_offices/runs/silver_member_offices_20260716T170342Z
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
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_debate_records/snapshot_date=2026-07-16/run_id=silver_debate_records_20260716T170345Z/silver_debate_records.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_debate_records/snapshot_date=2026-07-16/run_id=silver_debate_records_20260716T170345Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_debate_records/run_id=silver_debate_records_20260716T170345Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_debate_records/runs/silver_debate_records_20260716T170345Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_debate_records/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_debate_sections =====
TABLE=silver_debate_sections
MODE=incremental
ROWS=526
COLUMNS=9
PRIMARY_KEY=debate_section_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_debate_sections/snapshot_date=2026-07-16/run_id=silver_debate_sections_20260716T170350Z/silver_debate_sections.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_debate_sections/snapshot_date=2026-07-16/run_id=silver_debate_sections_20260716T170350Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_debate_sections/run_id=silver_debate_sections_20260716T170350Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_debate_sections/runs/silver_debate_sections_20260716T170350Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_debate_sections/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_speeches =====
TABLE=silver_speeches
MODE=incremental
ROWS=5664
COLUMNS=18
PRIMARY_KEY=speech_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_speeches/snapshot_date=2026-07-16/run_id=silver_speeches_20260716T170353Z/silver_speeches.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_speeches/snapshot_date=2026-07-16/run_id=silver_speeches_20260716T170353Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_speeches/run_id=silver_speeches_20260716T170353Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_speeches/runs/silver_speeches_20260716T170353Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_speeches/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_divisions =====
TABLE=silver_divisions
MODE=incremental
ROWS=52
COLUMNS=14
PRIMARY_KEY=division_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_divisions/snapshot_date=2026-07-16/run_id=silver_divisions_20260716T170408Z/silver_divisions.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_divisions/snapshot_date=2026-07-16/run_id=silver_divisions_20260716T170408Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_divisions/run_id=silver_divisions_20260716T170408Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_divisions/runs/silver_divisions_20260716T170408Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_divisions/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_division_tallies =====
TABLE=silver_division_tallies
MODE=incremental
ROWS=156
COLUMNS=7
PRIMARY_KEY=division_tally_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_division_tallies/snapshot_date=2026-07-16/run_id=silver_division_tallies_20260716T170411Z/silver_division_tallies.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_division_tallies/snapshot_date=2026-07-16/run_id=silver_division_tallies_20260716T170411Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_division_tallies/run_id=silver_division_tallies_20260716T170411Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_division_tallies/runs/silver_division_tallies_20260716T170411Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_division_tallies/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_member_votes =====
TABLE=silver_member_votes
MODE=incremental
ROWS=7642
COLUMNS=11
PRIMARY_KEY=member_vote_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_member_votes/snapshot_date=2026-07-16/run_id=silver_member_votes_20260716T170414Z/silver_member_votes.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_member_votes/snapshot_date=2026-07-16/run_id=silver_member_votes_20260716T170414Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_member_votes/run_id=silver_member_votes_20260716T170414Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_member_votes/runs/silver_member_votes_20260716T170414Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_member_votes/latest/manifest.json
DQ_STATUS=pass
===== TABLE silver_questions =====
TABLE=silver_questions
MODE=incremental
ROWS=8082
COLUMNS=19
PRIMARY_KEY=question_id
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_questions/snapshot_date=2026-07-16/run_id=silver_questions_20260716T170418Z/silver_questions.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_questions/snapshot_date=2026-07-16/run_id=silver_questions_20260716T170418Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_questions/run_id=silver_questions_20260716T170418Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/silver_questions/runs/silver_questions_20260716T170418Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_questions/latest/manifest.json
DQ_STATUS=pass
===== TABLE gold_current_members =====
TABLE=gold_current_members
MODE=incremental
ROWS=174
COLUMNS=7
PRIMARY_KEY=member_code
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/gold_csv/gold_current_members/snapshot_date=2026-07-16/run_id=gold_current_members_20260716T170443Z/gold_current_members.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/gold/gold_current_members/snapshot_date=2026-07-16/run_id=gold_current_members_20260716T170443Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/gold_current_members/run_id=gold_current_members_20260716T170443Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/gold_current_members/runs/gold_current_members_20260716T170443Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/gold_current_members/latest/manifest.json
DQ_STATUS=pass
===== TABLE gold_member_activity_yearly =====
TABLE=gold_member_activity_yearly
MODE=incremental
ROWS=174
COLUMNS=13
PRIMARY_KEY=member_code,year
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/gold_csv/gold_member_activity_yearly/snapshot_date=2026-07-16/run_id=gold_member_activity_yearly_20260716T170445Z/gold_member_activity_yearly.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/gold/gold_member_activity_yearly/snapshot_date=2026-07-16/run_id=gold_member_activity_yearly_20260716T170445Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/gold_member_activity_yearly/run_id=gold_member_activity_yearly_20260716T170445Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/gold_member_activity_yearly/runs/gold_member_activity_yearly_20260716T170445Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/gold_member_activity_yearly/latest/manifest.json
DQ_STATUS=pass
===== TABLE gold_member_activity_monthly =====
TABLE=gold_member_activity_monthly
MODE=incremental
ROWS=348
COLUMNS=6
PRIMARY_KEY=member_code,year_month
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=true
BATCH_ID=batch6-diagnostic-29518153776-1
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/gold_csv/gold_member_activity_monthly/snapshot_date=2026-07-16/run_id=gold_member_activity_monthly_20260716T170447Z/gold_member_activity_monthly.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/gold/gold_member_activity_monthly/snapshot_date=2026-07-16/run_id=gold_member_activity_monthly_20260716T170447Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/gold_member_activity_monthly/run_id=gold_member_activity_monthly_20260716T170447Z.json
REVIEW_LOCAL_DIR=oireachtas_review_output/gold_member_activity_monthly/runs/gold_member_activity_monthly_20260716T170447Z
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/gold_member_activity_monthly/latest/manifest.json
DQ_STATUS=pass
===== TABLE gold_content_fact_pool =====
ERROR: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
```
