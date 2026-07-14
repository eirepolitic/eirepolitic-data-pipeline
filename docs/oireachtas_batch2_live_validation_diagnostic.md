# Oireachtas Batch 2 live validation diagnostic

- Run ID: 29310980032
- Commit: cf02f9280c941826bcc6618560aa6deca42734ab
- Exit code: 0
- Rows: 176
- Production publishing enabled: false

```text
TABLE=silver_members
MODE=full
ROWS=176
COLUMNS=15
PRIMARY_KEY=member_code
PRIMARY_KEY_UNIQUE=true
PUBLISH_LATEST=false
CSV_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_members/snapshot_date=2026-07-14/run_id=silver_members_20260714T061923Z/silver_members.csv
PARQUET_KEY=s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_members/snapshot_date=2026-07-14/run_id=silver_members_20260714T061923Z/part-00000.parquet
MANIFEST_KEY=s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_members/run_id=silver_members_20260714T061923Z.json
REVIEW_LOCAL_DIR=artifacts/oireachtas-batch2-live/review/silver_members/latest
REVIEW_SAMPLE_RAW_URL=https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_members/latest/manifest.json
DQ_STATUS=pass
```
