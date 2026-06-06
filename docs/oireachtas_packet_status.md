# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-06  
**Current packet:** T03 — `silver_parties`  

This file is a compact handoff/status companion to `docs/oireachtas_unified_data_model_plan.md`.

---

## Completed packets

### F01 — Foundation package skeleton

**Status:** confirmed  
**Completed:** 2026-06-02  

Files created:

- `extract/oireachtas/__init__.py`
- `extract/oireachtas/normalize.py`
- `extract/oireachtas/schemas.py`
- `extract/oireachtas/build_table.py`
- `configs/oireachtas/api_params.yml`
- `configs/oireachtas/tables.yml`

Validation:

```text
python -m extract.oireachtas.build_table --help -> returncode 0
python -m extract.oireachtas.build_table --list-tables -> returncode 0
python -m extract.oireachtas.build_table --table silver_houses --json -> returncode 0
```

Notes:

- No legacy extraction/processing/workflow files were modified.
- F01 branch was merged to `main` in PR #22.

---

### F02 — S3 + review-branch smoke test

**Status:** confirmed  
**Completed:** 2026-06-02  

Files created/modified:

- `extract/oireachtas/io_s3.py`
- `extract/oireachtas/review.py`
- `extract/oireachtas/build_table.py`
- `.github/workflows/oireachtas_table_test.yml`

Successful workflow run:

```text
run_id=26832499568
run_number=2
conclusion=success
url=https://github.com/eirepolitic/eirepolitic-data-pipeline/actions/runs/26832499568
```

Verified outputs:

```text
s3://eirepolitic-data/processed/oireachtas_unified/review/_smoke/latest/manifest.json
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/_smoke/latest/manifest.json
```

Notes:

- Workflow was patched to force `ca-central-1` after the first run inherited `us-east-2` from repo secrets.
- Review branch publishing works.
- S3 PutObject/GetObject works for the unified review prefix.

---

### F03 — API discovery client

**Status:** confirmed  
**Completed:** 2026-06-02  

Files created/modified:

- `extract/oireachtas/client.py`
- `extract/oireachtas/discovery.py`
- `extract/oireachtas/build_table.py`
- `.github/workflows/oireachtas_table_test.yml`
- `configs/oireachtas/api_params.yml`
- `docs/oireachtas_packet_status.md`

Successful workflow run:

```text
run_id=26832847170
run_number=3
conclusion=success
url=https://github.com/eirepolitic/eirepolitic-data-pipeline/actions/runs/26832847170
```

Review outputs verified:

```text
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/_discovery/latest/manifest.json
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/_discovery/latest/sample.csv
```

Discovery summary:

```text
endpoint_count=9
ok_count=9
failed_count=0
```

Endpoints confirmed HTTP 200:

- `/houses`
- `/members`
- `/debates`
- `/divisions`
- `/votes`
- `/questions`
- `/legislation`
- `/parties`
- `/constituencies`

Important discovery finding:

```text
/divisions and /votes both returned HTTP 200 for Dáil 34 January 2025.
Both returned result_wrapper_keys=contextDate,division.
Both returned schema_hash=99138f2da33a4956.
Decision: use /divisions as canonical documented endpoint and keep /votes as compatibility fallback.
```

Notes:

- `parties?limit=5` returned 11 rows and `constituencies?limit=5` returned 8 rows, so builders must not assume exact page size.

---

### T01 — `silver_houses`

**Status:** confirmed  
**Completed:** 2026-06-02  

Files created/modified:

- `extract/oireachtas/table_houses.py`
- `extract/oireachtas/build_table.py`
- `.github/workflows/oireachtas_table_test.yml`
- `docs/oireachtas_packet_status.md`

Successful final workflow run:

```text
run_id=26847237939
run_number=5
conclusion=success
url=https://github.com/eirepolitic/eirepolitic-data-pipeline/actions/runs/26847237939
```

Review outputs verified:

```text
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_houses/latest/manifest.json
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_houses/latest/sample.csv
```

Final manifest summary:

```text
table=silver_houses
mode=test
status=success
dq_status=pass
raw_rows=25
output_rows=25
primary_key=house_uri
primary_key_unique=true
endpoint=/houses
url=https://api.oireachtas.ie/v1/houses?limit=25
run_id=silver_houses_20260602T205114Z
```

Final S3 outputs:

```text
s3://eirepolitic-data/raw/oireachtas_unified/api/houses/snapshot_date=2026-06-02/run_id=silver_houses_20260602T205114Z/page-00000.json
s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_houses/snapshot_date=2026-06-02/run_id=silver_houses_20260602T205114Z/silver_houses.csv
s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_houses/snapshot_date=2026-06-02/run_id=silver_houses_20260602T205114Z/part-00000.parquet
s3://eirepolitic-data/processed/oireachtas_unified/latest/csv/silver_houses.csv
s3://eirepolitic-data/processed/oireachtas_unified/latest/parquet/silver_houses.parquet
s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_houses/run_id=silver_houses_20260602T205114Z.json
s3://eirepolitic-data/processed/oireachtas_unified/review/silver_houses/latest/sample.csv
s3://eirepolitic-data/processed/oireachtas_unified/review/silver_houses/latest/schema.json
s3://eirepolitic-data/processed/oireachtas_unified/review/silver_houses/latest/manifest.json
```

Important fix:

- First T01 run used `chamber=house` from API `chamberType`.
- Parser was patched to use `houseCode`, producing `dail`, `seanad`, and `dail & seanad`.

---

### T02 — `silver_constituencies`

**Status:** confirmed  
**Completed:** 2026-06-06  

Files created/modified:

- `extract/oireachtas/table_constituencies.py`
- `extract/oireachtas/build_table.py`
- `.github/workflows/oireachtas_table_test.yml`
- `docs/oireachtas_packet_status.md`

Successful final workflow run:

```text
run_id=27069529002
run_number=9
conclusion=success
url=https://github.com/eirepolitic/eirepolitic-data-pipeline/actions/runs/27069529002
```

Review outputs verified:

```text
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_constituencies/latest/manifest.json
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_constituencies/latest/sample.csv
```

Final manifest summary:

```text
table=silver_constituencies
mode=test
status=success
dq_status=pass
raw_rows=43
output_rows=43
primary_key=constituency_uri
primary_key_unique=true
endpoint=/constituencies
params=chamber:dail, house_no:34, limit:25
url=https://api.oireachtas.ie/v1/constituencies?limit=25&chamber=dail&house_no=34
run_id=silver_constituencies_20260606T175022Z
```

Final S3 outputs:

```text
s3://eirepolitic-data/raw/oireachtas_unified/api/constituencies/snapshot_date=2026-06-06/run_id=silver_constituencies_20260606T175022Z/page-00000.json
s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_constituencies/snapshot_date=2026-06-06/run_id=silver_constituencies_20260606T175022Z/silver_constituencies.csv
s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_constituencies/snapshot_date=2026-06-06/run_id=silver_constituencies_20260606T175022Z/part-00000.parquet
s3://eirepolitic-data/processed/oireachtas_unified/latest/csv/silver_constituencies.csv
s3://eirepolitic-data/processed/oireachtas_unified/latest/parquet/silver_constituencies.parquet
s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_constituencies/run_id=silver_constituencies_20260606T175022Z.json
s3://eirepolitic-data/processed/oireachtas_unified/review/silver_constituencies/latest/sample.csv
s3://eirepolitic-data/processed/oireachtas_unified/review/silver_constituencies/latest/schema.json
s3://eirepolitic-data/processed/oireachtas_unified/review/silver_constituencies/latest/manifest.json
```

Important fixes/discoveries:

- Initial parser produced one blank generated row because `/constituencies` uses `constituencyOrPanel`, not `constituency`.
- Workflow was patched to publish review output even when table DQ fails, so parser diagnostics can be inspected from the review branch.
- Parser was patched to handle `constituencyOrPanel` and `representCode`.
- CLI was patched to pass `chamber` and `house_no` filters into the constituency builder.
- Final sample contains 43 Dáil 34 constituencies, including Cork South-Central, Roscommon-Galway, Tipperary North, Mayo, Dublin Mid-West, Clare, Meath East, Dún Laoghaire, Offaly, and Dublin Central.
- `house_uri` points to `https://data.oireachtas.ie/ie/oireachtas/house/dail/34`, matching the confirmed `silver_houses` key.
- `date_start`/`date_end` are blank in this API payload; `is_current=True` is expected for open-ended rows.

Handoff:

```text
Continue from main.
Next packet: T03 — silver_parties.
Workflow defaults currently point to table=silver_constituencies and mode=test. For T03, update workflow defaults to table=silver_parties and mode=test before dispatching, or dispatch manually with inputs outside this tool.
```

---

## Next packet

### T03 — `silver_parties`

Goal:

- build the party dimension from `/parties`;
- write CSV, Parquet, schema, manifest, and DQ outputs;
- publish and inspect review sample;
- verify party URI/name/code fields are populated and primary key is unique.

Expected files:

- likely `extract/oireachtas/table_parties.py`
- updates to `extract/oireachtas/build_table.py`
- possible update to `.github/workflows/oireachtas_table_test.yml` defaults for dispatch
- updates to this status file after successful run

Expected workflow command:

```bash
python -m extract.oireachtas.build_table --table silver_parties --mode test --limit 25 --write-review-sample
```
