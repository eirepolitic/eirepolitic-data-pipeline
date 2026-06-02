# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-02  
**Current packet:** T02 — `silver_constituencies`  

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

Validation performed:

```bash
python -m extract.oireachtas.build_table --help
```

Result:

```text
returncode 0
```

Notes:

- No legacy extraction/processing/workflow files were modified.
- The F01 branch was merged to `main` in PR #22.

Relevant commit:

- `de90c2904b19c3851358af38addeee35cb053cff` — squash merge for plan/F01/F02 base work.

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

S3 smoke object verified by workflow:

```text
s3://eirepolitic-data/processed/oireachtas_unified/review/_smoke/latest/manifest.json
```

Review branch object verified by assistant:

```text
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

Review outputs verified by assistant:

```text
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/_discovery/latest/manifest.json
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/_discovery/latest/sample.csv
```

Discovery result summary:

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
- F03 confirmed enough endpoint shape to begin T01.

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

Earlier T01 run:

```text
run_id=26847165828
conclusion=success
```

That first T01 output was patched because `chamber` used the API's generic `chamberType=house`. Final run uses `houseCode` for `chamber`, producing values like `dail`, `seanad`, and `dail & seanad`.

Review outputs verified by assistant:

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

Sample inspection notes:

- Current rows include `34th Dáil`, `27th Seanad`, and combined `34th Dáil & 27th Seanad`.
- `is_current` correctly appears true for open-ended current houses and false for ended houses.
- `date_start` and `date_end` parse to ISO date strings where present.
- `house_uri` is populated and unique.
- `chamber` now contains useful values, not the generic API value `house`.

Handoff:

```text
Continue from main.
Next packet: T02 — silver_constituencies.
Workflow defaults currently point to table=silver_houses and mode=test. For T02, update workflow defaults to table=silver_constituencies and mode=test before dispatching, or dispatch manually with inputs outside this tool.
```

---

## Next packet

### T02 — `silver_constituencies`

Goal:

- build the constituency dimension from `/constituencies`;
- write CSV, Parquet, schema, manifest, and DQ outputs;
- publish and inspect review sample;
- verify `house_uri` joins are plausible against `silver_houses` where available.

Expected files:

- likely `extract/oireachtas/table_constituencies.py`
- updates to `extract/oireachtas/build_table.py`
- possible update to `.github/workflows/oireachtas_table_test.yml` defaults for dispatch
- updates to this status file after successful run

Expected workflow command:

```bash
python -m extract.oireachtas.build_table --table silver_constituencies --mode test --limit 25 --write-review-sample
```
