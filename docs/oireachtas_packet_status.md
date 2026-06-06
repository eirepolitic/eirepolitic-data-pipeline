# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-06  
**Current packet:** T05 — `silver_member_memberships`  

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

Important finding:

```text
/divisions and /votes both returned HTTP 200 for Dáil 34 January 2025.
Both returned result_wrapper_keys=contextDate,division.
Both returned schema_hash=99138f2da33a4956.
Decision: use /divisions as canonical documented endpoint and keep /votes as compatibility fallback.
```

Notes:

- Some endpoints do not strictly honour `limit`, so builders must not assume exact page size.

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

Important fixes/discoveries:

- Initial parser produced one blank generated row because `/constituencies` uses `constituencyOrPanel`, not `constituency`.
- Workflow was patched to publish review output even when table DQ fails, so parser diagnostics can be inspected from the review branch.
- Parser was patched to handle `constituencyOrPanel` and `representCode`.
- CLI was patched to pass `chamber` and `house_no` filters into the constituency builder.
- Final sample contains 43 Dáil 34 constituencies.
- `house_uri` points to `https://data.oireachtas.ie/ie/oireachtas/house/dail/34`, matching the confirmed `silver_houses` key.

---

### T03 — `silver_parties`

**Status:** confirmed  
**Completed:** 2026-06-06  

Files created/modified:

- `extract/oireachtas/table_parties.py`
- `extract/oireachtas/build_table.py`
- `.github/workflows/oireachtas_table_test.yml`
- `docs/oireachtas_packet_status.md`

Successful final workflow run:

```text
run_id=27069711527
run_number=12
conclusion=success
url=https://github.com/eirepolitic/eirepolitic-data-pipeline/actions/runs/27069711527
```

Review outputs verified:

```text
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_parties/latest/manifest.json
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_parties/latest/sample.csv
```

Final manifest summary:

```text
table=silver_parties
mode=test
status=success
dq_status=pass
raw_rows=11
output_rows=11
primary_key=party_uri
primary_key_unique=true
endpoint=/parties
params=chamber:dail, house_no:34, limit:25
url=https://api.oireachtas.ie/v1/parties?limit=25&chamber=dail&house_no=34
run_id=silver_parties_20260606T175826Z
write_errors=[]
```

Important fixes/discoveries:

- `/parties` uses wrapper shape `party` plus parent `house`.
- Unfiltered `/parties` returned Dáil 31 rows, so the builder and CLI were patched to pass `chamber` and `house_no` filters.
- Final sample contains 11 Dáil 34 party rows.
- Workflow review publishing was patched to preserve existing review folders instead of wiping the branch each run.

---

### T04 — `silver_members`

**Status:** confirmed  
**Completed:** 2026-06-06  

Files created/modified:

- `extract/oireachtas/table_members.py`
- `extract/oireachtas/build_table.py`
- `.github/workflows/oireachtas_table_test.yml`
- `docs/oireachtas_packet_status.md`

Successful final workflow run:

```text
run_id=27070132888
run_number=14
conclusion=success
url=https://github.com/eirepolitic/eirepolitic-data-pipeline/actions/runs/27070132888
```

Earlier T04 run:

```text
run_id=27070101440
run_number=13
conclusion=success
```

The first T04 run passed mechanically but left `latest_party_name` and `latest_constituency_name` blank. Parser was patched to extract party and constituency from `membership.parties[]` and `membership.represents[]` arrays.

Review outputs verified:

```text
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_members/latest/manifest.json
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/silver_members/latest/sample.csv
```

Final manifest summary:

```text
table=silver_members
mode=test
status=success
dq_status=pass
raw_rows=25
output_rows=25
primary_key=member_code
primary_key_unique=true
endpoint=/members
params=chamber:dail, house_no:34, limit:25
url=https://api.oireachtas.ie/v1/members?limit=25&chamber=dail&house_no=34
run_id=silver_members_20260606T181728Z
write_errors=[]
```

Final S3 outputs:

```text
s3://eirepolitic-data/raw/oireachtas_unified/api/members/snapshot_date=2026-06-06/run_id=silver_members_20260606T181728Z/page-00000.json
s3://eirepolitic-data/processed/oireachtas_unified/silver_csv/silver_members/snapshot_date=2026-06-06/run_id=silver_members_20260606T181728Z/silver_members.csv
s3://eirepolitic-data/processed/oireachtas_unified/silver/silver_members/snapshot_date=2026-06-06/run_id=silver_members_20260606T181728Z/part-00000.parquet
s3://eirepolitic-data/processed/oireachtas_unified/latest/csv/silver_members.csv
s3://eirepolitic-data/processed/oireachtas_unified/latest/parquet/silver_members.parquet
s3://eirepolitic-data/processed/oireachtas_unified/manifests/silver_members/run_id=silver_members_20260606T181728Z.json
s3://eirepolitic-data/processed/oireachtas_unified/review/silver_members/latest/sample.csv
s3://eirepolitic-data/processed/oireachtas_unified/review/silver_members/latest/schema.json
s3://eirepolitic-data/processed/oireachtas_unified/review/silver_members/latest/manifest.json
```

Important fixes/discoveries:

- `/members` uses wrapper shape `member` and nested `member.memberships[].membership`.
- Member membership context contains nested arrays: `parties[]`, `represents[]`, `committees[]`, and `offices[]`.
- Parser now extracts current/latest party and constituency from those arrays.
- Final sample includes populated `member_code`, `member_uri`, names, `latest_party_name`, `latest_constituency_name`, and `latest_house_no=34`.
- Example final sample rows include Ciarán Ahern/Labour Party/Dublin South-West, William Aird/Fine Gael/Laois, Catherine Ardagh/Fianna Fáil/Dublin South-Central, and Ivana Bacik/Labour Party/Dublin Bay South.
- Some stable member codes can retain historical chamber letters, e.g. Senate-origin member codes, while the latest membership context correctly points to Dáil 34.

Handoff:

```text
Continue from main.
Next packet: T05 — silver_member_memberships.
Workflow defaults currently point to table=silver_members and mode=test. For T05, update workflow defaults to table=silver_member_memberships and mode=test before dispatching, or dispatch manually with inputs outside this tool.
```

---

## Next packet

### T05 — `silver_member_memberships`

Goal:

- build the time-aware member-to-house bridge from `/members`;
- write CSV, Parquet, schema, manifest, and DQ outputs;
- publish and inspect review sample;
- verify `membership_id`, `member_code`, `house_uri`, `house_no`, `house_code`, `membership_start`, `membership_end`, and `is_current` are populated correctly.

Expected files:

- likely `extract/oireachtas/table_member_memberships.py`
- updates to `extract/oireachtas/build_table.py`
- possible update to `.github/workflows/oireachtas_table_test.yml` defaults for dispatch
- updates to this status file after successful run

Expected workflow command:

```bash
python -m extract.oireachtas.build_table --table silver_member_memberships --mode test --limit 25 --write-review-sample
```
