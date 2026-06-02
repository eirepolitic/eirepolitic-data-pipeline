# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-02  
**Current packet:** T01 — `silver_houses`  

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

Additional local checks:

```bash
python -m extract.oireachtas.build_table --list-tables
python -m extract.oireachtas.build_table --table silver_houses --json
```

Result:

```text
returncode 0
```

Notes:

- The Python runtime printed unrelated `artifact_tool` spreadsheet warmup warnings to stderr in the local test environment, but the CLI commands exited successfully with return code 0.
- No legacy extraction/processing/workflow files were modified.
- The F01 branch was merged to `main` in PR #22.

Relevant commits:

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

Manifest confirmed:

```json
{
  "aws_region": "ca-central-1",
  "bucket": "eirepolitic-data",
  "manifest_key": "processed/oireachtas_unified/review/_smoke/latest/manifest.json",
  "mode": "test",
  "review_branch": "oireachtas-review-output",
  "status": "success",
  "table": "_smoke"
}
```

Notes:

- First smoke run `26832405873` also succeeded, but used `AWS_REGION=us-east-2` from repo secrets.
- Workflow was patched to force `ca-central-1`.
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

Config updated:

```text
configs/oireachtas/api_params.yml
endpoint_aliases.divisions.canonical=/divisions
endpoint_aliases.divisions.fallback=/votes
```

Notes:

- Discovery is one-page/payload-shape only, not a data table build.
- `parties?limit=5` returned 11 rows and `constituencies?limit=5` returned 8 rows, so some endpoints do not strictly honour `limit` as expected. Builders must not assume exact page size.
- F03 confirmed enough endpoint shape to begin T01.

Handoff:

```text
Continue from main.
Next packet: T01 — silver_houses.
Workflow defaults currently point to _discovery/discover because they were changed for F03 dispatch-tool limitations.
For T01, update workflow defaults to table=silver_houses and mode=test before dispatching, or dispatch manually with inputs outside this tool.
```

---

## Next packet

### T01 — `silver_houses`

Goal:

- build the first real silver table from `/houses`;
- write CSV, Parquet, schema, manifest, and DQ outputs;
- publish a review sample to `oireachtas-review-output`;
- inspect the sample before marking confirmed.

Expected files:

- likely new table builder module under `extract/oireachtas/`
- updates to `extract/oireachtas/build_table.py`
- possible updates to `.github/workflows/oireachtas_table_test.yml` defaults for dispatch
- updates to this status file after successful run

Expected workflow command:

```bash
python -m extract.oireachtas.build_table --table silver_houses --mode test --limit 25 --write-review-sample
```
