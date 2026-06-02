# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-02  
**Current packet:** F03 — API discovery client  

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

Handoff:

```text
Continue from main.
Next packet after F01: F02 — S3 + review-branch smoke test.
Do not start table builds until F02 and F03 are complete.
```

---

### F02 — S3 + review-branch smoke test

**Status:** confirmed  
**Completed:** 2026-06-02  

Files created/modified:

- `extract/oireachtas/io_s3.py`
- `extract/oireachtas/review.py`
- `extract/oireachtas/build_table.py`
- `.github/workflows/oireachtas_table_test.yml`

Workflow:

```text
Oireachtas Table Test (Manual)
```

Successful run:

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

Review branch object verified by assistant via GitHub tool:

```text
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/_smoke/latest/manifest.json
```

Manifest contents confirmed:

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
- Workflow was patched to force `ca-central-1` to match the bucket/Instagram preview convention.
- Second smoke run `26832499568` succeeded and produced the final accepted evidence.
- Review branch publishing works.
- S3 PutObject/GetObject works for the unified review prefix.

Handoff:

```text
Continue from main.
Next packet: F03 — API discovery client.
Do not start table builds until F03 is complete.
```

---

## Next packet

### F03 — API discovery client

Goal:

- build shared Oireachtas API client;
- discover payload shapes for documented endpoints;
- test `/divisions` and `/votes` behaviour;
- publish endpoint payload-shape summaries to `oireachtas-review-output`.

Expected files:

- `extract/oireachtas/client.py`
- `extract/oireachtas/discovery.py`
- updates to `extract/oireachtas/build_table.py`
- possible updates to `configs/oireachtas/api_params.yml`
- possible updates to `docs/oireachtas_unified_data_model_plan.md`

Expected workflow:

```bash
python -m extract.oireachtas.build_table --table _discovery --mode discover
```
