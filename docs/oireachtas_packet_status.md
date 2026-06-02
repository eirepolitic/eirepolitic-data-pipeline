# Oireachtas Unified Build Packet Status

**Branch:** `gpt/docs-oireachtas_unified_data_model_plan.md-a4926f5c`  
**Last updated:** 2026-06-02  
**Current packet:** F02 — S3 + review-branch smoke test  

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
- GitHub branch verification confirmed all expected F01 files exist under `extract/oireachtas/` and `configs/oireachtas/`.

Relevant commits:

- `8b6fec9deed58b1bd7094b5f67efe28499e529f6` — `extract/oireachtas/__init__.py`
- `25bd5b6a621ac1e3e6777883d7a9329264b04dc0` — `extract/oireachtas/normalize.py`
- `bfbdcce303976a53faf839358359ed61346628bb` — `extract/oireachtas/schemas.py`
- `f168282b048661b0a14360ecf1c7f2b71843f053` — `extract/oireachtas/build_table.py`
- `ca28128c737514705f24d71bdf475c2d2b9789f1` — `configs/oireachtas/api_params.yml`
- `f8541fb315e6df34fcdfcf9abb577b6b6c31a06f` — `configs/oireachtas/tables.yml`

Handoff:

```text
Continue from branch gpt/docs-oireachtas_unified_data_model_plan.md-a4926f5c.
Next packet: F02 — S3 + review-branch smoke test.
Do not start table builds until F02 and F03 are complete.
```

---

## Next packet

### F02 — S3 + review-branch smoke test

Goal:

- prove S3 Put/Get on the unified review prefix;
- create/update `oireachtas-review-output` branch;
- publish raw GitHub review URL in workflow summary.

Expected files:

- `extract/oireachtas/io_s3.py`
- `extract/oireachtas/review.py`
- `.github/workflows/oireachtas_table_test.yml`

Expected S3 smoke object:

```text
s3://eirepolitic-data/processed/oireachtas_unified/review/_smoke/latest/manifest.json
```

Expected review branch object:

```text
https://raw.githubusercontent.com/eirepolitic/eirepolitic-data-pipeline/oireachtas-review-output/review/_smoke/latest/manifest.json
```
