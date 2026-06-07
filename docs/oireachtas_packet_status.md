# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-07  
**Current packet:** T08 — `silver_member_offices`

This is the compact operational handoff for `docs/oireachtas_unified_data_model_plan.md`. Continue from `main`. Existing legacy pipelines remain untouched while unified replacements are built and validated table-by-table.

## Shared implementation state

- CLI: `python -m extract.oireachtas.build_table`
- Registry: `configs/oireachtas/tables.yml`
- Manual test workflow: `.github/workflows/oireachtas_table_test.yml`
- AWS region: `ca-central-1`
- S3 bucket: `eirepolitic-data`
- Review branch: `oireachtas-review-output`
- Review publishing preserves existing table folders and runs even after table/DQ failure when local review output exists.
- Standard outputs per confirmed table: raw API JSON, partitioned CSV, partitioned Parquet, latest CSV/Parquet pointers, run manifest, and review sample/schema/manifest.

## Completed foundation packets

### F01 — Package/registry skeleton

Confirmed files include `extract/oireachtas/*` foundation files and `configs/oireachtas/*.yml`.

### F02 — S3 and review-branch smoke test

- Run: `26832499568`
- Result: success
- Confirmed S3 PutObject/GetObject and review-branch publishing.

### F03 — API discovery

- Run: `26832847170`
- Result: success
- Endpoints confirmed HTTP 200: `/houses`, `/members`, `/debates`, `/divisions`, `/votes`, `/questions`, `/legislation`, `/parties`, `/constituencies`.
- `/divisions` is canonical; `/votes` remains compatibility fallback.
- API endpoints do not always honour `limit` exactly.

## Confirmed table packets

### T01 — `silver_houses`

- Builder: `extract/oireachtas/table_houses.py`
- Final run: `26847237939`
- Rows: 25
- PK: `house_uri`, unique
- DQ: pass
- Important fix: use `houseCode` for chamber rather than generic `chamberType=house`.

### T02 — `silver_constituencies`

- Builder: `extract/oireachtas/table_constituencies.py`
- Final run: `27069529002`
- Rows: 43 for Dáil 34
- PK: `constituency_uri`, unique
- DQ: pass
- API shape: result wrapper `constituencyOrPanel`; code field `representCode`.
- `house_uri` joins to `silver_houses.house_uri`.

### T03 — `silver_parties`

- Builder: `extract/oireachtas/table_parties.py`
- Final run: `27069711527`
- Rows: 11 for Dáil 34
- PK: `party_uri`, unique
- DQ: pass
- API shape: wrapper `party` plus parent `house`.
- Must pass `chamber=dail` and `house_no=34`; unfiltered endpoint returned older-house rows.

### T04 — `silver_members`

- Builder: `extract/oireachtas/table_members.py`
- Final run: `27070132888`
- Rows: 25 test members for Dáil 34
- PK: `member_code`, unique
- DQ: pass
- API shape: `member.memberships[].membership` with nested `parties[]`, `represents[]`, `committees[]`, and `offices[]`.
- Parser extracts latest party, constituency, and house context.

### T05 — `silver_member_memberships`

- Builder: `extract/oireachtas/table_member_memberships.py`
- Final run: `27070298915`
- Rows: 25
- PK: `membership_id`, unique
- DQ: pass
- Grain: one row per `member.memberships[].membership`.
- API membership URI is used as `membership_id` when present.
- `member_code` joins to `silver_members`; `house_uri` joins to `silver_houses`.

### T06 — `silver_member_parties`

- Builder: `extract/oireachtas/table_member_parties.py`
- Final run: `27097902733`
- Rows: 25
- PK: `member_party_id`, unique
- DQ: pass
- Grain: one row per `membership.parties[].party`.
- `membership_id` joins to `silver_member_memberships`; `member_code` joins to `silver_members`; `party_uri` joins to `silver_parties`.
- Party rows have `party_start=2024-11-29`, open-ended `party_end`, and `is_current=True` for Dáil 34 test sample.

### T07 — `silver_member_constituencies`

- Builder: `extract/oireachtas/table_member_constituencies.py`
- Final run: `27098119595`
- Run number: 17
- Result: success
- Rows: 25
- PK: `member_constituency_id`, unique
- DQ: pass
- Endpoint: `/members?limit=25&chamber=dail&house_no=34`
- Grain: one row per `membership.represents[].represent` record.
- `membership_id` uses the parent membership URI and joins to `silver_member_memberships.membership_id`.
- `member_code` joins to `silver_members.member_code`.
- `constituency_uri` joins to `silver_constituencies.constituency_uri`.
- Represents usually omit their own dates, so parser uses `represent.dateRange` where present and falls back to parent `membership.dateRange`.
- Final sample has populated constituency URI/name, `represent_start=2024-11-29`, open-ended `represent_end`, and `is_current=True` for Dáil 34.
- S3 run ID: `silver_member_constituencies_20260607T162359Z`
- Review:
  - `review/silver_member_constituencies/latest/manifest.json`
  - `review/silver_member_constituencies/latest/sample.csv`

## Next packet

### T08 — `silver_member_offices`

Goal:

- build the time-aware member office/role bridge from `/members`;
- explode `member.memberships[].membership.offices[]` where present;
- write raw JSON, CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample;
- validate `member_office_id`, `membership_id`, `member_code`, `office_uri`, `office_name`, office dates, and current flag.

Expected files:

- `extract/oireachtas/table_member_offices.py`
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `silver_member_offices`
- update this status file after validation

Expected command:

```bash
python -m extract.oireachtas.build_table \
  --table silver_member_offices \
  --mode test \
  --chamber dail \
  --house-no 34 \
  --limit 25 \
  --write-review-sample
```

Handoff instruction:

```text
Continue from main.
Start T08 — silver_member_offices.
Workflow default currently points to silver_member_constituencies.
Use the confirmed member bridge builders as the implementation pattern.
```
