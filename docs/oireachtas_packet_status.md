# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-07  
**Current packet:** T09 — `silver_source_files`

This is the compact operational handoff for `docs/oireachtas_unified_data_model_plan.md`. Continue from `main`. Existing legacy pipelines remain untouched while unified replacements are built and validated table-by-table.

## Shared implementation state

- CLI: `python -m extract.oireachtas.build_table`
- Registry: `configs/oireachtas/tables.yml`
- Manual test workflow: `.github/workflows/oireachtas_table_test.yml`
- AWS region: `ca-central-1`
- S3 bucket: `eirepolitic-data`
- Review branch: `oireachtas-review-output`
- Review publishing preserves existing table folders and runs after table/DQ failure when local review output exists.
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

### T07 — `silver_member_constituencies`

- Builder: `extract/oireachtas/table_member_constituencies.py`
- Final run: `27098119595`
- Rows: 25
- PK: `member_constituency_id`, unique
- DQ: pass
- Grain: one row per `membership.represents[].represent`.
- `membership_id` joins to `silver_member_memberships`; `member_code` joins to `silver_members`; `constituency_uri` joins to `silver_constituencies`.
- Represents usually omit dates, so parser falls back to parent membership dates.

### T08 — `silver_member_offices`

- Builder: `extract/oireachtas/table_member_offices.py`
- Initial run: `27098251798`
- Final run: `27098313330`
- Run number: 19
- Result: success
- Raw members: 176 for Dáil 34
- Output rows: 77
- PK: `member_office_id`, unique
- DQ: pass
- Endpoint: `/members?limit=200&chamber=dail&house_no=34`
- Grain: one row per `membership.offices[]` office-history record.
- `membership_id` uses the parent membership URI and joins to `silver_member_memberships.membership_id`.
- `member_code` joins to `silver_members.member_code`.
- Actual office shape is:

```text
officeName.showAs
officeName.uri
dateRange.start
dateRange.end
```

- `officeName.uri` is commonly null, so stable generated `office_uri` and `member_office_id` values are used.
- Initial run produced 70 rows but failed DQ because `officeName` is a nested object rather than text.
- Parser was patched to read nested `officeName.showAs`; final run produced 77 named office rows.
- Sample office values include:
  - Minister for Housing, Local Government and Heritage
  - Minister for Enterprise, Tourism and Employment
  - Minister of State at the Department of Justice
  - Minister of State at the Department of the Taoiseach and other departments
- Office history correctly preserves closed and open-ended date ranges and sets `is_current` accordingly.
- S3 run ID: `silver_member_offices_20260607T163158Z`
- Review:
  - `review/silver_member_offices/latest/manifest.json`
  - `review/silver_member_offices/latest/sample.csv`

## Next packet

### T09 — `silver_source_files`

Goal:

- build a reusable inventory of source XML/PDF/document files discovered in API `formats` fields;
- determine which parent endpoint(s) expose formats consistently, beginning with `/debates`, `/questions`, and `/legislation`;
- normalize format type, source URI/URL, content type, S3 destination key, download status, byte size, and hash metadata;
- create deterministic `source_file_id` values;
- write CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample;
- separate metadata discovery from actual download logic where useful so downstream debate/question/bill tables can reuse the file inventory.

Expected files:

- `extract/oireachtas/table_source_files.py`
- likely updates to `extract/oireachtas/client.py` or a dedicated source-file helper
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `silver_source_files`
- reset workflow test limit from `200` to an appropriate small endpoint-specific value
- update this status file after validation

Handoff instruction:

```text
Continue from main.
Start T09 — silver_source_files.
Workflow default currently points to silver_member_offices with limit=200.
First inspect formats payload shapes from debates, questions, and legislation before finalizing the source-file grain.
```
