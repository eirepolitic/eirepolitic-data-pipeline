# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-07  
**Current packet:** T10 — `silver_debate_records`

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
- Rows: 77 from 176 Dáil 34 members
- PK: `member_office_id`, unique
- DQ: pass
- Grain: one row per `membership.offices[]` office-history record.
- Actual office shape is `officeName.showAs`, `officeName.uri`, `dateRange.start`, `dateRange.end`.
- `officeName.uri` is commonly null, so stable generated office identifiers are used.

### T09 — `silver_source_files`

- Builder: `extract/oireachtas/table_source_files.py`
- Initial run: `27098586209`
- Final run: `27098621113`
- Run number: 21
- Result: success
- Rows: 25
- PK: `source_file_id`, unique
- DQ: pass
- Metadata-only inventory. No PDF/XML download is performed in this packet.
- Test window: `2025-01-01` to `2025-01-31`, `limit=10`.
- Endpoint coverage:
  - `/debates`: 2 raw debate records, 4 file rows.
  - `/questions`: 10 raw question records, 10 file rows.
  - `/legislation`: 10 raw legislation records, 11 file rows.
- Actual `formats` shapes observed:
  - debate: `debateRecord.formats.pdf.uri`, `debateRecord.formats.xml.uri`, plus nested debate-section formats that can be null placeholders.
  - questions: `question.debateSection.formats.xml.uri`; some `pdf` keys are null.
  - legislation: `bill.versions[].version.formats.pdf.uri`, `bill.relatedDocs[].relatedDoc.formats.pdf.uri`, with `xml` keys sometimes null.
- Initial run failed DQ because null-only placeholder containers such as `{pdf:null, xml:null}` were emitted as blank `.bin` rows.
- Parser was patched to skip null-only/locatorless format records and require `format_uri` or `format_url`.
- `download_status` is fixed to `not_downloaded`; `downloaded_at_utc`, `byte_size`, and `etag_or_hash` stay blank until a later download packet.
- S3 target keys are deterministic and sanitized under `raw/oireachtas_unified/source_files/{source_entity_type}/...`.
- S3 run ID: `silver_source_files_20260607T164455Z`
- Review:
  - `review/silver_source_files/latest/manifest.json`
  - `review/silver_source_files/latest/sample.csv`

## Next packet

### T10 — `silver_debate_records`

Goal:

- build debate metadata records from `/debates`;
- normalize debate-level identity, date, chamber/house fields, title/show_as, source XML/PDF URIs/URLs, and source file joins;
- reuse the source-file ID logic from T09 where possible so `source_file_id_xml` and `source_file_id_pdf` align with `silver_source_files`;
- write raw JSON, CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample;
- validate `debate_id`, `debate_uri`, `debate_date`, `house_uri`, `house_no`, source XML/PDF fields, and source file IDs.

Expected files:

- `extract/oireachtas/table_debate_records.py`
- possible shared helper extraction from `table_source_files.py` if reuse becomes worthwhile
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `silver_debate_records`
- update this status file after validation

Expected command:

```bash
python -m extract.oireachtas.build_table \
  --table silver_debate_records \
  --mode test \
  --chamber dail \
  --house-no 34 \
  --date-start 2025-01-01 \
  --date-end 2025-01-31 \
  --limit 10 \
  --write-review-sample
```

Handoff instruction:

```text
Continue from main.
Start T10 — silver_debate_records.
Workflow default currently points to silver_source_files.
Use the confirmed /debates payload and T09 source-file format logic.
```
