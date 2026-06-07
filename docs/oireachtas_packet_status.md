# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-07  
**Current packet:** T12 — `silver_speeches`

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
- Joins: `membership_id` → `silver_member_memberships`; `member_code` → `silver_members`; `party_uri` → `silver_parties`.

### T07 — `silver_member_constituencies`

- Builder: `extract/oireachtas/table_member_constituencies.py`
- Final run: `27098119595`
- Rows: 25
- PK: `member_constituency_id`, unique
- DQ: pass
- Grain: one row per `membership.represents[].represent`.
- Represents usually omit dates, so parser falls back to parent membership dates.

### T08 — `silver_member_offices`

- Builder: `extract/oireachtas/table_member_offices.py`
- Initial run: `27098251798`
- Final run: `27098313330`
- Rows: 77 from 176 Dáil 34 members
- PK: `member_office_id`, unique
- DQ: pass
- Grain: one row per `membership.offices[]` office-history record.
- Actual office shape: `officeName.showAs`, `officeName.uri`, `dateRange.start`, `dateRange.end`.
- `officeName.uri` is commonly null, so stable generated office identifiers are used.

### T09 — `silver_source_files`

- Builder: `extract/oireachtas/table_source_files.py`
- Initial run: `27098586209`
- Final run: `27098621113`
- Rows: 25
- PK: `source_file_id`, unique
- DQ: pass
- Metadata-only inventory; no PDF/XML download yet.
- Test window: `2025-01-01` to `2025-01-31`, `limit=10`.
- Endpoint coverage: `/debates` 4 rows, `/questions` 10 rows, `/legislation` 11 rows.
- Parser skips null-only format containers such as `{pdf:null, xml:null}`.
- `download_status=not_downloaded`; byte/hash/download-time fields remain blank until a download packet.

### T10 — `silver_debate_records`

- Builder: `extract/oireachtas/table_debate_records.py`
- Final run: `27098769263`
- Rows: 2
- PK: `debate_id`, unique
- DQ: pass
- Grain: one row per `debateRecord` result.
- `debate_id` uses `debateRecord.uri`.
- `house_uri`, `house_no=34`, and `house_code=dail` join to `silver_houses`.
- XML/PDF source-file IDs use the same T09 hash formula and were verified to align with T09 output.

### T11 — `silver_debate_sections`

- Builder: `extract/oireachtas/table_debate_sections.py`
- Final run: `27099679458`
- Run number: 23
- Result: success
- Raw debate rows: 2
- Output section rows: 8
- PK: `debate_section_id`, unique
- DQ: pass
- Endpoint: `/debates?chamber_id=/ie/oireachtas/house/dail/34&lang=en&date_start=2025-01-01&date_end=2025-01-31&limit=10`
- Grain: one row per `debateRecord.debateSections[].debateSection`.
- `debate_section_id` uses the section URI when available.
- `debate_id` joins to `silver_debate_records.debate_id`.
- `section_eid` uses `debateSectionId` values such as `dbsect_2`, `dbsect_7`, and `dbsect_19`.
- `section_order` is deterministic and restarts at 1 for each debate.
- `heading` and `show_as` are populated from `showAs` for the confirmed API shape.
- Parent section values were null in this test sample and remain blank.
- Section counts matched API metadata: 6 sections for 2025-01-23 and 2 for 2025-01-22.
- S3 run ID: `silver_debate_sections_20260607T172953Z`
- Review:
  - `review/silver_debate_sections/latest/manifest.json`
  - `review/silver_debate_sections/latest/sample.csv`

## Next packet

### T12 — `silver_speeches`

Goal:

- download or stream the debate XML referenced by `silver_debate_records.source_xml_uri`;
- parse Akoma Ntoso debate XML into atomic speech rows;
- populate `speech_id`, `debate_id`, `debate_section_id`, `debate_date`, `speech_order`, speaker reference/name/member code, match method/confidence, normalized speech text, hashes/counts/language, `source_file_id`, XML source key, and snapshot date;
- preserve joins to `silver_debate_records`, `silver_debate_sections`, `silver_members`, and `silver_source_files`;
- store downloaded XML in the deterministic source-file S3 key where practical and update source metadata in a later reconciliation packet if needed;
- write CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample;
- validate text extraction, section assignment, speaker extraction, stable speech keys, order, and source-file joins.

Expected implementation strategy:

1. Inspect existing legacy XML download/parsing pipelines in the repo for working Akoma Ntoso parsing methods.
2. Add a reusable HTTP binary/text fetch method or dedicated source-file downloader with retries and content checks.
3. Test against one confirmed debate XML first, likely 2025-01-22 or 2025-01-23.
4. Publish raw XML diagnostics and a small speech sample.
5. Patch namespaces, speaker references, section IDs, and text normalization based on actual XML.
6. Expand to both January test debates only after one-file parsing is confirmed.

Expected files:

- `extract/oireachtas/table_speeches.py`
- likely `extract/oireachtas/xml_debates.py` or another shared parser module
- possible update to `extract/oireachtas/client.py` for non-JSON downloads
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `silver_speeches`
- update this status file after validation

Handoff instruction:

```text
Continue from main.
Start T12 — silver_speeches.
Workflow default currently points to silver_debate_sections.
First inspect existing repo XML parsers/downloaders before implementing new Akoma Ntoso parsing logic.
```
