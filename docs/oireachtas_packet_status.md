# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-11  
**Current packet:** T19 — `silver_bill_stages`

This is the compact operational handoff for `docs/oireachtas_unified_data_model_plan.md`. Continue from `main`. Existing legacy pipelines remain untouched while unified replacements are built and validated table-by-table.

## Shared implementation state

- CLI: `python -m extract.oireachtas.build_table`
- Registry: `configs/oireachtas/tables.yml`
- Manual test workflow: `.github/workflows/oireachtas_table_test.yml`
- AWS region: `ca-central-1`
- S3 bucket: `eirepolitic-data`
- Review branch: `oireachtas-review-output`
- Review publishing preserves existing table folders and runs after table/DQ failure when local review output exists.
- Standard confirmed outputs: raw API/source files, partitioned CSV, partitioned Parquet, latest CSV/Parquet pointers, run manifest, review sample/schema/manifest.

## Completed foundation packets

- **F01** package/registry skeleton.
- **F02** S3 and review-branch smoke test, run `26832499568`, success.
- **F03** API discovery, run `26832847170`, success. Confirmed `/houses`, `/members`, `/debates`, `/divisions`, `/votes`, `/questions`, `/legislation`, `/parties`, `/constituencies`. `/divisions` is canonical; `/votes` is fallback.

## Confirmed table packets

### T01 — `silver_houses`
- Run `26847237939`; 25 rows; PK `house_uri`; DQ pass.

### T02 — `silver_constituencies`
- Run `27069529002`; 43 Dáil 34 rows; PK `constituency_uri`; DQ pass.

### T03 — `silver_parties`
- Run `27069711527`; 11 Dáil 34 rows; PK `party_uri`; DQ pass.

### T04 — `silver_members`
- Run `27070132888`; 25 test rows; PK `member_code`; DQ pass.

### T05 — `silver_member_memberships`
- Run `27070298915`; 25 rows; PK `membership_id`; DQ pass.

### T06 — `silver_member_parties`
- Run `27097902733`; 25 rows; PK `member_party_id`; DQ pass.

### T07 — `silver_member_constituencies`
- Run `27098119595`; 25 rows; PK `member_constituency_id`; DQ pass.

### T08 — `silver_member_offices`
- Final run `27098313330`; 77 rows; PK `member_office_id`; DQ pass.

### T09 — `silver_source_files`
- Final run `27098621113`; 25 rows; PK `source_file_id`; DQ pass.

### T10 — `silver_debate_records`
- Run `27098769263`; 2 rows; PK `debate_id`; DQ pass.

### T11 — `silver_debate_sections`
- Run `27099679458`; 8 rows; PK `debate_section_id`; DQ pass.

### T12 — `silver_speeches`
- Final run `27222202849`; 357 rows; PK `speech_id`; DQ pass.
- Speaker member-code enrichment: 344/357 rows, 96.36%.

### T13 — `silver_divisions`
- Final run `27222935479`; 3 rows; PK `division_id`; DQ pass.
- Canonical `/divisions` used; `/votes` fallback not used.

### T14 — `silver_division_tallies`
- Final run `27236879805`; 9 rows; PK `division_tally_id`; DQ pass.
- API tally values matched member-array lengths for all rows.

### T15 — `silver_member_votes`
- Final run `27291681684`; 512 rows; PK `member_vote_id`; DQ pass.
- Expected rows from T14 tallies: 512.
- No duplicate member vote per division.
- `party_name_at_vote` and `constituency_name_at_vote` remain blank because the division payload does not expose them.

### T16 — `silver_questions`
- Final run `27292008182`; 10 rows; PK `question_id`; DQ pass.
- API shape: `question` wrapper, `question.uri`, `question.date`, `question.questionNumber`, `question.questionType`, `question.showAs`, `question.by`, `question.to`, `question.debateSection`, and `question.debateSection.formats`.
- XML source IDs align with T09; answer text and PDF fields are blank because not exposed/null in the API response.

### T17 — `silver_bills`
- Builder: `extract/oireachtas/table_bills.py`
- Final run: `27325455277`
- Final run ID: `silver_bills_20260611T051524Z`
- Rows: 10; PK `bill_id`; DQ pass.
- Endpoint: `/legislation?chamber=dail&house_no=34&date_start=2025-01-01&date_end=2025-01-31&limit=10`.
- Confirmed wrapper: `bill`; identity: `bill.uri`.
- Confirmed nested shapes: `bill.versions`, `bill.stages`, `bill.events`, `bill.debates`, `bill.relatedDocs`, `bill.sponsors`, `bill.amendmentLists`.
- Review:
  - `review/silver_bills/latest/manifest.json`
  - `review/silver_bills/latest/sample.csv`
  - `review/silver_bills/latest/dq.json`

### T18 — `silver_bill_versions`
- Builder: `extract/oireachtas/table_bill_versions.py`
- CLI/workflow updates:
  - `extract/oireachtas/build_table.py`
  - `.github/workflows/oireachtas_table_test.yml`
- Final run: `27326809702`
- Run number: 34
- Result: success
- Raw legislation rows: 10
- Bills with versions: 10
- Output rows: 10
- PK: `bill_version_id`, unique
- DQ: pass
- Endpoint: `/legislation?chamber=dail&house_no=34&date_start=2025-01-01&date_end=2025-01-31&limit=10`.
- Source shape used:
  - wrapper: `bill`;
  - bill join key: `bill.uri` -> `bill_id`;
  - versions: `bill.versions[].version`;
  - fields: `version.uri`, `version.showAs`, `version.date`, `version.docType`, `version.lang`, `version.formats.pdf.uri`, `version.formats.xml.uri`.
- Normalized fields:
  - `bill_version_id`;
  - `bill_id` preserving join to `silver_bills.bill_id`;
  - `version_label`;
  - `version_date`;
  - PDF/XML format URIs and absolute URLs;
  - T09-compatible `source_file_id_pdf` / `source_file_id_xml`;
  - source-file S3 keys;
  - `snapshot_date`.
- Null XML/PDF formats are supported. In the test sample, PDF was populated for all 10 rows and XML was null for all rows.
- Source-file IDs were marked deterministic and compatible with the T09 hashing pattern.
- Final run ID: `silver_bill_versions_20260611T055143Z`.
- Review:
  - `review/silver_bill_versions/latest/manifest.json`
  - `review/silver_bill_versions/latest/sample.csv`
  - `review/silver_bill_versions/latest/dq.json`

## Next packet

### T19 — `silver_bill_stages`

Goal:

- build one row per bill stage/progress event from `bill.stages[].event`;
- normalize `bill_stage_id`, `bill_id`, `stage_name`, `stage_date`, house URI/name, stage outcome, order within bill, and snapshot date;
- preserve join to `silver_bills.bill_id`;
- use confirmed T17 shape: `bill.stages[].event.{uri,showAs,stageURI,stageOutcome,stageCompleted,progressStage,dates,house,chamber}`;
- publish raw JSON, CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample;
- validate row count, primary-key uniqueness, bill joins, stage labels, stage dates, and order fields.

Expected files:

- `extract/oireachtas/table_bill_stages.py`
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `silver_bill_stages`
- update this status file after validation

Suggested test command:

```bash
python -m extract.oireachtas.build_table \
  --table silver_bill_stages \
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
Start T19 — silver_bill_stages.
Workflow default currently points to silver_bill_versions.
Use bill.stages[].event from the confirmed T17 payload.
```
