# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-10  
**Current packet:** T18 — `silver_bill_versions`

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
- CLI/workflow updates:
  - `extract/oireachtas/build_table.py`
  - `.github/workflows/oireachtas_table_test.yml`
- Final run: `27325455277`
- Run number: 33
- Result: success
- Raw legislation rows: 10
- Output rows: 10
- PK: `bill_id`, unique
- DQ: pass
- Endpoint: `/legislation?chamber=dail&house_no=34&date_start=2025-01-01&date_end=2025-01-31&limit=10`.
- Confirmed API shape:
  - wrapper: `bill`;
  - identity: `bill.uri`;
  - number/year: `bill.billNo`, `bill.billYear`;
  - titles: `bill.shortTitleEn`, `bill.longTitleEn`, `bill.shortTitleGa`, `bill.longTitleGa`;
  - origin house: `bill.originHouse.{uri,showAs}` and `bill.originHouseURI`;
  - bill type: `bill.billType`;
  - status: `bill.status`;
  - method: `bill.method`;
  - latest stage: `bill.mostRecentStage.event`;
  - nested collections: `bill.versions`, `bill.stages`, `bill.events`, `bill.debates`, `bill.relatedDocs`, `bill.sponsors`, and `bill.amendmentLists`.
- All bill IDs, numbers, years, titles, origin house URIs, bill types, statuses, introduced dates, latest-event dates, and source endpoint fields were populated.
- `bill_type_values`: `Public`.
- `status_values`: `Current`, `Lapsed`.
- `origin_house_values`: `Seanad Éireann` for the sample.
- Dates are derived from all parseable dates in the bill payload:
  - `introduced_date` = earliest collected date;
  - `last_event_date` = latest collected date, including `lastUpdated`.
- Confirmed nested shapes for next packets:
  - versions: `bill.versions[].version.{uri,showAs,date,docType,lang,formats}`;
  - stages: `bill.stages[].event.{uri,showAs,stageURI,stageOutcome,stageCompleted,progressStage,dates,house,chamber}`;
  - related docs: `bill.relatedDocs[].relatedDoc.{uri,showAs,date,docType,lang,formats}`;
  - sponsors: `bill.sponsors[].sponsor.{by,as,isPrimary}`;
  - debates: `bill.debates[].{uri,showAs,date,debateSectionId,chamber}`;
  - events: `bill.events[].event.{uri,eventURI,showAs,dates,chamber}`.
- First sample row: Child Maintenance Bill 2026, bill 35/2026, status `Current`, introduced date `2025-01-21`, latest event date `2026-04-27`.
- Final run ID: `silver_bills_20260611T051524Z`.
- Review:
  - `review/silver_bills/latest/manifest.json`
  - `review/silver_bills/latest/sample.csv`
  - `review/silver_bills/latest/dq.json`

## Next packet

### T18 — `silver_bill_versions`

Goal:

- build one row per bill version/document from `bill.versions[].version`;
- normalize `bill_version_id`, `bill_id`, `version_label`, `version_date`, XML/PDF format URIs/URLs, T09-compatible source-file IDs, S3 target keys, and snapshot date;
- support format fields under `version.formats.{pdf,xml}` and keep null formats blank;
- preserve join to `silver_bills.bill_id`;
- publish raw JSON, CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample;
- validate primary-key uniqueness, bill joins, version labels/dates, and at least one source format per version.

Expected files:

- `extract/oireachtas/table_bill_versions.py`
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `silver_bill_versions`
- update this status file after validation

Suggested test command:

```bash
python -m extract.oireachtas.build_table \
  --table silver_bill_versions \
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
Start T18 — silver_bill_versions.
Workflow default currently points to silver_bills.
Use bill.versions[].version from the confirmed T17 payload; keep relatedDocs for a later source-doc packet unless the model is revised.
```
