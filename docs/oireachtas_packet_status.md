# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-10  
**Current packet:** T17 — `silver_bills`

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

- Builder: `extract/oireachtas/table_questions.py`
- CLI/workflow updates:
  - `extract/oireachtas/build_table.py`
  - `.github/workflows/oireachtas_table_test.yml`
- Final run: `27292008182`
- Run number: 32
- Result: success
- Raw question rows: 10
- Output rows: 10
- PK: `question_id`, unique
- DQ: pass
- Endpoint: `/questions?chamber=dail&house_no=34&date_start=2025-01-01&date_end=2025-01-31&limit=10`.
- Confirmed API shape:
  - wrapper: `question`;
  - identity: `question.uri`;
  - date/number/type: `question.date`, `question.questionNumber`, `question.questionType`;
  - text: `question.showAs`;
  - asker: `question.by.{memberCode,showAs,uri}`;
  - recipient: `question.to.{showAs,roleCode,roleType,uri}`;
  - debate section: `question.debateSection.{uri,debateSectionId,showAs}`;
  - formats: `question.debateSection.formats.{xml,pdf}`.
- All 10 rows were `written` questions dated 2025-01-22.
- All question IDs, dates, numbers, texts, asker member codes/names, recipients, debate-section IDs, XML URIs and XML source-file IDs were populated.
- XML source-file IDs use the same T09 hash formula and matched the known T09 sample IDs, including:
  - `pq_1` → `source_file:319b4198732329ee813cf7e4`;
  - `pq_2` → `source_file:a478282a299281cb94f781b3`;
  - `pq_3` → `source_file:75d7aaebfb0dfe95e5245fc9`;
  - `pq_4` → `source_file:213f2c82a61debdb4586bd48`;
  - `pq_5` → `source_file:58dd1ddab0310b6eaaa2318e`;
  - `pq_6` → `source_file:c99911fbbe6b272b50adb08b`.
- The API response did not include answer text, so `answer_text` is blank for all 10 rows.
- PDF formats were null, so PDF URI/URL/source-file fields are blank.
- Question debate-section URIs point to `/writtens/dbsect_*`; they are stable source identifiers but are not present in the T11 oral/main debate-section test sample. A later written-question XML packet may expand section coverage and answer text extraction.
- Final run ID: `silver_questions_20260610T165735Z`.
- Review:
  - `review/silver_questions/latest/manifest.json`
  - `review/silver_questions/latest/sample.csv`
  - `review/silver_questions/latest/dq.json`

## Next packet

### T17 — `silver_bills`

Goal:

- build one row per bill from `/legislation`;
- normalize bill identity, number/year, titles, origin house, bill type, status, introduction date, and latest event date;
- preserve joins to `silver_houses` through `origin_house_uri`;
- inspect sponsors, stages, versions, debates, related documents, and events for T18/T19;
- publish raw JSON, CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample;
- validate primary-key uniqueness, bill number/year/title, origin house, status, and date fields.

Expected files:

- `extract/oireachtas/table_bills.py`
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `silver_bills`
- update this status file after validation

Suggested test command:

```bash
python -m extract.oireachtas.build_table \
  --table silver_bills \
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
Start T17 — silver_bills.
Workflow default currently points to silver_questions.
Use /legislation and inspect actual bill/date/event/version shapes before finalizing parsing.
```
