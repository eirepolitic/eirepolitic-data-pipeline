# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-09  
**Current packet:** T13 — `silver_divisions`

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
- Builder: `extract/oireachtas/table_houses.py`
- Run `26847237939`; 25 rows; PK `house_uri`; DQ pass.

### T02 — `silver_constituencies`
- Builder: `extract/oireachtas/table_constituencies.py`
- Run `27069529002`; 43 Dáil 34 rows; PK `constituency_uri`; DQ pass.

### T03 — `silver_parties`
- Builder: `extract/oireachtas/table_parties.py`
- Run `27069711527`; 11 Dáil 34 rows; PK `party_uri`; DQ pass.

### T04 — `silver_members`
- Builder: `extract/oireachtas/table_members.py`
- Run `27070132888`; 25 test rows; PK `member_code`; DQ pass.

### T05 — `silver_member_memberships`
- Builder: `extract/oireachtas/table_member_memberships.py`
- Run `27070298915`; 25 rows; PK `membership_id`; DQ pass.

### T06 — `silver_member_parties`
- Builder: `extract/oireachtas/table_member_parties.py`
- Run `27097902733`; 25 rows; PK `member_party_id`; DQ pass.

### T07 — `silver_member_constituencies`
- Builder: `extract/oireachtas/table_member_constituencies.py`
- Run `27098119595`; 25 rows; PK `member_constituency_id`; DQ pass.

### T08 — `silver_member_offices`
- Builder: `extract/oireachtas/table_member_offices.py`
- Final run `27098313330`; 77 rows from 176 Dáil 34 members; PK `member_office_id`; DQ pass.
- Actual office name shape: `officeName.showAs`.

### T09 — `silver_source_files`
- Builder: `extract/oireachtas/table_source_files.py`
- Final run `27098621113`; 25 rows; PK `source_file_id`; DQ pass.
- Metadata-only source inventory across debates, questions, and legislation.
- Null-only format containers are skipped.

### T10 — `silver_debate_records`
- Builder: `extract/oireachtas/table_debate_records.py`
- Run `27098769263`; 2 rows; PK `debate_id`; DQ pass.
- XML/PDF source IDs align with T09.

### T11 — `silver_debate_sections`
- Builder: `extract/oireachtas/table_debate_sections.py`
- Run `27099679458`; 8 rows; PK `debate_section_id`; DQ pass.
- Section counts matched API metadata: 6 for 2025-01-23 and 2 for 2025-01-22.

### T12 — `silver_speeches`

- Builders/helpers:
  - `extract/oireachtas/table_speeches.py`
  - `extract/oireachtas/xml_debates.py`
- CLI/workflow updates:
  - `extract/oireachtas/build_table.py`
  - `.github/workflows/oireachtas_table_test.yml`
- Initial run: `27222041321`
- Final run: `27222202849`
- Run number: 25
- Result: success
- Raw debate rows: 2
- Output speech rows: 357
- PK: `speech_id`, unique
- DQ: pass
- Test window: `2025-01-01` to `2025-01-31`, Dáil 34.
- Downloaded and persisted two Akoma Ntoso XML files:
  - 2025-01-23: 276,286 bytes; 132 speeches.
  - 2025-01-22: 103,413 bytes; 225 speeches.
- XML files are stored under deterministic T09-compatible keys in `raw/oireachtas_unified/source_files/debate/...`.
- `source_file_id` values align with T09/T10:
  - `source_file:2e3f8d29d92f6ad336f74a71`
  - `source_file:a7530d36b8e66d9be3c0c883`
- Parser uses namespace-agnostic stdlib XML handling and supports:
  - `debateSection` nesting;
  - top-level section joins while excluding nested `division`, `ta`, `nil`, `staon`, and `prelude` sections;
  - `speech` ordering per debate;
  - speaker labels from `<from>`;
  - member codes from `TLCPerson.href` `/member/id/...` values;
  - clean speech text excluding speaker label and recorded time;
  - deterministic text hashes, word counts, and character counts.
- Speaker member-code enrichment:
  - 344 of 357 rows matched;
  - 96.36% coverage;
  - unmatched rows are mainly collective/interjection speakers such as `#` / `Deputies`.
- `language` populated for all 357 rows as `en` for the requested English extraction.
- All speech rows have populated joins to `silver_debate_records`, `silver_debate_sections`, and `silver_source_files`.
- Final run ID: `silver_speeches_20260609T165808Z`.
- Review:
  - `review/silver_speeches/latest/manifest.json`
  - `review/silver_speeches/latest/sample.csv`

## Next packet

### T13 — `silver_divisions`

Goal:

- build division/vote event rows from canonical `/divisions`;
- use `/votes` only as compatibility fallback if required;
- normalize `division_id`, `vote_id`, date, chamber/house, committee code, subject, outcome, debate links, and result hash;
- preserve joins to `silver_houses`, `silver_debate_records`, and `silver_debate_sections` where supplied;
- publish raw JSON, CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample;
- inspect the actual division wrapper and nested tally/member-vote structures for T14/T15.

Expected files:

- `extract/oireachtas/table_divisions.py`
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `silver_divisions`
- update this status file after validation

Suggested test command:

```bash
python -m extract.oireachtas.build_table \
  --table silver_divisions \
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
Start T13 — silver_divisions.
Workflow default currently points to silver_speeches.
Use /divisions as canonical and retain /votes only as fallback.
```
