# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-09  
**Current packet:** T14 — `silver_division_tallies`

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
- Final run: `27222202849`
- Raw debate rows: 2
- Output speech rows: 357
- PK: `speech_id`, unique
- DQ: pass
- Speaker member-code enrichment: 344 of 357 rows, 96.36%.
- XML files persisted under deterministic T09-compatible S3 keys.
- Source IDs align with T09/T10.

### T13 — `silver_divisions`

- Builder: `extract/oireachtas/table_divisions.py`
- CLI/workflow updates:
  - `extract/oireachtas/build_table.py`
  - `.github/workflows/oireachtas_table_test.yml`
- Initial runtime-error run: `27222580162`
- First successful payload run: `27222697330`
- Compact-diagnostics run: `27222821976`
- Final run: `27222935479`
- Run number: 29
- Result: success
- Raw rows: 3
- Output rows: 3
- PK: `division_id`, unique
- DQ: pass
- Test window: `2025-01-01` to `2025-01-31`, Dáil 34.
- Canonical endpoint used: `/divisions`.
- `/votes` fallback was not used.
- Corrected runtime import from `ResponseSummary` to `ApiResponseSummary`.
- Confirmed API shape:
  - result wrapper: `division`;
  - stable event URI: `division.uri`;
  - vote ID: `division.voteId`;
  - date/time: `division.date`, `division.datetime`;
  - house: `division.house.{uri,houseNo,houseCode,committeeCode}`;
  - chamber: `division.chamber.{uri,showAs}`;
  - event subject: `division.subject.showAs`;
  - outcome: `division.outcome`;
  - debate: `division.debate.{uri,showAs,debateSection}`.
- `division.debate.debateSection` is a scalar section EID such as `dbsect_2`; the parser derives the full URI matching `silver_debate_sections.debate_section_id`.
- Correct event subjects and section joins validated for:
  - `vote_164` → `dbsect_2`;
  - `vote_2` → `dbsect_7`;
  - `vote_3` → `dbsect_13`.
- Nested tally shape confirmed for T14/T15:
  - `division.tallies.taVotes.{showAs,tally,members[]}`;
  - `division.tallies.nilVotes.{showAs,tally,members[]}`;
  - `division.tallies.staonVotes.{showAs,tally,members[]}`;
  - each member entry wraps `member.{memberCode,showAs,uri}`.
- First test division counts: Tá 95, Níl 77, Staon 0.
- Final run ID: `silver_divisions_20260609T171109Z`.
- Review:
  - `review/silver_divisions/latest/manifest.json`
  - `review/silver_divisions/latest/sample.csv`
  - `review/silver_divisions/latest/dq.json`

## Next packet

### T14 — `silver_division_tallies`

Goal:

- build one row per division and vote category from `division.tallies`;
- normalize `division_tally_id`, `division_id`, `vote_code`, `vote_label`, `show_as`, `member_count`, and snapshot date;
- support confirmed categories `taVotes`, `nilVotes`, and `staonVotes` without hard-failing if additional categories appear;
- preserve exact join to `silver_divisions.division_id`;
- validate deterministic IDs, three category rows per standard division, non-negative counts, and equality between API `tally` and `members[]` length where applicable;
- publish raw JSON, CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample.

Expected files:

- `extract/oireachtas/table_division_tallies.py`
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `silver_division_tallies`
- update this status file after validation

Suggested test command:

```bash
python -m extract.oireachtas.build_table \
  --table silver_division_tallies \
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
Start T14 — silver_division_tallies.
Workflow default currently points to silver_divisions.
Use the confirmed division.tallies shape from T13.
```
