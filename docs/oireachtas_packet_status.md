# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-10  
**Current packet:** T16 — `silver_questions`

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

- Builder: `extract/oireachtas/table_member_votes.py`
- CLI/workflow updates:
  - `extract/oireachtas/build_table.py`
  - `.github/workflows/oireachtas_table_test.yml`
- Final run: `27291681684`
- Run number: 31
- Result: success
- Raw division rows: 3
- Output member-vote rows: 512
- Expected rows from T14 tallies: 512
- Division count: 3
- Distinct member codes: 172
- PK: `member_vote_id`, unique
- DQ: pass
- Canonical endpoint `/divisions` used; `/votes` fallback not used.
- Grain: one row per `division.tallies.*.members[].member` vote.
- Stable member-vote IDs use a hash of `division_id`, `member_code`, and normalized `vote_code`.
- Confirmed vote codes present in this sample: `ta` and `nil`; `staon` had zero member rows in all three divisions.
- Per-division totals:
  - `vote_164`: 172 rows = 95 Tá + 77 Níl.
  - `vote_2`: 171 rows = 95 Tá + 76 Níl.
  - `vote_3`: 169 rows = 97 Tá + 72 Níl.
- No duplicate member codes within any division.
- Exactly one vote row per member per division.
- All member codes, member names, vote IDs, division IDs, dates, vote codes, and vote labels are populated.
- `party_name_at_vote` and `constituency_name_at_vote` are blank because the division payload does not provide them; these remain for later temporal enrichment.
- Final run ID: `silver_member_votes_20260610T165200Z`.
- Review:
  - `review/silver_member_votes/latest/manifest.json`
  - `review/silver_member_votes/latest/sample.csv`
  - `review/silver_member_votes/latest/dq.json`
- Operational note: several workflow dispatch/write attempts briefly returned GitHub `401 Requires authentication`; retry succeeded without any configuration change. Treat this as a transient action-authentication failure unless it recurs consistently.

## Next packet

### T16 — `silver_questions`

Goal:

- build one row per parliamentary question from `/questions`;
- inspect and normalize question identity, date, number, type/status, subject/text, asker/member references, department/recipient, answer metadata, debate-section links, and source XML/PDF references;
- preserve joins to `silver_members`, `silver_debate_sections`, and `silver_source_files` where the API provides stable identifiers;
- reuse T09 source-file ID hashing for question XML/PDF references;
- publish raw JSON, CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample;
- validate primary-key uniqueness, dates, question text/subject, member references, and source-file joins.

Expected files:

- `extract/oireachtas/table_questions.py`
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `silver_questions`
- update this status file after validation

Suggested test command:

```bash
python -m extract.oireachtas.build_table \
  --table silver_questions \
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
Start T16 — silver_questions.
Workflow default currently points to silver_member_votes.
Inspect the actual /questions wrapper and nested answer/member/format shapes before finalizing parsing.
```
