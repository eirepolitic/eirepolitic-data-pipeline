# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-09  
**Current packet:** T15 — `silver_member_votes`

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
- Final run: `27222935479`
- Raw rows: 3
- Output rows: 3
- PK: `division_id`, unique
- DQ: pass
- Canonical endpoint `/divisions` used; `/votes` fallback not used.
- Confirmed event subject, outcome, house, debate and debate-section parsing.
- `division.debate.debateSection` is a scalar EID; parser derives the full T11-compatible section URI.
- Confirmed tally/member shape under `division.tallies.{taVotes,nilVotes,staonVotes}`.

### T14 — `silver_division_tallies`

- Builder: `extract/oireachtas/table_division_tallies.py`
- CLI/workflow updates:
  - `extract/oireachtas/build_table.py`
  - `.github/workflows/oireachtas_table_test.yml`
- Final run: `27236879805`
- Run number: 30
- Result: success
- Raw division rows: 3
- Output tally rows: 9
- Division count: 3
- PK: `division_tally_id`, unique
- DQ: pass
- Canonical endpoint `/divisions` used; `/votes` fallback not used.
- Grain: one row per division and tally category.
- Confirmed standard categories for every division:
  - `ta` / `yes` / `Tá`
  - `nil` / `no` / `Níl`
  - `staon` / `abstain` / `Staon`
- Stable tally IDs use a hash of `division_id` and normalized `vote_code`.
- Counts validated as non-negative.
- API `tally` values matched `members[]` lengths for all 9 rows; mismatch list was empty.
- Test counts:
  - `vote_164`: Tá 95, Níl 77, Staon 0.
  - `vote_2`: Tá 95, Níl 76, Staon 0.
  - `vote_3`: Tá 97, Níl 72, Staon 0.
- Generic category fallback is supported for future API categories beyond the three confirmed values.
- Final run ID: `silver_division_tallies_20260609T212658Z`.
- Review:
  - `review/silver_division_tallies/latest/manifest.json`
  - `review/silver_division_tallies/latest/sample.csv`
  - `review/silver_division_tallies/latest/dq.json`

## Next packet

### T15 — `silver_member_votes`

Goal:

- build one row per member vote from `division.tallies.*.members[].member`;
- normalize `member_vote_id`, `division_id`, `vote_id`, `division_date`, `member_code`, `member_name`, `vote_code`, `vote_label`, party/constituency-at-vote fields, and snapshot date;
- use confirmed normalized vote codes `ta`, `nil`, and `staon`, while supporting additional categories generically;
- preserve joins to `silver_divisions.division_id` and `silver_members.member_code`;
- derive deterministic member-vote IDs from division, member, and vote code;
- validate row counts equal the sum of T14 tally counts, member codes are populated and unique within a division, and each member has at most one vote category per division;
- populate party and constituency at vote only where the division payload provides them; otherwise leave blank for a later temporal enrichment join;
- publish raw JSON, CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample.

Expected files:

- `extract/oireachtas/table_member_votes.py`
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `silver_member_votes`
- update this status file after validation

Suggested test command:

```bash
python -m extract.oireachtas.build_table \
  --table silver_member_votes \
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
Start T15 — silver_member_votes.
Workflow default currently points to silver_division_tallies.
Use division.tallies.*.members[].member from the confirmed T13/T14 payload.
```
