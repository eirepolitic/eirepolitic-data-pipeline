# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-11  
**Current packet:** G02 — `gold_member_activity_yearly`

This is the compact operational handoff for `docs/oireachtas_unified_data_model_plan.md`. Continue from `main`. Existing legacy pipelines remain untouched while unified replacements are built and validated table-by-table.

## Shared implementation state

- CLI: `python -m extract.oireachtas.build_table`
- Registry: `configs/oireachtas/tables.yml`
- Manual test workflow: `.github/workflows/oireachtas_table_test.yml`
- AWS region: `ca-central-1`
- S3 bucket: `eirepolitic-data`
- Review branch: `oireachtas-review-output`
- Review publishing preserves existing table folders and runs after table/DQ failure when local review output exists.
- Standard confirmed outputs: raw API/source files, partitioned CSV, partitioned Parquet, latest CSV/Parquet pointers, run manifest, review sample/schema/manifest/DQ.

## Completed foundation packets

- **F01** package/registry skeleton.
- **F02** S3 and review-branch smoke test, run `26832499568`, success.
- **F03** API discovery, run `26832847170`, success. Confirmed `/houses`, `/members`, `/debates`, `/divisions`, `/votes`, `/questions`, `/legislation`, `/parties`, `/constituencies`. `/divisions` is canonical; `/votes` is fallback.

## Confirmed table packets

- **T01 — `silver_houses`**: run `26847237939`; 25 rows; PK `house_uri`; DQ pass.
- **T02 — `silver_constituencies`**: run `27069529002`; 43 rows; PK `constituency_uri`; DQ pass.
- **T03 — `silver_parties`**: run `27069711527`; 11 rows; PK `party_uri`; DQ pass.
- **T04 — `silver_members`**: run `27070132888`; 25 rows; PK `member_code`; DQ pass.
- **T05 — `silver_member_memberships`**: run `27070298915`; 25 rows; PK `membership_id`; DQ pass.
- **T06 — `silver_member_parties`**: run `27097902733`; 25 rows; PK `member_party_id`; DQ pass.
- **T07 — `silver_member_constituencies`**: run `27098119595`; 25 rows; PK `member_constituency_id`; DQ pass.
- **T08 — `silver_member_offices`**: run `27098313330`; 77 rows; PK `member_office_id`; DQ pass.
- **T09 — `silver_source_files`**: run `27098621113`; 25 rows; PK `source_file_id`; DQ pass.
- **T10 — `silver_debate_records`**: run `27098769263`; 2 rows; PK `debate_id`; DQ pass.
- **T11 — `silver_debate_sections`**: run `27099679458`; 8 rows; PK `debate_section_id`; DQ pass.
- **T12 — `silver_speeches`**: run `27222202849`; 357 rows; PK `speech_id`; DQ pass; speaker member-code enrichment 344/357 rows, 96.36%.
- **T13 — `silver_divisions`**: run `27222935479`; 3 rows; PK `division_id`; DQ pass.
- **T14 — `silver_division_tallies`**: run `27236879805`; 9 rows; PK `division_tally_id`; DQ pass.
- **T15 — `silver_member_votes`**: run `27291681684`; 512 rows; PK `member_vote_id`; DQ pass.
- **T16 — `silver_questions`**: run `27292008182`; 10 rows; PK `question_id`; DQ pass.
- **T17 — `silver_bills`**: run `27325455277`; 10 rows; PK `bill_id`; DQ pass. Confirmed nested legislation shapes for versions, stages, related docs, sponsors, debates, and events.
- **T18 — `silver_bill_versions`**: run `27326814396`; 10 rows; PK `bill_version_id`; DQ pass; PDF source rows 10, XML source rows 0.
- **T19 — `silver_bill_stages`**: run `27327648268`; 16 rows; PK `bill_stage_id`; DQ pass.
- **T20 — `silver_bill_related_docs`**: run `27328140775`; 1 row; PK `related_doc_id`; DQ pass; T09-compatible source-file IDs pass.
- **T21 — `silver_bill_sponsors`**: run `27328994935`; 36 rows; PK `bill_sponsor_id`; DQ pass.
- **T22 — `silver_bill_debates`**: run `27356675022`; 12 rows; PK `bill_debate_id`; DQ pass.
- **T23 — `silver_bill_events`**: run `27359680226`; 19 rows; PK `bill_event_id`; DQ pass.

### G01 — `gold_current_members`

- Builder: `extract/oireachtas/table_gold_current_members.py`
- CLI/workflow updates:
  - `extract/oireachtas/build_table.py`
  - `.github/workflows/oireachtas_table_test.yml`
- Final run: `27362834154`
- Run number: 41
- Result: success
- Input latest silver rows:
  - `silver_members`: 25
  - `silver_member_memberships`: 25
  - `silver_member_parties`: 25
  - `silver_member_constituencies`: 25
  - `silver_member_offices`: 77
- Current bridge rows selected:
  - memberships: 25
  - parties: 25
  - constituencies: 25
  - office member rows: 42
- Output rows: 10, limited by workflow test default `limit=10`.
- PK: `member_code`, unique
- DQ: pass
- Final run ID: `gold_current_members_20260611T164637Z`.
- Normalized columns:
  - `member_code`
  - `full_name`
  - `party_name`
  - `constituency_name`
  - `house_no`
  - `office_name`
  - `snapshot_date`
- DQ checks passed:
  - row count > 0;
  - required columns present;
  - primary key non-null and unique;
  - full name populated;
  - party name populated;
  - constituency name populated;
  - house number populated;
  - `office_name` optional for members without current office records.
- Review:
  - `review/gold_current_members/latest/manifest.json`
  - `review/gold_current_members/latest/sample.csv`
  - `review/gold_current_members/latest/dq.json`
- Sample includes current members such as Ciarán Ahern, William Aird, Catherine Ardagh, Ivana Bacik, Cathy Bennett, Grace Boland, Richard Boyd Barrett, Tom Brabazon, John Brady, and Brian Brennan.

## Next packet

### G02 — `gold_member_activity_yearly`

Goal:

- add yearly member activity mart from confirmed silver speech and vote tables plus `gold_current_members`;
- build one row per `member_code` + year;
- calculate speech count, debate-day count, division count, votes-cast count, vote participation percentage, Tá/Níl/Staon-style vote counts where labels exist, and deterministic ranks;
- preserve traceable joins to `silver_speeches.speaker_member_code` and `silver_member_votes.member_code`;
- publish CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample;
- validate row count > 0, primary key uniqueness on `member_code + year`, member_code populated, numeric metrics populated, and rank fields populated.

Expected files:

- `extract/oireachtas/table_gold_member_activity_yearly.py`
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `gold_member_activity_yearly`
- update this status file after validation

Suggested test command:

```bash
python -m extract.oireachtas.build_table \
  --table gold_member_activity_yearly \
  --mode test \
  --limit 25 \
  --write-review-sample
```

Handoff instruction:

```text
Continue from main.
Start G02 — gold_member_activity_yearly.
Workflow default currently points to gold_current_members.
Use latest CSV inputs from gold_current_members, silver_speeches, silver_member_votes, and silver_divisions where useful.
```
