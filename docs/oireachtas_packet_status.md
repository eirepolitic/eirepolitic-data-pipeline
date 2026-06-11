# Oireachtas Unified Build Packet Status

**Branch:** `main`  
**Last updated:** 2026-06-11  
**Current packet:** G03 — `gold_member_activity_monthly`

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

## Confirmed gold packets

### G01 — `gold_current_members`

- Builder: `extract/oireachtas/table_gold_current_members.py`
- Final run: `27362834154`; run number 41; success.
- Input latest silver rows: members 25, memberships 25, parties 25, constituencies 25, offices 77.
- Output rows: 10, limited by workflow test default `limit=10`.
- PK `member_code`, unique; DQ pass.
- Final run ID: `gold_current_members_20260611T164637Z`.
- Review: `review/gold_current_members/latest/{manifest.json,sample.csv,dq.json}`.

### G02 — `gold_member_activity_yearly`

- Builder: `extract/oireachtas/table_gold_member_activity_yearly.py`
- CLI/workflow updates:
  - `extract/oireachtas/build_table.py`
  - `.github/workflows/oireachtas_table_test.yml`
- Final run: `27363058073`
- Run number: 42
- Result: success
- Input latest rows:
  - `gold_current_members`: 10
  - `silver_speeches`: 357
  - `silver_member_votes`: 512
  - `silver_divisions`: 3
- Speech metric rows: 46
- Vote metric rows: 172
- Division-year rows: 1
- Member-year rows before limit: 173
- Output rows: 10, limited by workflow test default `limit=10`.
- Year values: `2025`
- PK: `member_code`, `year`, unique
- DQ: pass
- Final run ID: `gold_member_activity_yearly_20260611T165033Z`.
- Normalized columns:
  - `member_code`
  - `year`
  - `speech_count`
  - `debate_day_count`
  - `division_count`
  - `votes_cast_count`
  - `vote_participation_pct`
  - `ta_count`
  - `nil_count`
  - `staon_count`
  - `speech_rank`
  - `vote_participation_rank`
  - `snapshot_date`
- DQ checks passed:
  - row count > 0;
  - required columns present;
  - primary key non-null and unique;
  - member code and year populated;
  - numeric metrics populated;
  - rank fields populated.
- Sample leaders include Verona Murphy, Matt Carthy, Paul McAuliffe, Pearse Doherty, Louise O'Reilly, Mary Lou McDonald, Pádraig Mac Lochlainn, Richard Boyd Barrett, Hildegarde Naughton, and Paul Murphy.
- Review:
  - `review/gold_member_activity_yearly/latest/manifest.json`
  - `review/gold_member_activity_yearly/latest/sample.csv`
  - `review/gold_member_activity_yearly/latest/dq.json`

## Next packet

### G03 — `gold_member_activity_monthly`

Goal:

- add monthly member activity mart from confirmed `silver_speeches` and `silver_member_votes` inputs;
- build one row per `member_code` + `year_month`;
- calculate speech count, debate-day count, and votes-cast count;
- publish CSV, Parquet, latest pointers, manifest, schema, DQ, and review sample;
- validate row count > 0, primary key uniqueness on `member_code + year_month`, member code populated, month populated, and numeric metrics populated.

Expected files:

- `extract/oireachtas/table_gold_member_activity_monthly.py`
- update `extract/oireachtas/build_table.py`
- update `.github/workflows/oireachtas_table_test.yml` default to `gold_member_activity_monthly`
- update this status file after validation

Suggested test command:

```bash
python -m extract.oireachtas.build_table \
  --table gold_member_activity_monthly \
  --mode test \
  --limit 25 \
  --write-review-sample
```

Handoff instruction:

```text
Continue from main.
Start G03 — gold_member_activity_monthly.
Workflow default currently points to gold_member_activity_yearly.
Use latest CSV inputs from silver_speeches and silver_member_votes, optionally gold_current_members for current-member grid.
```
